"""Registry-first Gateway, Query Normalizer, 두 제한 Writer를 쓰는 LangGraph.

모든 입력은 strict Gateway의 세 route 중 하나로 분류된다. Registry 초안은
``competency_query``의 결정적 Query Normalizer가 active registry로 확정하고,
범위 밖 질문은 내용을 답하지 않는 Scope Writer가 처리한다. 두 Writer의 출력은
모두 전체 검증과 checkpoint 반영이 끝난 뒤에만 공개된다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import ExitStack, asynccontextmanager, contextmanager
from functools import lru_cache
from typing import Any, Literal, NotRequired

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from competency_query import (
    GroundedAnswerContext,
    ItemField,
    MAX_ANSWER_CHARS,
    NormalizationOutcome,
    ParsedRegistryQuery,
    QueryIntent,
    RegistryQueryPlan,
    RegistryQueryResult,
    build_grounded_answer_context,
    execute_registry_query,
    normalize_registry_query,
    render_grounded_fallback,
    validate_grounded_answer,
    validate_parsed_query,
)
from competency_registry import RegistrySnapshot, load_active_registry
from llm_gateway import (
    FIXED_FAILURE_MESSAGE,
    MAX_LLM_API_CALLS_PER_TURN,
    GatewayDecision,
    LlmCallBudget,
    MetaResponseDraft,
    MetaRouteDecision,
    ModelRole,
    OutOfScopeResponseDraft,
    OutOfScopeRouteDecision,
    RegistryRouteDecision,
    ScopeTopic,
    ainvoke_with_budget,
    astream_with_budget,
    capability_manifest_for_prompt,
    create_chat_model,
    invoke_with_budget,
    recent_conversation_context,
    selected_answer_model_name,
    selected_entry_model_name,
    validate_gateway_decision,
)
from scope_response import (
    REDIRECT_VARIANT_COUNT,
    SCOPE_CATEGORIES,
    definition_claim_is_absent,
    prefers_english,
    sanitize_topic_summary,
    scope_fallback_draft,
    scope_template_answer,
    validate_registry_scope_note,
    validate_scope_draft,
)


load_dotenv()

LOGGER = logging.getLogger(__name__)
_RUNTIME_METRIC_LOCK = threading.Lock()
_RUNTIME_METRICS: Counter[str] = Counter()
_SAFE_METRIC_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")


def _record_runtime_metric(name: str, label: str = "all") -> None:
    """Record one aggregate event without accepting user-controlled text."""

    if not _SAFE_METRIC_COMPONENT.fullmatch(name) or not _SAFE_METRIC_COMPONENT.fullmatch(label):
        return
    key = f"{name}.{label}"
    with _RUNTIME_METRIC_LOCK:
        _RUNTIME_METRICS[key] += 1
        total = _RUNTIME_METRICS[key]
    LOGGER.info("runtime_metric name=%s label=%s total=%d", name, label, total)


def runtime_metric_snapshot() -> dict[str, int]:
    """Return process-local enum/rule/count aggregates for operations."""

    with _RUNTIME_METRIC_LOCK:
        return dict(sorted(_RUNTIME_METRICS.items()))

GatewayRoute = Literal[
    "registry_query",
    "meta_conversation",
    "out_of_scope",
    "fixed_failure",
]
ScopeMode = Literal[
    "greeting",
    "thanks",
    "farewell",
    "bot_identity",
    "capability_help",
    "out_of_scope",
]
RegistryAnswerMode = Literal[
    "result",
    "candidates",
    "clarification",
    "unregistered",
    "guidance",
]


class SemanticSelection(BaseModel):
    """의미 검색 LLM이 반환할 엄격한 구조화 형식."""

    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_names: list[str] = Field(default_factory=list, max_length=3)
    confidence: Literal["high", "medium", "low"] = "low"
    auto_select: bool = False

    @model_validator(mode="after")
    def validate_auto_select(self) -> "SemanticSelection":
        if self.auto_select and (
            self.confidence != "high" or len(self.candidate_names) != 1
        ):
            raise ValueError(
                "auto_select에는 high confidence의 단일 후보가 필요합니다."
            )
        return self


class CompetencyState(MessagesState):
    """PostgreSQL checkpoint에 저장되는 JSON-safe graph 상태."""

    raw_query: NotRequired[str]
    candidate_ids: NotRequired[list[str]]
    candidate_names: NotRequired[list[str]]
    semantic_query: NotRequired[str]
    gateway_decision: NotRequired[dict[str, Any]]
    gateway_route: NotRequired[GatewayRoute]
    registry_query_draft: NotRequired[dict[str, Any]]
    normalization_issue: NotRequired[dict[str, Any]]
    query_plan: NotRequired[dict[str, Any]]
    result_ids: NotRequired[list[str]]
    query_result: NotRequired[dict[str, Any]]
    clarification_prompt: NotRequired[str]
    registry_answer_mode: NotRequired[RegistryAnswerMode]
    registry_answer: NotRequired[str]
    acknowledge_greeting: NotRequired[bool]
    scope_mode: NotRequired[ScopeMode]
    scope_topic_category: NotRequired[str]
    scope_topic_summary: NotRequired[str]
    llm_call_count: NotRequired[int]
    gateway_attempts: NotRequired[int]
    writer_attempts: NotRequired[int]
    writer_failed: NotRequired[bool]
    scope_writer_attempts: NotRequired[int]
    scope_writer_failed: NotRequired[bool]
    next_route: NotRequired[str]
    response_mode: NotRequired[
        Literal[
            "llm",
            "guidance_partial",
            "registry_fallback",
            "scope_template",
            "scope_fallback",
            "failure",
        ]
    ]
    last_scope_answer: NotRequired[str]

    # 후속 질문은 정식 이름이 아니라 현재 snapshot에서 재검증할 stable ID를 쓴다.
    last_query_plan: NotRequired[dict[str, Any]]
    last_result_ids: NotRequired[list[str]]
    last_candidate_ids: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# Runtime registry와 JSON-safe helpers
# ---------------------------------------------------------------------------

_registry_snapshot: RegistrySnapshot | None = None


def _require_registry() -> RegistrySnapshot:
    if _registry_snapshot is None:
        raise RuntimeError("역량 레지스트리가 초기화되지 않았습니다.")
    return _registry_snapshot


def _mutable_json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_json_copy(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json_copy(member) for member in value]
    return value


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        dumped = _mutable_json_copy(value)
    else:
        raise TypeError("질의 모델을 JSON 객체로 변환할 수 없습니다.")
    if not isinstance(dumped, dict):
        raise TypeError("질의 모델은 JSON 객체여야 합니다.")
    return dumped


def _snapshot_item_by_id(snapshot: RegistrySnapshot, item_id: str) -> Mapping[str, Any] | None:
    id_lookup = getattr(snapshot, "id_lookup", None)
    if isinstance(id_lookup, Mapping):
        item = id_lookup.get(item_id)
        if isinstance(item, Mapping):
            return item
    for item in snapshot.document.get("items", ()):
        if isinstance(item, Mapping) and item.get("id") == item_id:
            return item
    return None


def validate_registry_names(names: Sequence[str]) -> list[str]:
    registry = _require_registry()
    validated: list[str] = []
    seen_ids: set[str] = set()
    for raw_name in names:
        name = str(raw_name).strip()
        item = registry.lookup.get(name) or registry.canonical_lookup.get(name)
        if item is None:
            continue
        item_id = str(item["id"])
        if item_id in seen_ids:
            continue
        validated.append(str(item["name"]))
        seen_ids.add(item_id)
    return validated


def _valid_current_ids(ids: Sequence[str]) -> list[str]:
    snapshot = _require_registry()
    valid: list[str] = []
    seen: set[str] = set()
    for raw_id in ids:
        item_id = str(raw_id)
        if item_id in seen or _snapshot_item_by_id(snapshot, item_id) is None:
            continue
        valid.append(item_id)
        seen.add(item_id)
    return valid


def _safe_custom_event(payload: dict[str, Any]) -> None:
    """graph 밖의 focused unit test에서도 node를 직접 호출할 수 있게 한다."""

    try:
        get_stream_writer()(payload)
    except (LookupError, RuntimeError):
        return


# ---------------------------------------------------------------------------
# OpenAI Gateway, selector and dual writers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _gateway_for(model_name: str):
    """Return the strict entry model; ``model_name`` is the cache key."""

    return create_chat_model(
        ModelRole.ENTRY,
        model_name=model_name,
    ).with_structured_output(
        GatewayDecision,
        method="json_schema",
        strict=True,
    )


@lru_cache(maxsize=4)
def _semantic_selector_for(model_name: str):
    return create_chat_model(
        ModelRole.ENTRY,
        model_name=model_name,
    ).with_structured_output(
        SemanticSelection,
        method="json_schema",
        strict=True,
    )


@lru_cache(maxsize=4)
def _registry_writer_for(model_name: str) -> ChatOpenAI:
    return create_chat_model(ModelRole.ANSWER, model_name=model_name)


@lru_cache(maxsize=8)
def _scope_writer_for(model_name: str, mode: str):
    schema = OutOfScopeResponseDraft if mode == "out_of_scope" else MetaResponseDraft
    return create_chat_model(ModelRole.ANSWER, model_name=model_name).with_structured_output(
        schema,
        method="json_schema",
        strict=True,
    )


def _instrument_and_node_catalog(snapshot: RegistrySnapshot) -> tuple[list[str], list[str]]:
    instruments: list[str] = []
    node_types: list[str] = []
    for item in snapshot.document.get("items", ()):
        if not isinstance(item, Mapping):
            continue
        instrument = f"{item.get('instrument', '')}: {item.get('instrument_label', '')}".strip()
        node_type = str(item.get("level", "")).strip()
        if instrument and instrument not in instruments:
            instruments.append(instrument)
        if node_type and node_type not in node_types:
            node_types.append(node_type)
    return instruments, node_types


def _safe_previous_context(state: CompetencyState) -> dict[str, Any]:
    snapshot = _require_registry()
    previous_ids = _valid_current_ids(list(state.get("last_result_ids", [])))[:20]
    previous_candidates = _valid_current_ids(
        list(state.get("last_candidate_ids", []))
    )[:3]
    previous_plan_raw = state.get("last_query_plan") or {}
    previous_plan = {
        key: _mutable_json_copy(previous_plan_raw[key])
        for key in (
            "intent",
            "node_types",
            "hierarchy_tiers",
            "relation",
            "related_tier",
            "fields",
            "filters",
            "group_by",
            "scope",
            "max_depth",
        )
        if key in previous_plan_raw
    }
    return {
        "previous_results": [
            {"id": item_id, "name": str(snapshot.id_lookup[item_id]["name"])}
            for item_id in previous_ids
        ],
        "previous_candidates": [
            {"id": item_id, "name": str(snapshot.id_lookup[item_id]["name"])}
            for item_id in previous_candidates
        ],
        "previous_plan": previous_plan,
    }


def _gateway_prompt(state: CompetencyState) -> str:
    snapshot = _require_registry()
    instruments, node_types = _instrument_and_node_catalog(snapshot)
    previous = _safe_previous_context(state)

    return f"""
당신은 모든 입력을 가장 먼저 세 갈래로만 분류하는 LLM Gateway입니다.
사용자에게 답하지 말고 strict JSON schema만 반환하세요. 이름ㆍtierㆍ관계ㆍ필터의
최종 확정은 active registry를 읽는 Python Query Normalizer가 합니다. 정의, 개수,
위계 또는 일반 지식 답변을 만들지 마세요.

정상 route는 정확히 세 개입니다.
- registry_query: 등록 역량의 정의ㆍ목록ㆍ위계ㆍ관계ㆍ집계ㆍ비교ㆍ행동 기반 후보,
  비개인화된 활용 제안, 또는 역량명처럼 보이는 미등록 용어의 조회.
- meta_conversation: greeting, thanks, farewell, bot_identity, capability_help만 허용.
- out_of_scope: 날짜ㆍ시각ㆍ날씨ㆍ뉴스ㆍ일반 개념ㆍ외부 작업ㆍ개인 평가ㆍ채용 판단ㆍ
  전문 조언ㆍ위험 요청을 포함한 그 밖의 모든 실질적인 비역량 질문.

registry_query에서는 사용자가 실제로 쓴 target과 constraint의 짧은 표현을 보존한
초안만 만드세요. canonical name, stable ID, 정확한 enum이나 최종 filter를 발명하지
마세요. 모호함을 임의로 해결하지 말고 Query Normalizer가 판단하게 하세요.

target_mentions에는 등록 역량의 이름처럼 보이는 표현만 넣으세요. 목록ㆍ종류ㆍ전체ㆍ
리스트ㆍ위계ㆍ구조ㆍ개수처럼 범위나 형식을 가리키는 말은 target이 아니라 constraint
입니다. 이름이 없는 질문이면 target_mentions를 비워 두세요.

우선순위:
- 인사와 역량 질문이 섞이면 registry_query이며 acknowledge_greeting=true입니다.
- 역량 질문과 범위 밖 요청이 섞이면 registry_query를 우선하고
  out_of_scope_remainder에 category와 답이 아닌 짧은 주제 명사구를 넣습니다.
- 이름 없이 행동 특징으로 등록 역량을 찾으면 semantic_search hint입니다.
- 미등록 역량처럼 보이는 용어의 정의 요청도 registry_query입니다.
- 일반적이고 비개인화된 역량 활용 제안은 registry_query의 guidance answer mode입니다.

지원 intent hint:
item_lookup, semantic_search, catalog_query, hierarchy_query, relation_query,
aggregate_query, comparison_query

topic summary에는 답, 수치, URL, 절차나 조언을 넣지 말고 사용자의 주제만 짧게
요약하세요.

현재 검사 catalog: {instruments or ['없음']}
현재 node type: {node_types or ['없음']}
허용된 정식 이름과 별칭:
{snapshot.name_catalog}

현재 thread의 재검증된 stable-ID 문맥:
{json.dumps(previous, ensure_ascii=False)}

이 챗봇의 단일 capability manifest:
{capability_manifest_for_prompt()}
""".strip()


def _gateway_model_input(state: CompetencyState) -> list[tuple[str, str]]:
    model_input: list[tuple[str, str]] = [("system", _gateway_prompt(state))]
    for turn in recent_conversation_context(state.get("messages", [])):
        role = "human" if turn.role == "user" else "assistant"
        model_input.append((role, turn.content))
    return model_input


def _writer_recent_context(state: CompetencyState) -> str:
    turns = recent_conversation_context(state.get("messages", []))
    return json.dumps(
        [turn.model_dump(mode="json") for turn in turns],
        ensure_ascii=False,
    )


def _budget_from_state(state: CompetencyState) -> LlmCallBudget:
    used = int(state.get("llm_call_count", 0) or 0)
    used = min(max(used, 0), MAX_LLM_API_CALLS_PER_TURN)
    return LlmCallBudget(used_calls=used)


# ---------------------------------------------------------------------------
# LangGraph nodes: parsing, validation and deterministic execution
# ---------------------------------------------------------------------------

def _new_turn_updates(query: str) -> dict[str, Any]:
    """Reset only transient fields; stable follow-up context remains intact."""

    return {
        "raw_query": query,
        "candidate_ids": [],
        "candidate_names": [],
        "semantic_query": "",
        "gateway_decision": {},
        "gateway_route": "fixed_failure",
        "registry_query_draft": {},
        "normalization_issue": {},
        "query_plan": {},
        "result_ids": [],
        "query_result": {},
        "clarification_prompt": "",
        "registry_answer_mode": "result",
        "registry_answer": "",
        "acknowledge_greeting": False,
        "scope_mode": "out_of_scope",
        "scope_topic_category": "",
        "scope_topic_summary": "",
        "llm_call_count": 0,
        "gateway_attempts": 0,
        "writer_attempts": 0,
        "writer_failed": False,
        "scope_writer_attempts": 0,
        "scope_writer_failed": False,
        "next_route": "",
        "response_mode": "failure",
    }


def llm_gateway(state: CompetencyState) -> dict[str, Any]:
    """Run the sole natural-language entry classifier and query structurer."""

    query = str(state["messages"][-1].content).strip()
    model_input = _gateway_model_input(state)
    updates = _new_turn_updates(query)
    budget = LlmCallBudget()
    _safe_custom_event({"type": "status", "stage": "질문을 이해하는 중"})

    for attempt in range(1, 3):
        try:
            raw_decision = invoke_with_budget(
                _gateway_for(selected_entry_model_name()),
                model_input,
                budget=budget,
            )
            decision = validate_gateway_decision(raw_decision)
            updates.update(
                {
                    "gateway_decision": _model_dump(decision),
                    "gateway_attempts": attempt,
                    "llm_call_count": budget.used_calls,
                }
            )
            return updates
        except Exception:
            continue

    updates.update(
        {
            "gateway_attempts": budget.used_calls,
            "llm_call_count": budget.used_calls,
            "gateway_route": "fixed_failure",
        }
    )
    return updates


def validate_gateway_decision_node(state: CompetencyState) -> dict[str, Any]:
    """Validate the stored union again and expose only an enum-like route."""

    try:
        decision = validate_gateway_decision(state.get("gateway_decision") or {})
    except Exception:
        _record_runtime_metric("gateway_route", "fixed_failure")
        return {"gateway_route": "fixed_failure"}

    if isinstance(decision, RegistryRouteDecision):
        _record_runtime_metric("gateway_route", "registry_query")
        remainder = decision.draft.out_of_scope_remainder
        update: dict[str, Any] = {
            "gateway_route": "registry_query",
            "registry_query_draft": _model_dump(decision.draft),
            "registry_answer_mode": (
                "guidance"
                if decision.draft.answer_mode
                == "registry_facts_with_general_guidance"
                else "result"
            ),
            "acknowledge_greeting": decision.draft.acknowledge_greeting,
        }
        if remainder is not None:
            update["scope_topic_category"] = remainder.category
            update["scope_topic_summary"] = remainder.summary
        return update
    if isinstance(decision, MetaRouteDecision):
        _record_runtime_metric("gateway_route", "meta_conversation")
        return {
            "gateway_route": "meta_conversation",
            "scope_mode": decision.kind,
        }
    if isinstance(decision, OutOfScopeRouteDecision):
        _record_runtime_metric("gateway_route", "out_of_scope")
        return {
            "gateway_route": "out_of_scope",
            "scope_mode": "out_of_scope",
            "scope_topic_category": decision.topic.category,
            "scope_topic_summary": decision.topic.summary,
        }
    _record_runtime_metric("gateway_route", "fixed_failure")
    return {"gateway_route": "fixed_failure"}


def route_after_gateway(
    state: CompetencyState,
) -> Literal[
    "normalize_registry_query",
    "write_scope_answer",
    "fixed_failure_message",
]:
    route = state.get("gateway_route", "fixed_failure")
    if route == "registry_query":
        return "normalize_registry_query"
    if route in {"meta_conversation", "out_of_scope"}:
        return "write_scope_answer"
    return "fixed_failure_message"


def _validation_plan(validation: Any) -> RegistryQueryPlan | None:
    if isinstance(validation, RegistryQueryPlan):
        return validation
    candidate = getattr(validation, "plan", None)
    if candidate is None and isinstance(validation, Mapping):
        candidate = validation.get("plan")
    if candidate is None:
        return None
    try:
        return RegistryQueryPlan.model_validate(candidate)
    except Exception:
        return None


def normalize_registry_query_node(state: CompetencyState) -> dict[str, Any]:
    """Turn the untrusted Gateway draft into one registry-authoritative outcome."""

    try:
        normalized = normalize_registry_query(
            raw_query=str(state.get("raw_query", "") or ""),
            draft=state.get("registry_query_draft") or {},
            snapshot=_require_registry(),
            previous_result_ids=_valid_current_ids(list(state.get("last_result_ids", []))),
        )
    except Exception:
        return {"next_route": "fixed_failure_message"}

    for rule_id in normalized.applied_rule_ids:
        _record_runtime_metric("normalizer_rule", str(rule_id))
    issue = normalized.issue or normalized.unregistered_target
    if issue is not None:
        _record_runtime_metric("normalization_issue", str(issue.code.value))

    if normalized.outcome == NormalizationOutcome.CLARIFICATION:
        issue = normalized.issue
        if issue is None:
            return {"next_route": "fixed_failure_message"}
        options = [option.label for option in issue.options]
        prompt = issue.question
        if options:
            prompt += " 선택지는 " + ", ".join(options[:3]) + "입니다."
        return {
            "query_plan": {},
            "normalization_issue": _model_dump(issue),
            "clarification_prompt": prompt,
            "registry_answer_mode": "clarification",
            "next_route": "write_registry_answer",
        }
    if normalized.outcome == NormalizationOutcome.UNREGISTERED_TARGET:
        unknown = normalized.unregistered_target
        if unknown is None:
            return {"next_route": "fixed_failure_message"}
        return {
            "query_plan": {},
            "normalization_issue": _model_dump(unknown),
            "clarification_prompt": unknown.question,
            "registry_answer_mode": "unregistered",
            "next_route": "write_registry_answer",
        }
    if normalized.outcome == NormalizationOutcome.SEMANTIC_CANDIDATES:
        request = normalized.semantic_request
        if request is None:
            return {"next_route": "fixed_failure_message"}
        raw_query = str(state.get("raw_query", "") or "")
        fields = (
            [ItemField.DEFINITION]
            if re.search(r"정의|뜻|의미|what\s+is|definition|meaning", raw_query, re.I)
            else []
        )
        semantic_plan = RegistryQueryPlan(
            intent=QueryIntent.SEMANTIC_SEARCH,
            user_question=raw_query,
            fields=fields,
            semantic_query=request.semantic_query,
        )
        return {
            "query_plan": _model_dump(semantic_plan),
            "normalization_issue": {},
            "semantic_query": request.semantic_query,
            "next_route": "find_semantic_candidates",
        }
    plan = normalized.plan
    if normalized.outcome != NormalizationOutcome.PLAN or plan is None:
        return {"next_route": "fixed_failure_message"}
    intent = str(getattr(plan.intent, "value", plan.intent))
    if intent == QueryIntent.SEMANTIC_SEARCH.value:
        next_route = "find_semantic_candidates"
    elif intent == QueryIntent.ITEM_LOOKUP.value:
        next_route = "find_competencies"
    else:
        next_route = "execute_registry_query"
    return {
        "query_plan": _model_dump(plan),
        "normalization_issue": {},
        "semantic_query": plan.semantic_query or state.get("raw_query", ""),
        "next_route": next_route,
    }


def route_after_normalization(
    state: CompetencyState,
) -> Literal[
    "find_competencies",
    "find_semantic_candidates",
    "execute_registry_query",
    "write_registry_answer",
    "fixed_failure_message",
]:
    route = state.get("next_route", "write_registry_answer")
    if route in {
        "find_competencies",
        "find_semantic_candidates",
        "execute_registry_query",
    }:
        return route  # type: ignore[return-value]
    if route == "fixed_failure_message":
        return "fixed_failure_message"
    return "write_registry_answer"


def _result_item_ids(result: RegistryQueryResult) -> list[str]:
    dumped = _model_dump(result)
    ordered: list[str] = []
    seen: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
            return
        if isinstance(value, list):
            if key in {"item_ids", "target_ids", "result_ids"}:
                for member in value:
                    item_id = str(member)
                    if item_id not in seen and _snapshot_item_by_id(_require_registry(), item_id):
                        ordered.append(item_id)
                        seen.add(item_id)
            else:
                for member in value:
                    visit(member)

    visit(dumped)
    return ordered


def _execute_plan(plan: RegistryQueryPlan) -> dict[str, Any]:
    _safe_custom_event({"type": "status", "stage": "레지스트리를 조회하는 중"})
    result = execute_registry_query(plan, _require_registry())
    result_ids = _result_item_ids(result)
    update: dict[str, Any] = {
        "query_plan": _model_dump(plan),
        "query_result": _model_dump(result),
        "result_ids": result_ids,
        "candidate_ids": [],
        "candidate_names": [],
        "clarification_prompt": "",
    }
    if result.clarification:
        update["registry_answer_mode"] = "clarification"
        update["clarification_prompt"] = result.clarification
    return update


def execute_registry_query_node(state: CompetencyState) -> dict[str, Any]:
    try:
        plan = RegistryQueryPlan.model_validate(state.get("query_plan") or {})
        return _execute_plan(plan)
    except Exception:
        return {
            "query_result": {},
            "result_ids": [],
            "registry_answer_mode": "clarification",
            "clarification_prompt": (
                "요청한 조건으로 레지스트리를 조회하지 못했습니다. "
                "대상이나 범위를 조금 더 구체적으로 말해 주세요."
            ),
        }


def find_competencies(state: CompetencyState) -> dict[str, Any]:
    """Execute a validated item lookup without any legacy item-dict fallback."""

    try:
        plan = RegistryQueryPlan.model_validate(state.get("query_plan") or {})
        if plan.intent != QueryIntent.ITEM_LOOKUP:
            raise ValueError("item lookup plan required")
        return _execute_plan(plan)
    except Exception:
        return {
            "query_result": {},
            "result_ids": [],
            "registry_answer_mode": "clarification",
            "clarification_prompt": (
                "현재 레지스트리에서 요청한 역량을 안전하게 확정하지 못했습니다."
            ),
        }


def find_semantic_candidates(state: CompetencyState) -> dict[str, Any]:
    registry = _require_registry()
    semantic_query = state.get("semantic_query") or state.get("raw_query", "")
    system_prompt = f"""
당신은 사용자의 설명과 역량 레지스트리를 비교하는 검색기입니다.
아래 레지스트리에서 의미상 가장 가까운 정식 역량명을 최대 3개 고르세요.
confidence와 auto_select를 명시하세요. auto_select=true는 단 하나의 후보가
명백하고 confidence=high일 때만 허용합니다. 후보가 한 개라는 사실만으로 자동
선택하지 마세요. 확신할 후보가 없으면 빈 목록을 반환하고, 목록 밖 이름이나
정의를 만들지 마세요.

역량 레지스트리:
{registry.semantic_catalog}
""".strip()
    budget = _budget_from_state(state)
    try:
        raw_selection = invoke_with_budget(
            _semantic_selector_for(selected_entry_model_name()),
            [("system", system_prompt), ("human", semantic_query)],
            budget=budget,
        )
        selection = SemanticSelection.model_validate(raw_selection)
        candidates = validate_registry_names(selection.candidate_names)[:3]
    except Exception:
        return {
            "llm_call_count": budget.used_calls,
            "next_route": "fixed_failure_message",
        }

    candidate_ids = [str(registry.canonical_lookup[name]["id"]) for name in candidates]
    if selection.auto_select and selection.confidence == "high" and len(candidates) == 1:
        try:
            semantic_plan = RegistryQueryPlan.model_validate(state.get("query_plan") or {})
            parsed = ParsedRegistryQuery(
                intent=QueryIntent.ITEM_LOOKUP,
                target_names=candidates,
                fields=semantic_plan.fields,
            )
            validation = validate_parsed_query(
                parsed,
                registry,
                previous_result_ids=_valid_current_ids(
                    list(state.get("last_result_ids", []))
                ),
                user_question=state.get("raw_query", ""),
            )
            plan = _validation_plan(validation)
            if plan is None:
                raise ValueError("semantic target validation failed")
            return {
                "query_plan": _model_dump(plan),
                "candidate_ids": [],
                "candidate_names": [],
                "llm_call_count": budget.used_calls,
                "next_route": "find_competencies",
            }
        except Exception:
            return {
                "registry_answer_mode": "clarification",
                "clarification_prompt": (
                    "의미상 가까운 후보를 현재 레지스트리에서 안전하게 확정하지 못했습니다."
                ),
                "llm_call_count": budget.used_calls,
                "next_route": "write_registry_answer",
            }
    if candidates:
        return {
            "candidate_ids": candidate_ids,
            "candidate_names": candidates,
            "registry_answer_mode": "candidates",
            "llm_call_count": budget.used_calls,
            "next_route": "write_registry_answer",
        }
    return {
        "candidate_ids": [],
        "candidate_names": [],
        "registry_answer_mode": "unregistered",
        "llm_call_count": budget.used_calls,
        "next_route": "write_registry_answer",
    }


def route_after_candidates(
    state: CompetencyState,
) -> Literal["find_competencies", "write_registry_answer", "fixed_failure_message"]:
    route = state.get("next_route", "fixed_failure_message")
    if route == "find_competencies":
        return "find_competencies"
    if route == "write_registry_answer":
        return "write_registry_answer"
    return "fixed_failure_message"


# ---------------------------------------------------------------------------
# Grounded response writer and deterministic final-message nodes
# ---------------------------------------------------------------------------

def _context_with_question(context: GroundedAnswerContext, question: str) -> GroundedAnswerContext:
    if "user_question" not in type(context).model_fields:
        return context
    return context.model_copy(update={"user_question": question})


def _chunk_text(chunk: AIMessageChunk | Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") in {None, "text"}:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _answer_is_valid(validation: Any) -> bool:
    if isinstance(validation, bool):
        return validation
    for key in ("valid", "is_valid", "ok"):
        value = getattr(validation, key, None)
        if value is None and isinstance(validation, Mapping):
            value = validation.get(key)
        if isinstance(value, bool):
            return value
    return False


def _writer_context_json(context: GroundedAnswerContext) -> str:
    """writer에 불필요한 stable ID 없이 선택된 사실만 직렬화한다."""

    payload = context.model_dump(mode="json")
    for fact in payload.get("facts", []):
        if isinstance(fact, dict):
            fact.pop("item_id", None)
    return json.dumps(payload, ensure_ascii=False)


def _clarification_context(state: CompetencyState) -> GroundedAnswerContext:
    snapshot = _require_registry()
    mode = state.get("registry_answer_mode", "clarification")
    allowed_names: list[str] = []
    if mode == "unregistered":
        message = state.get("clarification_prompt", "").strip() or (
            "현재 레지스트리에서 입력한 용어와 일치하는 등록 역량을 찾지 "
            "못했습니다. 정확한 역량명이나 행동 특징을 알려 주세요."
        )
        intent = QueryIntent.SEMANTIC_SEARCH
    else:
        message = state.get("clarification_prompt", "").strip() or (
            "역량 질문의 대상이나 조회 범위를 조금 더 구체적으로 알려 주세요."
        )
        intent = QueryIntent.ITEM_LOOKUP
        issue = state.get("normalization_issue") or {}
        raw_options = issue.get("options", []) if isinstance(issue, Mapping) else []
        option_labels = [
            str(option.get("label", ""))
            for option in raw_options
            if isinstance(option, Mapping)
        ]
        allowed_names = validate_registry_names(option_labels)
    return GroundedAnswerContext(
        intent=intent,
        user_question=state.get("raw_query", ""),
        clarification=message,
        allowed_names=allowed_names,
        allowed_numbers=sorted(
            {
                int(value)
                for value in re.findall(
                    r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])",
                    message,
                )
            }
        ),
        registry_names=list(snapshot.canonical_names),
    )


def _candidate_context(state: CompetencyState) -> GroundedAnswerContext:
    snapshot = _require_registry()
    candidate_ids = _valid_current_ids(list(state.get("candidate_ids", [])))[:3]
    if not candidate_ids:
        raise ValueError("validated semantic candidates are required")
    plan = RegistryQueryPlan(
        intent=QueryIntent.ITEM_LOOKUP,
        user_question=state.get("raw_query", ""),
        target_ids=candidate_ids,
        fields=[ItemField.DEFINITION],
    )
    result = RegistryQueryResult(
        kind="items",
        item_ids=candidate_ids,
        counts={"total": len(candidate_ids)},
    )
    return build_grounded_answer_context(plan, result, snapshot)


def _registry_grounding_context(state: CompetencyState) -> GroundedAnswerContext:
    mode = state.get("registry_answer_mode", "result")
    if mode in {"clarification", "unregistered"}:
        return _clarification_context(state)
    if mode == "candidates":
        return _candidate_context(state)
    plan = RegistryQueryPlan.model_validate(state.get("query_plan") or {})
    result = RegistryQueryResult.model_validate(state.get("query_result") or {})
    context = build_grounded_answer_context(plan, result, _require_registry())
    return _context_with_question(context, state.get("raw_query", ""))


def _registry_reference_answer(
    context: GroundedAnswerContext,
    mode: RegistryAnswerMode,
) -> str:
    reference = render_grounded_fallback(context)
    if mode == "candidates":
        reference += (
            "\n원하는 후보의 정확한 역량명을 입력해 주세요."
        )
    return reference


def _safe_writer_plan_summary(state: CompetencyState) -> dict[str, Any]:
    raw = state.get("query_plan") or {}
    return {
        key: _mutable_json_copy(raw[key])
        for key in (
            "intent",
            "instrument_ids",
            "node_types",
            "hierarchy_tiers",
            "relation",
            "related_tier",
            "fields",
            "filters",
            "group_by",
            "scope",
            "max_depth",
        )
        if key in raw
    }


def _safe_writer_result_summary(state: CompetencyState) -> dict[str, Any]:
    raw = state.get("query_result") or {}
    groups: list[dict[str, Any]] = []
    for group in raw.get("groups", []) if isinstance(raw, Mapping) else []:
        if isinstance(group, Mapping):
            groups.append(
                {
                    "label": str(group.get("label", "")),
                    "count": int(group.get("count", 0) or 0),
                }
            )
    return {
        "kind": str(raw.get("kind", "")) if isinstance(raw, Mapping) else "",
        "groups": groups,
        "counts": _mutable_json_copy(raw.get("counts", {}))
        if isinstance(raw, Mapping)
        else {},
        "truncated": bool(raw.get("truncated", False))
        if isinstance(raw, Mapping)
        else False,
        "clarification": str(raw.get("clarification") or "")
        if isinstance(raw, Mapping)
        else "",
    }


def _mixed_scope_topic(state: CompetencyState) -> ScopeTopic | None:
    draft = state.get("registry_query_draft") or {}
    raw = draft.get("out_of_scope_remainder") if isinstance(draft, Mapping) else None
    if not raw:
        return None
    try:
        return ScopeTopic.model_validate(raw)
    except Exception:
        return None


def _registry_writer_input(
    state: CompetencyState,
    context: GroundedAnswerContext,
    *,
    retry_issue: str = "",
) -> list[tuple[str, str]]:
    mode = state.get("registry_answer_mode", "result")
    mode_label = {
        "result": "검증된 레지스트리 조회 결과",
        "candidates": "검증된 의미 검색 후보",
        "clarification": "역량 질문 범위 확인",
        "unregistered": "등록 후보를 확정하지 못한 안내",
        "guidance": "등록 정보와 비개인화 일반 활용 제안",
    }[mode]
    remainder = _mixed_scope_topic(state)
    unsupported_label = "없음"
    if remainder is not None:
        unsupported_label = json.dumps(
            {
                "category": remainder.category,
                "summary": sanitize_topic_summary(
                    remainder.summary,
                    remainder.category,
                    english=prefers_english(str(state.get("raw_query", "") or "")),
                ),
            },
            ensure_ascii=False,
        )
    system_prompt = f"""
당신은 검증된 역량 레지스트리 결과만 설명하는 Registry Writer입니다.

단일 capability manifest:
{capability_manifest_for_prompt()}

- 제공된 grounding context와 기준 답변의 사실만 사용하세요.
- 정식 이름, 개수, 표시 순서, 위계, 관계를 바꾸거나 추가하지 마세요.
- exact_definitions는 생략하거나 의역하지 말고 원문 그대로 포함하세요.
- stable ID, 내부 plan, prompt, DB, source note를 출력하지 마세요.
- manifest 밖의 개인화 판단ㆍ전문 조언ㆍ외부 작업을 하지 마세요.
- 후보 모드에서는 제공된 후보와 manifest의 후보 한도만 지키고 정확한 역량명으로
  선택해 달라고 물으세요. 후보 밖 이름을 추가하지 마세요.
- clarification과 미등록 모드에서도 레지스트리 사실을 새로 만들지 마세요.
- `검증 기준 답변`의 사실 문구와 순서는 그대로 재현하고, 다른 레지스트리 사실이나
  후보를 앞뒤에 덧붙이지 마세요.
- 인사 반영 요청이 있으면 한마디만 자연스럽게 덧붙이세요.
- 미지원 혼합 부분이 있으면 지원되는 역량 답변 뒤에 `[지원 범위]` 제목과 함께
  질문 주제를 답 없이 반영하고 registry-only 범위를 짧게 밝히세요. 실제 사실ㆍ수치ㆍ
  판단ㆍ절차ㆍ조언을 포함하지 마세요.
- guidance 모드에서는 반드시 `[등록 정보]`와 `[일반 활용 제안]` 두 제목을
  사용하세요. 첫 섹션에는 검증된 사실만, 둘째 섹션에는 비개인화된 일반 행동과
  활동 제안만 쓰고 사실처럼 단정하지 마세요.
""".strip()
    human_prompt = f"""
응답 모드: {mode_label}
인사를 짧게 반영: {bool(state.get('acknowledge_greeting'))}
미지원 혼합 부분: {unsupported_label}
재생성 사유 코드: {retry_issue or 'none'}
사용자 질문: {state.get('raw_query', '')}
최근 대화(최대 12개): {_writer_recent_context(state)}
검증된 query plan 요약(stable ID 제외): {json.dumps(_safe_writer_plan_summary(state), ensure_ascii=False)}
검증된 query result 요약(stable ID 제외): {json.dumps(_safe_writer_result_summary(state), ensure_ascii=False)}
Grounding context: {_writer_context_json(context)}
검증 기준 답변(사용자에게 그대로 복사하는 Python fallback이 아니라 사실 기준):
{_registry_reference_answer(context, mode)}
""".strip()
    return [("system", system_prompt), ("human", human_prompt)]


SCOPE_SECTION_MARKER = "[지원 범위]"


def _split_guidance_answer(
    answer: str,
) -> tuple[str, str, str, str] | None:
    facts_marker = "[등록 정보]"
    guidance_marker = "[일반 활용 제안]"
    scope_marker = SCOPE_SECTION_MARKER
    if (
        answer.count(facts_marker) != 1
        or answer.count(guidance_marker) != 1
        or answer.count(scope_marker) > 1
    ):
        return None
    facts_marker_start = answer.index(facts_marker)
    prefix = answer[:facts_marker_start].strip()
    facts_start = facts_marker_start + len(facts_marker)
    guidance_start = answer.index(guidance_marker)
    if facts_start >= guidance_start:
        return None
    facts = answer[facts_start:guidance_start].strip()
    guidance_content_start = guidance_start + len(guidance_marker)
    if scope_marker in answer:
        scope_start = answer.index(scope_marker)
        if scope_start <= guidance_content_start:
            return None
        guidance = answer[guidance_content_start:scope_start].strip()
        scope = answer[scope_start + len(scope_marker):].strip()
    else:
        guidance = answer[guidance_content_start:].strip()
        scope = ""
    if not facts or not guidance:
        return None
    return prefix, facts, guidance, scope




# Personalized judgement, professional advice and unfounded effect claims are
# out of scope everywhere a writer produces free prose, so the guidance section
# and the main registry body share one list.
_PERSONALIZATION_RISK_TOKENS = (
    "당신",
    "사용자",
    "현재 수준",
    "현재 상태",
    "점수",
    "등급",
    "진단",
    "채용",
    "합격",
    "적합",
    "취업",
    "가능성",
    "뛰어",
    "우수",
    "부족",
    "평균 이하",
    "연구 결과",
    "생산성",
    "your score",
    "your level",
    "diagnosis",
    "hiring",
    "job fit",
    "research shows",
    "의료",
    "법률",
    "금융",
    "투자",
    "medical",
    "legal advice",
    "financial advice",
    "%",
)


def _registry_body_without_scope_note(answer: str) -> str:
    """Drop the mixed-input suffix, which ``validate_registry_scope_note`` owns."""

    head, _, _ = answer.partition(SCOPE_SECTION_MARKER)
    return head


def _unverified_prose(answer: str, context: GroundedAnswerContext) -> str:
    """Return the answer with every verbatim-reproduced fact span removed.

    Registered definitions and the rendered hierarchy are reproduced word for
    word by contract, so they are facts rather than writer prose.  Everything
    that remains is treated as prose even when it looks like a fact line: a
    fabricated sentence must not escape the check merely by opening with a
    registered name.
    """

    remainder = answer
    verified = [
        span
        for span in (*context.exact_definitions.values(), context.hierarchy_text or "")
        if span
    ]
    for span in sorted(verified, key=len, reverse=True):
        remainder = remainder.replace(span, " ")
    return remainder


def _registry_prose_is_safe(answer: str, context: GroundedAnswerContext) -> bool:
    """Reject interpretive prose that introduces no new name or number.

    ``validate_grounded_answer`` already blocks unlisted registry names and
    unlisted numbers.  The remaining gap is an editorial or definitional
    sentence assembled purely from allowed vocabulary, which the previous
    exact-copy requirement used to block as a side effect.
    """

    prose = _unverified_prose(answer, context)
    if not definition_claim_is_absent(prose):
        return False
    lowered = prose.casefold()
    if any(token.casefold() in lowered for token in _PERSONALIZATION_RISK_TOKENS):
        return False
    # A registered name as the topic of a new sentence is how a pseudo
    # definition gets in.  The deterministic rendering never uses that form —
    # it attaches facts with 의/에는/이(가) — so banning it costs nothing.
    return not any(
        re.search(
            rf"(?:^|\n|(?<=[.!?])\s)\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?"
            rf"{re.escape(name)}\s*(?:은|는)\s",
            prose,
        )
        for name in context.allowed_names
    )


_KOREAN_GREETING_TOKENS = ("안녕", "반갑")
# Word boundaries matter here: a bare ``hi`` substring also fires on
# "hierarchy", which is ordinary registry vocabulary rather than a greeting.
_ENGLISH_GREETING_PATTERN = re.compile(
    r"\b(?:hello|hi|nice to meet|welcome)\b",
    re.IGNORECASE,
)


def _is_greeting_text(text: str) -> bool:
    segment = re.sub(r"\s+", " ", text).strip()
    return any(token in segment for token in _KOREAN_GREETING_TOKENS) or bool(
        _ENGLISH_GREETING_PATTERN.search(segment)
    )


def _registry_greeting_is_safe(text: str) -> bool:
    segment = re.sub(r"\s+", " ", text).strip()
    if not segment or len(segment) > 160:
        return False
    if re.search(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])", segment):
        return False
    if any(name in segment for name in _require_registry().canonical_names):
        return False
    lowered = segment.casefold()
    if any(
        token in lowered
        for token in (
            "정의",
            "위계",
            "부모",
            "자식",
            "능력",
            "특성",
            "뜻",
            "의미",
            "ability",
            "trait",
            "means",
            "system prompt",
        )
    ):
        return False
    return _is_greeting_text(segment)


def _first_sentence(text: str) -> tuple[str, str]:
    """Split the leading sentence from the rest of a block of prose."""

    boundary = re.search(r"(?<=[.!?])\s+|\n", text)
    if boundary is None:
        return text.strip(), ""
    return text[: boundary.start()].strip(), text[boundary.end() :].strip()


def _registry_structure_is_valid(
    answer: str,
    *,
    acknowledge_greeting: bool,
    scope_topic: ScopeTopic | None,
) -> bool:
    """Check only the optional sections that surround the registry body.

    The body itself is owned by ``validate_grounded_answer`` and
    ``_registry_prose_is_safe``.  Requiring it to reproduce the deterministic
    rendering word for word would reduce the writer to a copier and turn every
    formatting difference into a failure.
    """

    if answer.count(SCOPE_SECTION_MARKER) > 1:
        return False
    head, marker, scope_note = answer.partition(SCOPE_SECTION_MARKER)
    if scope_topic is None:
        if marker:
            return False
    elif not marker or not validate_registry_scope_note(
        scope_note.strip(),
        category=scope_topic.category,
        summary=scope_topic.summary,
    ):
        return False

    body = head.strip()
    if not body:
        return False
    opening, remainder = _first_sentence(body)
    if acknowledge_greeting:
        return bool(remainder) and _registry_greeting_is_safe(opening)
    # An unrequested greeting is still a writer addition, so it stays blocked.
    return not _is_greeting_text(opening)


def _guidance_is_safe(
    guidance: str,
    context: GroundedAnswerContext,
    snapshot: RegistrySnapshot,
) -> bool:
    lowered = guidance.casefold()
    forbidden = (
        "당신의 점수",
        "역량 점수는",
        "진단 결과",
        "채용해야",
        "직무가 적합",
        "합격 가능",
        "등록된 정의",
        "등록 경로",
        "등록된 부모",
        "등록된 자식",
        "직접 하위 항목",
        "위계 구조",
        "검사 도구:",
        "node type:",
        "분석 포함:",
        "능력입니다",
        "특성입니다",
        "태도입니다",
        "뜻입니다",
        "의미입니다",
        "부르",
        "is the ability",
        "is a trait",
        "is an attitude",
        "means the ability",
        "is defined as",
        " called ",
        "known as",
        " means ",
    )
    if any(token.casefold() in lowered for token in forbidden):
        return False
    if any(
        definition and definition in guidance
        for definition in context.exact_definitions.values()
    ):
        return False
    if any(token.casefold() in lowered for token in _PERSONALIZATION_RISK_TOKENS):
        return False
    numeric_scan = re.sub(
        r"(?<![A-Za-z0-9])\d+\s*(?:분|시간|회|minutes?|hours?|times?)(?![A-Za-z0-9])",
        "",
        guidance,
        flags=re.IGNORECASE,
    )
    if re.search(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])", numeric_scan):
        return False
    suggestion_tokens = (
        "수 있습니다",
        "해 보",
        "연습",
        "확인",
        "돌아보",
        "시도",
        "참고",
        "consider",
        " can ",
        "can ",
        "could",
        "try",
        "practice",
        "review",
    )
    for sentence in (
        part.strip()
        for part in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", guidance)
        if part.strip()
    ):
        if not any(token in sentence.casefold() for token in suggestion_tokens):
            return False
    if re.search(
        r"(?:^|\n|(?<=[.!?])\s+)[가-힣A-Za-z]{1,24}(?:성|력|능력)(?:은|는)\s+",
        guidance,
    ):
        return False
    if re.search(
        r"[가-힣A-Za-z]{1,24}(?:성|력|능력)\s*[:(]",
        guidance,
    ):
        return False
    if any(token in guidance for token in ("(", ")", ":")):
        return False
    # The guidance section is deliberately name-free.  Registry names belong
    # to the exact, validated facts section; allowing one as the subject of a
    # new sentence would let a writer append a contradictory pseudo-definition.
    for name in snapshot.canonical_names:
        if name in guidance:
            return False
    return True


def _registry_answer_is_valid(
    answer: str,
    context: GroundedAnswerContext,
    mode: RegistryAnswerMode,
    *,
    acknowledge_greeting: bool = False,
    scope_topic: ScopeTopic | None = None,
) -> bool:
    candidate = answer.strip()
    if not candidate or len(candidate) > MAX_ANSWER_CHARS:
        return False
    lowered = candidate.casefold()
    internal_tokens = (
        "postgresql://",
        "database_url",
        "openai_api_key",
        "grounding context",
        "system prompt",
        "target_ids",
        "query_plan",
        "current_information",
        "sensitive_advice",
        "unsafe_or_other_unsupported",
        "registry_facts_with_general_guidance",
    )
    grounded_text = "\n".join(
        [
            *context.allowed_names,
            *context.exact_definitions.values(),
            context.hierarchy_text or "",
        ]
    ).casefold()
    if any(
        token in lowered and token not in grounded_text
        for token in internal_tokens
    ):
        return False
    snapshot = _require_registry()
    internal_ids = {
        str(item_id).casefold()
        for item_id in snapshot.id_lookup
        if len(str(item_id)) >= 4
    }
    internal_ids.update(
        str(item.get("instrument", "")).casefold()
        for item in snapshot.document.get("items", ())
        if isinstance(item, Mapping) and len(str(item.get("instrument", ""))) >= 4
    )
    if any(
        internal_id
        and internal_id in lowered
        and internal_id not in grounded_text
        for internal_id in internal_ids
    ):
        return False
    if mode == "guidance":
        sections = _split_guidance_answer(candidate)
        if sections is None:
            return False
        prefix, facts, guidance, scope = sections
        greeting_valid = (
            _registry_greeting_is_safe(prefix)
            if acknowledge_greeting
            else not prefix
        )
        scope_valid = (
            validate_registry_scope_note(
                scope,
                category=scope_topic.category,
                summary=scope_topic.summary,
            )
            if scope_topic is not None
            else not scope
        )
        return (
            greeting_valid
            and scope_valid
            and _answer_is_valid(validate_grounded_answer(facts, context))
            and _registry_prose_is_safe(facts, context)
            and _guidance_is_safe(guidance, context, _require_registry())
        )
    if not _registry_structure_is_valid(
        candidate,
        acknowledge_greeting=acknowledge_greeting,
        scope_topic=scope_topic,
    ):
        return False
    if not _answer_is_valid(validate_grounded_answer(candidate, context)):
        return False
    # The scope suffix is a different genre with its own validator, so the
    # prose gate only sees the greeting and the registry body.
    if not _registry_prose_is_safe(
        _registry_body_without_scope_note(candidate),
        context,
    ):
        return False
    if mode == "candidates":
        return (
            ("역량명" in candidate or "이름" in candidate)
            and any(token in candidate for token in ("선택", "입력", "말씀"))
        )
    if mode == "unregistered":
        return any(token in candidate for token in ("찾지", "확정", "등록"))
    if mode == "clarification":
        return (
            all(str(number) in candidate for number in context.allowed_numbers)
            and any(token in candidate for token in ("구체", "확인", "선택", "알려", "줄여"))
        )
    return True


def _guidance_partial_answer(
    answer: str,
    context: GroundedAnswerContext,
    *,
    acknowledge_greeting: bool,
    scope_topic: ScopeTopic | None,
) -> str | None:
    """Salvage the validated facts when only the suggestion section fails.

    ``_guidance_is_safe`` is deliberately strict, so a single risky phrase in
    ``[일반 활용 제안]`` used to discard the already validated ``[등록 정보]``
    section as well.  Dropping just the suggestion turns the answer back into
    an ordinary registry result, which is then held to the ordinary contract.
    """

    sections = _split_guidance_answer(answer)
    if sections is None:
        return None
    prefix, facts, _suggestion, scope = sections
    salvaged = "\n".join(part for part in (prefix, facts) if part)
    if scope:
        salvaged = f"{salvaged}\n{SCOPE_SECTION_MARKER}\n{scope}"
    if not _registry_answer_is_valid(
        salvaged,
        context,
        "result",
        acknowledge_greeting=acknowledge_greeting,
        scope_topic=scope_topic,
    ):
        return None
    return salvaged


async def write_registry_answer(state: CompetencyState) -> dict[str, Any]:
    """Buffer and validate Registry Writer output before any public delta."""

    try:
        context = _registry_grounding_context(state)
    except Exception:
        return {"registry_answer": "", "writer_failed": True}

    _safe_custom_event({"type": "status", "stage": "답변을 작성하는 중"})
    mode = state.get("registry_answer_mode", "result")
    budget = _budget_from_state(state)
    attempts = int(state.get("writer_attempts", 0) or 0)
    retry_issue = ""

    while attempts < 2 and budget.remaining_calls > 0:
        attempts += 1
        chunks: list[str] = []
        length = 0
        try:
            async for chunk in astream_with_budget(
                _registry_writer_for(selected_answer_model_name()),
                # Rebuilt every attempt so a retry carries why the previous
                # draft was rejected, mirroring the Scope Writer.
                _registry_writer_input(state, context, retry_issue=retry_issue),
                budget=budget,
            ):
                text = _chunk_text(chunk)
                if not text:
                    continue
                length += len(text)
                if length > MAX_ANSWER_CHARS:
                    raise ValueError("registry answer exceeded safe length")
                chunks.append(text)
            candidate = "".join(chunks).strip()
            if _registry_answer_is_valid(
                candidate,
                context,
                mode,
                acknowledge_greeting=bool(state.get("acknowledge_greeting")),
                scope_topic=_mixed_scope_topic(state),
            ):
                return {
                    "registry_answer": candidate,
                    "writer_attempts": attempts,
                    "llm_call_count": budget.used_calls,
                    "writer_failed": False,
                }
            retry_issue = "registry_draft_validation_failed"
        except asyncio.CancelledError:
            raise
        except Exception:
            retry_issue = "registry_writer_request_failed"

    return {
        "registry_answer": "",
        "writer_attempts": attempts,
        "llm_call_count": budget.used_calls,
        "writer_failed": True,
    }


def validate_registry_answer_node(state: CompetencyState) -> dict[str, Any]:
    """Revalidate the buffered answer, then publish and checkpoint it once.

    A writer that never produced a publishable draft is recovered with the
    deterministic grounded rendering rather than the shared failure notice.
    That rendering is built from the same validated context, so it cannot
    introduce a fact the writer was denied.  ``FIXED_FAILURE_MESSAGE`` is
    reserved for real system failures, such as a grounding context that
    cannot be built at all.
    """

    answer = state.get("registry_answer", "").strip()
    mode = state.get("registry_answer_mode", "result")
    try:
        context = _registry_grounding_context(state)
    except Exception:
        # Without a context neither validation nor deterministic recovery is
        # possible, so this is the one registry path that stays a hard failure.
        return {
            "registry_answer": "",
            "writer_failed": True,
            "next_route": "fixed_failure_message",
        }

    acknowledge_greeting = bool(state.get("acknowledge_greeting"))
    scope_topic = _mixed_scope_topic(state)
    try:
        valid = _registry_answer_is_valid(
            answer,
            context,
            mode,
            acknowledge_greeting=acknowledge_greeting,
            scope_topic=scope_topic,
        )
    except Exception:
        valid = False

    try:
        salvaged = (
            None
            if valid or mode != "guidance"
            else _guidance_partial_answer(
                answer,
                context,
                acknowledge_greeting=acknowledge_greeting,
                scope_topic=scope_topic,
            )
        )
    except Exception:
        salvaged = None

    response_mode: Literal["llm", "guidance_partial", "registry_fallback"]
    if valid:
        published = answer
        response_mode = "llm"
    elif salvaged is not None:
        published = salvaged
        response_mode = "guidance_partial"
        _record_runtime_metric("guidance_partial", "all")
    else:
        # ``render_grounded_fallback`` already caps its own length and may
        # legitimately return the over-length notice, so only emptiness can
        # still block publication here.
        published = _registry_reference_answer(context, mode).strip()
        if not published:
            return {
                "registry_answer": "",
                "writer_failed": True,
                "next_route": "fixed_failure_message",
            }
        response_mode = "registry_fallback"
        _record_runtime_metric("registry_fallback", mode)

    _safe_custom_event(
        {"type": "delta", "text": published, "commit_required": True}
    )
    _record_runtime_metric(
        "llm_calls",
        f"registry.{int(state.get('llm_call_count', 0) or 0)}",
    )
    update: dict[str, Any] = {
        "messages": AIMessage(content=published),
        "registry_answer": "",
        # The recovery path still terminates normally; the flag stays true so
        # operations can separate a writer failure from a clean writer turn.
        "writer_failed": response_mode != "llm",
        "response_mode": response_mode,
        "next_route": END,
    }
    if mode == "candidates":
        candidate_ids = _valid_current_ids(list(state.get("candidate_ids", [])))[:3]
        update["last_candidate_ids"] = candidate_ids
        update["candidate_names"] = [
            str(_require_registry().id_lookup[item_id]["name"])
            for item_id in candidate_ids
        ]
    else:
        update["candidate_ids"] = []
        update["candidate_names"] = []
        update["last_candidate_ids"] = []
    if mode in {"result", "guidance"}:
        current_ids = _valid_current_ids(list(state.get("result_ids", [])))
        update["last_query_plan"] = _mutable_json_copy(
            state.get("query_plan") or {}
        )
        update["last_result_ids"] = current_ids
    return update


def route_after_registry_validation(
    state: CompetencyState,
) -> Literal["fixed_failure_message", "__end__"]:
    # ``writer_failed`` stays true on the deterministic recovery path, which
    # still publishes and terminates normally.  Mirroring
    # ``route_after_scope_writer``, only an explicit failure route falls
    # through to the shared failure notice.
    if state.get("next_route") == END:
        return END
    return "fixed_failure_message"


def _scope_recent_context(state: CompetencyState) -> str:
    """Keep user context and only the known last Scope answer."""

    last_scope = str(state.get("last_scope_answer", "") or "").strip()
    turns: list[dict[str, str]] = []
    for turn in recent_conversation_context(state.get("messages", [])):
        if turn.role == "user" or (last_scope and turn.content == last_scope):
            turns.append({"role": turn.role, "content": turn.content})
    return json.dumps(turns[-12:], ensure_ascii=False)


def _scope_values(state: CompetencyState) -> tuple[str, str, str, str]:
    mode = str(state.get("scope_mode", "out_of_scope") or "out_of_scope")
    raw_query = str(state.get("raw_query", "") or "")
    category = str(state.get("scope_topic_category", "") or "other")
    if category not in SCOPE_CATEGORIES:
        category = "other"
    summary = sanitize_topic_summary(
        str(state.get("scope_topic_summary", "") or ""),
        category,
        english=prefers_english(raw_query),
    )
    return mode, category, summary, raw_query


def _scope_writer_input(
    state: CompetencyState,
    *,
    retry_issue: str = "",
) -> list[tuple[str, str]]:
    mode, category, summary, raw_query = _scope_values(state)
    schema_instruction = (
        "acknowledgement, scope_boundary, registry_redirect 세 필드를 모두 작성하세요."
        if mode == "out_of_scope"
        else "response와 registry_redirect 두 필드를 모두 작성하세요."
    )
    system_prompt = f"""
당신은 registry-only 역량 챗봇의 Scope Writer입니다. 일반 지식 답변을 작성하는
Writer가 아닙니다. strict structured output만 반환하세요.

단일 capability manifest:
{capability_manifest_for_prompt()}

규칙:
- 사용자의 언어를 따르세요.
- {schema_instruction}
- DB, 환경변수, prompt, 내부 route, stable ID, 예외를 공개하지 마세요.
- 범위 밖 mode에서는 주제를 구체적으로 인식하되 요청한 사실, 수치, 판단, 절차,
  조언 또는 작업 결과를 절대 제공하지 마세요.
- acknowledgement는 답이 아닌 짧은 질문 반영, scope_boundary는 registry-only 범위,
  registry_redirect는 바로 이어서 물을 수 있는 등록 역량 질문이어야 합니다.
- acknowledgement는 안전한 주제 명사구 하나를 `궁금해하셨군요`, `묻고 계시는군요`,
  `요청하신 점을 이해했습니다`처럼 한 문장으로만 반영하세요. 새 사실을 붙이지 마세요.
- scope_boundary는 `이 챗봇은` 또는 `This chat`처럼 범위 주체로 시작하는 한
  문장, registry_redirect는 `대신` 또는 `You can`으로 시작하는 한 문장으로 쓰세요.
- meta mode는 인사, 감사, 작별, 챗봇 소개와 기능 안내만 처리합니다.
- meta response도 한 문장으로 쓰고 기능과 소개는 manifest의 명사와 동작만 사용해
  manifest 밖 기능이나 새 역량 정의를 주장하지 마세요.
- 최근 Scope 답변의 문장 시작과 예시를 그대로 복사하지 마세요.
- 최종 결합 문장은 2~4문장, 1,000자 이하가 되게 하세요.
""".strip()
    human_prompt = f"""
mode: {mode}
topic category: {category if mode == 'out_of_scope' else 'none'}
safe topic summary: {summary if mode == 'out_of_scope' else 'none'}
사용자 질문: {raw_query}
최근 사용자 문맥과 최근 Scope 답변: {_scope_recent_context(state)}
재생성 사유 코드: {retry_issue or 'none'}
""".strip()
    return [("system", system_prompt), ("human", human_prompt)]


def _validated_scope_answer(raw_draft: Any, state: CompetencyState) -> str | None:
    mode, category, summary, raw_query = _scope_values(state)
    return validate_scope_draft(
        raw_draft,
        mode=mode,
        category=category,
        summary=summary,
        raw_query=raw_query,
    )


def _redirect_variant(state: CompetencyState) -> int:
    """Rotate the out-of-scope redirect deterministically, once per turn.

    ``messages`` grows by exactly two entries per completed turn (one human,
    one assistant), so the turn index is what must drive the rotation.  A bare
    ``len(messages) % 4`` only ever produces two of the four variants.

    Determinism matters beyond testability: a re-executed or resumed node must
    reproduce the same string, or the public delta and the checkpointed
    ``AIMessage`` stop matching in ``run_competency_stream``.
    """

    turn_index = max(len(state.get("messages", ())) - 1, 0) // 2
    return turn_index % REDIRECT_VARIANT_COUNT


def _scope_template_update(state: CompetencyState) -> dict[str, Any]:
    """Answer an out-of-scope turn from a template, with no model call.

    The only model-derived text that survives is the topic summary, and
    ``sanitize_topic_summary`` has already reduced it to a short noun phrase or
    a fixed category label.  Re-validating deterministic output on the request
    path would just move a unit-test assertion into the user's turn, so the
    template's safety is proven in ``test_scope_response`` instead.
    """

    _, category, summary, raw_query = _scope_values(state)
    budget = _budget_from_state(state)
    answer = scope_template_answer(
        category=category,
        summary=summary,
        raw_query=raw_query,
        variant=_redirect_variant(state),
    )
    _safe_custom_event({"type": "delta", "text": answer, "commit_required": True})
    _record_runtime_metric("scope_template", category)
    _record_runtime_metric("llm_calls", f"scope.{budget.used_calls}")
    return {
        "messages": AIMessage(content=answer),
        "candidate_ids": [],
        "candidate_names": [],
        "last_candidate_ids": [],
        "scope_writer_attempts": 0,
        "scope_writer_failed": False,
        "llm_call_count": budget.used_calls,
        "response_mode": "scope_template",
        "last_scope_answer": answer,
        "next_route": END,
    }


async def write_scope_answer(state: CompetencyState) -> dict[str, Any]:
    """Answer out-of-scope from a template; keep the model for meta turns."""

    _safe_custom_event({"type": "status", "stage": "답변을 작성하는 중"})
    mode, category, summary, raw_query = _scope_values(state)
    if mode == "out_of_scope":
        return _scope_template_update(state)

    budget = _budget_from_state(state)
    attempts = int(state.get("scope_writer_attempts", 0) or 0)
    retry_issue = ""

    while attempts < 2 and budget.remaining_calls > 0:
        attempts += 1
        try:
            raw_draft = await ainvoke_with_budget(
                _scope_writer_for(selected_answer_model_name(), mode),
                _scope_writer_input(state, retry_issue=retry_issue),
                budget=budget,
            )
            answer = _validated_scope_answer(raw_draft, state)
            if answer is None:
                if attempts == 1:
                    _record_runtime_metric("scope_first_failure", category)
                retry_issue = "scope_draft_validation_failed"
                continue
            _safe_custom_event(
                {"type": "delta", "text": answer, "commit_required": True}
            )
            if attempts > 1:
                _record_runtime_metric("scope_retry_success", category)
            _record_runtime_metric("llm_calls", f"scope.{budget.used_calls}")
            return {
                "messages": AIMessage(content=answer),
                "candidate_ids": [],
                "candidate_names": [],
                "last_candidate_ids": [],
                "scope_writer_attempts": attempts,
                "scope_writer_failed": False,
                "llm_call_count": budget.used_calls,
                "response_mode": "llm",
                "last_scope_answer": answer,
                "next_route": END,
            }
        except asyncio.CancelledError:
            raise
        except Exception:
            if attempts == 1:
                _record_runtime_metric("scope_first_failure", category)
            retry_issue = "scope_writer_request_failed"

    fallback_draft = scope_fallback_draft(
        mode=mode,
        category=category,
        summary=summary,
        raw_query=raw_query,
    )
    # The meta draft is built from ``mode`` and the query language only, so a
    # second attempt with a blanked summary would revalidate the same string.
    fallback = _validated_scope_answer(fallback_draft, state)
    if fallback is None:
        return {
            "scope_writer_attempts": attempts,
            "scope_writer_failed": True,
            "llm_call_count": budget.used_calls,
            "next_route": "fixed_failure_message",
        }
    _safe_custom_event(
        {"type": "delta", "text": fallback, "commit_required": True}
    )
    _record_runtime_metric("scope_fallback", category)
    _record_runtime_metric("llm_calls", f"scope.{budget.used_calls}")
    return {
        "messages": AIMessage(content=fallback),
        "candidate_ids": [],
        "candidate_names": [],
        "last_candidate_ids": [],
        "scope_writer_attempts": attempts,
        "scope_writer_failed": True,
        "llm_call_count": budget.used_calls,
        "response_mode": "scope_fallback",
        "last_scope_answer": fallback,
        "next_route": END,
    }


def route_after_scope_writer(
    state: CompetencyState,
) -> Literal["fixed_failure_message", "__end__"]:
    if state.get("next_route") == END:
        return END
    return "fixed_failure_message"


def fixed_failure_message(state: CompetencyState) -> dict[str, Any]:
    """Return the shared system-failure message after route recovery is exhausted."""

    _safe_custom_event(
        {"type": "delta", "text": FIXED_FAILURE_MESSAGE, "commit_required": True}
    )
    _record_runtime_metric("fixed_failure", "all")
    _record_runtime_metric(
        "llm_calls",
        f"failure.{int(state.get('llm_call_count', 0) or 0)}",
    )
    return {
        "messages": AIMessage(content=FIXED_FAILURE_MESSAGE),
        "candidate_ids": [],
        "candidate_names": [],
        "last_candidate_ids": [],
        "registry_answer": "",
        "writer_failed": True,
        "response_mode": "failure",
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(CompetencyState)
builder.add_node("llm_gateway", llm_gateway)
builder.add_node("validate_gateway_decision", validate_gateway_decision_node)
builder.add_node("normalize_registry_query", normalize_registry_query_node)
builder.add_node("find_semantic_candidates", find_semantic_candidates)
builder.add_node("find_competencies", find_competencies)
builder.add_node("execute_registry_query", execute_registry_query_node)
builder.add_node("write_registry_answer", write_registry_answer)
builder.add_node("validate_registry_answer", validate_registry_answer_node)
builder.add_node("write_scope_answer", write_scope_answer)
builder.add_node("fixed_failure_message", fixed_failure_message)

builder.add_edge(START, "llm_gateway")
builder.add_edge("llm_gateway", "validate_gateway_decision")
builder.add_conditional_edges("validate_gateway_decision", route_after_gateway)
builder.add_conditional_edges("normalize_registry_query", route_after_normalization)
builder.add_conditional_edges("find_semantic_candidates", route_after_candidates)
builder.add_edge("find_competencies", "write_registry_answer")
builder.add_edge("execute_registry_query", "write_registry_answer")
builder.add_edge("write_registry_answer", "validate_registry_answer")
builder.add_conditional_edges(
    "validate_registry_answer",
    route_after_registry_validation,
)
builder.add_conditional_edges("write_scope_answer", route_after_scope_writer)
builder.add_edge("fixed_failure_message", END)


# ---------------------------------------------------------------------------
# Runtime/checkpointer lifecycle and per-thread concurrency
# ---------------------------------------------------------------------------

class _AsyncCheckpointAdapter(BaseCheckpointSaver):
    """동기 PostgresSaver를 LangGraph ``astream``에서도 사용하게 한다."""

    def __init__(self, delegate: BaseCheckpointSaver) -> None:
        super().__init__(serde=delegate.serde)
        self.delegate = delegate

    @property
    def config_specs(self):  # type: ignore[override]
        return self.delegate.config_specs

    def get_tuple(self, config):
        return self.delegate.get_tuple(config)

    def list(self, config, *, filter=None, before=None, limit=None) -> Iterator[Any]:
        return self.delegate.list(config, filter=filter, before=before, limit=limit)

    def put(self, config, checkpoint, metadata, new_versions):
        return self.delegate.put(config, checkpoint, metadata, new_versions)

    def put_writes(self, config, writes, task_id, task_path="") -> None:
        self.delegate.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self.delegate.delete_thread(thread_id)

    def get_next_version(self, current, channel):
        return self.delegate.get_next_version(current, channel)

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.delegate.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None) -> AsyncIterator[Any]:
        values = await asyncio.to_thread(
            lambda: list(self.delegate.list(config, filter=filter, before=before, limit=limit))
        )
        for value in values:
            yield value

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(
            self.delegate.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(self, config, writes, task_id, task_path="") -> None:
        await asyncio.to_thread(self.delegate.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delegate.delete_thread, thread_id)


_runtime_init_lock = threading.RLock()
# 이전 외부 test/import 호환용 alias. 네트워크 실행 전체에는 사용하지 않는다.
_runtime_lock = _runtime_init_lock
_runtime_stack: ExitStack | None = None
_runtime_active_users = 0
_runtime_close_pending = False
_thread_locks_guard = threading.Lock()
# A same-thread burst should not spin at a fixed 100 Hz while a long turn runs.
_THREAD_LOCK_MIN_DELAY = 0.01
_THREAD_LOCK_MAX_DELAY = 0.1


class _ThreadLockEntry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


_thread_locks: dict[str, _ThreadLockEntry] = {}
app: Any | None = None


def _get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")
    return database_url


def _open_checkpointer(runtime_stack: ExitStack, database_url: str) -> BaseCheckpointSaver:
    return runtime_stack.enter_context(PostgresSaver.from_conn_string(database_url))


def initialize_competency_runtime() -> None:
    global app, _registry_snapshot, _runtime_stack

    with _runtime_init_lock:
        if _runtime_close_pending:
            raise RuntimeError("역량 챗봇 runtime이 종료 중입니다.")
        if app is not None:
            return
        _registry_snapshot = None
        database_url = _get_database_url()
        runtime_stack = ExitStack()
        try:
            registry_snapshot = load_active_registry(database_url)
            _registry_snapshot = registry_snapshot
            saver = _open_checkpointer(runtime_stack, database_url)
            compiled_app = builder.compile(
                checkpointer=(
                    _AsyncCheckpointAdapter(saver)
                    if isinstance(saver, BaseCheckpointSaver)
                    else saver
                )
            )
        except Exception:
            app = None
            _runtime_stack = None
            _registry_snapshot = None
            runtime_stack.close()
            raise
        _runtime_stack = runtime_stack
        app = compiled_app


def close_competency_runtime() -> None:
    global app, _registry_snapshot, _runtime_stack, _runtime_close_pending

    runtime_stack: ExitStack | None = None
    with _runtime_init_lock:
        if _runtime_active_users:
            # Do not close the PostgreSQL checkpointer underneath an active
            # graph call.  The final runtime lease performs the actual close.
            _runtime_close_pending = True
            return
        runtime_stack = _runtime_stack
        app = None
        _registry_snapshot = None
        _runtime_stack = None
        _runtime_close_pending = False
        with _thread_locks_guard:
            _thread_locks.clear()
    if runtime_stack is not None:
        runtime_stack.close()


def _release_runtime_execution() -> None:
    global app, _registry_snapshot, _runtime_stack
    global _runtime_active_users, _runtime_close_pending

    runtime_stack: ExitStack | None = None
    with _runtime_init_lock:
        _runtime_active_users -= 1
        if _runtime_active_users < 0:  # pragma: no cover - defensive invariant
            _runtime_active_users = 0
            raise RuntimeError("runtime lease count가 올바르지 않습니다.")
        if _runtime_active_users == 0 and _runtime_close_pending:
            runtime_stack = _runtime_stack
            app = None
            _registry_snapshot = None
            _runtime_stack = None
            _runtime_close_pending = False
            with _thread_locks_guard:
                _thread_locks.clear()
    if runtime_stack is not None:
        runtime_stack.close()


@contextmanager
def _runtime_execution() -> Iterator[Any]:
    """Keep one compiled runtime/checkpointer alive for a complete request."""

    global _runtime_active_users

    initialize_competency_runtime()
    with _runtime_init_lock:
        if app is None or _runtime_close_pending:
            raise RuntimeError("역량 챗봇 runtime을 사용할 수 없습니다.")
        compiled_app = app
        _runtime_active_users += 1
    try:
        yield compiled_app
    finally:
        _release_runtime_execution()


@asynccontextmanager
async def _async_runtime_execution() -> AsyncIterator[Any]:
    """Async counterpart whose acquisition never spans network execution."""

    global _runtime_active_users

    initialize_competency_runtime()
    with _runtime_init_lock:
        if app is None or _runtime_close_pending:
            raise RuntimeError("역량 챗봇 runtime을 사용할 수 없습니다.")
        compiled_app = app
        _runtime_active_users += 1
    try:
        yield compiled_app
    finally:
        _release_runtime_execution()


@contextmanager
def _thread_execution(thread_id: str) -> Iterator[None]:
    with _thread_locks_guard:
        entry = _thread_locks.setdefault(thread_id, _ThreadLockEntry())
        entry.users += 1
    entry.lock.acquire()
    try:
        yield
    finally:
        entry.lock.release()
        with _thread_locks_guard:
            entry.users -= 1
            if _thread_locks.get(thread_id) is entry and entry.users == 0:
                _thread_locks.pop(thread_id, None)


@asynccontextmanager
async def _async_thread_execution(thread_id: str) -> AsyncIterator[None]:
    """동일 thread는 직렬화하되 event loop 자체는 점유하지 않는다."""

    with _thread_locks_guard:
        entry = _thread_locks.setdefault(thread_id, _ThreadLockEntry())
        entry.users += 1

    acquired = False
    delay = _THREAD_LOCK_MIN_DELAY
    try:
        while not acquired:
            # Non-blocking acquisition leaves no worker-thread call behind
            # when this coroutine is cancelled while waiting.  The sync API
            # shares this same ``threading.Lock``, so an ``asyncio.Lock`` would
            # stop serialising /api/chat against /api/chat/stream.
            acquired = entry.lock.acquire(blocking=False)
            if not acquired:
                await asyncio.sleep(delay)
                delay = min(delay * 2, _THREAD_LOCK_MAX_DELAY)
        yield
    finally:
        if acquired:
            entry.lock.release()
        with _thread_locks_guard:
            entry.users -= 1
            if _thread_locks.get(thread_id) is entry and entry.users == 0:
                _thread_locks.pop(thread_id, None)


def _run_async(coro_factory):
    """sync API에서도 동일한 async graph 경로를 호출한다."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    outcome: list[Any] = []
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            outcome.append(asyncio.run(coro_factory()))
        except BaseException as error:  # pragma: no cover - rare embedded-loop path
            failure.append(error)

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join()
    if failure:
        raise failure[0]
    return outcome[0]


def _graph_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


async def _ainvoke_compiled(
    compiled_app: Any,
    graph_input: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """실제 compiled graph와 최소 기존 test double을 모두 지원한다."""

    if hasattr(compiled_app, "ainvoke"):
        return await compiled_app.ainvoke(graph_input, config=config)
    return await asyncio.to_thread(compiled_app.invoke, graph_input, config=config)


def run_competency(question: str, thread_id: str) -> dict[str, Any]:
    with _runtime_execution() as compiled_app:
        with _thread_execution(thread_id):
            return _run_async(
                lambda: _ainvoke_compiled(
                    compiled_app,
                    {"messages": [HumanMessage(content=question)]},
                    _graph_config(thread_id),
                )
            )


def _stream_parts(event: Any) -> tuple[str, Any]:
    if isinstance(event, tuple) and len(event) == 2:
        return str(event[0]), event[1]
    if isinstance(event, Mapping) and event.get("type") in {"custom", "values"}:
        return str(event["type"]), event.get("data")
    return "custom", event


def _last_answer(state: Mapping[str, Any]) -> str:
    for message in reversed(list(state.get("messages", []))):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
            return message.content
    raise RuntimeError("LangGraph 응답에서 최종 답변을 찾지 못했습니다.")


async def run_competency_stream(question: str, thread_id: str) -> AsyncIterator[dict[str, Any]]:
    """공개 SSE 계층이 소비할 안전한 event stream을 생성한다."""

    final_state: dict[str, Any] = {}
    emitted_answer_event = False
    pending_commit_delta: str | None = None
    try:
        async with _async_runtime_execution() as compiled_app:
            yield {"type": "start", "thread_id": thread_id}
            async with _async_thread_execution(thread_id):
                async for raw_event in compiled_app.astream(
                    {"messages": [HumanMessage(content=question)]},
                    config=_graph_config(thread_id),
                    stream_mode=["custom", "values"],
                ):
                    mode, data = _stream_parts(raw_event)
                    if mode == "values" and isinstance(data, Mapping):
                        final_state = dict(data)
                        if pending_commit_delta is not None:
                            try:
                                committed_answer = _last_answer(final_state)
                            except RuntimeError:
                                committed_answer = ""
                            if committed_answer == pending_commit_delta:
                                emitted_answer_event = True
                                yield {
                                    "type": "delta",
                                    "text": pending_commit_delta,
                                }
                                pending_commit_delta = None
                        continue
                    if mode != "custom" or not isinstance(data, Mapping):
                        continue
                    event_type = data.get("type")
                    if event_type == "status" and data.get("stage") in {
                        "질문을 이해하는 중",
                        "레지스트리를 조회하는 중",
                        "답변을 작성하는 중",
                    }:
                        yield {"type": "status", "stage": str(data["stage"])}
                    elif event_type == "delta" and isinstance(data.get("text"), str):
                        if data.get("commit_required") is True:
                            if pending_commit_delta is not None:
                                raise RuntimeError(
                                    "checkpoint-backed 공개 delta가 중복되었습니다."
                                )
                            pending_commit_delta = str(data["text"])
                        else:
                            emitted_answer_event = True
                            yield {"type": "delta", "text": data["text"]}
                    elif event_type == "replace" and isinstance(data.get("answer"), str):
                        emitted_answer_event = True
                        yield {"type": "replace", "answer": data["answer"]}

            answer = _last_answer(final_state)
            if pending_commit_delta is not None:
                if answer != pending_commit_delta:
                    raise RuntimeError(
                        "공개 답변과 checkpoint가 일치하지 않습니다."
                    )
                emitted_answer_event = True
                yield {"type": "delta", "text": pending_commit_delta}
                pending_commit_delta = None
            candidates = validate_registry_names(
                list(final_state.get("candidate_names", []))
            )[:3]
            if not emitted_answer_event:
                # Test doubles or future deterministic terminal nodes also fill
                # the browser bubble from their committed final message.
                yield {"type": "delta", "text": answer}
            yield {
                "type": "done",
                "thread_id": thread_id,
                "answer": answer,
                "candidates": candidates,
            }
    except asyncio.CancelledError:
        raise
    except Exception:
        yield {
            "type": "error",
            "code": "chat_failed",
            "message": FIXED_FAILURE_MESSAGE,
            "retryable": True,
        }


def get_competency_state(thread_id: str) -> dict[str, Any]:
    with _runtime_execution() as compiled_app:
        snapshot = compiled_app.get_state(_graph_config(thread_id))
        return dict(snapshot.values) if snapshot.values else {}


def ask_competency(question: str, thread_id: str = "competency-chat-1") -> str:
    result = run_competency(question, thread_id)
    return str(result["messages"][-1].content)
