"""등록된 역량을 자연어로 탐색하고 대화 맥락을 유지하는 LangGraph.

LLM은 제한된 질의 계획을 만들고, 실제 이름ㆍ정의ㆍ위계ㆍ개수는
``competency_query``의 결정적 Python 실행기가 active registry에서 계산한다.
응답 LLM의 token delta는 custom stream으로만 내보내며 checkpoint에는 검증을
통과한 최종 ``AIMessage`` 하나만 저장한다.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
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
from pydantic import BaseModel, ConfigDict, Field

from competency_query import (
    GroundedAnswerContext,
    ItemField,
    MAX_ANSWER_CHARS,
    ParsedRegistryQuery,
    QueryIntent,
    RegistryQueryPlan,
    RegistryQueryResult,
    build_grounded_answer_context,
    detect_deterministic_query,
    execute_registry_query,
    render_grounded_fallback,
    validate_grounded_answer,
    validate_parsed_query,
)
from competency_registry import RegistrySnapshot, load_active_registry


load_dotenv()

RequestedField = Literal["definition", "children", "path"]
QueryType = Literal["named_lookup", "semantic_search", "out_of_scope"]
LlmRoute = Literal[
    "validate_query_plan",
    "find_competencies",
    "find_semantic_candidates",
    "execute_registry_query",
    "answer_help",
    "clarify_query",
    "handle_unknown",
    "handle_out_of_scope",
]


class ParsedNaturalLanguageQuery(BaseModel):
    """이전 버전 import와 테스트를 위한 호환 모델.

    새 parser의 실제 schema는 ``ParsedRegistryQuery``다. 이 모델이 반환되는
    stub도 안전하게 새 schema/기존 경로로 변환해 기존 확장 지점을 깨지 않는다.
    """

    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    competency_names: list[str] = Field(default_factory=list)
    requested_fields: list[RequestedField] = Field(default_factory=list)
    semantic_query: str | None = None
    reuse_previous_fields: bool = False


class SemanticSelection(BaseModel):
    """의미 검색 LLM이 반환할 엄격한 구조화 형식."""

    model_config = ConfigDict(extra="forbid")

    candidate_names: list[str] = Field(default_factory=list, max_length=3)


class CompetencyState(MessagesState):
    """PostgreSQL checkpoint에 저장되는 JSON-safe graph 상태."""

    raw_query: NotRequired[str]
    requested_fields: NotRequired[list[str]]
    resolved_names: NotRequired[list[str]]
    matched_items: NotRequired[list[dict[str, Any]]]
    candidate_names: NotRequired[list[str]]
    semantic_query: NotRequired[str]
    llm_route: NotRequired[LlmRoute]

    # LLM의 이름 기반 raw plan과 검증된 stable-ID plan을 분리한다.
    parsed_query: NotRequired[dict[str, Any]]
    query_plan: NotRequired[dict[str, Any]]
    result_ids: NotRequired[list[str]]
    query_result: NotRequired[dict[str, Any]]
    clarification_prompt: NotRequired[str]
    query_route: NotRequired[str]
    response_mode: NotRequired[Literal["llm", "fallback"]]

    # 후속 질문은 정식 이름이 아니라 현재 snapshot에서 재검증할 stable ID를 쓴다.
    last_query_plan: NotRequired[dict[str, Any]]
    last_result_ids: NotRequired[list[str]]
    last_resolved_names: NotRequired[list[str]]


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


def _mutable_registry_item(item: Mapping[str, Any]) -> dict[str, Any]:
    copied = _mutable_json_copy(item)
    assert isinstance(copied, dict)
    return copied


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


DEFAULT_FIELDS: list[RequestedField] = ["definition", "children", "path"]
EXACT_NAME_SEPARATOR = re.compile(r"[,，\n]+")


def extract_exact_registered_names(query: str) -> list[str]:
    """입력 전체가 정확한 등록 이름/별칭 목록일 때만 정식 이름을 반환한다."""

    registry = _require_registry()
    stripped = query.strip()
    if not stripped:
        return []

    raw_names = [part.strip() for part in EXACT_NAME_SEPARATOR.split(stripped)]
    if not raw_names or any(not name for name in raw_names):
        return []

    resolved: list[str] = []
    seen_ids: set[str] = set()
    for raw_name in raw_names:
        item = registry.lookup.get(raw_name)
        if item is None:
            return []
        item_id = str(item["id"])
        if item_id in seen_ids:
            continue
        resolved.append(str(item["name"]))
        seen_ids.add(item_id)
    return resolved


def normalize_requested_fields(fields: Sequence[str]) -> list[RequestedField]:
    allowed = {"definition", "children", "path"}
    normalized: list[RequestedField] = []
    for field in fields:
        if field not in allowed or field in normalized:
            continue
        normalized.append(field)  # type: ignore[arg-type]
    return normalized


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
# OpenAI structured parser/selector/writer
# ---------------------------------------------------------------------------

def selected_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "").strip() or "gpt-5.6-luna"


def _chat_model(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, max_retries=1, timeout=30)


@lru_cache(maxsize=4)
def _query_parser_for(model_name: str):
    return _chat_model(model_name).with_structured_output(
        ParsedRegistryQuery,
        method="json_schema",
        strict=True,
    )


@lru_cache(maxsize=4)
def _semantic_selector_for(model_name: str):
    return _chat_model(model_name).with_structured_output(
        SemanticSelection,
        method="json_schema",
        strict=True,
    )


@lru_cache(maxsize=4)
def _response_writer_for(model_name: str) -> ChatOpenAI:
    return _chat_model(model_name)


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


def _parser_prompt(state: CompetencyState) -> str:
    snapshot = _require_registry()
    instruments, node_types = _instrument_and_node_catalog(snapshot)
    previous_ids = _valid_current_ids(list(state.get("last_result_ids", [])))[:20]
    previous_names = [
        str(item["name"])
        for item_id in previous_ids
        if (item := _snapshot_item_by_id(snapshot, item_id)) is not None
    ]
    if "last_result_ids" not in state and not previous_names:
        # 이전 버전 checkpoint/extension이 name 문맥만 저장했을 때의 읽기 호환.
        # 새 generalized flow는 아래 값을 다시 stable ID plan으로 canonicalize한다.
        previous_names = validate_registry_names(
            list(state.get("last_resolved_names", []))
        )[:20]
    previous_candidates = validate_registry_names(list(state.get("candidate_names", [])))
    previous_plan_raw = state.get("last_query_plan") or {}
    previous_plan = {
        key: _mutable_json_copy(previous_plan_raw[key])
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
        if key in previous_plan_raw
    }

    return f"""
당신은 역량 레지스트리 질문을 제한된 QueryPlan으로 바꾸는 parser입니다.
사용자에게 답하지 말고 지정된 strict JSON schema만 반환하세요. 등록 정보나 정의를
만들지 말고, 이름ㆍ검사ㆍnode type은 아래 허용 catalog만 사용하세요.

지원 intent:
item_lookup, semantic_search, catalog_query, hierarchy_query, relation_query,
aggregate_query, comparison_query, help, out_of_scope

핵심 의미 규칙:
- target 없는 상위/중위/하위/최하위요인은 필기검사의 공식 tier입니다.
- target이 있는 '<역량>의 상위요인/하위요인'은 직접 parent/children입니다.
- '모든 상위/하위'는 ancestors/descendants입니다.
- '<역량>이 속한 중위요인'처럼 '속한'은 related_tier입니다.
- root/leaf는 공식 상위/최하위 tier와 다릅니다.
- 영상면접 factor는 요인, item은 세부항목이며 필기 4단계 tier를 적용하지 않습니다.
- '그중'은 previous_result scope로 제한하고 이전 결과가 없으면 임의 보완하지 않습니다.

예시:
- 전체 역량 목록 -> catalog_query, scope=all
- 전체 위계 구조 -> hierarchy_query, scope=all
- 상위요인은 몇 개야? -> aggregate_query, hierarchy_tiers=[upper]
- 자기긍정의 상위요인은? -> relation_query, relation=parent
- 자기긍정의 모든 상위요인은? -> relation_query, relation=ancestors
- 자기긍정이 속한 중위요인은? -> relation_query, related_tier=middle
- 영상면접 세부항목 목록 -> catalog_query, node_types=[item]
- 성실성과 자기긍정을 비교 -> comparison_query

현재 검사 catalog: {instruments or ['없음']}
현재 node type: {node_types or ['없음']}
허용된 정식 이름과 별칭:
{snapshot.name_catalog}

현재 thread의 안전한 이전 문맥:
- 이전 결과 이름(최대 20개): {previous_names or '없음'}
- 기존 의미검색 후보: {previous_candidates or '없음'}
- 이전 plan 요약: {previous_plan or '없음'}
""".strip()


def _legacy_parser_update(
    parsed: ParsedNaturalLanguageQuery,
    state: CompetencyState,
) -> dict[str, Any]:
    """오래된 parser stub/plugin을 안전한 기존 경로로만 제한한다."""

    previous_fields = normalize_requested_fields(list(state.get("requested_fields", [])))
    fields = normalize_requested_fields(parsed.requested_fields)
    if not fields:
        fields = previous_fields if parsed.reuse_previous_fields and previous_fields else DEFAULT_FIELDS.copy()

    if parsed.query_type == "named_lookup":
        names = validate_registry_names(parsed.competency_names)
        if names:
            registry_parsed = ParsedRegistryQuery(
                intent=QueryIntent.ITEM_LOOKUP,
                target_names=names,
                fields=[ItemField(field) for field in fields],
            )
            validated = validate_parsed_query(
                registry_parsed,
                _require_registry(),
                previous_result_ids=_valid_current_ids(
                    list(state.get("last_result_ids", []))
                ),
                user_question=state.get("raw_query", ""),
            )
            plan = _validation_plan(validated)
            if plan is None:
                return {
                    "candidate_names": [],
                    "llm_route": "handle_unknown",
                }
            return {
                "resolved_names": names,
                "requested_fields": fields,
                "candidate_names": [],
                "query_plan": _model_dump(plan),
                "llm_route": "find_competencies",
            }
        return {"candidate_names": [], "llm_route": "handle_unknown"}
    if parsed.query_type == "semantic_search":
        return {
            "semantic_query": (parsed.semantic_query or state.get("raw_query", "")).strip(),
            "requested_fields": fields,
            "candidate_names": [],
            "llm_route": "find_semantic_candidates",
        }
    return {"candidate_names": [], "llm_route": "handle_out_of_scope"}


# ---------------------------------------------------------------------------
# LangGraph nodes: parsing, validation and deterministic execution
# ---------------------------------------------------------------------------

def interpret_query(state: CompetencyState) -> dict[str, Any]:
    query = str(state["messages"][-1].content).strip()
    resolved_names = extract_exact_registered_names(query)
    previous_candidates = validate_registry_names(list(state.get("candidate_names", [])))
    previous_fields = normalize_requested_fields(list(state.get("requested_fields", [])))
    selected_previous_candidate = any(name in previous_candidates for name in resolved_names)

    _safe_custom_event({"type": "status", "stage": "질문을 이해하는 중"})
    updates: dict[str, Any] = {
        "raw_query": query,
        "resolved_names": resolved_names,
        "matched_items": [],
        "semantic_query": "",
        "parsed_query": {},
        "query_plan": {},
        "result_ids": [],
        "query_result": {},
        "llm_route": "handle_unknown",
        "query_route": "",
        "clarification_prompt": "",
        "response_mode": "fallback",
    }

    if resolved_names:
        updates["requested_fields"] = (
            previous_fields
            if selected_previous_candidate and previous_fields
            else DEFAULT_FIELDS.copy()
        )
        updates["candidate_names"] = []

    detected = detect_deterministic_query(query, _require_registry())
    if detected is None and resolved_names:
        # Exact lookup itself is authoritative even if whitespace-normalized
        # labels collide. Never fall back to checkpointing whole item dicts.
        exact_fields = [
            ItemField(field)
            for field in updates.get("requested_fields", DEFAULT_FIELDS)
        ]
        validated = validate_parsed_query(
            ParsedRegistryQuery(
                intent=QueryIntent.ITEM_LOOKUP,
                target_names=resolved_names,
                fields=exact_fields,
            ),
            _require_registry(),
            user_question=query,
        )
        detected = validated.plan
        if detected is None:
            updates["clarification_prompt"] = (
                validated.clarification
                or "현재 레지스트리 구성에서 해당 역량을 안전하게 조회할 수 없습니다."
            )
            updates["query_route"] = "clarification"
    if detected is not None:
        updates["query_plan"] = _model_dump(detected)
        updates["query_route"] = str(getattr(detected, "intent", ""))

    return updates


def _intent_from_plan_dict(plan_data: Mapping[str, Any]) -> str:
    intent = plan_data.get("intent", "")
    return str(getattr(intent, "value", intent))


def _route_for_intent(intent: str) -> LlmRoute:
    if intent == "item_lookup":
        return "find_competencies"
    if intent == "semantic_search":
        return "find_semantic_candidates"
    if intent == "help":
        return "answer_help"
    if intent == "out_of_scope":
        return "handle_out_of_scope"
    if intent in {
        "catalog_query",
        "hierarchy_query",
        "relation_query",
        "aggregate_query",
        "comparison_query",
    }:
        return "execute_registry_query"
    return "handle_unknown"


def route_after_interpret(state: CompetencyState) -> LlmRoute | Literal["llm_interpret_query"]:
    plan_data = state.get("query_plan") or {}
    if plan_data:
        return _route_for_intent(_intent_from_plan_dict(plan_data))
    if state.get("resolved_names"):
        # An exact name without a validated stable-ID plan indicates an
        # inconsistent registry edge case.  Never route it into the legacy
        # whole-item checkpoint path.
        return "clarify_query"
    return "llm_interpret_query"


def llm_interpret_query(state: CompetencyState) -> dict[str, Any]:
    query = state.get("raw_query", "")
    try:
        parsed = _query_parser_for(selected_model_name()).invoke(
            [("system", _parser_prompt(state)), ("human", query)]
        )
    except Exception:
        return {
            "candidate_names": [],
            "clarification_prompt": (
                "질문의 범위를 확정하지 못했습니다. 역량명이나 원하는 목록, "
                "위계, 관계, 개수, 비교 범위를 조금 더 구체적으로 말해 주세요."
            ),
            "llm_route": "clarify_query",
        }

    if isinstance(parsed, ParsedNaturalLanguageQuery):
        return _legacy_parser_update(parsed, state)

    try:
        parsed_query = ParsedRegistryQuery.model_validate(parsed)
    except Exception:
        return {
            "clarification_prompt": "질문의 대상과 범위를 조금 더 구체적으로 말해 주세요.",
            "llm_route": "clarify_query",
        }

    intent = str(getattr(parsed_query.intent, "value", parsed_query.intent))
    if intent == "semantic_search":
        semantic_fields = normalize_requested_fields(
            [str(getattr(field, "value", field)) for field in parsed_query.fields]
        )
        return {
            "parsed_query": _model_dump(parsed_query),
            "semantic_query": (parsed_query.semantic_query or query).strip(),
            "requested_fields": semantic_fields or DEFAULT_FIELDS.copy(),
            "candidate_names": [],
            "llm_route": "find_semantic_candidates",
        }
    if intent == "help":
        return {"parsed_query": _model_dump(parsed_query), "llm_route": "answer_help"}
    if intent == "out_of_scope":
        return {"parsed_query": _model_dump(parsed_query), "llm_route": "handle_out_of_scope"}
    return {"parsed_query": _model_dump(parsed_query), "llm_route": "validate_query_plan"}


def route_after_llm(state: CompetencyState) -> LlmRoute:
    return state.get("llm_route", "handle_unknown")


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


def _validation_message(validation: Any) -> str:
    for key in ("clarification", "clarification_prompt", "message", "error"):
        value = getattr(validation, key, None)
        if value is None and isinstance(validation, Mapping):
            value = validation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "질문의 대상과 범위를 확정하지 못했습니다. 조금 더 구체적으로 말해 주세요."


def validate_query_plan_node(state: CompetencyState) -> dict[str, Any]:
    try:
        parsed = ParsedRegistryQuery.model_validate(state.get("parsed_query") or {})
        validation = validate_parsed_query(
            parsed,
            _require_registry(),
            previous_result_ids=_valid_current_ids(list(state.get("last_result_ids", []))),
            user_question=state.get("raw_query", ""),
        )
        plan = _validation_plan(validation)
    except Exception:
        plan = None
        validation = None

    if plan is None:
        return {
            "query_plan": {},
            "clarification_prompt": _validation_message(validation),
            "llm_route": "clarify_query",
        }
    return {
        "query_plan": _model_dump(plan),
        "query_route": str(getattr(plan.intent, "value", plan.intent)),
        "llm_route": _route_for_intent(str(getattr(plan.intent, "value", plan.intent))),
    }


def route_after_plan_validation(state: CompetencyState) -> LlmRoute:
    return state.get("llm_route", "clarify_query")


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
    names = [
        str(item["name"])
        for item_id in result_ids
        if (item := _snapshot_item_by_id(_require_registry(), item_id)) is not None
    ]
    return {
        "query_plan": _model_dump(plan),
        "query_result": _model_dump(result),
        "result_ids": result_ids,
        "resolved_names": names,
        "matched_items": [],
        "candidate_names": [],
        "clarification_prompt": "",
    }


def execute_registry_query_node(state: CompetencyState) -> dict[str, Any]:
    try:
        plan = RegistryQueryPlan.model_validate(state.get("query_plan") or {})
        return _execute_plan(plan)
    except Exception:
        return {
            "query_result": {},
            "result_ids": [],
            "clarification_prompt": (
                "요청한 조건으로 레지스트리를 조회하지 못했습니다. "
                "대상이나 범위를 조금 더 구체적으로 말해 주세요."
            ),
        }


def find_competencies(state: CompetencyState) -> dict[str, Any]:
    """기존 exact/named 경로를 유지하되 graph에서는 stable ID만 저장한다."""

    plan_data = state.get("query_plan") or {}
    if plan_data:
        try:
            plan = RegistryQueryPlan.model_validate(plan_data)
            fields = list(state.get("requested_fields", []))
            if fields and hasattr(plan, "fields"):
                plan = plan.model_copy(
                    update={
                        "fields": [
                            ItemField(field)
                            for field in fields
                            if field in {member.value for member in ItemField}
                        ]
                    }
                )
            return _execute_plan(plan)
        except Exception:
            pass

    # 호환용 직접 node 호출 경로. 새 graph flow에서는 detector가 plan을 만든다.
    registry = _require_registry()
    matched_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for name in state.get("resolved_names", []):
        item = registry.canonical_lookup.get(name)
        if item is None or str(item["id"]) in seen_ids:
            continue
        matched_items.append(_mutable_registry_item(item))
        seen_ids.add(str(item["id"]))
    return {
        "matched_items": matched_items,
        "last_resolved_names": [str(item["name"]) for item in matched_items],
        "candidate_names": [],
    }


def route_after_execution(state: CompetencyState) -> Literal["write_streamed_answer", "clarify_query"]:
    if state.get("query_result") and state.get("query_plan"):
        return "write_streamed_answer"
    return "clarify_query"


def route_after_item_lookup(
    state: CompetencyState,
) -> Literal["write_streamed_answer", "produce_answer", "clarify_query"]:
    if state.get("query_result") and state.get("query_plan"):
        return "write_streamed_answer"
    if state.get("matched_items"):
        return "produce_answer"
    return "clarify_query"


def find_semantic_candidates(state: CompetencyState) -> dict[str, Any]:
    registry = _require_registry()
    semantic_query = state.get("semantic_query") or state.get("raw_query", "")
    system_prompt = f"""
당신은 사용자의 설명과 역량 레지스트리를 비교하는 검색기입니다.
아래 레지스트리에서 의미상 가장 가까운 정식 역량명을 최대 3개 고르세요.
확신할 후보가 없으면 빈 목록을 반환하세요. 목록 밖 이름이나 정의를 만들지 마세요.

역량 레지스트리:
{registry.semantic_catalog}
""".strip()
    try:
        selection = _semantic_selector_for(selected_model_name()).invoke(
            [("system", system_prompt), ("human", semantic_query)]
        )
        candidates = validate_registry_names(selection.candidate_names)[:3]
    except Exception:
        candidates = []
    return {"candidate_names": candidates}


def route_after_candidates(state: CompetencyState) -> Literal["present_candidates", "handle_unknown"]:
    return "present_candidates" if state.get("candidate_names") else "handle_unknown"


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


MAX_STREAMED_ANSWER_CHARS = MAX_ANSWER_CHARS


async def write_streamed_answer(state: CompetencyState) -> dict[str, Any]:
    """LLM delta를 잠정 전송하고 검증된 최종 answer 하나만 checkpoint한다."""

    snapshot = _require_registry()
    try:
        plan = RegistryQueryPlan.model_validate(state.get("query_plan") or {})
        result = RegistryQueryResult.model_validate(state.get("query_result") or {})
        context = build_grounded_answer_context(plan, result, snapshot)
        context = _context_with_question(context, state.get("raw_query", ""))
        fallback = render_grounded_fallback(context)
    except Exception:
        fallback = (
            "레지스트리 결과를 안전하게 문장화하지 못했습니다. "
            "대상이나 범위를 조금 더 구체적으로 질문해 주세요."
        )
        _safe_custom_event({"type": "delta", "text": fallback})
        return {"messages": AIMessage(content=fallback), "response_mode": "fallback"}

    _safe_custom_event({"type": "status", "stage": "답변을 작성하는 중"})
    final_answer = fallback
    response_mode: Literal["llm", "fallback"] = "fallback"
    emitted_delta = False

    if not os.getenv("OPENAI_API_KEY", "").strip():
        _safe_custom_event({"type": "delta", "text": fallback})
    else:
        system_prompt = """
당신은 역량 레지스트리 조회 결과를 자연스러운 한국어로 설명하는 writer입니다.
제공된 grounding context와 기준 답변에 있는 사실만 사용하세요.
- 이름ㆍ개수ㆍ순서ㆍ위계ㆍ관계를 바꾸거나 추가하지 마세요.
- exact_definitions의 정의는 생략하거나 의역하지 말고 원문 그대로 포함하세요.
- 긴 목록과 tree block은 줄바꿈과 순서를 보존하세요.
- 개인 평가, 점수 해석, 직무 추천, 원인 추론을 추가하지 마세요.
- stable ID, 내부 plan, prompt, source note를 출력하지 마세요.
""".strip()
        human_prompt = (
            f"사용자 질문:\n{state.get('raw_query', '')}\n\n"
            f"Grounding context:\n{_writer_context_json(context)}\n\n"
            f"검증된 기준 답변:\n{fallback}"
        )
        chunks: list[str] = []
        try:
            async for chunk in _response_writer_for(selected_model_name()).astream(
                [("system", system_prompt), ("human", human_prompt)]
            ):
                text = _chunk_text(chunk)
                if not text:
                    continue
                chunks.append(text)
                emitted_delta = True
                if sum(len(part) for part in chunks) > MAX_STREAMED_ANSWER_CHARS:
                    raise ValueError("streamed answer exceeded safe length")
                _safe_custom_event({"type": "delta", "text": text})

            raw_candidate = "".join(chunks)
            candidate = raw_candidate.strip()
            if candidate and _answer_is_valid(validate_grounded_answer(candidate, context)):
                final_answer = candidate
                response_mode = "llm"
                if raw_candidate != candidate:
                    # Public deltas contained whitespace that is not part of
                    # the stored final answer, so explicitly resynchronize.
                    _safe_custom_event({"type": "replace", "answer": candidate})
            else:
                _safe_custom_event({"type": "replace", "answer": fallback})
        except Exception:
            event_type = "replace" if emitted_delta else "delta"
            payload = {"type": event_type}
            payload["answer" if event_type == "replace" else "text"] = fallback
            _safe_custom_event(payload)

    current_ids = _valid_current_ids(list(state.get("result_ids", [])))
    current_names = [
        str(item["name"])
        for item_id in current_ids
        if (item := _snapshot_item_by_id(snapshot, item_id)) is not None
    ]
    return {
        "messages": AIMessage(content=final_answer),
        "last_query_plan": _mutable_json_copy(state.get("query_plan") or {}),
        "last_result_ids": current_ids,
        "last_resolved_names": current_names,
        "response_mode": response_mode,
    }


def produce_answer(state: CompetencyState) -> dict[str, Any]:
    """이전 버전의 직접 node 호출을 위한 registry-backed fallback."""

    requested_fields = set(state.get("requested_fields", DEFAULT_FIELDS))
    matched_items = state.get("matched_items", [])
    if not matched_items:
        return {"messages": AIMessage(content="확정된 역량을 레지스트리에서 찾지 못했습니다.")}

    answers: list[str] = []
    for item in matched_items:
        name = str(item["name"])
        sentences: list[str] = []
        definition = item.get("definition")
        if "definition" in requested_fields:
            if item.get("definition_status") == "not_provided" or not definition:
                sentences.append(f"{name}은(는) 원본 문서에 독립적인 정의가 제공되어 있지 않습니다.")
            else:
                sentences.append(f"{name}의 정의는 다음과 같습니다. {definition}")
        else:
            sentences.append(name)
        if "children" in requested_fields:
            children = item.get("children", [])
            sentences.append(
                f"직접 하위요인은 {', '.join(children)}입니다."
                if children
                else "등록된 직접 하위요인은 없습니다."
            )
        if "path" in requested_fields and item.get("path"):
            sentences.append(f"{name}의 위계 구조: {' > '.join(item['path'])}입니다.")
        answers.append(" ".join(sentences))
    return {"messages": AIMessage(content="\n\n".join(answers))}


def present_candidates(state: CompetencyState) -> dict[str, Any]:
    registry = _require_registry()
    lines = ["입력한 표현과 관련된 역량 후보를 찾았습니다."]
    for number, name in enumerate(state.get("candidate_names", []), start=1):
        item = registry.canonical_lookup[name]
        definition = item.get("definition") or "독립적인 정의가 제공되어 있지 않음"
        lines.append(f"{number}. {name}: {definition}")
    lines.append("원하는 후보의 번호(예: 1번) 또는 정확한 역량명을 입력해 주세요.")
    return {"messages": AIMessage(content="\n".join(lines))}


def answer_help(_: CompetencyState) -> dict[str, Any]:
    return {
        "candidate_names": [],
        "messages": AIMessage(
            content=(
                "역량 레지스트리에서 다음과 같은 질문을 할 수 있습니다.\n"
                "- 전체 역량 목록 또는 필기/영상면접별 목록\n"
                "- 상위ㆍ중위ㆍ하위ㆍ최하위요인 목록과 개수\n"
                "- 전체 또는 특정 역량 아래의 위계 구조\n"
                "- 직접 부모, 모든 조상, 자식, 후손, 형제 역량\n"
                "- 검사별ㆍnode type별 집계와 최대 3개 역량 비교\n"
                "- 행동 설명에 가까운 역량 후보 찾기\n"
                "예: ‘자기긍정이 속한 중위요인은?’, ‘영상면접 세부항목 목록’"
            )
        )
    }


def clarify_query(state: CompetencyState) -> dict[str, Any]:
    message = state.get("clarification_prompt", "").strip() or (
        "질문의 대상과 범위를 확정하지 못했습니다. 역량명이나 원하는 목록, "
        "위계, 관계, 개수, 비교 범위를 조금 더 구체적으로 말해 주세요."
    )
    return {
        "candidate_names": [],
        "messages": AIMessage(content=message),
    }


def handle_unknown(state: CompetencyState) -> dict[str, Any]:
    query = state.get("raw_query", "")
    return {
        "candidate_names": [],
        "messages": AIMessage(
            content=(
                f"입력에서 등록된 역량이나 관련 후보를 찾지 못했습니다: {query}\n"
                "역량명 또는 찾고 싶은 행동의 특징을 조금 더 구체적으로 말해 주세요."
            )
        )
    }


def handle_out_of_scope(_: CompetencyState) -> dict[str, Any]:
    return {
        "candidate_names": [],
        "messages": AIMessage(
            content=(
                "이 챗봇은 역량 레지스트리의 정의, 목록, 위계, 관계, 개수, 비교와 "
                "관련 역량 찾기 질문에 답합니다. 역량에 관한 질문으로 다시 입력해 주세요."
            )
        )
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(CompetencyState)
builder.add_node("interpret_query", interpret_query)
builder.add_node("llm_interpret_query", llm_interpret_query)
builder.add_node("validate_query_plan", validate_query_plan_node)
builder.add_node("find_semantic_candidates", find_semantic_candidates)
builder.add_node("find_competencies", find_competencies)
builder.add_node("execute_registry_query", execute_registry_query_node)
builder.add_node("write_streamed_answer", write_streamed_answer)
builder.add_node("produce_answer", produce_answer)
builder.add_node("present_candidates", present_candidates)
builder.add_node("answer_help", answer_help)
builder.add_node("clarify_query", clarify_query)
builder.add_node("handle_unknown", handle_unknown)
builder.add_node("handle_out_of_scope", handle_out_of_scope)

builder.add_edge(START, "interpret_query")
builder.add_conditional_edges("interpret_query", route_after_interpret)
builder.add_conditional_edges("llm_interpret_query", route_after_llm)
builder.add_conditional_edges("validate_query_plan", route_after_plan_validation)
builder.add_conditional_edges("find_semantic_candidates", route_after_candidates)
builder.add_conditional_edges("find_competencies", route_after_item_lookup)
builder.add_conditional_edges("execute_registry_query", route_after_execution)
builder.add_edge("write_streamed_answer", END)
builder.add_edge("produce_answer", END)
builder.add_edge("present_candidates", END)
builder.add_edge("answer_help", END)
builder.add_edge("clarify_query", END)
builder.add_edge("handle_unknown", END)
builder.add_edge("handle_out_of_scope", END)


# ---------------------------------------------------------------------------
# Runtime/checkpointer lifecycle and per-thread concurrency
# ---------------------------------------------------------------------------

class ThreadBusyError(RuntimeError):
    """같은 thread_id에 이미 실행 중인 요청이 있을 때 발생한다."""


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
    try:
        while not acquired:
            # Non-blocking acquisition leaves no worker-thread call behind
            # when this coroutine is cancelled while waiting.
            acquired = entry.lock.acquire(blocking=False)
            if not acquired:
                await asyncio.sleep(0.01)
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
                        emitted_answer_event = True
                        yield {"type": "delta", "text": data["text"]}
                    elif event_type == "replace" and isinstance(data.get("answer"), str):
                        emitted_answer_event = True
                        yield {"type": "replace", "answer": data["answer"]}

            answer = _last_answer(final_state)
            candidates = validate_registry_names(list(final_state.get("candidate_names", [])))
            if not emitted_answer_event:
                # help/semantic candidates/clarification처럼 writer를 거치지 않는
                # final node도 브라우저가 점진 응답 bubble을 채울 수 있게 한다.
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
            "message": "답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.",
            "retryable": True,
        }


def get_competency_state(thread_id: str) -> dict[str, Any]:
    with _runtime_execution() as compiled_app:
        snapshot = compiled_app.get_state(_graph_config(thread_id))
        return dict(snapshot.values) if snapshot.values else {}


def ask_competency(question: str, thread_id: str = "competency-chat-1") -> str:
    result = run_competency(question, thread_id)
    return str(result["messages"][-1].content)
