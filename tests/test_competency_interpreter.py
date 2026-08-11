"""LLM Gateway, Registry/Scope writers, checkpoint and concurrency tests."""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

import competency_interpreter
from competency_query import (
    ItemField,
    ParsedRegistryQuery,
    QueryIntent,
    RegistryQueryPlan,
    build_grounded_answer_context,
    execute_registry_query,
    render_grounded_fallback,
    validate_parsed_query,
)
from competency_registry import RegistrySnapshot, build_registry_snapshot
from llm_gateway import (
    FIXED_FAILURE_MESSAGE,
    MetaResponseDraft,
    OutOfScopeResponseDraft,
    ScopeTopic,
)


def make_synthetic_registry(version_id: int = 1) -> RegistrySnapshot:
    """Build a small, graph-correct registry without production files or DB."""

    items = [
        {
            "id": "cognition",
            "name": "인지력",
            "aliases": [],
            "definition": "정보를 이해하는 능력",
            "definition_status": "provided",
            "children": ["수인지력"],
            "children_ids": ["numeric_cognition"],
            "path": ["인지력"],
            "parent_name": None,
            "parent_id": None,
        },
        {
            "id": "numeric_cognition",
            "name": "수인지력",
            "aliases": [],
            "definition": "수를 인지하는 능력",
            "definition_status": "provided",
            "children": [],
            "children_ids": [],
            "path": ["인지력", "수인지력"],
            "parent_name": "인지력",
            "parent_id": "cognition",
        },
        {
            "id": "expression",
            "name": "의사표현",
            "aliases": ["표현능력_의사표현"],
            "definition": "의사를 명확히 표현하는 능력",
            "definition_status": "provided",
            "children": [],
            "children_ids": [],
            "path": ["의사표현"],
            "parent_name": None,
            "parent_id": None,
        },
        {
            "id": "responsibility",
            "name": "책임성",
            "aliases": [],
            "definition": "맡은 일을 책임지고 수행하는 특성",
            "definition_status": "provided",
            "children": [],
            "children_ids": [],
            "path": ["책임성"],
            "parent_name": None,
            "parent_id": None,
        },
        {
            "id": "achievement",
            "name": "성취추구",
            "aliases": [],
            "definition": "높은 성과를 추구하는 특성",
            "definition_status": "provided",
            "children": [],
            "children_ids": [],
            "path": ["성취추구"],
            "parent_name": None,
            "parent_id": None,
        },
    ]
    for item in items:
        item.update(
            {
                "instrument": "synthetic",
                "instrument_label": "합성 검사",
                "level": "factor",
                "analysis_included": False,
                "notes": [],
                "source_section": "테스트",
            }
        )

    source_hash = f"{version_id:064x}"
    document = {
        "schema_version": "1.0",
        "source": {
            "file": "synthetic.md",
            "sha256": source_hash,
            "encoding": "utf-8",
        },
        "rules": {"synthetic": "합성 레지스트리 규칙"},
        "validation": {
            "status": "passed",
            "counts": {"total": len(items)},
        },
        "items": items,
    }
    return build_registry_snapshot(
        {
            "id": version_id,
            "source_filename": "synthetic.md",
            "source_sha256": source_hash,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(items),
        }
    )


@pytest.fixture(autouse=True)
def synthetic_registry(monkeypatch: pytest.MonkeyPatch) -> RegistrySnapshot:
    competency_interpreter.close_competency_runtime()
    snapshot = make_synthetic_registry()
    monkeypatch.setattr(competency_interpreter, "_registry_snapshot", snapshot)
    yield snapshot
    competency_interpreter.close_competency_runtime()


def registry_decision(
    *,
    intent: str = "item_lookup",
    target_mentions: list[str] | None = None,
    constraint_mentions: list[dict[str, str]] | None = None,
    semantic_description: str | None = None,
    reuse_previous_result: bool = False,
    answer_mode: str = "registry_facts",
    acknowledge_greeting: bool = False,
    out_of_scope_remainder: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "decision": {
            "route": "registry_query",
            "draft": {
                "intent_hint": intent,
                "target_mentions": target_mentions or [],
                "constraint_mentions": constraint_mentions or [],
                "semantic_description": semantic_description,
                "reuse_previous_result": reuse_previous_result,
                "answer_mode": answer_mode,
                "acknowledge_greeting": acknowledge_greeting,
                "out_of_scope_remainder": out_of_scope_remainder,
            },
        }
    }


def meta_decision(kind: str) -> dict[str, Any]:
    return {"decision": {"route": "meta_conversation", "kind": kind}}


def scope_decision(category: str, summary: str) -> dict[str, Any]:
    return {
        "decision": {
            "route": "out_of_scope",
            "topic": {"category": category, "summary": summary},
        }
    }


def valid_scope_draft(summary: str) -> OutOfScopeResponseDraft:
    return OutOfScopeResponseDraft(
        acknowledgement=f"{summary}에 관해 질문하신 점을 이해했습니다",
        scope_boundary=(
            "이 챗봇은 등록된 역량 정보만 다루므로 그 요청의 내용을 직접 "
            "답하거나 판단하지 않습니다"
        ),
        registry_redirect=(
            "대신 등록 역량의 정의나 위계 관계를 질문하거나 행동 특징으로 "
            "역량 후보를 찾을 수 있습니다"
        ),
    )


def valid_meta_draft(kind: str) -> MetaResponseDraft:
    responses = {
        "greeting": "안녕하세요, 반갑습니다",
        "thanks": "고맙습니다, 도움이 되었다니 기쁩니다",
        "farewell": "안녕히 가세요, 다음에 또 뵙겠습니다",
        "bot_identity": "저는 등록된 역량 정보를 안내하는 역량 챗봇입니다",
        "capability_help": (
            "등록 역량의 정의·위계·관계·집계·비교와 행동 기반 후보 찾기를 "
            "도와드릴 수 있습니다"
        ),
    }
    return MetaResponseDraft(
        response=responses[kind],
        registry_redirect=(
            "등록된 역량명을 질문하거나 행동 특징을 설명해 역량 후보를 찾아보세요"
        ),
    )


class SequenceGateway:
    def __init__(self, outputs: list[object], trace: list[str] | None = None) -> None:
        self.outputs = list(outputs)
        self.trace = trace if trace is not None else []
        self.inputs: list[object] = []

    def invoke(self, model_input: object) -> object:
        self.trace.append("gateway")
        self.inputs.append(model_input)
        if not self.outputs:
            raise AssertionError("unexpected gateway call")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


class SequenceSelector:
    def __init__(self, outputs: list[object], trace: list[str] | None = None) -> None:
        self.outputs = list(outputs)
        self.trace = trace if trace is not None else []

    def invoke(self, _: object) -> object:
        self.trace.append("semantic_selector")
        if not self.outputs:
            raise AssertionError("unexpected semantic selector call")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def _reference_answer(model_input: object) -> str:
    messages = list(model_input)  # type: ignore[arg-type]
    human_prompt = str(messages[-1][1])
    marker = "사실 기준):\n"
    assert marker in human_prompt
    return human_prompt.split(marker, 1)[1].strip()


class StreamingWriter:
    def __init__(
        self,
        answer: str | Any,
        *,
        role: str,
        trace: list[str] | None = None,
        chunks: int = 1,
    ) -> None:
        self.answer = answer
        self.role = role
        self.trace = trace if trace is not None else []
        self.chunks = chunks
        self.inputs: list[object] = []

    async def astream(self, model_input: object):
        self.trace.append(self.role)
        self.inputs.append(model_input)
        answer = self.answer(model_input) if callable(self.answer) else self.answer
        if isinstance(answer, BaseException):
            raise answer
        text = str(answer)
        if self.chunks <= 1:
            yield AIMessageChunk(content=text)
            return
        width = max(1, len(text) // self.chunks)
        for start in range(0, len(text), width):
            yield AIMessageChunk(content=text[start : start + width])


class SequenceScopeWriter:
    """Return one fully buffered structured draft per Scope Writer request."""

    def __init__(
        self,
        outputs: list[object],
        trace: list[str] | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.trace = trace if trace is not None else []
        self.inputs: list[object] = []

    async def ainvoke(self, model_input: object) -> object:
        self.trace.append("scope_writer")
        self.inputs.append(model_input)
        if not self.outputs:
            raise AssertionError("unexpected scope writer call")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def _patch_models(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gateway: object,
    registry_writer: object | None = None,
    scope_writer: object | None = None,
    selector: object | None = None,
) -> None:
    monkeypatch.setattr(competency_interpreter, "_gateway_for", lambda _: gateway)
    if registry_writer is not None:
        monkeypatch.setattr(
            competency_interpreter,
            "_registry_writer_for",
            lambda _: registry_writer,
        )
    if scope_writer is not None:
        monkeypatch.setattr(
            competency_interpreter,
            "_scope_writer_for",
            lambda _model_name, _mode: scope_writer,
        )
    if selector is not None:
        monkeypatch.setattr(
            competency_interpreter,
            "_semantic_selector_for",
            lambda _: selector,
        )


def _compiled() -> Any:
    return competency_interpreter.builder.compile(checkpointer=InMemorySaver())


def _invoke(compiled: Any, question: str, thread_id: str = "test-thread") -> dict[str, Any]:
    return asyncio.run(
        compiled.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}},
        )
    )


def _validated_registry_state(
    *,
    names: list[str] | None = None,
    fields: list[ItemField] | None = None,
    llm_call_count: int = 1,
) -> dict[str, Any]:
    parsed = ParsedRegistryQuery(
        intent=QueryIntent.ITEM_LOOKUP,
        target_names=names or ["책임성"],
        fields=fields or [ItemField.DEFINITION],
    )
    validation = validate_parsed_query(
        parsed,
        competency_interpreter._require_registry(),
        user_question="책임성의 정의를 알려줘",
    )
    assert validation.plan is not None
    result = execute_registry_query(
        validation.plan,
        competency_interpreter._require_registry(),
    )
    return {
        "messages": [HumanMessage(content="책임성의 정의를 알려줘")],
        "raw_query": "책임성의 정의를 알려줘",
        "query_plan": validation.plan.model_dump(mode="json"),
        "query_result": result.model_dump(mode="json"),
        "result_ids": list(result.item_ids),
        "registry_answer_mode": "result",
        "llm_call_count": llm_call_count,
        "writer_attempts": 0,
    }


def test_graph_starts_at_gateway_and_has_no_legacy_terminal_nodes() -> None:
    graph = competency_interpreter.builder.compile().get_graph()

    assert [(edge.source, edge.target) for edge in graph.edges if edge.source == "__start__"] == [
        ("__start__", "llm_gateway")
    ]
    assert {
        "interpret_query",
        "llm_interpret_query",
        "write_streamed_answer",
        "produce_answer",
        "present_candidates",
        "answer_help",
        "clarify_query",
        "handle_unknown",
        "handle_out_of_scope",
        "write_general_answer",
    }.isdisjoint(graph.nodes)


def test_runtime_metrics_accept_only_safe_aggregate_components() -> None:
    before = competency_interpreter.runtime_metric_snapshot().get(
        "normalizer_rule.deterministic_fast_path",
        0,
    )

    competency_interpreter._record_runtime_metric(
        "normalizer_rule",
        "deterministic_fast_path",
    )
    competency_interpreter._record_runtime_metric(
        "normalizer_rule",
        "raw user question",
    )

    metrics = competency_interpreter.runtime_metric_snapshot()
    assert metrics["normalizer_rule.deterministic_fast_path"] == before + 1
    assert "raw user question" not in repr(metrics)


def test_exact_name_calls_gateway_first_and_direct_path_uses_two_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    gateway = SequenceGateway(
        [
            registry_decision(
                target_mentions=["책임성"],
                constraint_mentions=[{"kind": "field", "text": "정의"}],
            )
        ],
        trace,
    )
    writer = StreamingWriter(_reference_answer, role="registry_writer", trace=trace)
    _patch_models(monkeypatch, gateway=gateway, registry_writer=writer)

    result = _invoke(_compiled(), "책임성의 정의를 알려줘")

    assert trace == ["gateway", "registry_writer"]
    assert result["llm_call_count"] == 2
    assert result["gateway_attempts"] == 1
    assert result["last_result_ids"] == ["responsibility"]
    assert "맡은 일을 책임지고 수행하는 특성" in result["messages"][-1].content


@pytest.mark.parametrize(
    ("intent", "question", "draft_kwargs"),
    [
        ("catalog_query", "전체 역량 목록을 알려줘", {}),
        ("hierarchy_query", "전체 역량의 위계 구조를 알려줘", {}),
        (
            "relation_query",
            "인지력의 직접 하위요인을 알려줘",
            {
                "target_mentions": ["인지력"],
                "constraint_mentions": [
                    {"kind": "relation", "text": "직접 하위요인"}
                ],
            },
        ),
        ("aggregate_query", "전체 역량 개수를 알려줘", {}),
        (
            "comparison_query",
            "책임성과 성취추구의 정의를 비교해 줘",
            {
                "target_mentions": ["책임성", "성취추구"],
                "constraint_mentions": [{"kind": "field", "text": "정의"}],
            },
        ),
    ],
)
def test_gateway_preserves_deterministic_registry_intents(
    monkeypatch: pytest.MonkeyPatch,
    intent: str,
    question: str,
    draft_kwargs: dict[str, Any],
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [registry_decision(intent=intent, **draft_kwargs)], trace
        ),
        registry_writer=StreamingWriter(
            _reference_answer,
            role="registry_writer",
            trace=trace,
        ),
    )

    result = _invoke(_compiled(), question)

    assert trace == ["gateway", "registry_writer"]
    assert result["llm_call_count"] == 2
    assert result["query_plan"]["intent"] == intent
    assert result["query_result"]
    assert result["messages"][-1].content != FIXED_FAILURE_MESSAGE


def test_registry_clarification_preserves_grounded_numeric_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [
                registry_decision(
                    intent="comparison_query",
                    target_mentions=[
                        "책임성",
                        "성취추구",
                        "인지력",
                        "수인지력",
                    ],
                    constraint_mentions=[{"kind": "field", "text": "정의"}],
                )
            ]
        ),
        registry_writer=StreamingWriter(
            _reference_answer,
            role="registry_writer",
        ),
    )

    result = _invoke(
        _compiled(),
        "책임성, 성취추구, 인지력, 수인지력의 정의를 한꺼번에 비교해 줘",
    )

    assert result["registry_answer_mode"] == "clarification"
    assert result["llm_call_count"] == 2
    assert "3개까지" in result["messages"][-1].content
    assert result["messages"][-1].content != FIXED_FAILURE_MESSAGE


@pytest.mark.parametrize(
    ("question", "decision", "draft", "expected_route", "expected_mode"),
    [
        (
            "안녕하세요",
            meta_decision("greeting"),
            valid_meta_draft("greeting"),
            "meta_conversation",
            "greeting",
        ),
        (
            "고마워",
            meta_decision("thanks"),
            valid_meta_draft("thanks"),
            "meta_conversation",
            "thanks",
        ),
        (
            "다음에 봐",
            meta_decision("farewell"),
            valid_meta_draft("farewell"),
            "meta_conversation",
            "farewell",
        ),
        (
            "너는 누구야?",
            meta_decision("bot_identity"),
            valid_meta_draft("bot_identity"),
            "meta_conversation",
            "bot_identity",
        ),
        (
            "무엇을 할 수 있어?",
            meta_decision("capability_help"),
            valid_meta_draft("capability_help"),
            "meta_conversation",
            "capability_help",
        ),
        (
            "오늘 날씨 어때?",
            scope_decision("weather", "오늘 날씨"),
            valid_scope_draft("오늘 날씨"),
            "out_of_scope",
            "out_of_scope",
        ),
    ],
)
def test_meta_and_scope_routes_use_structured_scope_writer_in_two_calls(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    decision: dict[str, Any],
    draft: object,
    expected_route: str,
    expected_mode: str,
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway([decision], trace),
        scope_writer=SequenceScopeWriter([draft], trace),
    )

    result = _invoke(_compiled(), question)

    assert trace == ["gateway", "scope_writer"]
    assert result["llm_call_count"] == 2
    assert result["gateway_route"] == expected_route
    assert result["scope_mode"] == expected_mode
    assert result["response_mode"] == "llm"
    assert result["messages"][-1].content != FIXED_FAILURE_MESSAGE


def test_f24_mixed_greeting_and_registry_question_prioritizes_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = SequenceGateway(
        [
            registry_decision(
                target_mentions=["책임성"],
                constraint_mentions=[{"kind": "field", "text": "정의"}],
                acknowledge_greeting=True,
            )
        ]
    )

    def greeted_reference(model_input: object) -> str:
        return f"안녕하세요. {_reference_answer(model_input)}"

    writer = StreamingWriter(greeted_reference, role="registry_writer")
    _patch_models(monkeypatch, gateway=gateway, registry_writer=writer)

    result = _invoke(_compiled(), "안녕, 책임성 정의를 알려줘")

    assert result["gateway_route"] == "registry_query"
    assert result["acknowledge_greeting"] is True
    assert result["messages"][-1].content.startswith("안녕하세요.")
    assert "맡은 일을 책임지고 수행하는 특성" in result["messages"][-1].content
    assert "인사를 짧게 반영: True" in str(writer.inputs[0])


def test_guidance_mode_separates_registry_fact_from_general_suggestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = SequenceGateway(
        [
            registry_decision(
                target_mentions=["책임성"],
                constraint_mentions=[{"kind": "field", "text": "정의"}],
                answer_mode="registry_facts_with_general_guidance",
            )
        ]
    )

    def guidance_answer(model_input: object) -> str:
        return (
            f"[등록 정보]\n{_reference_answer(model_input)}\n\n"
            "[일반 활용 제안]\n업무를 시작하기 전에 할 일을 확인하고 완료 여부를 돌아볼 수 있습니다."
        )

    _patch_models(
        monkeypatch,
        gateway=gateway,
        registry_writer=StreamingWriter(guidance_answer, role="registry_writer"),
    )

    result = _invoke(_compiled(), "책임성을 높이려면 어떻게 해야 해?")

    answer = result["messages"][-1].content
    assert result["llm_call_count"] == 2
    assert result["registry_answer_mode"] == "guidance"
    assert "[등록 정보]" in answer and "[일반 활용 제안]" in answer
    assert "맡은 일을 책임지고 수행하는 특성" in answer


def test_f26_high_confidence_semantic_match_uses_at_most_three_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [
                registry_decision(
                    intent="semantic_search",
                    semantic_description="맡은 일을 끝까지 수행",
                    constraint_mentions=[{"kind": "field", "text": "정의"}],
                )
            ],
            trace,
        ),
        selector=SequenceSelector(
            [
                {
                    "candidate_names": ["책임성"],
                    "confidence": "high",
                    "auto_select": True,
                }
            ],
            trace,
        ),
        registry_writer=StreamingWriter(
            _reference_answer,
            role="registry_writer",
            trace=trace,
        ),
    )

    result = _invoke(_compiled(), "맡은 일을 끝까지 수행하는 역량은?")

    assert trace == ["gateway", "semantic_selector", "registry_writer"]
    assert result["llm_call_count"] == 3
    assert result["candidate_names"] == []
    assert result["last_result_ids"] == ["responsibility"]
    assert "맡은 일을 책임지고 수행하는 특성" in result["messages"][-1].content


@pytest.mark.parametrize(
    "selection",
    [
        {"candidate_names": ["책임성"], "confidence": "low", "auto_select": False},
        {
            "candidate_names": ["책임성", "없는 역량", "성취추구"],
            "confidence": "medium",
            "auto_select": False,
        },
    ],
)
def test_f06_f29_multiple_semantic_matches_publish_only_validated_candidates(
    monkeypatch: pytest.MonkeyPatch,
    selection: dict[str, Any],
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [
                registry_decision(
                    intent="semantic_search",
                    semantic_description="주도적으로 일함",
                    constraint_mentions=[{"kind": "field", "text": "역량"}],
                )
            ],
            trace,
        ),
        selector=SequenceSelector([selection], trace),
        registry_writer=StreamingWriter(
            _reference_answer,
            role="registry_writer",
            trace=trace,
        ),
    )

    result = _invoke(_compiled(), "주도적으로 일하는 역량은?")

    assert trace == ["gateway", "semantic_selector", "registry_writer"]
    assert result["llm_call_count"] == 3
    assert result["registry_answer_mode"] == "candidates"
    assert result["candidate_names"] == [
        name for name in selection["candidate_names"] if name != "없는 역량"
    ][:3]
    assert "없는 역량" not in result["messages"][-1].content
    assert "정확한 역량명" in result["messages"][-1].content


def test_gateway_schema_error_retries_once_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    gateway = SequenceGateway(
        [
            {"route": "not-a-route"},
            meta_decision("greeting"),
        ],
        trace,
    )
    _patch_models(
        monkeypatch,
        gateway=gateway,
        scope_writer=SequenceScopeWriter([valid_meta_draft("greeting")], trace),
    )

    result = _invoke(_compiled(), "안녕하세요")

    assert trace == ["gateway", "gateway", "scope_writer"]
    assert result["gateway_attempts"] == 2
    assert result["llm_call_count"] == 3
    assert result["messages"][-1].content != FIXED_FAILURE_MESSAGE


def test_f21_gateway_final_schema_failure_returns_only_fixed_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [{"route": "bad"}, {"route": "still-bad"}],
            trace,
        ),
    )

    result = _invoke(_compiled(), "해석할 수 없는 질문")

    assert trace == ["gateway", "gateway"]
    assert result["llm_call_count"] == 2
    assert result["messages"][-1].content == FIXED_FAILURE_MESSAGE
    assert result["response_mode"] == "failure"


def test_registry_writer_buffers_all_tokens_until_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _validated_registry_state()
    context = competency_interpreter._registry_grounding_context(state)
    expected = render_grounded_fallback(context)
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(competency_interpreter, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(
        competency_interpreter,
        "_registry_writer_for",
        lambda _: StreamingWriter(expected, role="registry_writer", chunks=3),
    )

    draft = asyncio.run(competency_interpreter.write_registry_answer(state))

    assert draft["writer_failed"] is False
    assert not any(event["type"] in {"delta", "replace"} for event in events)

    committed = competency_interpreter.validate_registry_answer_node({**state, **draft})
    assert committed["messages"] == AIMessage(content=expected)
    assert [event for event in events if event["type"] == "delta"] == [
        {"type": "delta", "text": expected, "commit_required": True}
    ]


def test_registry_delta_waits_for_matching_checkpoint_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = "검증되고 체크포인트된 Registry 답변"
    checkpoint_value_seen = False

    class CommitOrderedApp:
        async def astream(self, *_: object, **__: object):
            nonlocal checkpoint_value_seen
            yield "custom", {
                "type": "delta",
                "text": answer,
                "commit_required": True,
            }
            checkpoint_value_seen = True
            yield "values", {
                "messages": [AIMessage(content=answer)],
                "candidate_names": [],
            }

    monkeypatch.setattr(competency_interpreter, "app", CommitOrderedApp())
    monkeypatch.setattr(
        competency_interpreter,
        "initialize_competency_runtime",
        lambda: None,
    )

    async def collect() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for event in competency_interpreter.run_competency_stream(
            "질문",
            "registry-commit-order",
        ):
            if event.get("type") == "delta":
                assert checkpoint_value_seen
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert [event for event in events if event.get("type") == "delta"] == [
        {"type": "delta", "text": answer}
    ]
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == answer


def test_registry_delta_is_checkpointed_before_public_stream_can_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    compiled = competency_interpreter.builder.compile(checkpointer=checkpointer)
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [
                registry_decision(
                    target_mentions=["책임성"],
                    constraint_mentions=[{"kind": "field", "text": "정의"}],
                )
            ]
        ),
        registry_writer=StreamingWriter(
            _reference_answer,
            role="registry_writer",
        ),
    )
    monkeypatch.setattr(competency_interpreter, "app", compiled)
    monkeypatch.setattr(
        competency_interpreter,
        "initialize_competency_runtime",
        lambda: None,
    )
    public_answer = ""

    async def cancel_after_registry_delta() -> None:
        nonlocal public_answer
        stream = competency_interpreter.run_competency_stream(
            "책임성의 정의를 알려줘",
            "registry-cancel-after-commit",
        )
        async for event in stream:
            if event.get("type") == "delta":
                public_answer = str(event["text"])
                await stream.aclose()
                return
        raise AssertionError("committed registry delta was not emitted")

    asyncio.run(cancel_after_registry_delta())
    saved = compiled.get_state(
        {"configurable": {"thread_id": "registry-cancel-after-commit"}}
    ).values

    assert public_answer
    assert isinstance(saved["messages"][-1], AIMessage)
    assert saved["messages"][-1].content == public_answer
    assert saved["registry_answer"] == ""


def test_changed_registry_definition_is_never_public_and_ends_in_fixed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _validated_registry_state()
    altered = "책임성의 정의는 맡은 일을 대충 수행하는 특성입니다."
    events: list[dict[str, Any]] = []
    writer = StreamingWriter(altered, role="registry_writer")
    monkeypatch.setattr(competency_interpreter, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(competency_interpreter, "_registry_writer_for", lambda _: writer)

    draft = asyncio.run(competency_interpreter.write_registry_answer(state))
    validation = competency_interpreter.validate_registry_answer_node({**state, **draft})
    final = competency_interpreter.fixed_failure_message({**state, **draft, **validation})

    assert draft["writer_failed"] is True
    assert draft["writer_attempts"] == 2
    assert draft["llm_call_count"] == 3
    assert len(writer.inputs) == 2
    assert altered not in str(events)
    assert final["messages"].content == FIXED_FAILURE_MESSAGE
    assert events[-1] == {
        "type": "delta",
        "text": FIXED_FAILURE_MESSAGE,
        "commit_required": True,
    }


def test_registry_validator_rejects_contradiction_after_exact_reference() -> None:
    state = _validated_registry_state()
    context = competency_interpreter._registry_grounding_context(state)
    reference = competency_interpreter._registry_reference_answer(context, "result")

    assert not competency_interpreter._registry_answer_is_valid(
        f"{reference}\n하지만 실제 정의는 맡은 일을 피하는 특성입니다.",
        context,
        "result",
    )


@pytest.mark.parametrize(
    "guidance",
    [
        "책임성은 맡은 일을 피하는 특성입니다.",
        "혁신민첩성은 빠르게 혁신하는 태도입니다.",
        "빠른 혁신을 혁신민첩성이라 부르며 업무에서 시도할 수 있습니다.",
        "Consider rapid innovation, also called Innovagility.",
        "업무에서 혁신민첩성(빠른 혁신을 추구하는 자세)을 연습해 볼 수 있습니다.",
        "당신의 현재 수준은 95점입니다.",
        "당신은 채용에 적합하고 합격할 것입니다.",
        "이 활동은 생산성을 99% 높인다는 연구 결과가 있습니다.",
        "현재 상태가 아주 뛰어납니다. 매일 진행 상황을 확인해 볼 수 있습니다.",
        "취업 가능성이 높아 보입니다. 매일 점검해 볼 수 있습니다.",
    ],
)
def test_guidance_validator_rejects_registry_fact_in_suggestion_section(
    guidance: str,
) -> None:
    state = _validated_registry_state()
    state["registry_answer_mode"] = "guidance"
    context = competency_interpreter._registry_grounding_context(state)
    reference = competency_interpreter._registry_reference_answer(context, "guidance")
    answer = (
        f"[등록 정보]\n{reference}\n\n"
        f"[일반 활용 제안]\n{guidance}"
    )

    assert not competency_interpreter._registry_answer_is_valid(
        answer,
        context,
        "guidance",
    )


@pytest.mark.parametrize(
    ("mode", "answer_builder", "kwargs"),
    [
        (
            "result",
            lambda reference: (
                "안녕하세요. 혁신민첩성은 빠르게 혁신하는 능력입니다.\n"
                f"{reference}"
            ),
            {"acknowledge_greeting": True},
        ),
        (
            "result",
            lambda reference: (
                f"{reference}\n[지원 범위]\n"
                "날씨 요청에 관해 질문하신 점은 이해하지만 등록 역량 정보만 "
                "다루므로 날씨를 직접 답하지 않습니다. "
                "혁신민첩성은 빠르게 혁신하는 능력입니다."
            ),
            {
                "scope_topic": ScopeTopic(
                    category="weather",
                    summary="오늘 날씨",
                )
            },
        ),
        (
            "guidance",
            lambda reference: (
                f"[등록 정보]\n{reference}\n\n[일반 활용 제안]\n"
                "혁신민첩성은 빠르게 혁신하는 능력입니다."
            ),
            {},
        ),
    ],
)
def test_registry_framing_rejects_invented_unregistered_definition(
    mode: str,
    answer_builder: Any,
    kwargs: dict[str, Any],
) -> None:
    state = _validated_registry_state()
    state["registry_answer_mode"] = mode
    context = competency_interpreter._registry_grounding_context(state)
    reference = competency_interpreter._registry_reference_answer(context, mode)

    assert not competency_interpreter._registry_answer_is_valid(
        answer_builder(reference),
        context,
        mode,
        **kwargs,
    )


def test_candidate_validator_rejects_writer_injected_unregistered_candidate() -> None:
    state = {
        "messages": [HumanMessage(content="주도적으로 일하는 역량은?")],
        "raw_query": "주도적으로 일하는 역량은?",
        "registry_answer_mode": "candidates",
        "candidate_ids": ["responsibility", "achievement"],
    }
    context = competency_interpreter._registry_grounding_context(state)
    reference = competency_interpreter._registry_reference_answer(context, "candidates")

    assert not competency_interpreter._registry_answer_is_valid(
        f"{reference}\n3. 혁신민첩성: 빠르게 혁신하는 능력",
        context,
        "candidates",
    )


def test_mixed_registry_out_of_scope_requires_safe_scope_note() -> None:
    state = _validated_registry_state()
    context = competency_interpreter._registry_grounding_context(state)
    reference = competency_interpreter._registry_reference_answer(context, "result")
    topic = ScopeTopic(category="weather", summary="오늘 날씨")
    scope = (
        "오늘 날씨 요청에 관해 질문하신 점은 이해하지만 이 챗봇은 등록 역량 "
        "정보만 다루므로 날씨를 직접 답하지 않습니다."
    )

    assert not competency_interpreter._registry_answer_is_valid(
        reference,
        context,
        "result",
        scope_topic=topic,
    )
    assert competency_interpreter._registry_answer_is_valid(
        f"{reference}\n[지원 범위]\n{scope}",
        context,
        "result",
        scope_topic=topic,
    )


@pytest.mark.parametrize(
    "guidance",
    [
        "매일 10분 동안 해야 하는 일을 확인해 볼 수 있습니다.",
        "매일 해야 하는 일을 확인할 수 있습니다.",
        "동료와 협력하려는 행동을 시도해 볼 수 있습니다.",
        "회의 전에는 안건을 확인해 볼 수 있습니다.",
        "체크리스트는 완료 여부를 돌아보는 데 참고할 수 있습니다.",
    ],
)
def test_guidance_validator_accepts_safe_action_suggestions(guidance: str) -> None:
    state = _validated_registry_state()
    state["registry_answer_mode"] = "guidance"
    context = competency_interpreter._registry_grounding_context(state)
    reference = competency_interpreter._registry_reference_answer(context, "guidance")
    answer = f"[등록 정보]\n{reference}\n\n[일반 활용 제안]\n{guidance}"

    assert competency_interpreter._registry_answer_is_valid(
        answer,
        context,
        "guidance",
    )


def test_f04_alias_definition_remains_registry_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [
                registry_decision(
                    target_mentions=["표현능력_의사표현"],
                    constraint_mentions=[{"kind": "field", "text": "정의"}],
                )
            ]
        ),
        registry_writer=StreamingWriter(_reference_answer, role="registry_writer"),
    )

    result = _invoke(_compiled(), "표현능력_의사표현의 정의를 알려줘")

    assert result["last_result_ids"] == ["expression"]
    assert "의사를 명확히 표현하는 능력" in result["messages"][-1].content
    assert result["llm_call_count"] == 2


def test_f05_unknown_competency_like_term_never_uses_general_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [registry_decision(target_mentions=["정의감"])]
        ),
        registry_writer=StreamingWriter(_reference_answer, role="registry_writer"),
    )

    result = _invoke(_compiled(), "정의감은 뭐야?")
    answer = result["messages"][-1].content

    assert result["gateway_route"] == "registry_query"
    assert result["registry_answer_mode"] == "unregistered"
    assert "등록된 정식 이름이나 별칭으로 확인되지 않습니다" in answer
    assert "정의감은" not in answer
    assert result["response_mode"] == "llm"


def test_f07_ambiguous_relation_explains_the_actual_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [
                registry_decision(
                    intent="relation_query",
                    target_mentions=["인지력"],
                    constraint_mentions=[{"kind": "relation", "text": "하위 항목"}],
                )
            ]
        ),
        registry_writer=StreamingWriter(_reference_answer, role="registry_writer"),
    )

    result = _invoke(_compiled(), "인지력의 하위 항목을 알려줘")
    answer = result["messages"][-1].content

    assert result["normalization_issue"]["code"] == "ambiguous_relation"
    assert "직접 하위요인" in answer
    assert "모든 하위요인" in answer
    assert answer != FIXED_FAILURE_MESSAGE


@pytest.mark.parametrize(
    ("question", "category", "summary", "forbidden"),
    [
        ("오늘이 며칠이야?", "date_time", "오늘 날짜", "2026"),
        ("지금 몇 시야?", "date_time", "현재 시각", "오후"),
        ("오늘 날씨 어때?", "weather", "오늘 날씨", "맑"),
        ("최신 뉴스가 뭐야?", "news_current_events", "최신 뉴스", "속보"),
        ("양자역학은 뭐야?", "general_knowledge", "양자역학 개념", "입자"),
    ],
)
def test_f08_f12_substantive_non_registry_questions_get_no_answer_scope_redirect(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    category: str,
    summary: str,
    forbidden: str,
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway([scope_decision(category, summary)], trace),
        scope_writer=SequenceScopeWriter([valid_scope_draft(summary)], trace),
    )

    result = _invoke(_compiled(), question)
    answer = result["messages"][-1].content

    assert trace == ["gateway", "scope_writer"]
    assert result["gateway_route"] == "out_of_scope"
    assert result["response_mode"] == "llm"
    assert result["llm_call_count"] == 2
    assert summary in answer
    assert "등록" in answer and "역량" in answer
    assert forbidden not in answer
    assert answer != FIXED_FAILURE_MESSAGE


@pytest.mark.parametrize(
    ("question", "category", "summary", "forbidden"),
    [
        ("내 역량 점수를 추정해줘", "personal_assessment", "개인 역량 평가", "점입니다"),
        ("이 지원자를 채용해야 해?", "employment_decision", "채용 판단", "채용해야"),
        ("이 약을 얼마나 먹어야 해?", "professional_advice", "의료 조언", "복용"),
        ("소송을 제기할까?", "professional_advice", "법률 조언", "제기하세요"),
        ("이 주식을 살까?", "professional_advice", "금융 조언", "매수"),
    ],
)
def test_f17_sensitive_requests_get_topic_specific_boundary_without_judgment(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    category: str,
    summary: str,
    forbidden: str,
) -> None:
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway([scope_decision(category, summary)]),
        scope_writer=SequenceScopeWriter([valid_scope_draft(summary)]),
    )

    result = _invoke(_compiled(), question)
    answer = result["messages"][-1].content

    assert summary in answer
    assert forbidden not in answer
    assert result["response_mode"] == "llm"
    assert result["messages"][-1].content != FIXED_FAILURE_MESSAGE


def _scope_state(*, raw_query: str = "오늘 날씨 어때?") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=raw_query)],
        "raw_query": raw_query,
        "scope_mode": "out_of_scope",
        "scope_topic_category": "weather",
        "scope_topic_summary": "오늘 날씨",
        "llm_call_count": 1,
        "scope_writer_attempts": 0,
    }


def test_f18_scope_actual_answer_is_rejected_before_one_valid_answer_is_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = OutOfScopeResponseDraft(
        acknowledgement="오늘 날씨 질문의 답은 맑고 28도입니다",
        scope_boundary="이 챗봇은 등록 역량 정보만 다룹니다",
        registry_redirect="대신 등록 역량의 정의를 질문해 보세요",
    )
    safe = valid_scope_draft("오늘 날씨")
    writer = SequenceScopeWriter([unsafe, safe])
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(competency_interpreter, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(
        competency_interpreter,
        "_scope_writer_for",
        lambda _model_name, _mode: writer,
    )

    update = asyncio.run(competency_interpreter.write_scope_answer(_scope_state()))

    assert update["scope_writer_attempts"] == 2
    assert update["llm_call_count"] == 3
    assert update["response_mode"] == "llm"
    assert "28도" not in str(events)
    assert [event for event in events if event["type"] == "delta"] == [
        {
            "type": "delta",
            "text": update["messages"].content,
            "commit_required": True,
        }
    ]


def test_f19_scope_first_request_failure_then_success_publishes_only_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = SequenceScopeWriter(
        [RuntimeError("provider failed"), valid_scope_draft("오늘 날씨")]
    )
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(competency_interpreter, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(
        competency_interpreter,
        "_scope_writer_for",
        lambda _model_name, _mode: writer,
    )

    update = asyncio.run(competency_interpreter.write_scope_answer(_scope_state()))

    assert update["scope_writer_attempts"] == 2
    assert update["llm_call_count"] == 3
    assert update["response_mode"] == "llm"
    assert len([event for event in events if event["type"] == "delta"]) == 1


def test_f20_two_scope_failures_use_category_fallback_not_fixed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = SequenceScopeWriter(
        [RuntimeError("first"), RuntimeError("second")]
    )
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(competency_interpreter, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(
        competency_interpreter,
        "_scope_writer_for",
        lambda _model_name, _mode: writer,
    )

    update = asyncio.run(competency_interpreter.write_scope_answer(_scope_state()))
    answer = update["messages"].content

    assert update["scope_writer_attempts"] == 2
    assert update["llm_call_count"] == 3
    assert update["response_mode"] == "scope_fallback"
    assert "날씨" in answer and "등록" in answer and "역량" in answer
    assert answer != FIXED_FAILURE_MESSAGE
    assert events == [
        {"type": "status", "stage": "답변을 작성하는 중"},
        {"type": "delta", "text": answer, "commit_required": True},
    ]


@pytest.mark.parametrize(
    ("category", "summary", "raw_query"),
    [
        ("date_time", "오늘 오늘", "오늘이 며칠이야?"),
        ("date_time", "오늘 현재", "현재 시각 알려줘"),
        ("weather", "weather sunny", "What is the weather?"),
        ("news_current_events", "president resigned", "Latest news?"),
    ],
)
def test_exhausted_scope_budget_still_uses_validated_topic_fallback(
    category: str,
    summary: str,
    raw_query: str,
) -> None:
    state = _scope_state(raw_query=raw_query)
    state.update(
        {
            "scope_topic_category": category,
            "scope_topic_summary": summary,
            "llm_call_count": 3,
            "scope_writer_attempts": 2,
        }
    )

    update = asyncio.run(competency_interpreter.write_scope_answer(state))

    assert update["next_route"] == END
    assert update["response_mode"] == "scope_fallback"
    assert update["messages"].content != FIXED_FAILURE_MESSAGE


def test_f22_repeated_scope_turns_receive_last_scope_answer_as_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = valid_scope_draft("오늘 날씨")
    second = OutOfScopeResponseDraft(
        acknowledgement="오늘 날씨를 묻고 계시는군요",
        scope_boundary=(
            "이 챗봇은 등록된 역량 정보만 다루므로 오늘 날씨 내용은 직접 "
            "답하지 않습니다"
        ),
        registry_redirect=(
            "대신 등록 역량의 관계를 질문하거나 행동 특징으로 후보를 찾을 수 "
            "있습니다"
        ),
    )
    writer = SequenceScopeWriter([first, second])
    gateway = SequenceGateway(
        [
            scope_decision("weather", "오늘 날씨"),
            scope_decision("weather", "오늘 날씨"),
        ]
    )
    _patch_models(monkeypatch, gateway=gateway, scope_writer=writer)
    compiled = _compiled()

    first_result = _invoke(compiled, "오늘 날씨 어때?", "repeat-scope")
    second_result = _invoke(compiled, "오늘 날씨 다시 알려줘", "repeat-scope")

    assert first_result["messages"][-1].content != second_result["messages"][-1].content
    assert first_result["messages"][-1].content in str(writer.inputs[1])
    assert second_result["response_mode"] == "llm"


def test_f23_english_scope_question_gets_english_boundary_and_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = OutOfScopeResponseDraft(
        acknowledgement="You're asking about today's weather",
        scope_boundary=(
            "This chatbot is limited to registered competency information"
        ),
        registry_redirect=(
            "You can instead ask for a registered competency definition or hierarchy"
        ),
    )
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway([scope_decision("weather", "weather request")]),
        scope_writer=SequenceScopeWriter([draft]),
    )

    result = _invoke(_compiled(), "What is the weather today?")
    answer = result["messages"][-1].content

    assert "weather" in answer.casefold()
    assert "registered competency" in answer.casefold()
    assert not any("가" <= char <= "힣" for char in answer)
    assert result["response_mode"] == "llm"


def test_f28_stream_nonstream_and_history_answers_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled()
    gateway = SequenceGateway(
        [
            scope_decision("weather", "오늘 날씨"),
            scope_decision("weather", "오늘 날씨"),
        ]
    )
    writer = SequenceScopeWriter(
        [valid_scope_draft("오늘 날씨"), valid_scope_draft("오늘 날씨")]
    )
    _patch_models(monkeypatch, gateway=gateway, scope_writer=writer)
    monkeypatch.setattr(competency_interpreter, "app", compiled)
    monkeypatch.setattr(competency_interpreter, "initialize_competency_runtime", lambda: None)

    async def collect_stream() -> list[dict[str, Any]]:
        return [
            event
            async for event in competency_interpreter.run_competency_stream(
                "오늘 날씨 어때?",
                "byte-stream",
            )
        ]

    stream_events = asyncio.run(collect_stream())
    stream_answer = next(
        event["answer"] for event in stream_events if event["type"] == "done"
    )
    nonstream_state = competency_interpreter.run_competency(
        "오늘 날씨 어때?",
        "byte-nonstream",
    )
    history_state = competency_interpreter.get_competency_state("byte-stream")

    assert stream_answer == nonstream_state["messages"][-1].content
    assert stream_answer == history_state["messages"][-1].content
    assert [
        event["text"] for event in stream_events if event["type"] == "delta"
    ] == [stream_answer]


def test_f25_registry_plus_scope_uses_registry_writer_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []

    def mixed_answer(model_input: object) -> str:
        return (
            f"{_reference_answer(model_input)}\n[지원 범위]\n"
            "오늘 날씨 요청에 관해 질문하신 점은 이해하지만 이 챗봇은 등록 "
            "역량 정보만 다루므로 날씨를 직접 답하지 않습니다."
        )

    class UnexpectedScopeWriter:
        async def ainvoke(self, _: object) -> object:
            raise AssertionError("mixed registry input must not call Scope Writer")

    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [
                registry_decision(
                    target_mentions=["책임성"],
                    constraint_mentions=[{"kind": "field", "text": "정의"}],
                    out_of_scope_remainder={
                        "category": "weather",
                        "summary": "오늘 날씨",
                    },
                )
            ],
            trace,
        ),
        registry_writer=StreamingWriter(
            mixed_answer,
            role="registry_writer",
            trace=trace,
        ),
        scope_writer=UnexpectedScopeWriter(),
    )

    result = _invoke(_compiled(), "책임성의 정의와 오늘 날씨를 알려줘")
    answer = result["messages"][-1].content

    assert trace == ["gateway", "registry_writer"]
    assert "맡은 일을 책임지고 수행하는 특성" in answer
    assert "[지원 범위]" in answer and "날씨를 직접 답하지 않습니다" in answer
    assert result["llm_call_count"] == 2


def test_legacy_general_route_is_ignored_by_new_turn_and_router() -> None:
    updates = competency_interpreter._new_turn_updates("안녕하세요")
    assert "general_route" not in updates
    assert competency_interpreter.route_after_gateway(
        {
            "gateway_route": "meta_conversation",
            "general_route": "simple_concept",
        }
    ) == "write_scope_answer"


def test_same_thread_follow_up_uses_revalidated_stable_id_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContextGateway:
        def __init__(self) -> None:
            self.inputs: list[object] = []

        def invoke(self, model_input: object) -> object:
            self.inputs.append(model_input)
            return registry_decision(
                target_mentions=["책임성"] if len(self.inputs) == 1 else [],
                constraint_mentions=(
                    [{"kind": "field", "text": "정의"}]
                    if len(self.inputs) == 1
                    else []
                ),
                reuse_previous_result=len(self.inputs) > 1,
            )

    gateway = ContextGateway()
    _patch_models(
        monkeypatch,
        gateway=gateway,
        registry_writer=StreamingWriter(_reference_answer, role="registry_writer"),
    )
    compiled = _compiled()

    first = _invoke(compiled, "책임성의 정의를 알려줘", "follow-up")
    second = _invoke(compiled, "그 역량을 다시 설명해줘", "follow-up")

    second_system_prompt = str(list(gateway.inputs[1])[0][1])
    assert first["last_result_ids"] == ["responsibility"]
    assert second["last_result_ids"] == ["responsibility"]
    assert '"id": "responsibility"' in second_system_prompt
    assert '"name": "책임성"' in second_system_prompt


def test_stale_stable_id_is_not_recovered_from_a_name_string() -> None:
    prompt = competency_interpreter._gateway_prompt(
        {
            "messages": [HumanMessage(content="그 역량은?")],
            "last_result_ids": ["deleted-id"],
        }
    )

    assert '"previous_results": []' in prompt
    assert "deleted-id" not in prompt


def test_different_threads_do_not_share_gateway_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = SequenceGateway(
        [
            registry_decision(
                target_mentions=["책임성"],
                constraint_mentions=[{"kind": "field", "text": "정의"}],
            ),
            meta_decision("greeting"),
        ]
    )
    _patch_models(
        monkeypatch,
        gateway=gateway,
        registry_writer=StreamingWriter(_reference_answer, role="registry_writer"),
        scope_writer=SequenceScopeWriter([valid_meta_draft("greeting")]),
    )
    compiled = _compiled()

    _invoke(compiled, "책임성의 정의를 알려줘", "thread-a")
    _invoke(compiled, "안녕하세요", "thread-b")

    second_system_prompt = str(list(gateway.inputs[1])[0][1])
    assert '"previous_results": []' in second_system_prompt
    assert '"id": "responsibility"' not in second_system_prompt


def test_new_turn_resets_transient_state_but_keeps_last_context() -> None:
    updates = competency_interpreter._new_turn_updates("새 질문")

    assert updates["query_plan"] == {}
    assert updates["query_result"] == {}
    assert updates["candidate_names"] == []
    assert updates["llm_call_count"] == 0
    assert "last_query_plan" not in updates
    assert "last_result_ids" not in updates
    assert "last_candidate_ids" not in updates


def test_scope_generation_cancellation_leaves_no_partial_ai_checkpoint_or_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    compiled = competency_interpreter.builder.compile(checkpointer=checkpointer)
    writer_started = asyncio.Event()
    observed: list[dict[str, Any]] = []

    class BlockingScopeWriter:
        async def ainvoke(self, _: object) -> object:
            writer_started.set()
            await asyncio.Future()
            return valid_scope_draft("오늘 날씨")

    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [scope_decision("weather", "오늘 날씨")]
        ),
        scope_writer=BlockingScopeWriter(),
    )
    monkeypatch.setattr(competency_interpreter, "app", compiled)
    monkeypatch.setattr(competency_interpreter, "initialize_competency_runtime", lambda: None)

    async def cancel_during_scope_generation() -> None:
        async def collect() -> None:
            async for event in competency_interpreter.run_competency_stream(
                "오늘 날씨 어때?",
                "cancel-scope",
            ):
                observed.append(event)

        task = asyncio.create_task(collect())
        await writer_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_during_scope_generation())
    saved = compiled.get_state(
        {"configurable": {"thread_id": "cancel-scope"}}
    ).values
    assert not any(isinstance(message, AIMessage) for message in saved.get("messages", []))
    assert not any(event.get("type") in {"delta", "replace"} for event in observed)


def test_runtime_initializes_registry_before_checkpointer(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    events: list[str] = []
    compiled_app = object()

    def load_registry(database_url: str) -> RegistrySnapshot:
        assert database_url == "postgresql://test"
        events.append("registry")
        return synthetic_registry

    def open_checkpointer(_: object, database_url: str) -> object:
        assert database_url == "postgresql://test"
        assert competency_interpreter._registry_snapshot is synthetic_registry
        events.append("checkpointer")
        return object()

    class BuilderStub:
        def compile(self, *, checkpointer: object) -> object:
            assert checkpointer is not None
            events.append("compile")
            return compiled_app

    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(competency_interpreter, "load_active_registry", load_registry)
    monkeypatch.setattr(competency_interpreter, "_open_checkpointer", open_checkpointer)
    monkeypatch.setattr(competency_interpreter, "builder", BuilderStub())

    competency_interpreter.initialize_competency_runtime()

    assert events == ["registry", "checkpointer", "compile"]
    assert competency_interpreter._registry_snapshot is synthetic_registry
    assert competency_interpreter.app is compiled_app


def test_runtime_failure_rolls_back_registry_and_resources(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    events: list[str] = []

    def open_checkpointer(runtime_stack: object, _: str) -> object:
        runtime_stack.callback(events.append, "closed")
        return object()

    class FailingBuilder:
        def compile(self, *, checkpointer: object) -> object:
            assert checkpointer is not None
            raise RuntimeError("compile failed")

    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(competency_interpreter, "load_active_registry", lambda _: synthetic_registry)
    monkeypatch.setattr(competency_interpreter, "_open_checkpointer", open_checkpointer)
    monkeypatch.setattr(competency_interpreter, "builder", FailingBuilder())

    with pytest.raises(RuntimeError, match="compile failed"):
        competency_interpreter.initialize_competency_runtime()

    assert events == ["closed"]
    assert competency_interpreter._registry_snapshot is None
    assert competency_interpreter._runtime_stack is None
    assert competency_interpreter.app is None


def test_close_clears_registry_snapshot(synthetic_registry: RegistrySnapshot) -> None:
    assert competency_interpreter._registry_snapshot is synthetic_registry
    competency_interpreter.close_competency_runtime()
    assert competency_interpreter._registry_snapshot is None


def test_close_defers_resource_shutdown_until_active_async_request_releases(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    closed: list[str] = []
    runtime_stack = ExitStack()
    runtime_stack.callback(closed.append, "closed")
    compiled_app = object()
    monkeypatch.setattr(competency_interpreter, "app", compiled_app)
    monkeypatch.setattr(competency_interpreter, "_runtime_stack", runtime_stack)
    monkeypatch.setattr(competency_interpreter, "_registry_snapshot", synthetic_registry)
    monkeypatch.setattr(competency_interpreter, "_runtime_active_users", 0)
    monkeypatch.setattr(competency_interpreter, "_runtime_close_pending", False)
    monkeypatch.setattr(competency_interpreter, "initialize_competency_runtime", lambda: None)

    async def scenario() -> None:
        async with competency_interpreter._async_runtime_execution() as leased_app:
            assert leased_app is compiled_app
            competency_interpreter.close_competency_runtime()
            assert closed == []
            assert competency_interpreter.app is compiled_app
            with pytest.raises(RuntimeError, match="사용할 수 없습니다"):
                async with competency_interpreter._async_runtime_execution():
                    pass
        assert closed == ["closed"]
        assert competency_interpreter.app is None
        assert competency_interpreter._registry_snapshot is None

    asyncio.run(scenario())


def test_registry_helpers_fail_clearly_before_initialization() -> None:
    competency_interpreter.close_competency_runtime()
    with pytest.raises(RuntimeError, match="역량 레지스트리가 초기화되지 않았습니다"):
        competency_interpreter.validate_registry_names(["책임성"])


def test_mutable_copy_does_not_modify_frozen_snapshot(
    synthetic_registry: RegistrySnapshot,
) -> None:
    copied = competency_interpreter._mutable_json_copy(
        synthetic_registry.canonical_lookup["책임성"]
    )
    json.dumps(copied, ensure_ascii=False)
    copied["definition"] = "상태에서만 변경"
    copied["path"].append("추가 경로")

    frozen = synthetic_registry.canonical_lookup["책임성"]
    assert frozen["definition"] == "맡은 일을 책임지고 수행하는 특성"
    assert frozen["path"] == ("책임성",)


def test_repeated_requests_load_registry_only_once(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    load_calls: list[str] = []

    class CompiledAppStub:
        def invoke(self, state: dict, *, config: dict) -> dict:
            assert config["configurable"]["thread_id"]
            return state

    class BuilderStub:
        def compile(self, *, checkpointer: object) -> CompiledAppStub:
            assert checkpointer is not None
            return CompiledAppStub()

    def load_registry(database_url: str) -> RegistrySnapshot:
        load_calls.append(database_url)
        return synthetic_registry

    competency_interpreter.close_competency_runtime()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(competency_interpreter, "load_active_registry", load_registry)
    monkeypatch.setattr(competency_interpreter, "_open_checkpointer", lambda *_: object())
    monkeypatch.setattr(competency_interpreter, "builder", BuilderStub())

    competency_interpreter.run_competency("첫 질문", "thread-1")
    competency_interpreter.run_competency("둘째 질문", "thread-2")

    assert load_calls == ["postgresql://test"]


def test_close_then_reinitialize_loads_new_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    first_snapshot = make_synthetic_registry(version_id=1)
    second_snapshot = make_synthetic_registry(version_id=2)
    snapshots = iter([first_snapshot, second_snapshot])
    loaded_version_ids: list[int] = []

    class BuilderStub:
        def compile(self, *, checkpointer: object) -> object:
            assert checkpointer is not None
            return object()

    def load_registry(_: str) -> RegistrySnapshot:
        snapshot = next(snapshots)
        loaded_version_ids.append(snapshot.version_id)
        return snapshot

    competency_interpreter.close_competency_runtime()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(competency_interpreter, "load_active_registry", load_registry)
    monkeypatch.setattr(competency_interpreter, "_open_checkpointer", lambda *_: object())
    monkeypatch.setattr(competency_interpreter, "builder", BuilderStub())

    competency_interpreter.initialize_competency_runtime()
    assert competency_interpreter._registry_snapshot is first_snapshot
    competency_interpreter.close_competency_runtime()
    competency_interpreter.initialize_competency_runtime()

    assert competency_interpreter._registry_snapshot is second_snapshot
    assert loaded_version_ids == [1, 2]


def test_same_thread_streams_are_serialized_without_blocking_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0

    class SlowFinalApp:
        async def astream(self, *_: object, **__: object):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            yield "values", {"messages": [AIMessage(content="완료")], "candidate_names": []}

    monkeypatch.setattr(competency_interpreter, "app", SlowFinalApp())
    monkeypatch.setattr(competency_interpreter, "initialize_competency_runtime", lambda: None)

    async def collect(thread_id: str) -> list[dict]:
        return [
            event
            async for event in competency_interpreter.run_competency_stream("질문", thread_id)
        ]

    async def run_pair() -> tuple[list[dict], list[dict]]:
        return await asyncio.gather(collect("same-thread"), collect("same-thread"))

    first, second = asyncio.run(run_pair())
    assert max_active == 1
    assert first[-1]["type"] == "done"
    assert second[-1]["type"] == "done"


def test_cancelled_lock_waiter_does_not_block_next_same_thread_request() -> None:
    async def scenario() -> None:
        first = competency_interpreter._async_thread_execution("cancel-waiter")
        await first.__aenter__()

        async def wait_for_lock() -> None:
            async with competency_interpreter._async_thread_execution("cancel-waiter"):
                return

        waiter = asyncio.create_task(wait_for_lock())
        await asyncio.sleep(0.03)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await first.__aexit__(None, None, None)

        async with asyncio.timeout(0.2):
            async with competency_interpreter._async_thread_execution("cancel-waiter"):
                pass

    asyncio.run(scenario())
    assert "cancel-waiter" not in competency_interpreter._thread_locks


def test_different_thread_streams_can_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0

    class SlowFinalApp:
        async def astream(self, *_: object, **__: object):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            yield "values", {"messages": [AIMessage(content="완료")], "candidate_names": []}

    monkeypatch.setattr(competency_interpreter, "app", SlowFinalApp())
    monkeypatch.setattr(competency_interpreter, "initialize_competency_runtime", lambda: None)

    async def collect(thread_id: str) -> list[dict]:
        return [
            event
            async for event in competency_interpreter.run_competency_stream("질문", thread_id)
        ]

    async def run_pair() -> None:
        await asyncio.gather(collect("thread-a"), collect("thread-b"))

    asyncio.run(run_pair())
    assert max_active == 2
