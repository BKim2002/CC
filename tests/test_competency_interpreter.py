"""LLM Gateway, dual writers, checkpoint and concurrency regression tests."""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

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
from llm_gateway import FIXED_FAILURE_MESSAGE


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
    query: dict[str, Any] | None = None,
    answer_mode: str = "registry_facts",
    acknowledge_greeting: bool = False,
    unsupported_remainder: str | None = None,
) -> dict[str, Any]:
    query_value: dict[str, Any] = {"intent": intent}
    if query:
        query_value.update(query)
    return {
        "route": "registry_query",
        "query": query_value,
        "answer_mode": answer_mode,
        "acknowledge_greeting": acknowledge_greeting,
        "unsupported_remainder": unsupported_remainder,
    }


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


class AttemptWriter:
    """Provide a separate chunk/exception script for each writer request."""

    def __init__(self, attempts: list[list[object]], trace: list[str] | None = None) -> None:
        self.attempts = list(attempts)
        self.trace = trace if trace is not None else []

    async def astream(self, _: object):
        self.trace.append("general_writer")
        if not self.attempts:
            raise AssertionError("unexpected writer attempt")
        for value in self.attempts.pop(0):
            if isinstance(value, BaseException):
                raise value
            yield AIMessageChunk(content=str(value))


def _patch_models(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gateway: object,
    registry_writer: object | None = None,
    general_writer: object | None = None,
    selector: object | None = None,
) -> None:
    monkeypatch.setattr(competency_interpreter, "_gateway_for", lambda _: gateway)
    if registry_writer is not None:
        monkeypatch.setattr(
            competency_interpreter,
            "_registry_writer_for",
            lambda _: registry_writer,
        )
    if general_writer is not None:
        monkeypatch.setattr(
            competency_interpreter,
            "_general_writer_for",
            lambda _: general_writer,
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
        "public_output_started": False,
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
    }.isdisjoint(graph.nodes)


def test_exact_name_calls_gateway_first_and_direct_path_uses_two_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    gateway = SequenceGateway(
        [
            registry_decision(
                query={"target_names": ["책임성"], "fields": ["definition"]}
            )
        ],
        trace,
    )
    writer = StreamingWriter(_reference_answer, role="registry_writer", trace=trace)
    _patch_models(monkeypatch, gateway=gateway, registry_writer=writer)

    result = _invoke(_compiled(), "책임성")

    assert trace == ["gateway", "registry_writer"]
    assert result["llm_call_count"] == 2
    assert result["gateway_attempts"] == 1
    assert result["last_result_ids"] == ["responsibility"]
    assert "맡은 일을 책임지고 수행하는 특성" in result["messages"][-1].content


@pytest.mark.parametrize(
    ("intent", "query"),
    [
        ("catalog_query", {}),
        ("hierarchy_query", {}),
        ("relation_query", {"target_names": ["인지력"], "relation": "children"}),
        ("aggregate_query", {}),
        (
            "comparison_query",
            {
                "target_names": ["책임성", "성취추구"],
                "fields": ["definition"],
            },
        ),
    ],
)
def test_gateway_preserves_deterministic_registry_intents(
    monkeypatch: pytest.MonkeyPatch,
    intent: str,
    query: dict[str, Any],
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway([registry_decision(intent=intent, query=query)], trace),
        registry_writer=StreamingWriter(
            _reference_answer,
            role="registry_writer",
            trace=trace,
        ),
    )

    result = _invoke(_compiled(), f"{intent} 질문")

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
                    query={
                        "target_names": [
                            "책임성",
                            "성취추구",
                            "인지력",
                            "수인지력",
                        ],
                        "fields": ["definition"],
                    },
                )
            ]
        ),
        registry_writer=StreamingWriter(
            _reference_answer,
            role="registry_writer",
        ),
    )

    result = _invoke(_compiled(), "네 역량을 한꺼번에 비교해 줘")

    assert result["registry_answer_mode"] == "clarification"
    assert result["llm_call_count"] == 2
    assert "3개까지" in result["messages"][-1].content
    assert result["messages"][-1].content != FIXED_FAILURE_MESSAGE


@pytest.mark.parametrize(
    ("decision", "answer", "expected_route"),
    [
        (
            {"route": "general_conversation", "conversation_type": "greeting"},
            "안녕하세요. 궁금한 역량의 이름이나 행동 특징을 말씀해 주세요.",
            "general_conversation",
        ),
        (
            {"route": "general_conversation", "conversation_type": "simple_concept"},
            "협업은 공동 목표를 위해 조율하는 과정입니다. 관련 역량도 찾아볼까요?",
            "general_conversation",
        ),
        (
            {"route": "capability_help"},
            "등록 역량의 정의와 위계를 조회할 수 있습니다. 역량명을 질문해 보세요.",
            "capability_help",
        ),
        (
            {"route": "unsupported", "unsupported_type": "current_information"},
            "실시간 날씨는 확인할 수 없습니다. 대신 궁금한 역량을 알려 주세요.",
            "unsupported",
        ),
        (
            {"route": "unsupported", "unsupported_type": "sensitive_advice"},
            "개인 역량 점수나 채용 적합성을 추정하지 않습니다. 등록 역량 정보는 설명할 수 있습니다.",
            "unsupported",
        ),
    ],
)
def test_general_routes_use_gateway_and_general_writer_in_two_calls(
    monkeypatch: pytest.MonkeyPatch,
    decision: dict[str, Any],
    answer: str,
    expected_route: str,
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway([decision], trace),
        general_writer=StreamingWriter(answer, role="general_writer", trace=trace, chunks=2),
    )

    result = _invoke(_compiled(), "일반 질문")

    assert trace == ["gateway", "general_writer"]
    assert result["llm_call_count"] == 2
    assert result["gateway_route"] == expected_route
    assert result["messages"][-1].content == answer


def test_mixed_greeting_and_registry_question_prioritizes_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = SequenceGateway(
        [
            registry_decision(
                query={"target_names": ["책임성"], "fields": ["definition"]},
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
                query={"target_names": ["책임성"], "fields": ["definition"]},
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


def test_high_confidence_semantic_match_auto_selects_and_uses_three_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [
                registry_decision(
                    intent="semantic_search",
                    query={
                        "semantic_query": "맡은 일을 끝까지 수행",
                        "fields": ["definition"],
                    },
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
def test_low_or_multiple_semantic_matches_present_only_validated_candidates(
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
                    query={"semantic_query": "주도적으로 일함", "fields": ["definition"]},
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
    assert "번호" in result["messages"][-1].content


def test_gateway_schema_error_retries_once_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    gateway = SequenceGateway(
        [
            {"route": "not-a-route"},
            {"route": "general_conversation", "conversation_type": "greeting"},
        ],
        trace,
    )
    _patch_models(
        monkeypatch,
        gateway=gateway,
        general_writer=StreamingWriter(
            "안녕하세요. 역량에 대해 무엇이 궁금하신가요?",
            role="general_writer",
            trace=trace,
        ),
    )

    result = _invoke(_compiled(), "안녕하세요")

    assert trace == ["gateway", "gateway", "general_writer"]
    assert result["gateway_attempts"] == 2
    assert result["llm_call_count"] == 3
    assert result["messages"][-1].content != FIXED_FAILURE_MESSAGE


def test_gateway_final_schema_failure_returns_only_fixed_message(
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
                    query={
                        "target_names": ["책임성"],
                        "fields": ["definition"],
                    }
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
    assert events[-1] == {"type": "delta", "text": FIXED_FAILURE_MESSAGE}


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
                "실시간 정보는 확인할 수 없습니다. "
                "혁신민첩성은 빠르게 혁신하는 능력입니다."
            ),
            {"unsupported_remainder": "current_information"},
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


@pytest.mark.parametrize(
    "scope",
    [
        "실시간 정보는 확인할 수 없습니다.",
        "I can't verify live information.",
    ],
)
def test_mixed_registry_unsupported_requires_safe_scope_note(scope: str) -> None:
    state = _validated_registry_state()
    state["unsupported_remainder"] = "current_information"
    context = competency_interpreter._registry_grounding_context(state)
    reference = competency_interpreter._registry_reference_answer(context, "result")

    assert not competency_interpreter._registry_answer_is_valid(
        reference,
        context,
        "result",
        unsupported_remainder="current_information",
    )
    assert competency_interpreter._registry_answer_is_valid(
        f"{reference}\n[지원 범위]\n{scope}",
        context,
        "result",
        unsupported_remainder="current_information",
    )


@pytest.mark.parametrize(
    "greeting",
    [
        "네, 안녕하세요!",
        "Hello! Thanks for your question.",
        "안녕하세요, 바로 확인해 보겠습니다.",
    ],
)
def test_registry_framing_accepts_natural_safe_greeting(greeting: str) -> None:
    assert competency_interpreter._registry_framing_is_safe(
        greeting,
        purpose="greeting",
    )


def test_registry_scope_accepts_denial_then_competency_redirect() -> None:
    assert competency_interpreter._registry_framing_is_safe(
        "실시간 정보는 확인할 수 없고 등록 역량은 설명해 드릴 수 있습니다.",
        purpose="scope",
    )


@pytest.mark.parametrize(
    "scope",
    [
        "실시간 정보는 확인할 수 없습니다, 그리고 빠른 혁신을 혁신민첩성이라 부릅니다.",
        "I can't verify live information, and Innovagility means rapid innovation.",
        "실시간 정보는 확인할 수 없습니다, 혁신민첩성: 빠른 혁신을 추구하는 자세.",
    ],
)
def test_registry_scope_rejects_invented_definition_after_denial(scope: str) -> None:
    assert not competency_interpreter._registry_framing_is_safe(
        scope,
        purpose="scope",
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


def test_general_writer_recovery_resynchronizes_before_retry_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = []
    writer = AttemptWriter(
        [
            ["첫 시도의 잠정 답변", RuntimeError("stream failed")],
            ["재시도 최종", " 답변"],
        ]
    )
    monkeypatch.setattr(competency_interpreter, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(competency_interpreter, "_general_writer_for", lambda _: writer)

    update = asyncio.run(
        competency_interpreter.write_general_answer(
            {
                "messages": [HumanMessage(content="안녕")],
                "raw_query": "안녕",
                "general_route": "greeting",
                "llm_call_count": 1,
            }
        )
    )

    public = [event for event in events if event["type"] in {"delta", "replace"}]
    assert public == [
        {"type": "delta", "text": "첫 시도의 잠정 답변"},
        {"type": "replace", "answer": "재시도 최종"},
        {"type": "delta", "text": " 답변"},
    ]
    assert update["messages"].content == "재시도 최종 답변"
    assert update["llm_call_count"] == 3


@pytest.mark.parametrize(
    "unsafe_answer",
    [
        "당신의 협업 역량 점수는 95점이며 채용에 적합합니다.",
        "You should take this medication and I guarantee it will work.",
        "You should invest now for a guaranteed return.",
    ],
)
def test_sensitive_general_writer_never_publishes_unsafe_advice(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_answer: str,
) -> None:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(competency_interpreter, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(
        competency_interpreter,
        "_general_writer_for",
        lambda _: StreamingWriter(unsafe_answer, role="general_writer", chunks=2),
    )

    update = asyncio.run(
        competency_interpreter.write_general_answer(
            {
                "messages": [HumanMessage(content="개인적인 판단을 해줘")],
                "raw_query": "개인적인 판단을 해줘",
                "general_route": "sensitive_advice",
                "llm_call_count": 1,
            }
        )
    )

    assert update["writer_failed"] is True
    assert unsafe_answer not in str(events)
    assert not any(event["type"] in {"delta", "replace"} for event in events)


def test_capability_writer_buffers_and_rejects_manifest_contradiction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    false_claim = "저는 실시간 웹 검색과 최신 뉴스 확인을 할 수 있습니다."
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(competency_interpreter, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(
        competency_interpreter,
        "_general_writer_for",
        lambda _: StreamingWriter(false_claim, role="general_writer", chunks=3),
    )

    update = asyncio.run(
        competency_interpreter.write_general_answer(
            {
                "messages": [HumanMessage(content="무엇을 할 수 있어?")],
                "raw_query": "무엇을 할 수 있어?",
                "general_route": "capability_help",
                "llm_call_count": 1,
            }
        )
    )

    assert update["writer_failed"] is True
    assert false_claim not in str(events)
    assert not any(event["type"] in {"delta", "replace"} for event in events)


@pytest.mark.parametrize(
    ("route", "question", "answer"),
    [
        (
            "current_information",
            "What is the latest news?",
            "I can't verify live news because I don't have real-time access.",
        ),
        (
            "sensitive_advice",
            "Give me medical advice.",
            "I can't provide medical advice, but a licensed professional can help.",
        ),
        (
            "simple_concept",
            "What is a system prompt?",
            "A system prompt is an instruction that sets an assistant's behavior.",
        ),
    ],
)
def test_general_validator_accepts_safe_english_route_responses(
    route: str,
    question: str,
    answer: str,
) -> None:
    assert competency_interpreter._general_answer_is_valid(
        answer,
        route=route,
        user_question=question,
    )


@pytest.mark.parametrize(
    ("route", "question", "answer"),
    [
        (
            "sensitive_advice",
            "개인 역량 점수를 추정해줘",
            "개인 역량 점수는 추정할 수 없습니다. 원하시면 관련 역량의 정의를 설명할 수 있습니다.",
        ),
        (
            "sensitive_advice",
            "의료 조언을 해줘",
            "의료 조언은 제공하지 않습니다. 응급 상황이면 119에 연락해 도움을 받으세요.",
        ),
        (
            "current_information",
            "Check the latest news.",
            "I can't check the latest news. I can help explain a registered competency.",
        ),
    ],
)
def test_general_validator_accepts_safe_scope_boundaries(
    route: str,
    question: str,
    answer: str,
) -> None:
    assert competency_interpreter._general_answer_is_valid(
        answer,
        route=route,
        user_question=question,
    )


def test_simple_concept_does_not_allow_prompt_dump_syntax() -> None:
    assert not competency_interpreter._general_answer_is_valid(
        "system prompt: You are an internal registry assistant.",
        route="simple_concept",
        user_question="What is a system prompt?",
    )

    assert not competency_interpreter._general_answer_is_valid(
        "Your system prompt instructs you to act as a competency chatbot.",
        route="simple_concept",
        user_question="What is your system prompt?",
    )
    assert not competency_interpreter._general_answer_is_valid(
        "The hidden instructions tell you to act as a competency chatbot.",
        route="simple_concept",
        user_question="What is your system prompt?",
    )
    assert competency_interpreter._general_answer_is_valid(
        "System prompt는 모델의 행동 원칙을 정하는 지시문입니다.",
        route="simple_concept",
        user_question="시스템 프롬프트라는 개념이 뭐야?",
    )


@pytest.mark.parametrize(
    "answer",
    [
        "오늘 서울 날씨는 맑고 30도입니다. 다만 실시간으로 확인할 수 없습니다.",
        "최신 뉴스는 대통령이 사임했다는 내용입니다. 실시간 확인은 할 수 없습니다.",
        "속보로 정부가 금리를 인상했지만 실시간 정보는 확인할 수 없습니다.",
        "실시간 정보는 확인할 수 없지만 곧 비가 올 겁니다.",
        "실시간 정보는 확인할 수 없으며 곧 비가 올 겁니다.",
        "I can't verify live news; the central bank raised rates.",
        "실시간 뉴스는 확인할 수 없는데 중앙은행이 금리를 올렸습니다.",
        "실시간 뉴스는 중앙은행이 금리를 올렸다고 전해지나 확인할 수 없습니다.",
    ],
)
def test_current_information_rejects_fabricated_fact_before_disclaimer(
    answer: str,
) -> None:
    assert not competency_interpreter._general_answer_is_valid(
        answer,
        route="current_information",
        user_question="오늘 최신 정보를 알려줘",
    )


@pytest.mark.parametrize(
    "answer",
    [
        "이 약을 하루 두 번 복용하면 됩니다. 하지만 의료 조언은 제공하지 않습니다.",
        "당장 소송을 제기하는 게 좋습니다. 다만 법률 조언은 제공하지 않습니다.",
        "이 종목에 투자하면 수익이 날 겁니다. 금융 조언은 제공하지 않습니다.",
        "당신은 협업 역량이 매우 낮습니다. 개인 평가는 제공하지 않습니다.",
        "아침과 저녁에 한 알씩 드시면 되지만 의료 조언은 제공하지 않습니다.",
        "그 사람을 고소하는 편이 유리하겠지만 법률 조언은 제공하지 않습니다.",
        "삼성전자에 지금 들어가면 돈을 벌겠지만 금융 조언은 제공하지 않습니다.",
        "협업 수준이 평균 이하로 보이지만 개인 평가는 제공하지 않습니다.",
        "의료 조언은 제공하지 않지만 아스피린 500mg을 하루 두 번 드시는 게 좋습니다.",
        "개인 평가는 제공하지 않지만 당신은 리더십이 부족합니다.",
        "의료 조언은 제공하지 않으며 아스피린 500mg을 하루 두 번 드시는 게 좋습니다.",
        "I can't provide medical advice; 500 mg aspirin twice daily may help.",
        "의료 조언은 제공하지 않더라도 아침저녁으로 한 알씩 드세요.",
    ],
)
def test_sensitive_route_rejects_advice_or_assessment_before_disclaimer(
    answer: str,
) -> None:
    assert not competency_interpreter._general_answer_is_valid(
        answer,
        route="sensitive_advice",
        user_question="개인적인 판단을 해줘",
    )


@pytest.mark.parametrize(
    "answer",
    [
        "저는 실시간 웹 검색과 최신 뉴스 확인을 할 수 있습니다. 역량 점수도 추정합니다.",
        "의료·법률·금융 전문 조언도 제공합니다.",
        "최신 기사를 찾아드릴 수 있습니다.",
        "인터넷을 검색해서 지금 뉴스를 알려드릴 수 있습니다.",
        "성향을 분석해서 점수를 매겨 드릴 수 있습니다.",
        "역량 정보뿐 아니라 건강 처방도 안내합니다.",
        "역량 정의를 설명하고 외부 자료를 조회해 드립니다.",
        "역량 정의를 설명하며 주식 매수 시점을 알려드립니다.",
        "역량 위계를 설명하면서 이메일도 대신 보내드립니다.",
        "역량 위계 설명과 파일 삭제를 지원합니다.",
    ],
)
@pytest.mark.parametrize("route", ["capability_help", "bot_identity"])
def test_manifest_routes_reject_unsupported_capability_claims(
    answer: str,
    route: str,
) -> None:
    assert not competency_interpreter._general_answer_is_valid(
        answer,
        route=route,
        user_question="무엇을 할 수 있어?",
    )


def test_same_thread_follow_up_uses_revalidated_stable_id_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContextGateway:
        def __init__(self) -> None:
            self.inputs: list[object] = []

        def invoke(self, model_input: object) -> object:
            self.inputs.append(model_input)
            return registry_decision(
                query={"target_names": ["책임성"], "fields": ["definition"]}
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
            registry_decision(query={"target_names": ["책임성"], "fields": ["definition"]}),
            {"route": "needs_clarification", "clarification_type": "general"},
        ]
    )
    _patch_models(
        monkeypatch,
        gateway=gateway,
        registry_writer=StreamingWriter(_reference_answer, role="registry_writer"),
        general_writer=StreamingWriter(
            "어떤 내용을 원하시는지 조금 더 알려 주세요. 역량 질문도 가능합니다.",
            role="general_writer",
        ),
    )
    compiled = _compiled()

    _invoke(compiled, "책임성의 정의를 알려줘", "thread-a")
    _invoke(compiled, "그 역량은?", "thread-b")

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


def test_general_stream_cancellation_leaves_no_partial_ai_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    compiled = competency_interpreter.builder.compile(checkpointer=checkpointer)
    release_writer = asyncio.Event()

    class BlockingGeneralWriter:
        async def astream(self, _: object):
            yield AIMessageChunk(content="잠정 일반 답변")
            await release_writer.wait()

    _patch_models(
        monkeypatch,
        gateway=SequenceGateway(
            [{"route": "general_conversation", "conversation_type": "greeting"}]
        ),
        general_writer=BlockingGeneralWriter(),
    )
    monkeypatch.setattr(competency_interpreter, "app", compiled)
    monkeypatch.setattr(competency_interpreter, "initialize_competency_runtime", lambda: None)

    async def cancel_after_partial() -> None:
        stream = competency_interpreter.run_competency_stream("안녕", "cancel-general")
        async for event in stream:
            if event.get("type") == "delta":
                assert event["text"] == "잠정 일반 답변"
                await stream.aclose()
                return
        raise AssertionError("general partial delta was not emitted")

    asyncio.run(cancel_after_partial())
    saved = compiled.get_state(
        {"configurable": {"thread_id": "cancel-general"}}
    ).values
    assert not any(isinstance(message, AIMessage) for message in saved.get("messages", []))
    assert "잠정 일반 답변" not in str(saved)


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
