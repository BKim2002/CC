"""Focused tests for the gateway/writer shared contracts."""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

import llm_gateway
from llm_gateway import (
    CAPABILITY_MANIFEST,
    DEFAULT_OPENAI_MODEL,
    FIXED_FAILURE_MESSAGE,
    MAX_LLM_API_CALLS_PER_TURN,
    ConstraintMention,
    GatewayDecision,
    LlmCallBudget,
    LlmCallBudgetExceeded,
    MetaResponseDraft,
    MetaRouteDecision,
    ModelRole,
    OutOfScopeResponseDraft,
    OutOfScopeRouteDecision,
    RegistryQueryDraft,
    RegistryRouteDecision,
    ScopeTopic,
    ainvoke_with_budget,
    capability_manifest_for_prompt,
    create_chat_model,
    invoke_with_budget,
    recent_conversation_context,
    selected_answer_model_name,
    selected_entry_model_name,
    validate_gateway_decision,
)


def _registry_decision(**draft_overrides: object) -> dict[str, object]:
    draft: dict[str, object] = {
        "intent_hint": "item_lookup",
        "target_mentions": ["책임성"],
        "constraint_mentions": [],
        "semantic_description": None,
        "reuse_previous_result": False,
        "answer_mode": "registry_facts",
        "acknowledge_greeting": False,
        "out_of_scope_remainder": None,
    }
    draft.update(draft_overrides)
    return {
        "route": "registry_query",
        "draft": draft,
    }


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (_registry_decision(), RegistryRouteDecision),
        (
            {"route": "meta_conversation", "kind": "greeting"},
            MetaRouteDecision,
        ),
        (
            {
                "route": "out_of_scope",
                "topic": {"category": "date_time", "summary": "오늘 날짜"},
            },
            OutOfScopeRouteDecision,
        ),
    ],
)
def test_gateway_decision_is_a_strict_discriminated_union(
    value: dict[str, object],
    expected_type: type[object],
) -> None:
    decision = validate_gateway_decision(value)

    assert isinstance(decision, expected_type)


def test_gateway_decision_accepts_openai_wrapper_and_flat_test_double() -> None:
    flat = {"route": "meta_conversation", "kind": "greeting"}
    registry_flat = _registry_decision(
        constraint_mentions=[{"kind": "field", "text": "정의"}]
    )

    flat_decision = validate_gateway_decision(flat)
    wrapped_decision = validate_gateway_decision({"decision": flat})
    model_decision = validate_gateway_decision(
        GatewayDecision.model_validate({"decision": flat})
    )
    registry_decision = validate_gateway_decision(registry_flat)

    assert isinstance(flat_decision, MetaRouteDecision)
    assert wrapped_decision == flat_decision
    assert model_decision == flat_decision
    assert isinstance(registry_decision, RegistryRouteDecision)
    assert registry_decision.draft.constraint_mentions == [
        ConstraintMention(kind="field", text="정의"),
    ]


def test_gateway_decision_schema_has_object_root_and_nested_union() -> None:
    schema = GatewayDecision.model_json_schema()

    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert "oneOf" not in schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["decision"]
    assert len(schema["properties"]["decision"]["anyOf"]) == 3
    for definition_name in (
        "RegistryRouteDecision",
        "MetaRouteDecision",
        "OutOfScopeRouteDecision",
    ):
        assert schema["$defs"][definition_name]["additionalProperties"] is False


def test_gateway_decision_converts_to_openai_strict_object_schema() -> None:
    schema = to_strict_json_schema(GatewayDecision)

    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert "oneOf" not in schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["decision"]
    assert len(schema["properties"]["decision"]["anyOf"]) == 3


@pytest.mark.parametrize(
    "invalid_value",
    [
        {"route": "unknown"},
        {"route": "capability_help"},
        {
            "route": "meta_conversation",
            "kind": "small_talk",
        },
        {"route": "meta_conversation", "kind": "simple_concept"},
        {
            "route": "out_of_scope",
            "topic": {"category": "current_information", "summary": "날씨"},
        },
        {
            "route": "out_of_scope",
            "topic": {"category": "weather", "summary": 21},
        },
        {
            "route": "out_of_scope",
            "topic": {"category": "weather", "summary": "   "},
        },
        {
            "route": "out_of_scope",
            "topic": {"category": "weather", "summary": "x" * 161},
        },
        {
            "route": "out_of_scope",
            "topic": {
                "category": "weather",
                "summary": "현재 날씨",
                "answer": "맑음",
            },
        },
        _registry_decision(acknowledge_greeting=1),
        _registry_decision(reuse_previous_result=0),
        _registry_decision(semantic_description=123),
        _registry_decision(intent_hint="help"),
        _registry_decision(unrecognized=True),
        _registry_decision(
            constraint_mentions=[{"kind": "canonical_tier", "text": "하위요인"}]
        ),
        _registry_decision(
            constraint_mentions=[{"kind": "hierarchy_tier", "text": 3}]
        ),
        _registry_decision(canonical_ids=["written.conscientiousness"]),
    ],
)
def test_gateway_decision_rejects_unknown_or_non_strict_values(
    invalid_value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        validate_gateway_decision(invalid_value)

    with pytest.raises(ValidationError):
        validate_gateway_decision({"decision": invalid_value})


def test_registry_gateway_preserves_all_seven_registry_intents() -> None:
    supported = {
        "item_lookup",
        "semantic_search",
        "catalog_query",
        "hierarchy_query",
        "relation_query",
        "aggregate_query",
        "comparison_query",
    }

    for intent in supported:
        decision = validate_gateway_decision(
            _registry_decision(intent_hint=intent)
        )
        assert isinstance(decision, RegistryRouteDecision)
        assert decision.draft.intent_hint == intent


@pytest.mark.parametrize(
    "kind",
    ["greeting", "thanks", "farewell", "bot_identity", "capability_help"],
)
def test_meta_route_is_limited_to_supported_social_and_help_kinds(kind: str) -> None:
    decision = validate_gateway_decision(
        {"route": "meta_conversation", "kind": kind}
    )

    assert isinstance(decision, MetaRouteDecision)
    assert decision.kind == kind


@pytest.mark.parametrize(
    "category",
    [
        "date_time",
        "weather",
        "news_current_events",
        "professional_advice",
        "personal_assessment",
        "employment_decision",
        "general_knowledge",
        "external_action",
        "unsafe",
        "other",
    ],
)
def test_out_of_scope_route_supports_each_safe_topic_category(category: str) -> None:
    decision = validate_gateway_decision(
        {
            "route": "out_of_scope",
            "topic": {"category": category, "summary": "사용자가 요청한 주제"},
        }
    )

    assert isinstance(decision, OutOfScopeRouteDecision)
    assert decision.topic.category == category


def test_registry_draft_can_represent_mixed_greeting_and_scope_remainder() -> None:
    decision = validate_gateway_decision(
        _registry_decision(
            acknowledge_greeting=True,
            out_of_scope_remainder={
                "category": "date_time",
                "summary": "오늘 날짜",
            },
        )
    )

    assert isinstance(decision, RegistryRouteDecision)
    assert decision.draft.acknowledge_greeting is True
    assert decision.draft.out_of_scope_remainder == ScopeTopic(
        category="date_time",
        summary="오늘 날짜",
    )


def test_registry_draft_only_contains_untrusted_mentions_not_canonical_plan_fields() -> None:
    schema = RegistryQueryDraft.model_json_schema()

    assert set(schema["properties"]) == {
        "intent_hint",
        "target_mentions",
        "constraint_mentions",
        "semantic_description",
        "reuse_previous_result",
        "answer_mode",
        "acknowledge_greeting",
        "out_of_scope_remainder",
    }
    assert schema["additionalProperties"] is False


def test_capability_manifest_is_the_prompt_source_of_truth() -> None:
    rendered = json.loads(capability_manifest_for_prompt())

    assert rendered == CAPABILITY_MANIFEST.model_dump(mode="json")
    assert {entry.key for entry in CAPABILITY_MANIFEST.supported} == {
        "registry_information",
        "hierarchy_and_relations",
        "filters_and_aggregates",
        "comparison",
        "semantic_candidates",
        "general_guidance",
        "meta_conversation",
    }
    assert {entry.key for entry in CAPABILITY_MANIFEST.limitations} == {
        "invented_registry_facts",
        "non_registry_knowledge",
        "personal_assessment",
        "employment_fit",
        "current_information",
        "professional_advice",
    }
    supported_text = " ".join(
        entry.description for entry in CAPABILITY_MANIFEST.supported
    )
    limitation_text = " ".join(
        entry.description for entry in CAPABILITY_MANIFEST.limitations
    )
    assert "일반 개념" not in supported_text
    assert all(token in supported_text for token in ("인사", "감사", "작별", "사용법"))
    assert "범위 안내" in limitation_text
    assert CAPABILITY_MANIFEST.max_comparison_items == 3
    assert CAPABILITY_MANIFEST.max_semantic_candidates == 3


@pytest.mark.parametrize(
    ("environ", "expected_entry", "expected_answer"),
    [
        ({}, DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_MODEL),
        (
            {"OPENAI_MODEL": "legacy"},
            "legacy",
            "legacy",
        ),
        (
            {
                "OPENAI_MODEL": "legacy",
                "OPENAI_ENTRY_MODEL": "entry-role",
                "OPENAI_ANSWER_MODEL": "answer-role",
            },
            "entry-role",
            "answer-role",
        ),
        (
            {
                "OPENAI_MODEL": " legacy ",
                "OPENAI_ENTRY_MODEL": "   ",
                "OPENAI_ANSWER_MODEL": " answer-role ",
            },
            "legacy",
            "answer-role",
        ),
    ],
)
def test_role_model_precedence(
    environ: dict[str, str],
    expected_entry: str,
    expected_answer: str,
) -> None:
    assert selected_entry_model_name(environ=environ) == expected_entry
    assert selected_answer_model_name(environ=environ) == expected_answer


def test_chat_model_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class ChatModelStub:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_ENTRY_MODEL", "entry-role")
    monkeypatch.setattr(llm_gateway, "ChatOpenAI", ChatModelStub)

    model = create_chat_model(
        ModelRole.ENTRY,
        timeout=17,
        model_name=" explicit-entry ",
    )

    assert isinstance(model, ChatModelStub)
    assert captured == {
        "model": "explicit-entry",
        "max_retries": 0,
        "timeout": 17,
    }


def test_recent_context_keeps_only_last_twelve_user_assistant_messages() -> None:
    messages = [HumanMessage(content=f"message-{index}") for index in range(14)]
    messages.insert(5, SystemMessage(content="internal instructions"))
    messages.append(AIMessage(content="final-assistant"))

    context = recent_conversation_context(messages)

    assert len(context) == 12
    assert [turn.content for turn in context] == [
        *(f"message-{index}" for index in range(3, 14)),
        "final-assistant",
    ]
    assert all(turn.content != "internal instructions" for turn in context)


def test_recent_context_accepts_checkpoint_style_messages_and_text_blocks() -> None:
    context = recent_conversation_context(
        [
            {"type": "human", "content": "질문"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "첫 문장"},
                    {"type": "image", "url": "not-forwarded"},
                    " 두 번째 문장",
                ],
            },
            {"role": "tool", "content": "internal tool output"},
        ]
    )

    assert [turn.model_dump() for turn in context] == [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "첫 문장 두 번째 문장"},
    ]


def test_failed_requests_still_consume_the_three_call_budget() -> None:
    class FailingModel:
        def __init__(self) -> None:
            self.attempts = 0

        def invoke(self, _: object) -> object:
            self.attempts += 1
            raise TimeoutError("provider timeout")

    model = FailingModel()
    budget = LlmCallBudget()

    for expected_used in range(1, MAX_LLM_API_CALLS_PER_TURN + 1):
        with pytest.raises(TimeoutError):
            invoke_with_budget(model, [], budget=budget)
        assert budget.used_calls == expected_used

    with pytest.raises(LlmCallBudgetExceeded):
        invoke_with_budget(model, [], budget=budget)

    assert model.attempts == MAX_LLM_API_CALLS_PER_TURN
    assert budget.remaining_calls == 0
    assert LlmCallBudget.model_validate(budget.model_dump()).used_calls == 3


def test_fixed_failure_message_is_the_exact_public_constant() -> None:
    assert FIXED_FAILURE_MESSAGE == (
        "답변을 만드는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
    )


@pytest.mark.parametrize(
    ("model_type", "valid_value"),
    [
        (
            OutOfScopeResponseDraft,
            {
                "acknowledgement": "날짜를 물어보셨군요.",
                "scope_boundary": "이 챗봇은 역량 레지스트리 질문을 다룹니다.",
                "registry_redirect": "역량별 정의를 살펴볼까요?",
            },
        ),
        (
            MetaResponseDraft,
            {
                "response": "안녕하세요.",
                "registry_redirect": "어떤 역량을 찾아볼까요?",
            },
        ),
    ],
)
def test_scope_response_drafts_are_strict_and_reject_extra_fields(
    model_type: type[object],
    valid_value: dict[str, str],
) -> None:
    model = model_type.model_validate(valid_value)  # type: ignore[attr-defined]
    assert model.model_dump() == valid_value  # type: ignore[attr-defined]
    schema = to_strict_json_schema(model_type)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(valid_value)

    with pytest.raises(ValidationError):
        model_type.model_validate(  # type: ignore[attr-defined]
            {**valid_value, "internal_route": "out_of_scope"}
        )

    first_field = next(iter(valid_value))
    with pytest.raises(ValidationError):
        model_type.model_validate(  # type: ignore[attr-defined]
            {**valid_value, first_field: 123}
        )


@pytest.mark.parametrize(
    ("model_type", "valid_value"),
    [
        (
            OutOfScopeResponseDraft,
            {
                "acknowledgement": "주제를 확인했습니다.",
                "scope_boundary": "역량 레지스트리 범위만 지원합니다.",
                "registry_redirect": "역량 위계를 살펴볼까요?",
            },
        ),
        (
            MetaResponseDraft,
            {
                "response": "감사합니다.",
                "registry_redirect": "다른 역량도 찾아볼까요?",
            },
        ),
    ],
)
@pytest.mark.parametrize("blank", ["", " \t\r\n "])
def test_scope_response_drafts_reject_blank_text_fields(
    model_type: type[object],
    valid_value: dict[str, str],
    blank: str,
) -> None:
    for field_name in valid_value:
        with pytest.raises(ValidationError):
            model_type.model_validate(  # type: ignore[attr-defined]
                {**valid_value, field_name: blank}
            )


def test_failed_async_requests_still_consume_the_three_call_budget() -> None:
    class FailingAsyncModel:
        def __init__(self) -> None:
            self.attempts = 0

        async def ainvoke(self, _: object) -> object:
            self.attempts += 1
            raise TimeoutError("provider timeout")

    async def exercise() -> None:
        model = FailingAsyncModel()
        budget = LlmCallBudget()

        for expected_used in range(1, MAX_LLM_API_CALLS_PER_TURN + 1):
            with pytest.raises(TimeoutError):
                await ainvoke_with_budget(model, [], budget=budget)
            assert budget.used_calls == expected_used

        with pytest.raises(LlmCallBudgetExceeded):
            await ainvoke_with_budget(model, [], budget=budget)

        assert model.attempts == MAX_LLM_API_CALLS_PER_TURN
        assert budget.remaining_calls == 0

    asyncio.run(exercise())


def test_async_cancellation_propagates_after_claiming_the_budget() -> None:
    class CancellingModel:
        async def ainvoke(self, _: object) -> object:
            raise asyncio.CancelledError

    async def exercise() -> None:
        budget = LlmCallBudget()

        with pytest.raises(asyncio.CancelledError):
            await ainvoke_with_budget(CancellingModel(), [], budget=budget)

        assert budget.used_calls == 1

    asyncio.run(exercise())
