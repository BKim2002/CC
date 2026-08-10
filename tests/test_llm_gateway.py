"""Focused tests for the gateway/writer shared contracts."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

import llm_gateway
from competency_query import QueryIntent
from llm_gateway import (
    CAPABILITY_MANIFEST,
    DEFAULT_OPENAI_MODEL,
    FIXED_FAILURE_MESSAGE,
    MAX_LLM_API_CALLS_PER_TURN,
    CapabilityHelpDecision,
    GatewayDecision,
    GeneralConversationDecision,
    LlmCallBudget,
    LlmCallBudgetExceeded,
    ModelRole,
    NeedsClarificationDecision,
    RegistryQueryDecision,
    UnsupportedDecision,
    capability_manifest_for_prompt,
    create_chat_model,
    invoke_with_budget,
    recent_conversation_context,
    selected_answer_model_name,
    selected_entry_model_name,
    validate_gateway_decision,
)


def _registry_decision(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "route": "registry_query",
        "query": {
            "intent": "item_lookup",
            "target_names": ["책임성"],
        },
        "answer_mode": "registry_facts",
        "acknowledge_greeting": False,
        "unsupported_remainder": None,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (_registry_decision(), RegistryQueryDecision),
        (
            {"route": "general_conversation", "conversation_type": "greeting"},
            GeneralConversationDecision,
        ),
        ({"route": "capability_help"}, CapabilityHelpDecision),
        (
            {"route": "unsupported", "unsupported_type": "current_information"},
            UnsupportedDecision,
        ),
        (
            {"route": "needs_clarification", "clarification_type": "general"},
            NeedsClarificationDecision,
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
    flat = {"route": "general_conversation", "conversation_type": "greeting"}
    registry_flat = _registry_decision(
        query={
            "intent": "item_lookup",
            "target_names": ["책임성"],
            "fields": ["definition"],
        }
    )

    flat_decision = validate_gateway_decision(flat)
    wrapped_decision = validate_gateway_decision({"decision": flat})
    model_decision = validate_gateway_decision(
        GatewayDecision.model_validate({"decision": flat})
    )
    registry_decision = validate_gateway_decision(registry_flat)

    assert isinstance(flat_decision, GeneralConversationDecision)
    assert wrapped_decision == flat_decision
    assert model_decision == flat_decision
    assert isinstance(registry_decision, RegistryQueryDecision)
    assert [field.value for field in registry_decision.query.fields] == ["definition"]


def test_gateway_decision_schema_has_object_root_and_nested_union() -> None:
    schema = GatewayDecision.model_json_schema()

    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert "oneOf" not in schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["decision"]
    assert len(schema["properties"]["decision"]["anyOf"]) == 5
    for definition_name in (
        "RegistryQueryDecision",
        "GeneralConversationDecision",
        "CapabilityHelpDecision",
        "UnsupportedDecision",
        "NeedsClarificationDecision",
    ):
        assert schema["$defs"][definition_name]["additionalProperties"] is False


def test_gateway_decision_converts_to_openai_strict_object_schema() -> None:
    schema = to_strict_json_schema(GatewayDecision)

    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert "oneOf" not in schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["decision"]
    assert len(schema["properties"]["decision"]["anyOf"]) == 5


@pytest.mark.parametrize(
    "invalid_value",
    [
        {"route": "unknown"},
        {"route": "capability_help", "answer": "사용자에게 보여 줄 문장"},
        {
            "route": "general_conversation",
            "conversation_type": "weather",
        },
        _registry_decision(acknowledge_greeting=1),
        _registry_decision(
            query={"intent": "help", "target_names": []},
        ),
        _registry_decision(
            query={
                "intent": "item_lookup",
                "target_names": ["책임성"],
                "unrecognized": True,
            },
        ),
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
        QueryIntent.ITEM_LOOKUP,
        QueryIntent.SEMANTIC_SEARCH,
        QueryIntent.CATALOG_QUERY,
        QueryIntent.HIERARCHY_QUERY,
        QueryIntent.RELATION_QUERY,
        QueryIntent.AGGREGATE_QUERY,
        QueryIntent.COMPARISON_QUERY,
    }

    for intent in supported:
        decision = validate_gateway_decision(
            _registry_decision(query={"intent": intent.value})
        )
        assert isinstance(decision, RegistryQueryDecision)
        assert decision.query.intent is intent


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
        "general_conversation",
    }
    assert {entry.key for entry in CAPABILITY_MANIFEST.limitations} == {
        "invented_registry_facts",
        "personal_assessment",
        "employment_fit",
        "current_information",
        "professional_advice",
    }
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
