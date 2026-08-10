"""Shared contracts for the LLM gateway and the two response writers.

This module deliberately contains no registry lookup or LangGraph routing logic.
It owns the cross-cutting contracts that every graph node must agree on:

* the strict, discriminated gateway decision schema;
* the single capability manifest used by prompts and capability answers;
* role-specific model selection with retries disabled; and
* the per-turn *actual request* budget.

Keeping these contracts outside ``competency_interpreter`` lets graph nodes stay
small and prevents prompts, model configuration, and failure text from drifting.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Annotated, Any, AsyncIterator, Iterable, Literal, Mapping

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
)

from competency_query import ParsedRegistryQuery, QueryIntent


DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
ENTRY_MODEL_ENV = "OPENAI_ENTRY_MODEL"
ANSWER_MODEL_ENV = "OPENAI_ANSWER_MODEL"
LEGACY_MODEL_ENV = "OPENAI_MODEL"

RECENT_MESSAGE_LIMIT = 12
MAX_LLM_API_CALLS_PER_TURN = 3
FIXED_FAILURE_MESSAGE = "답변을 만드는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# ``help`` and ``out_of_scope`` deliberately do not belong to this type.  The
# gateway has dedicated routes for those cases, while all seven deterministic
# registry query intents remain available here.
RegistryGatewayIntent = Literal[
    QueryIntent.ITEM_LOOKUP,
    QueryIntent.SEMANTIC_SEARCH,
    QueryIntent.CATALOG_QUERY,
    QueryIntent.HIERARCHY_QUERY,
    QueryIntent.RELATION_QUERY,
    QueryIntent.AGGREGATE_QUERY,
    QueryIntent.COMPARISON_QUERY,
]
RegistryAnswerMode = Literal["registry_facts", "registry_facts_with_general_guidance"]
GeneralConversationKind = Literal[
    "greeting",
    "small_talk",
    "bot_identity",
    "simple_concept",
]
UnsupportedKind = Literal[
    "current_information",
    "sensitive_advice",
    "unsafe_or_other_unsupported",
]
ClarificationKind = Literal["registry", "general"]


class GatewayRegistryQuery(ParsedRegistryQuery):
    """Registry query accepted specifically from the gateway.

    Narrowing ``intent`` makes the JSON schema itself exclude the legacy help
    and out-of-scope intents instead of relying on a later Python branch.
    """

    intent: RegistryGatewayIntent


class RegistryQueryDecision(_StrictFrozenModel):
    route: Literal["registry_query"]
    query: GatewayRegistryQuery
    answer_mode: RegistryAnswerMode
    acknowledge_greeting: StrictBool
    unsupported_remainder: UnsupportedKind | None


class GeneralConversationDecision(_StrictFrozenModel):
    route: Literal["general_conversation"]
    conversation_type: GeneralConversationKind


class CapabilityHelpDecision(_StrictFrozenModel):
    route: Literal["capability_help"]


class UnsupportedDecision(_StrictFrozenModel):
    route: Literal["unsupported"]
    unsupported_type: UnsupportedKind


class NeedsClarificationDecision(_StrictFrozenModel):
    route: Literal["needs_clarification"]
    clarification_type: ClarificationKind


GatewayDecisionVariant = (
    RegistryQueryDecision
    | GeneralConversationDecision
    | CapabilityHelpDecision
    | UnsupportedDecision
    | NeedsClarificationDecision
)
_DISCRIMINATED_GATEWAY_DECISION = TypeAdapter(
    Annotated[GatewayDecisionVariant, Field(discriminator="route")]
)


class GatewayDecision(_StrictFrozenModel):
    """OpenAI-compatible object wrapper for the entry model's decision.

    OpenAI Structured Outputs requires the root schema to be an object.  The
    ordinary union therefore lives under ``decision`` instead of producing a
    root-level ``anyOf``/``oneOf``.  Each variant's literal ``route`` remains the
    strict discriminator when Pydantic validates the nested value.
    """

    decision: GatewayDecisionVariant


def validate_gateway_decision(value: Any) -> GatewayDecisionVariant:
    """Validate wrapper output while retaining flat test-double compatibility."""

    if isinstance(value, GatewayDecision):
        candidate: Any = value.decision
    elif isinstance(value, Mapping) and "decision" in value:
        candidate = GatewayDecision.model_validate(value).decision
    else:
        candidate = value
    if isinstance(candidate, Mapping):
        return _DISCRIMINATED_GATEWAY_DECISION.validate_json(
            json.dumps(candidate, ensure_ascii=False),
            strict=True,
        )
    return _DISCRIMINATED_GATEWAY_DECISION.validate_python(candidate, strict=True)


class CapabilitySpec(_StrictFrozenModel):
    key: StrictStr
    description: StrictStr


class CapabilityManifest(_StrictFrozenModel):
    """The only source used to tell a model what the chatbot can do."""

    supported: tuple[CapabilitySpec, ...]
    limitations: tuple[CapabilitySpec, ...]
    max_comparison_items: Literal[3]
    max_semantic_candidates: Literal[3]


CAPABILITY_MANIFEST = CapabilityManifest(
    supported=(
        CapabilitySpec(
            key="registry_information",
            description="역량 레지스트리의 정식 정의와 등록 정보를 조회합니다.",
        ),
        CapabilitySpec(
            key="hierarchy_and_relations",
            description=(
                "역량 목록과 위계, 부모·자식, 조상·후손, 형제 관계를 조회합니다."
            ),
        ),
        CapabilitySpec(
            key="filters_and_aggregates",
            description="검사와 node type별로 역량을 필터링하고 집계합니다.",
        ),
        CapabilitySpec(
            key="comparison",
            description="한 번에 최대 3개의 등록 역량을 비교합니다.",
        ),
        CapabilitySpec(
            key="semantic_candidates",
            description="행동 설명과 관련된 등록 역량 후보를 찾습니다.",
        ),
        CapabilitySpec(
            key="general_guidance",
            description="등록 사실과 구분해 비개인화된 일반 활용 제안을 제공합니다.",
        ),
        CapabilitySpec(
            key="general_conversation",
            description="인사와 챗봇 소개, 시사성이 없는 간단한 일반 개념에 답합니다.",
        ),
    ),
    limitations=(
        CapabilitySpec(
            key="invented_registry_facts",
            description="레지스트리에 없는 역량 정의나 등록 사실을 만들지 않습니다.",
        ),
        CapabilitySpec(
            key="personal_assessment",
            description="사용자의 개인 역량을 평가하거나 점수를 추정하지 않습니다.",
        ),
        CapabilitySpec(
            key="employment_fit",
            description="채용 판단이나 직무 적합성·직무 추천을 제공하지 않습니다.",
        ),
        CapabilitySpec(
            key="current_information",
            description="웹 검색, 날씨, 뉴스 같은 최신 정보를 확인하지 않습니다.",
        ),
        CapabilitySpec(
            key="professional_advice",
            description="의료·법률·금융 전문 조언을 제공하지 않습니다.",
        ),
    ),
    max_comparison_items=3,
    max_semantic_candidates=3,
)


def capability_manifest_for_prompt() -> str:
    """Serialize the manifest without maintaining a second prompt template."""

    return json.dumps(
        CAPABILITY_MANIFEST.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )


class ModelRole(str, Enum):
    ENTRY = "entry"
    ANSWER = "answer"


def _nonblank_environment_value(
    name: str,
    environ: Mapping[str, str],
) -> str | None:
    value = environ.get(name, "").strip()
    return value or None


def selected_model_name(
    role: ModelRole,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return role-specific -> legacy -> default model configuration."""

    source = os.environ if environ is None else environ
    role_environment_name = ENTRY_MODEL_ENV if role is ModelRole.ENTRY else ANSWER_MODEL_ENV
    return (
        _nonblank_environment_value(role_environment_name, source)
        or _nonblank_environment_value(LEGACY_MODEL_ENV, source)
        or DEFAULT_OPENAI_MODEL
    )


def selected_entry_model_name(*, environ: Mapping[str, str] | None = None) -> str:
    return selected_model_name(ModelRole.ENTRY, environ=environ)


def selected_answer_model_name(*, environ: Mapping[str, str] | None = None) -> str:
    return selected_model_name(ModelRole.ANSWER, environ=environ)


def create_chat_model(
    role: ModelRole,
    *,
    timeout: float = 30,
    model_name: str | None = None,
) -> ChatOpenAI:
    """Create a role model whose SDK cannot silently exceed the turn budget."""

    selected = (
        model_name.strip()
        if model_name is not None and model_name.strip()
        else selected_model_name(role)
    )
    return ChatOpenAI(
        model=selected,
        max_retries=0,
        timeout=timeout,
    )


class ConversationTurn(_StrictFrozenModel):
    role: Literal["user", "assistant"]
    content: StrictStr


def _message_role(message: BaseMessage | Mapping[str, Any]) -> Literal["user", "assistant"] | None:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raw_role: Any
    if isinstance(message, Mapping):
        raw_role = message.get("role", message.get("type"))
    else:
        raw_role = getattr(message, "type", None)
    if raw_role in {"user", "human"}:
        return "user"
    if raw_role in {"assistant", "ai"}:
        return "assistant"
    return None


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def recent_conversation_context(
    messages: Iterable[BaseMessage | Mapping[str, Any]],
) -> tuple[ConversationTurn, ...]:
    """Return at most the last 12 user/assistant messages in stable order."""

    turns: list[ConversationTurn] = []
    for message in messages:
        role = _message_role(message)
        if role is None:
            continue
        content_source = (
            message.get("content", "")
            if isinstance(message, Mapping)
            else getattr(message, "content", "")
        )
        content = _text_content(content_source).strip()
        if content:
            turns.append(ConversationTurn(role=role, content=content))
    return tuple(turns[-RECENT_MESSAGE_LIMIT:])


class LlmCallBudgetExceeded(RuntimeError):
    """Raised before an API request that would exceed the per-turn limit."""


class LlmCallBudget(BaseModel):
    """Mutable, serializable ledger counting actual SDK request attempts.

    A caller must claim immediately before ``invoke``/``stream``.  The claim is
    retained even when the provider call raises, so a recovery attempt consumes
    the next slot instead of resetting the budget.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    used_calls: StrictInt = Field(default=0, ge=0, le=MAX_LLM_API_CALLS_PER_TURN)

    @property
    def remaining_calls(self) -> int:
        return MAX_LLM_API_CALLS_PER_TURN - self.used_calls

    def claim(self) -> int:
        if self.used_calls >= MAX_LLM_API_CALLS_PER_TURN:
            raise LlmCallBudgetExceeded("LLM call budget exhausted")
        self.used_calls += 1
        return self.used_calls


def invoke_with_budget(
    model: Any,
    input_value: Any,
    *,
    budget: LlmCallBudget,
    **kwargs: Any,
) -> Any:
    budget.claim()
    return model.invoke(input_value, **kwargs)


async def astream_with_budget(
    model: Any,
    input_value: Any,
    *,
    budget: LlmCallBudget,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    budget.claim()
    async for chunk in model.astream(input_value, **kwargs):
        yield chunk
