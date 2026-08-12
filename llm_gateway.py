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
    field_validator,
)

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
ENTRY_MODEL_ENV = "OPENAI_ENTRY_MODEL"
ANSWER_MODEL_ENV = "OPENAI_ANSWER_MODEL"
LEGACY_MODEL_ENV = "OPENAI_MODEL"

RECENT_MESSAGE_LIMIT = 12
MAX_LLM_API_CALLS_PER_TURN = 3
FIXED_FAILURE_MESSAGE = "답변을 만드는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


RegistryIntentHint = Literal[
    "item_lookup",
    "semantic_search",
    "catalog_query",
    "hierarchy_query",
    "relation_query",
    "aggregate_query",
    "comparison_query",
]
RegistryAnswerMode = Literal["registry_facts", "registry_facts_with_general_guidance"]
ConstraintKind = Literal[
    "instrument",
    "node_type",
    "hierarchy_tier",
    "relation",
    "field",
    "scope",
    "filter",
    "group_by",
]
ScopeTopicCategory = Literal[
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
]
MetaConversationKind = Literal[
    "greeting",
    "thanks",
    "farewell",
    "bot_identity",
    "capability_help",
]


class ConstraintMention(_StrictFrozenModel):
    """An untrusted constraint phrase copied from the user's wording."""

    kind: ConstraintKind
    text: StrictStr


class ScopeTopic(_StrictFrozenModel):
    """A non-registry topic label, never an answer to the user's question."""

    category: ScopeTopicCategory
    summary: StrictStr = Field(min_length=1, max_length=160)

    @field_validator("summary")
    @classmethod
    def _reject_blank_summary(cls, value: StrictStr) -> StrictStr:
        if not value.strip():
            raise ValueError("scope topic summary must not be blank")
        return value


class _StrictNonBlankResponseModel(_StrictFrozenModel):
    """Strict structured-output base whose public prose fields cannot be blank."""

    @field_validator("*", check_fields=False)
    @classmethod
    def _reject_blank_text(cls, value: StrictStr) -> StrictStr:
        if not value.strip():
            raise ValueError("response text must not be blank")
        return value


class OutOfScopeResponseDraft(_StrictNonBlankResponseModel):
    """Buffered Scope Writer output for a substantive non-registry request."""

    acknowledgement: StrictStr
    scope_boundary: StrictStr
    registry_redirect: StrictStr


class MetaResponseDraft(_StrictNonBlankResponseModel):
    """Buffered Scope Writer output for a supported social or help turn."""

    response: StrictStr
    registry_redirect: StrictStr


class RegistryQueryDraft(_StrictFrozenModel):
    """Small, non-authoritative interpretation produced by the entry model.

    Canonical names, stable IDs, exact hierarchy enums, and final filters are
    deliberately absent.  The deterministic query normalizer owns those
    decisions after checking the raw query and active registry snapshot.
    """

    intent_hint: RegistryIntentHint
    target_mentions: list[StrictStr]
    constraint_mentions: list[ConstraintMention]
    semantic_description: StrictStr | None
    reuse_previous_result: StrictBool
    answer_mode: RegistryAnswerMode
    acknowledge_greeting: StrictBool
    out_of_scope_remainder: ScopeTopic | None


class RegistryRouteDecision(_StrictFrozenModel):
    route: Literal["registry_query"]
    draft: RegistryQueryDraft


class MetaRouteDecision(_StrictFrozenModel):
    route: Literal["meta_conversation"]
    kind: MetaConversationKind


class OutOfScopeRouteDecision(_StrictFrozenModel):
    route: Literal["out_of_scope"]
    topic: ScopeTopic


GatewayDecisionVariant = (
    RegistryRouteDecision
    | MetaRouteDecision
    | OutOfScopeRouteDecision
)
_DISCRIMINATED_GATEWAY_DECISION = TypeAdapter(
    Annotated[GatewayDecisionVariant, Field(discriminator="route")]
)


class GatewayDecision(_StrictFrozenModel):
    """OpenAI-compatible object wrapper for the entry model's decision.

    OpenAI Structured Outputs requires the root schema to be an object.  The
    ordinary union therefore lives under ``decision`` instead of producing a
    root-level ``anyOf``/``oneOf``.  After the provider response is received,
    :func:`validate_gateway_decision` revalidates the nested variant through a
    route-discriminated adapter.
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
            key="meta_conversation",
            description=(
                "인사·감사·짧은 작별, 챗봇 소개와 지원 기능 사용법 안내를 제공합니다."
            ),
        ),
    ),
    limitations=(
        CapabilitySpec(
            key="invented_registry_facts",
            description="레지스트리에 없는 역량 정의나 등록 사실을 만들지 않습니다.",
        ),
        CapabilitySpec(
            key="non_registry_knowledge",
            description=(
                "모든 실질적인 비역량 지식 질문에는 내용을 직접 답하지 않고 "
                "주제에 맞는 범위 안내와 역량 질문 전환을 제공합니다."
            ),
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


async def ainvoke_with_budget(
    model: Any,
    input_value: Any,
    *,
    budget: LlmCallBudget,
    **kwargs: Any,
) -> Any:
    """Await one model request after claiming its non-refundable budget slot."""

    budget.claim()
    return await model.ainvoke(input_value, **kwargs)


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
