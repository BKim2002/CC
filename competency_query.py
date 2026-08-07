"""Pure, registry-backed query planning, execution, and answer grounding.

This module intentionally contains no database, OpenAI, or LangGraph imports.
The application may ask an LLM to produce :class:`ParsedRegistryQuery`, but all
names, filters, relationships, counts, and answer facts are validated and
computed here from one immutable :class:`~competency_registry.RegistrySnapshot`.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
import re
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from competency_registry import RegistrySnapshot


MAX_RENDERED_ITEMS = 100
MAX_COMPARISON_ITEMS = 3
MAX_HIERARCHY_DEPTH = 100
MAX_ANSWER_CHARS = 20_000
# Leave room for Korean headings, numbering, truncation guidance, and model
# connective prose.  Detailed registry facts are limited before they enter the
# writer context, so an LLM cannot reconstruct omitted definitions or paths.
MAX_GROUNDED_FACT_CHARS = 16_000
NOT_PROVIDED_DEFINITION = "독립적인 정의가 제공되어 있지 않음"
WRITTEN_PROFILE_ID = "written_competency_v1"
WRITTEN_INSTRUMENT_ID = "written_competency"
VIDEO_INSTRUMENT_ID = "video_interview"


class _ValueEnum(str, Enum):
    """String enum whose values serialize cleanly into checkpoint JSON."""


class QueryIntent(_ValueEnum):
    ITEM_LOOKUP = "item_lookup"
    SEMANTIC_SEARCH = "semantic_search"
    CATALOG_QUERY = "catalog_query"
    HIERARCHY_QUERY = "hierarchy_query"
    RELATION_QUERY = "relation_query"
    AGGREGATE_QUERY = "aggregate_query"
    COMPARISON_QUERY = "comparison_query"
    HELP = "help"
    OUT_OF_SCOPE = "out_of_scope"


class HierarchyTier(_ValueEnum):
    OVERALL = "overall"
    UPPER = "upper"
    MIDDLE = "middle"
    LOWER = "lower"
    BOTTOM = "bottom"


class RelationType(_ValueEnum):
    PARENT = "parent"
    ANCESTORS = "ancestors"
    CHILDREN = "children"
    DESCENDANTS = "descendants"
    SIBLINGS = "siblings"


class ItemField(_ValueEnum):
    NAME = "name"
    INSTRUMENT_LABEL = "instrument_label"
    NODE_TYPE = "node_type"
    DEFINITION = "definition"
    PARENT = "parent"
    CHILDREN = "children"
    PATH = "path"
    ANALYSIS_INCLUDED = "analysis_included"


class GroupBy(_ValueEnum):
    NONE = "none"
    INSTRUMENT = "instrument"
    NODE_TYPE = "node_type"
    HIERARCHY_TIER = "hierarchy_tier"
    DEPTH = "depth"
    DEFINITION_STATUS = "definition_status"
    ANALYSIS_INCLUDED = "analysis_included"


class QueryScope(_ValueEnum):
    ALL = "all"
    TARGETS = "targets"
    SUBTREE = "subtree"
    PREVIOUS_RESULT = "previous_result"


class ResultKind(_ValueEnum):
    ITEMS = "items"
    HIERARCHY = "hierarchy"
    RELATIONS = "relations"
    AGGREGATE = "aggregate"
    COMPARISON = "comparison"
    HELP = "help"
    CLARIFICATION = "clarification"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ParsedRegistryQuery(_StrictModel):
    """Limited, untrusted shape accepted from the structured query parser."""

    intent: QueryIntent
    target_names: list[StrictStr] = Field(default_factory=list)
    instrument_labels: list[StrictStr] = Field(default_factory=list)
    node_types: list[StrictStr] = Field(default_factory=list)
    hierarchy_tiers: list[HierarchyTier] = Field(default_factory=list)
    relation: RelationType | None = None
    related_tier: HierarchyTier | None = None
    fields: list[ItemField] = Field(default_factory=list)
    root_only: StrictBool = False
    leaf_only: StrictBool = False
    has_parent: StrictBool | None = None
    has_children: StrictBool | None = None
    analysis_included: StrictBool | None = None
    definition_statuses: list[StrictStr] = Field(default_factory=list)
    group_by: GroupBy = GroupBy.NONE
    scope: QueryScope = QueryScope.ALL
    max_depth: StrictInt | None = None
    reuse_previous_result: StrictBool = False
    semantic_query: StrictStr | None = None


class QueryFilters(_StrictModel):
    root_only: StrictBool = False
    leaf_only: StrictBool = False
    has_parent: StrictBool | None = None
    has_children: StrictBool | None = None
    analysis_included: StrictBool | None = None
    definition_statuses: list[StrictStr] = Field(default_factory=list)


class RegistryQueryPlan(_StrictModel):
    """Canonical, checkpoint-safe plan containing stable IDs only."""

    intent: QueryIntent
    user_question: StrictStr = ""
    target_ids: list[StrictStr] = Field(default_factory=list)
    instrument_ids: list[StrictStr] = Field(default_factory=list)
    node_types: list[StrictStr] = Field(default_factory=list)
    hierarchy_tiers: list[HierarchyTier] = Field(default_factory=list)
    relation: RelationType | None = None
    related_tier: HierarchyTier | None = None
    fields: list[ItemField] = Field(default_factory=list)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    group_by: GroupBy = GroupBy.NONE
    scope: QueryScope = QueryScope.ALL
    max_depth: StrictInt | None = None
    previous_result_ids: list[StrictStr] = Field(default_factory=list)
    terminology_profile_id: StrictStr | None = None
    semantic_query: StrictStr | None = None


class PlanValidationResult(_StrictModel):
    plan: RegistryQueryPlan | None = None
    clarification: StrictStr | None = None
    errors: list[StrictStr] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.plan is not None and not self.errors


class ResultGroup(_StrictModel):
    key: StrictStr
    label: StrictStr
    item_ids: list[StrictStr] = Field(default_factory=list)
    count: StrictInt = 0


class RegistryQueryResult(_StrictModel):
    kind: ResultKind
    item_ids: list[StrictStr] = Field(default_factory=list)
    groups: list[ResultGroup] = Field(default_factory=list)
    counts: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    truncated: StrictBool = False
    clarification: StrictStr | None = None


class HierarchyTerminologyProfile(_StrictModel):
    profile_id: StrictStr
    instrument_id: StrictStr
    tier_levels: dict[HierarchyTier, StrictStr]


class HierarchyProfileValidation(_StrictModel):
    valid: StrictBool
    profile: HierarchyTerminologyProfile | None = None
    tier_item_ids: dict[HierarchyTier, list[StrictStr]] = Field(default_factory=dict)
    reason: StrictStr | None = None


class GroundedFact(_StrictModel):
    fact_type: StrictStr
    label: StrictStr
    value: Any
    item_id: StrictStr | None = None


class GroundedAnswerContext(_StrictModel):
    """Ephemeral facts passed to the response writer; never checkpoint this."""

    intent: QueryIntent
    user_question: StrictStr
    facts: list[GroundedFact] = Field(default_factory=list)
    exact_definitions: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    allowed_names: list[StrictStr] = Field(default_factory=list)
    allowed_numbers: list[StrictInt] = Field(default_factory=list)
    hierarchy_text: StrictStr | None = None
    truncated: StrictBool = False
    clarification: StrictStr | None = None
    required_names: list[StrictStr] = Field(default_factory=list, exclude=True, repr=False)
    registry_names: list[StrictStr] = Field(default_factory=list, exclude=True, repr=False)


class GroundedAnswerValidation(_StrictModel):
    valid: StrictBool
    errors: list[StrictStr] = Field(default_factory=list)


WRITTEN_COMPETENCY_V1 = HierarchyTerminologyProfile(
    profile_id=WRITTEN_PROFILE_ID,
    instrument_id=WRITTEN_INSTRUMENT_ID,
    tier_levels={
        HierarchyTier.OVERALL: "overall",
        HierarchyTier.UPPER: "L1",
        HierarchyTier.MIDDLE: "L2",
        HierarchyTier.LOWER: "L3",
        HierarchyTier.BOTTOM: "L4",
    },
)

TIER_LABELS: Mapping[HierarchyTier, str] = {
    HierarchyTier.OVERALL: "종합점수",
    HierarchyTier.UPPER: "상위요인",
    HierarchyTier.MIDDLE: "중위요인",
    HierarchyTier.LOWER: "하위요인",
    HierarchyTier.BOTTOM: "최하위요인",
}


def _items(snapshot: RegistrySnapshot) -> list[Mapping[str, Any]]:
    return list(snapshot.document.get("items", ()))


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _active_children(item: Mapping[str, Any], snapshot: RegistrySnapshot) -> list[str]:
    return [item_id for item_id in item.get("children_ids", ()) if item_id in snapshot.id_lookup]


def _instrument_catalog(snapshot: RegistrySnapshot) -> tuple[dict[str, str], dict[str, str]]:
    by_id: dict[str, str] = {}
    by_label: dict[str, str] = {}
    for item in _items(snapshot):
        instrument_id = item["instrument"]
        label = item["instrument_label"]
        previous = by_id.setdefault(instrument_id, label)
        if previous != label:
            raise ValueError(f"instrument {instrument_id!r}에 서로 다른 label이 등록되어 있습니다.")
        owner = by_label.setdefault(label, instrument_id)
        if owner != instrument_id:
            raise ValueError(f"instrument label {label!r}이 여러 ID에 등록되어 있습니다.")
    return by_id, by_label


def node_type_term(item: Mapping[str, Any]) -> str:
    """Return the user-facing node term without mixing written/video vocabularies."""

    if item.get("instrument") == VIDEO_INSTRUMENT_ID:
        if item.get("level") == "factor":
            return "요인"
        if item.get("level") == "item":
            return "세부항목"
    return str(item.get("level", "항목"))


def validate_hierarchy_terminology_profile(
    snapshot: RegistrySnapshot,
    profile_id: str = WRITTEN_PROFILE_ID,
) -> HierarchyProfileValidation:
    """Validate the version-specific written hierarchy contract.

    The expected counts are compatibility checks for this named profile, not
    query-time constants. A future registry must opt into a new profile rather
    than silently mapping unfamiliar levels to the old Korean tier labels.
    """

    if profile_id != WRITTEN_PROFILE_ID:
        return HierarchyProfileValidation(valid=False, reason="지원하지 않는 위계 용어 프로필입니다.")

    profile = WRITTEN_COMPETENCY_V1
    written = [item for item in _items(snapshot) if item["instrument"] == profile.instrument_id]
    if not written:
        return HierarchyProfileValidation(valid=False, reason="현재 레지스트리에 필기 역량검사가 없습니다.")

    expected_counts = {
        HierarchyTier.OVERALL: 1,
        HierarchyTier.UPPER: 3,
        HierarchyTier.MIDDLE: 10,
        HierarchyTier.LOWER: 30,
        HierarchyTier.BOTTOM: 9,
    }
    tier_item_ids: dict[HierarchyTier, list[str]] = {}
    for tier, level in profile.tier_levels.items():
        ids = [item["id"] for item in written if item["level"] == level]
        tier_item_ids[tier] = ids
        if len(ids) != expected_counts[tier]:
            return HierarchyProfileValidation(
                valid=False,
                reason=f"현재 필기검사의 {TIER_LABELS[tier]} 구성이 {profile_id} 프로필과 일치하지 않습니다.",
            )

    by_name = {item["name"]: item for item in written}
    overall = by_name.get("역검 종합점수")
    upper_names = [snapshot.id_lookup[item_id]["name"] for item_id in tier_item_ids[HierarchyTier.UPPER]]
    if overall is None or overall["level"] != "overall" or upper_names != ["성과 예측", "관계 예측", "적응 예측"]:
        return HierarchyProfileValidation(valid=False, reason="현재 필기검사의 종합점수 또는 상위요인 구성이 프로필과 일치하지 않습니다.")

    strategy = by_name.get("전략성")
    expected_strategy_children = ["기억력", "인지력", "분석력"]
    if strategy is None:
        return HierarchyProfileValidation(valid=False, reason="전략성 구조가 위계 용어 프로필과 일치하지 않습니다.")
    strategy_children = [snapshot.id_lookup[item_id]["name"] for item_id in _active_children(strategy, snapshot)]
    if strategy_children != expected_strategy_children:
        return HierarchyProfileValidation(valid=False, reason="전략성의 하위 구조가 위계 용어 프로필과 일치하지 않습니다.")
    for name in expected_strategy_children:
        item = by_name.get(name)
        children = _active_children(item, snapshot) if item is not None else []
        if item is None or item["level"] != "L3" or len(children) != 3 or any(snapshot.id_lookup[child]["level"] != "L4" for child in children):
            return HierarchyProfileValidation(valid=False, reason=f"{name}의 최하위요인 구조가 위계 용어 프로필과 일치하지 않습니다.")

    return HierarchyProfileValidation(valid=True, profile=profile, tier_item_ids=tier_item_ids)


def _normalized_text(value: str) -> str:
    value = value.replace("，", ",").replace("、", ",")
    return re.sub(r"\s+", " ", value.strip())


def _exact_target_ids(query: str, snapshot: RegistrySnapshot) -> list[str]:
    separator_preserving = query.replace("，", ",").replace("、", ",").strip()
    if not separator_preserving:
        return []
    parts = [_normalized_text(part) for part in re.split(r"[,\r\n]+", separator_preserving)]
    if not parts or any(not part for part in parts):
        return []
    normalized_lookup: dict[str, Mapping[str, Any]] = {}
    collisions: set[str] = set()
    for label, item in snapshot.lookup.items():
        key = _normalized_text(label)
        if key in normalized_lookup and normalized_lookup[key]["id"] != item["id"]:
            collisions.add(key)
        else:
            normalized_lookup[key] = item
    target_ids: list[str] = []
    for part in parts:
        key = _normalized_text(part)
        item = None if key in collisions else normalized_lookup.get(key)
        if item is None:
            return []
        target_ids.append(item["id"])
    return _unique(target_ids)


def detect_deterministic_query(query: str, snapshot: RegistrySnapshot) -> RegistryQueryPlan | None:
    """Return a safe plan for deliberately narrow fast paths, otherwise ``None``."""

    normalized = _normalized_text(query)
    target_ids = _exact_target_ids(query, snapshot)
    if target_ids:
        return RegistryQueryPlan(
            intent=QueryIntent.ITEM_LOOKUP,
            user_question=query,
            target_ids=target_ids,
            fields=[ItemField.DEFINITION, ItemField.CHILDREN, ItemField.PATH],
            scope=QueryScope.TARGETS,
        )

    compact = normalized.rstrip(".?!。？！")
    help_pattern = r"(?:사용법|도움말|어떤 질문(?:을 할 수 있(?:어|나요)|이 가능(?:해|한가요)))"
    if re.fullmatch(help_pattern, compact):
        return RegistryQueryPlan(intent=QueryIntent.HELP, user_question=query)

    polite = r"(?:\s*(?:보여\s*줘|보여\s*주세요|알려\s*줘|알려\s*주세요))?"
    if re.fullmatch(r"전체\s*역량\s*목록" + polite, compact):
        return RegistryQueryPlan(intent=QueryIntent.CATALOG_QUERY, user_question=query)
    if re.fullmatch(r"전체\s*역량\s*(?:수|개수)" + polite, compact):
        return RegistryQueryPlan(intent=QueryIntent.AGGREGATE_QUERY, user_question=query)
    if re.fullmatch(r"전체\s*위계\s*구조" + polite, compact):
        return RegistryQueryPlan(intent=QueryIntent.HIERARCHY_QUERY, user_question=query)

    tier_match = re.fullmatch(r"(상위|중위|하위|최하위)\s*요인\s*(목록|개수)" + polite, compact)
    if tier_match:
        tier = {
            "상위": HierarchyTier.UPPER,
            "중위": HierarchyTier.MIDDLE,
            "하위": HierarchyTier.LOWER,
            "최하위": HierarchyTier.BOTTOM,
        }[tier_match.group(1)]
        profile = validate_hierarchy_terminology_profile(snapshot)
        if not profile.valid:
            return None
        intent = QueryIntent.CATALOG_QUERY if tier_match.group(2) == "목록" else QueryIntent.AGGREGATE_QUERY
        return RegistryQueryPlan(
            intent=intent,
            user_question=query,
            instrument_ids=[WRITTEN_INSTRUMENT_ID],
            hierarchy_tiers=[tier],
            terminology_profile_id=WRITTEN_PROFILE_ID,
        )

    video_match = re.fullmatch(r"영상면접\s*(요인|세부항목)\s*목록" + polite, compact)
    if video_match:
        level = "factor" if video_match.group(1) == "요인" else "item"
        return RegistryQueryPlan(
            intent=QueryIntent.CATALOG_QUERY,
            user_question=query,
            instrument_ids=[VIDEO_INSTRUMENT_ID],
            node_types=[level],
        )

    root_leaf = re.fullmatch(r"(?:전체\s*)?(루트|말단)\s*항목\s*목록" + polite, compact)
    if root_leaf:
        return RegistryQueryPlan(
            intent=QueryIntent.CATALOG_QUERY,
            user_question=query,
            filters=QueryFilters(root_only=root_leaf.group(1) == "루트", leaf_only=root_leaf.group(1) == "말단"),
        )
    return None


def validate_parsed_query(
    parsed: ParsedRegistryQuery,
    snapshot: RegistrySnapshot,
    previous_result_ids: list[str] | None = None,
    *,
    user_question: str = "",
) -> PlanValidationResult:
    """Canonicalize parser names/labels and reject unsupported combinations."""

    errors: list[str] = []
    target_ids: list[str] = []
    for raw_name in parsed.target_names:
        name = raw_name.strip()
        item = snapshot.lookup.get(name)
        if item is None:
            errors.append(f"등록되지 않은 역량명입니다: {name}")
        else:
            target_ids.append(item["id"])
    target_ids = _unique(target_ids)

    try:
        instruments_by_id, instruments_by_label = _instrument_catalog(snapshot)
    except ValueError as exc:
        return PlanValidationResult(clarification="현재 검사 도구 구성을 확인할 수 없습니다.", errors=[str(exc)])
    instrument_ids: list[str] = []
    for raw_label in parsed.instrument_labels:
        label = raw_label.strip()
        instrument_id = instruments_by_label.get(label) or (label if label in instruments_by_id else None)
        if instrument_id is None:
            errors.append(f"등록되지 않은 검사 도구입니다: {label}")
        else:
            instrument_ids.append(instrument_id)
    instrument_ids = _unique(instrument_ids)

    current_node_types = {str(item["level"]) for item in _items(snapshot)}
    node_types = _unique(value.strip() for value in parsed.node_types)
    for node_type in node_types:
        if node_type not in current_node_types:
            errors.append(f"등록되지 않은 node type입니다: {node_type}")

    definition_statuses = _unique(value.strip() for value in parsed.definition_statuses)
    current_statuses = {str(item["definition_status"]) for item in _items(snapshot)}
    for status in definition_statuses:
        if status not in current_statuses:
            errors.append(f"등록되지 않은 정의 상태입니다: {status}")

    if parsed.max_depth is not None and not 1 <= parsed.max_depth <= MAX_HIERARCHY_DEPTH:
        errors.append(f"max_depth는 1 이상 {MAX_HIERARCHY_DEPTH} 이하여야 합니다.")
    if parsed.root_only and parsed.leaf_only:
        errors.append("루트 항목과 말단 항목 조건을 동시에 사용할 수 없습니다.")
    if parsed.root_only and parsed.has_parent is True:
        errors.append("루트 항목은 등록된 부모를 가질 수 없습니다.")
    if parsed.leaf_only and parsed.has_children is True:
        errors.append("말단 항목은 등록된 자식을 가질 수 없습니다.")
    if parsed.relation is not None and parsed.related_tier is not None:
        errors.append("구조 관계와 공식 단계 관계를 동시에 지정할 수 없습니다.")
    if (
        parsed.intent != QueryIntent.RELATION_QUERY
        and (parsed.relation is not None or parsed.related_tier is not None)
    ):
        errors.append("관계 조건은 관계 질의에서만 사용할 수 있습니다.")
    if parsed.intent != QueryIntent.AGGREGATE_QUERY and parsed.group_by != GroupBy.NONE:
        errors.append("group_by는 집계 질의에서만 사용할 수 있습니다.")
    if parsed.max_depth is not None and parsed.intent not in {
        QueryIntent.HIERARCHY_QUERY,
        QueryIntent.RELATION_QUERY,
    }:
        errors.append("max_depth는 위계 또는 관계 질의에서만 사용할 수 있습니다.")
    if parsed.hierarchy_tiers and parsed.intent not in {
        QueryIntent.CATALOG_QUERY,
        QueryIntent.HIERARCHY_QUERY,
        QueryIntent.AGGREGATE_QUERY,
    }:
        errors.append("공식 단계 필터는 목록, 위계 또는 집계 질의에서만 사용할 수 있습니다.")

    requested_previous = parsed.reuse_previous_result or parsed.scope == QueryScope.PREVIOUS_RESULT
    active_previous = _unique(item_id for item_id in (previous_result_ids or []) if item_id in snapshot.id_lookup)
    if requested_previous and not active_previous:
        errors.append("현재 레지스트리에서 다시 사용할 수 있는 이전 결과가 없습니다.")
    effective_targets = target_ids or (active_previous if requested_previous else [])
    if parsed.intent == QueryIntent.COMPARISON_QUERY and len(effective_targets) < 2:
        errors.append("비교하려면 서로 다른 역량을 두 개 이상 지정해야 합니다.")
    if parsed.intent == QueryIntent.COMPARISON_QUERY and len(effective_targets) > MAX_COMPARISON_ITEMS:
        return PlanValidationResult(clarification=f"한 번에 {MAX_COMPARISON_ITEMS}개까지 비교할 수 있습니다. 비교할 역량을 줄여 주세요.")
    if parsed.intent == QueryIntent.ITEM_LOOKUP and not effective_targets:
        errors.append("개별 역량 조회에는 하나 이상의 등록된 대상이 필요합니다.")
    if parsed.intent == QueryIntent.RELATION_QUERY and parsed.relation is None and parsed.related_tier is None:
        errors.append("관계 질의에는 부모, 조상, 자식, 후손, 형제 또는 공식 단계를 지정해야 합니다.")
    if (parsed.relation is not None or parsed.related_tier is not None) and not effective_targets:
        errors.append("관계 질의에는 하나 이상의 등록된 대상이 필요합니다.")
    if parsed.scope == QueryScope.SUBTREE and not target_ids:
        errors.append("부분 위계 조회에는 시작할 등록 항목이 필요합니다.")
    if parsed.scope == QueryScope.TARGETS and not target_ids:
        errors.append("대상 범위 조회에는 하나 이상의 등록 항목이 필요합니다.")

    terminology_profile_id: str | None = None
    if parsed.hierarchy_tiers or parsed.related_tier is not None or parsed.group_by == GroupBy.HIERARCHY_TIER:
        profile_validation = validate_hierarchy_terminology_profile(snapshot)
        if not profile_validation.valid:
            return PlanValidationResult(clarification=profile_validation.reason or "현재 검사 버전에는 요청한 공식 단계 용어를 적용할 수 없습니다.")
        if instrument_ids and any(value != WRITTEN_INSTRUMENT_ID for value in instrument_ids):
            errors.append("필기검사의 공식 4단계 용어를 영상면접이나 다른 검사에 적용할 수 없습니다.")
        terminology_profile_id = WRITTEN_PROFILE_ID
        if not instrument_ids and parsed.hierarchy_tiers:
            instrument_ids = [WRITTEN_INSTRUMENT_ID]
        if parsed.related_tier is not None and any(
            snapshot.id_lookup[item_id]["instrument"] != WRITTEN_INSTRUMENT_ID
            for item_id in effective_targets
        ):
            errors.append(
                "필기검사의 공식 단계 관계를 영상면접이나 다른 검사 대상에 적용할 수 없습니다."
            )

    if errors:
        return PlanValidationResult(clarification="질문의 대상이나 조건을 다시 확인해 주세요.", errors=errors)

    scope = QueryScope.PREVIOUS_RESULT if parsed.reuse_previous_result else parsed.scope
    filters = QueryFilters(
        root_only=parsed.root_only,
        leaf_only=parsed.leaf_only,
        has_parent=parsed.has_parent,
        has_children=parsed.has_children,
        analysis_included=parsed.analysis_included,
        definition_statuses=definition_statuses,
    )
    fields = list(dict.fromkeys(parsed.fields))
    if not fields and parsed.intent in {QueryIntent.ITEM_LOOKUP, QueryIntent.COMPARISON_QUERY}:
        fields = [ItemField.DEFINITION, ItemField.CHILDREN, ItemField.PATH]
    plan = RegistryQueryPlan(
        intent=parsed.intent,
        user_question=user_question,
        target_ids=effective_targets if parsed.intent in {QueryIntent.ITEM_LOOKUP, QueryIntent.RELATION_QUERY, QueryIntent.COMPARISON_QUERY} else target_ids,
        instrument_ids=instrument_ids,
        node_types=node_types,
        hierarchy_tiers=list(dict.fromkeys(parsed.hierarchy_tiers)),
        relation=parsed.relation,
        related_tier=parsed.related_tier,
        fields=fields,
        filters=filters,
        group_by=parsed.group_by,
        scope=scope,
        max_depth=parsed.max_depth,
        previous_result_ids=active_previous,
        terminology_profile_id=terminology_profile_id,
        semantic_query=parsed.semantic_query.strip() if parsed.semantic_query else None,
    )

    if parsed.related_tier is not None:
        related = _related_tier_ids(plan, snapshot)
        if not related:
            return PlanValidationResult(clarification="대상과 요청한 공식 단계 사이의 등록된 관계를 찾지 못했습니다.")
        if (
            scope != QueryScope.SUBTREE
            and any(
                len(_related_tier_ids_for_target(plan, target_id, snapshot)) > 1
                for target_id in effective_targets
            )
        ):
            return PlanValidationResult(clarification="요청한 단계에 여러 항목이 있습니다. '모두' 또는 '목록'이라고 범위를 명시해 주세요.")
    return PlanValidationResult(plan=plan)


def _descendant_ids(target_id: str, snapshot: RegistrySnapshot, max_depth: int | None = None) -> list[str]:
    found: list[str] = []
    seen = {target_id}

    def visit(item_id: str, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        item = snapshot.id_lookup.get(item_id)
        if item is None:
            return
        for child_id in _active_children(item, snapshot):
            if child_id in seen:
                continue
            seen.add(child_id)
            found.append(child_id)
            visit(child_id, depth + 1)

    visit(target_id, 1)
    return found


def _ancestor_ids(target_id: str, snapshot: RegistrySnapshot) -> list[str]:
    found_nearest: list[str] = []
    seen = {target_id}
    current = snapshot.id_lookup.get(target_id)
    while current is not None:
        parent_id = current.get("parent_id")
        if not isinstance(parent_id, str) or parent_id not in snapshot.id_lookup or parent_id in seen:
            break
        seen.add(parent_id)
        found_nearest.append(parent_id)
        current = snapshot.id_lookup[parent_id]
    return list(reversed(found_nearest))


def _depth(item_id: str, snapshot: RegistrySnapshot) -> int:
    return len(_ancestor_ids(item_id, snapshot))


def _related_tier_ids_for_target(
    plan: RegistryQueryPlan,
    target_id: str,
    snapshot: RegistrySnapshot,
) -> list[str]:
    if plan.related_tier is None:
        return []
    profile_validation = validate_hierarchy_terminology_profile(snapshot, plan.terminology_profile_id or WRITTEN_PROFILE_ID)
    if not profile_validation.valid or profile_validation.profile is None:
        return []
    level = profile_validation.profile.tier_levels[plan.related_tier]
    target = snapshot.id_lookup.get(target_id)
    if target is None or target["instrument"] != WRITTEN_INSTRUMENT_ID:
        return []
    lineage = [target_id, *reversed(_ancestor_ids(target_id, snapshot))]
    ancestor_match = next(
        (
            item_id
            for item_id in lineage
            if snapshot.id_lookup[item_id]["level"] == level
        ),
        None,
    )
    if ancestor_match is not None:
        return [ancestor_match]
    return [
        item_id
        for item_id in _descendant_ids(target_id, snapshot, plan.max_depth)
        if snapshot.id_lookup[item_id]["level"] == level
    ]


def _related_tier_ids(
    plan: RegistryQueryPlan,
    snapshot: RegistrySnapshot,
) -> list[str]:
    return _unique(
        item_id
        for target_id in plan.target_ids
        for item_id in _related_tier_ids_for_target(plan, target_id, snapshot)
    )


def _base_item_ids(plan: RegistryQueryPlan, snapshot: RegistrySnapshot) -> list[str]:
    ordered_ids = [item["id"] for item in _items(snapshot)]
    if plan.scope == QueryScope.TARGETS:
        return _unique(item_id for item_id in plan.target_ids if item_id in snapshot.id_lookup)
    if plan.scope == QueryScope.PREVIOUS_RESULT:
        if plan.intent in {
            QueryIntent.ITEM_LOOKUP,
            QueryIntent.RELATION_QUERY,
            QueryIntent.COMPARISON_QUERY,
        } and plan.target_ids:
            return _unique(
                item_id
                for item_id in plan.target_ids
                if item_id in snapshot.id_lookup
            )
        selected = set(item_id for item_id in plan.previous_result_ids if item_id in snapshot.id_lookup)
        return [item_id for item_id in ordered_ids if item_id in selected]
    if plan.scope == QueryScope.SUBTREE:
        selected: set[str] = set()
        for target_id in plan.target_ids:
            if target_id in snapshot.id_lookup:
                selected.add(target_id)
                selected.update(_descendant_ids(target_id, snapshot, plan.max_depth))
        return [item_id for item_id in ordered_ids if item_id in selected]
    if plan.intent in {QueryIntent.ITEM_LOOKUP, QueryIntent.COMPARISON_QUERY} and plan.target_ids:
        return _unique(item_id for item_id in plan.target_ids if item_id in snapshot.id_lookup)
    return ordered_ids


def _filtered_item_ids(plan: RegistryQueryPlan, snapshot: RegistrySnapshot) -> list[str]:
    ids = _base_item_ids(plan, snapshot)
    tier_levels: set[str] = set()
    if plan.hierarchy_tiers:
        validation = validate_hierarchy_terminology_profile(snapshot, plan.terminology_profile_id or WRITTEN_PROFILE_ID)
        if validation.valid and validation.profile is not None:
            tier_levels = {validation.profile.tier_levels[tier] for tier in plan.hierarchy_tiers}
        else:
            return []
    result: list[str] = []
    for item_id in ids:
        item = snapshot.id_lookup.get(item_id)
        if item is None:
            continue
        children = _active_children(item, snapshot)
        parent_exists = item.get("parent_id") in snapshot.id_lookup
        if plan.instrument_ids and item["instrument"] not in plan.instrument_ids:
            continue
        if plan.node_types and item["level"] not in plan.node_types:
            continue
        if tier_levels and (item["instrument"] != WRITTEN_INSTRUMENT_ID or item["level"] not in tier_levels):
            continue
        filters = plan.filters
        if filters.root_only and parent_exists:
            continue
        if filters.leaf_only and children:
            continue
        if filters.has_parent is not None and parent_exists is not filters.has_parent:
            continue
        if filters.has_children is not None and bool(children) is not filters.has_children:
            continue
        if filters.analysis_included is not None and item["analysis_included"] is not filters.analysis_included:
            continue
        if filters.definition_statuses and item["definition_status"] not in filters.definition_statuses:
            continue
        if (
            plan.intent == QueryIntent.HIERARCHY_QUERY
            and plan.max_depth is not None
            and plan.scope != QueryScope.SUBTREE
            and _depth(item_id, snapshot) > plan.max_depth
        ):
            continue
        result.append(item_id)
    return result


def _limit_ids(ids: list[str], limit: int = MAX_RENDERED_ITEMS) -> tuple[list[str], bool]:
    return ids[:limit], len(ids) > limit


def _estimated_item_output_chars(
    item_id: str,
    plan: RegistryQueryPlan,
    snapshot: RegistrySnapshot,
) -> int:
    """Conservatively estimate one item's rendered/context footprint."""

    item = snapshot.id_lookup[item_id]
    fields = set(plan.fields)
    name = str(item["name"])
    estimate = 120 + (len(name) * 3) + len(str(item["instrument_label"]))
    if ItemField.DEFINITION in fields:
        estimate += len(str(item.get("definition") or NOT_PROVIDED_DEFINITION)) + 40
    if ItemField.PARENT in fields:
        parent = snapshot.id_lookup.get(item.get("parent_id"))
        estimate += len(str(parent["name"])) if parent else 20
        estimate += 30
    if ItemField.CHILDREN in fields:
        estimate += 40 + sum(
            len(str(snapshot.id_lookup[child_id]["name"])) + 3
            for child_id in _active_children(item, snapshot)
        )
    if ItemField.PATH in fields:
        estimate += 40 + sum(len(str(part)) + 3 for part in item.get("path", ()))
    if ItemField.ANALYSIS_INCLUDED in fields:
        estimate += 30
    return estimate


def _limit_ids_for_output(
    ids: list[str],
    plan: RegistryQueryPlan,
    snapshot: RegistrySnapshot,
    *,
    limit: int = MAX_RENDERED_ITEMS,
    char_budget: int = MAX_GROUNDED_FACT_CHARS,
) -> tuple[list[str], bool, int]:
    """Apply both item-count and conservative rendered-character limits."""

    selected: list[str] = []
    used_chars = 0
    for item_id in ids:
        if len(selected) >= limit:
            break
        item_cost = _estimated_item_output_chars(item_id, plan, snapshot)
        if used_chars + item_cost > char_budget:
            break
        selected.append(item_id)
        used_chars += item_cost
    return selected, len(selected) < len(ids), used_chars


def _group_items(plan: RegistryQueryPlan, ids: list[str], snapshot: RegistrySnapshot) -> list[ResultGroup]:
    grouped: OrderedDict[str, tuple[str, list[str]]] = OrderedDict()
    profile = validate_hierarchy_terminology_profile(snapshot, plan.terminology_profile_id or WRITTEN_PROFILE_ID)
    reverse_tier = {
        level: tier
        for tier, level in (profile.profile.tier_levels.items() if profile.valid and profile.profile else [])
    }
    instrument_labels, _ = _instrument_catalog(snapshot)
    for item_id in ids:
        item = snapshot.id_lookup[item_id]
        if plan.group_by == GroupBy.INSTRUMENT:
            key, label = item["instrument"], instrument_labels[item["instrument"]]
        elif plan.group_by == GroupBy.NODE_TYPE:
            key, label = item["level"], node_type_term(item)
        elif plan.group_by == GroupBy.HIERARCHY_TIER:
            tier = reverse_tier.get(item["level"]) if item["instrument"] == WRITTEN_INSTRUMENT_ID else None
            key, label = (tier.value, TIER_LABELS[tier]) if tier is not None else ("not_applicable", "공식 단계 미적용")
        elif plan.group_by == GroupBy.DEPTH:
            key = str(_depth(item_id, snapshot))
            label = f"깊이 {key}"
        elif plan.group_by == GroupBy.DEFINITION_STATUS:
            key = item["definition_status"]
            label = key
        elif plan.group_by == GroupBy.ANALYSIS_INCLUDED:
            key = "included" if item["analysis_included"] else "excluded"
            label = "분석 포함" if item["analysis_included"] else "분석 제외"
        else:
            key, label = "total", "전체"
        if key not in grouped:
            grouped[key] = (label, [])
        grouped[key][1].append(item_id)
    return [ResultGroup(key=key, label=label, item_ids=members[:MAX_RENDERED_ITEMS], count=len(members)) for key, (label, members) in grouped.items()]


def _limit_groups_for_output(
    groups: list[ResultGroup],
) -> tuple[list[ResultGroup], bool]:
    """Bound aggregate group labels before they enter the writer context."""

    selected: list[ResultGroup] = []
    used_chars = 0
    for group in groups:
        if len(selected) >= MAX_RENDERED_ITEMS:
            break
        group_cost = 80 + (len(group.label) * 2)
        if used_chars + group_cost > MAX_GROUNDED_FACT_CHARS:
            break
        selected.append(group)
        used_chars += group_cost
    return selected, len(selected) < len(groups)


def _structural_relation_ids_for_target(
    plan: RegistryQueryPlan,
    target_id: str,
    snapshot: RegistrySnapshot,
) -> list[str]:
    target = snapshot.id_lookup.get(target_id)
    if target is None:
        return []
    relation = plan.relation
    if relation == RelationType.PARENT:
        parent_id = target.get("parent_id")
        return [parent_id] if parent_id in snapshot.id_lookup else []
    if relation == RelationType.ANCESTORS:
        return _ancestor_ids(target_id, snapshot)
    if relation == RelationType.CHILDREN:
        return _active_children(target, snapshot)
    if relation == RelationType.DESCENDANTS:
        return _descendant_ids(target_id, snapshot, plan.max_depth)
    if relation == RelationType.SIBLINGS:
        raw_parent_id = target.get("parent_id")
        parent_id = raw_parent_id if raw_parent_id in snapshot.id_lookup else None
        return [
            item["id"]
            for item in _items(snapshot)
            if item["instrument"] == target["instrument"]
            and (
                item.get("parent_id")
                if item.get("parent_id") in snapshot.id_lookup
                else None
            )
            == parent_id
            and item["id"] != target_id
        ]
    return []


def _relation_result(
    plan: RegistryQueryPlan,
    snapshot: RegistrySnapshot,
    *,
    related_tier: bool,
) -> RegistryQueryResult:
    groups: list[ResultGroup] = []
    all_ids: list[str] = []
    all_raw_ids: list[str] = []
    any_group_truncated = False
    remaining_slots = MAX_RENDERED_ITEMS
    remaining_chars = MAX_GROUNDED_FACT_CHARS
    for target_id in plan.target_ids:
        target = snapshot.id_lookup.get(target_id)
        if target is None:
            continue
        raw_ids = (
            _related_tier_ids_for_target(plan, target_id, snapshot)
            if related_tier
            else _structural_relation_ids_for_target(plan, target_id, snapshot)
        )
        relation_plan = plan.model_copy(
            update={"scope": QueryScope.TARGETS, "target_ids": raw_ids}
        )
        filtered_ids = _filtered_item_ids(relation_plan, snapshot)
        all_raw_ids.extend(filtered_ids)
        target_overhead = 80 + (len(str(target["name"])) * 2)
        if (
            len(groups) >= MAX_RENDERED_ITEMS
            or target_overhead > remaining_chars
        ):
            any_group_truncated = True
            continue
        remaining_chars = max(0, remaining_chars - target_overhead)
        limited_ids, group_truncated, used_chars = _limit_ids_for_output(
            filtered_ids,
            plan,
            snapshot,
            limit=remaining_slots,
            char_budget=remaining_chars,
        )
        remaining_slots -= len(limited_ids)
        remaining_chars = max(0, remaining_chars - used_chars)
        groups.append(
            ResultGroup(
                key=target_id,
                label=str(target["name"]),
                item_ids=limited_ids,
                count=len(filtered_ids),
            )
        )
        all_ids.extend(limited_ids)
        any_group_truncated = any_group_truncated or group_truncated
    unique_ids = _unique(all_ids)
    if plan.target_ids and not groups:
        return RegistryQueryResult(
            kind="clarification",
            clarification=(
                "요청한 관계 정보가 안전한 출력 길이를 초과합니다. "
                "대상 역량 수를 줄여 다시 질문해 주세요."
            ),
            truncated=True,
        )
    return RegistryQueryResult(
        kind="relations",
        item_ids=unique_ids,
        groups=groups,
        counts={"total": len(_unique(all_raw_ids))},
        truncated=any_group_truncated,
    )


def execute_registry_query(plan: RegistryQueryPlan, snapshot: RegistrySnapshot) -> RegistryQueryResult:
    """Execute every registry fact operation deterministically by stable ID."""

    if plan.intent == QueryIntent.HELP:
        return RegistryQueryResult(kind="help")
    if plan.intent == QueryIntent.OUT_OF_SCOPE:
        return RegistryQueryResult(kind="clarification", clarification="역량의 정의, 목록, 위계, 관계, 개수 또는 비교에 관해 질문해 주세요.")

    if plan.related_tier is not None:
        return _relation_result(plan, snapshot, related_tier=True)

    if plan.intent == QueryIntent.RELATION_QUERY or plan.relation is not None:
        return _relation_result(plan, snapshot, related_tier=False)

    ids = _filtered_item_ids(plan, snapshot)
    if plan.intent == QueryIntent.AGGREGATE_QUERY:
        raw_groups = _group_items(plan, ids, snapshot)
        groups, groups_truncated = _limit_groups_for_output(raw_groups)
        counts = {group.key: group.count for group in groups}
        counts["total"] = len(ids)
        limited, overall_truncated = _limit_ids(ids)
        return RegistryQueryResult(
            kind="aggregate",
            item_ids=limited,
            groups=groups,
            counts=counts,
            truncated=(
                groups_truncated
                or overall_truncated
                or any(group.count > len(group.item_ids) for group in groups)
            ),
        )

    limit = MAX_COMPARISON_ITEMS if plan.intent == QueryIntent.COMPARISON_QUERY else MAX_RENDERED_ITEMS
    limited, truncated, _ = _limit_ids_for_output(ids, plan, snapshot, limit=limit)
    if ids and not limited:
        return RegistryQueryResult(
            kind="clarification",
            clarification=(
                "요청한 정보가 안전한 출력 길이를 초과합니다. "
                "검사 도구, node type 또는 특정 역량으로 범위를 좁혀 다시 질문해 주세요."
            ),
            truncated=True,
        )
    if plan.intent == QueryIntent.COMPARISON_QUERY and len(limited) != len(ids):
        return RegistryQueryResult(
            kind="clarification",
            clarification=(
                "요청한 비교 정보가 안전한 출력 길이를 초과합니다. "
                "비교 필드를 줄이거나 더 짧은 범위로 다시 질문해 주세요."
            ),
            truncated=True,
        )
    if plan.intent == QueryIntent.COMPARISON_QUERY and len(limited) < 2:
        return RegistryQueryResult(
            kind="clarification",
            clarification=(
                "비교 조건을 적용한 뒤 남은 역량이 두 개 미만입니다. "
                "필터를 줄이거나 비교 대상을 다시 지정해 주세요."
            ),
        )
    if plan.intent == QueryIntent.HIERARCHY_QUERY:
        grouping_plan = plan.model_copy(update={"group_by": GroupBy.INSTRUMENT})
        groups = _group_items(grouping_plan, limited, snapshot)
        return RegistryQueryResult(kind="hierarchy", item_ids=limited, groups=groups, counts={"total": len(ids)}, truncated=truncated)
    if plan.intent == QueryIntent.COMPARISON_QUERY:
        return RegistryQueryResult(kind="comparison", item_ids=limited, counts={"total": len(ids)}, truncated=truncated)
    return RegistryQueryResult(kind="items", item_ids=limited, counts={"total": len(ids)}, truncated=truncated)


def _hierarchy_text(result: RegistryQueryResult, snapshot: RegistrySnapshot) -> str | None:
    if result.kind != "hierarchy" or not result.item_ids:
        return None
    selected = set(result.item_ids)
    instrument_labels, _ = _instrument_catalog(snapshot)
    lines: list[str] = []

    def render_node(item_id: str, prefix: str, is_last: bool, seen: set[str]) -> None:
        if item_id in seen:
            return
        seen.add(item_id)
        item = snapshot.id_lookup[item_id]
        lines.append(f"{prefix}{'└─' if is_last else '├─'} {item['name']}")
        child_ids = [child_id for child_id in _active_children(item, snapshot) if child_id in selected]
        child_prefix = prefix + ("   " if is_last else "│  ")
        for index, child_id in enumerate(child_ids):
            render_node(child_id, child_prefix, index == len(child_ids) - 1, seen)

    instruments = _unique(snapshot.id_lookup[item_id]["instrument"] for item_id in result.item_ids if item_id in snapshot.id_lookup)
    for instrument_id in instruments:
        if lines:
            lines.append("")
        lines.append(f"[{instrument_labels[instrument_id]}]")
        instrument_ids = [item_id for item_id in result.item_ids if snapshot.id_lookup[item_id]["instrument"] == instrument_id]
        roots = [item_id for item_id in instrument_ids if snapshot.id_lookup[item_id].get("parent_id") not in selected]
        seen: set[str] = set()
        for index, root_id in enumerate(roots):
            render_node(root_id, "", index == len(roots) - 1, seen)
        # Malformed/stale edges must not make active selected items disappear.
        for item_id in instrument_ids:
            if item_id not in seen:
                render_node(item_id, "", True, seen)
    return "\n".join(lines)


def build_grounded_answer_context(
    plan: RegistryQueryPlan,
    result: RegistryQueryResult,
    snapshot: RegistrySnapshot,
) -> GroundedAnswerContext:
    """Rehydrate only result facts from the current snapshot for one answer."""

    facts: list[GroundedFact] = []
    exact_definitions: dict[str, str] = {}
    allowed_names: list[str] = []
    required_names: list[str] = []
    allowed_numbers: list[int] = []
    fields = set(plan.fields)

    if result.clarification:
        return GroundedAnswerContext(
            intent=plan.intent,
            user_question=plan.user_question,
            clarification=result.clarification,
            truncated=result.truncated,
        )

    relation_target_ids = (
        [group.key for group in result.groups]
        if plan.intent == QueryIntent.RELATION_QUERY
        else []
    )
    for target_id in relation_target_ids:
        target = snapshot.id_lookup.get(target_id)
        if target is None:
            continue
        allowed_names.append(target["name"])
        if plan.intent == QueryIntent.RELATION_QUERY:
            required_names.append(target["name"])
        facts.append(
            GroundedFact(
                fact_type="relation_target",
                label=target["name"],
                value=(plan.relation.value if plan.relation is not None else plan.related_tier.value if plan.related_tier is not None else "target"),
                item_id=target_id,
            )
        )

    context_item_ids = [] if plan.intent == QueryIntent.AGGREGATE_QUERY else result.item_ids
    for item_id in context_item_ids:
        item = snapshot.id_lookup.get(item_id)
        if item is None:
            continue
        name = item["name"]
        allowed_names.append(name)
        required_names.append(name)
        item_value: dict[str, str] = {}
        if ItemField.INSTRUMENT_LABEL in fields:
            item_value["instrument"] = str(item["instrument_label"])
            allowed_names.append(item["instrument_label"])
        if ItemField.NODE_TYPE in fields:
            item_value["node_type"] = node_type_term(item)
        facts.append(
            GroundedFact(
                fact_type="item",
                label=name,
                value=item_value,
                item_id=item_id,
            )
        )
        if ItemField.DEFINITION in fields:
            definition = item.get("definition") or NOT_PROVIDED_DEFINITION
            exact_definitions[name] = definition
        if ItemField.PARENT in fields:
            parent = snapshot.id_lookup.get(item.get("parent_id"))
            parent_name = parent["name"] if parent else None
            if parent_name:
                allowed_names.append(parent_name)
                required_names.append(parent_name)
            facts.append(GroundedFact(fact_type="parent", label=name, value=parent_name, item_id=item_id))
        if ItemField.CHILDREN in fields:
            child_names = [snapshot.id_lookup[child_id]["name"] for child_id in _active_children(item, snapshot)]
            allowed_names.extend(child_names)
            required_names.extend(child_names)
            facts.append(GroundedFact(fact_type="children", label=name, value=child_names, item_id=item_id))
            allowed_numbers.append(len(child_names))
        if ItemField.PATH in fields:
            path = list(item.get("path", ()))
            allowed_names.extend(path)
            required_names.extend(path)
            facts.append(GroundedFact(fact_type="path", label=name, value=path, item_id=item_id))
        if ItemField.ANALYSIS_INCLUDED in fields:
            facts.append(GroundedFact(fact_type="analysis_included", label=name, value=item["analysis_included"], item_id=item_id))

    for group in result.groups:
        if result.kind == "relations":
            relation_names = [
                snapshot.id_lookup[item_id]["name"]
                for item_id in group.item_ids
                if item_id in snapshot.id_lookup
            ]
            facts.append(
                GroundedFact(
                    fact_type="relation_group",
                    label=group.label,
                    value={
                        "relation": (
                            plan.relation.value
                            if plan.relation is not None
                            else plan.related_tier.value
                            if plan.related_tier is not None
                            else "related"
                        ),
                        "item_names": relation_names,
                        "count": group.count,
                    },
                    item_id=group.key,
                )
            )
            allowed_names.append(group.label)
            allowed_names.extend(relation_names)
            required_names.append(group.label)
            required_names.extend(relation_names)
            allowed_numbers.append(group.count)
        elif plan.intent == QueryIntent.AGGREGATE_QUERY:
            facts.append(
                GroundedFact(
                    fact_type="group_count",
                    label=group.label,
                    value=group.count,
                )
            )
            allowed_names.append(group.label)
            allowed_numbers.append(group.count)
            if group.label != "전체":
                required_names.append(group.label)
    for key, value in result.counts.items():
        facts.append(GroundedFact(fact_type="count", label=key, value=value))
        allowed_numbers.append(value)
    hierarchy = _hierarchy_text(result, snapshot)
    # A registry definition may itself mention another registered item. That
    # occurrence is still grounded because the exact definition is mandatory.
    for definition in exact_definitions.values():
        allowed_names.extend(
            item["name"]
            for item in _items(snapshot)
            if item["name"] in definition
        )
    for value in [*allowed_names, *exact_definitions.values(), hierarchy or ""]:
        allowed_numbers.extend(
            int(match)
            for match in re.findall(
                r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])",
                value,
            )
        )
    if plan.intent == QueryIntent.HELP:
        allowed_numbers.append(MAX_COMPARISON_ITEMS)
    return GroundedAnswerContext(
        intent=plan.intent,
        user_question=plan.user_question,
        facts=facts,
        exact_definitions=exact_definitions,
        allowed_names=_unique(allowed_names),
        allowed_numbers=sorted(set(allowed_numbers)),
        hierarchy_text=hierarchy,
        truncated=result.truncated,
        clarification=result.clarification,
        required_names=_unique(required_names),
        registry_names=[item["name"] for item in _items(snapshot)],
    )


def _render_grounded_fallback_unbounded(context: GroundedAnswerContext) -> str:
    """Render a complete Korean answer using context facts only."""

    if context.clarification:
        return context.clarification
    if context.intent == QueryIntent.HELP:
        return (
            "역량의 정확한 이름이나 별칭을 입력하거나, 전체·검사별 목록, 위계 구조, 부모·자식·조상·후손·형제 관계, "
            "단계별 개수, 또는 최대 3개 역량의 등록 정보를 비교해 달라고 질문할 수 있습니다."
        )
    if context.intent == QueryIntent.OUT_OF_SCOPE:
        return "역량의 정의, 목록, 위계, 관계, 개수 또는 비교에 관해 질문해 주세요."
    if context.hierarchy_text:
        prefix = "현재 레지스트리에 등록된 위계 구조입니다."
        hierarchy_total = next(
            (
                fact.value
                for fact in context.facts
                if fact.fact_type == "count" and fact.label == "total"
            ),
            None,
        )
        suffix = (
            f"\n\n전체 조건 항목은 총 {hierarchy_total}개이며 결과가 길어 일부만 표시했습니다. "
            "검사 도구나 항목으로 범위를 좁혀 주세요."
            if context.truncated and hierarchy_total is not None
            else ""
        )
        return f"{prefix}\n\n{context.hierarchy_text}{suffix}"

    group_facts = [fact for fact in context.facts if fact.fact_type == "group_count"]
    item_facts = [fact for fact in context.facts if fact.fact_type == "item"]
    relation_targets = [fact for fact in context.facts if fact.fact_type == "relation_target"]
    relation_groups = [fact for fact in context.facts if fact.fact_type == "relation_group"]
    total = next((fact.value for fact in context.facts if fact.fact_type == "count" and fact.label == "total"), len(item_facts))
    facts_by_item: dict[str, list[GroundedFact]] = {}
    for fact in context.facts:
        if fact.item_id:
            facts_by_item.setdefault(fact.item_id, []).append(fact)

    def append_item_details(
        lines: list[str],
        *,
        include_name_only: bool = False,
    ) -> None:
        detailed_items = (
            item_facts
            if include_name_only
            else [
                fact
                for fact in item_facts
                if fact.label in context.exact_definitions
                or bool(fact.value)
                or any(
                    detail.fact_type
                    in {"parent", "children", "path", "analysis_included"}
                    for detail in facts_by_item.get(fact.item_id or "", [])
                )
            ]
        )
        for index, item_fact in enumerate(detailed_items, start=1):
            name = item_fact.label
            lines.append(f"\n{index}. {name}")
            if name in context.exact_definitions:
                if context.intent == QueryIntent.ITEM_LOOKUP:
                    lines.append(f"{name}의 등록된 정의는 다음과 같습니다: {context.exact_definitions[name]}")
                else:
                    lines.append(f"{name}의 등록된 정의: {context.exact_definitions[name]}")
            item_value = item_fact.value if isinstance(item_fact.value, Mapping) else {}
            if "instrument" in item_value:
                lines.append(f"{name}의 검사 도구: {item_value['instrument']}")
            if "node_type" in item_value:
                lines.append(f"{name}의 node type: {item_value['node_type']}")
            for detail in facts_by_item.get(item_fact.item_id or "", []):
                if detail.fact_type == "parent":
                    lines.append(
                        f"{name}의 등록된 부모: "
                        f"{detail.value or '등록된 부모 없음'}"
                    )
                elif detail.fact_type == "children":
                    children = ", ".join(detail.value) if detail.value else "등록된 자식 없음"
                    lines.append(f"{name}의 등록된 직접 하위 항목: {children}")
                elif detail.fact_type == "path":
                    lines.append(
                        f"{name}의 등록 경로: "
                        f"{' > '.join(detail.value) if detail.value else '등록된 경로 없음'}"
                    )
                elif detail.fact_type == "analysis_included":
                    lines.append(
                        f"{name}의 분석 포함: "
                        f"{'예' if detail.value else '아니요'}"
                    )
    if context.intent == QueryIntent.AGGREGATE_QUERY:
        if group_facts and not (len(group_facts) == 1 and group_facts[0].label == "전체"):
            lines = [f"현재 조건에 맞는 등록 항목은 총 {total}개입니다."]
            lines.extend(f"- {fact.label}: {fact.value}개" for fact in group_facts)
            if context.truncated:
                lines.append("결과가 길어 일부 그룹만 표시했습니다. 범위를 좁혀 주세요.")
            return "\n".join(lines)
        suffix = " 결과가 길어 범위를 좁혀 주세요." if context.truncated else ""
        return f"현재 조건에 맞는 등록 항목은 총 {total}개입니다.{suffix}"
    if context.intent == QueryIntent.RELATION_QUERY and relation_groups:
        relation_labels = {
            "parent": "직접 상위 항목",
            "ancestors": "모든 상위 경로 항목",
            "children": "직접 하위 항목",
            "descendants": "모든 하위 항목",
            "siblings": "형제 항목",
            "overall": "종합점수",
            "upper": "상위요인",
            "middle": "중위요인",
            "lower": "하위요인",
            "bottom": "최하위요인",
        }
        lines: list[str] = []
        for fact in relation_groups:
            value = fact.value if isinstance(fact.value, Mapping) else {}
            relation = str(value.get("relation", "related"))
            label = relation_labels.get(relation, "관련 항목")
            names = [str(name) for name in value.get("item_names", [])]
            count = int(value.get("count", len(names)))
            if names:
                if relation in {"overall", "upper", "middle", "lower", "bottom"}:
                    lines.append(
                        f"{fact.label}이(가) 속한 {label}은 총 {count}개입니다: "
                        + ", ".join(names)
                    )
                else:
                    lines.append(
                        f"{fact.label}의 {label}은 총 {count}개입니다: "
                        + ", ".join(names)
                    )
            else:
                if count:
                    lines.append(
                        f"{fact.label}의 {label}은 총 {count}개이지만 "
                        "안전한 출력 길이 제한으로 이름 목록을 생략했습니다."
                    )
                else:
                    lines.append(f"{fact.label}에는 등록된 {label}이 없습니다.")
        append_item_details(lines)
        if context.truncated:
            lines.append(
                f"전체 관계 결과는 중복을 제외해 총 {total}개이며, "
                "결과가 길어 일부만 표시했습니다. 범위를 좁혀 주세요."
            )
        return "\n".join(lines)
    if not item_facts:
        if context.intent == QueryIntent.RELATION_QUERY and relation_targets:
            target = relation_targets[0]
            relation_empty = {
                "parent": "등록된 직접 상위 항목이 없습니다.",
                "ancestors": "등록된 상위 경로 항목이 없습니다.",
                "children": "등록된 직접 하위 항목이 없습니다.",
                "descendants": "등록된 전체 하위 항목이 없습니다.",
                "siblings": "등록된 형제 항목이 없습니다.",
            }.get(str(target.value), "요청한 관계에 해당하는 등록 항목이 없습니다.")
            return f"{target.label}에는 {relation_empty}"
        if context.truncated and total:
            return (
                f"현재 조건에 맞는 등록 항목은 총 {total}개이지만 안전한 출력 길이를 초과합니다. "
                "검사 도구나 항목으로 범위를 좁혀 주세요."
            )
        return "현재 레지스트리에서 조건에 맞는 등록 항목을 찾지 못했습니다."

    lines: list[str] = []
    if context.intent == QueryIntent.COMPARISON_QUERY:
        lines.append(f"등록된 사실을 기준으로 총 {len(item_facts)}개 역량을 비교했습니다.")
    elif context.intent == QueryIntent.RELATION_QUERY and relation_targets:
        target = relation_targets[0]
        relation_label = {
            "parent": "직접 상위 항목",
            "ancestors": "모든 상위 경로 항목",
            "children": "직접 하위 항목",
            "descendants": "모든 하위 항목",
            "siblings": "형제 항목",
            "overall": "속한 종합점수",
            "upper": "속한 상위요인",
            "middle": "속한 중위요인",
            "lower": "속한 하위요인",
            "bottom": "속한 최하위요인",
        }.get(str(target.value), "관련 항목")
        lines.append(f"{target.label}의 {relation_label}은(는) 총 {total}개입니다.")
    else:
        lines.append(f"현재 조건에 맞는 등록 항목은 총 {total}개입니다.")
    append_item_details(lines, include_name_only=True)
    if context.truncated:
        lines.append("\n결과가 길어 일부만 표시했습니다. 검사 도구나 항목으로 범위를 좁혀 주세요.")
    return "\n".join(lines)


def render_grounded_fallback(context: GroundedAnswerContext) -> str:
    """Render a deterministic answer and enforce the public character cap."""

    answer = _render_grounded_fallback_unbounded(context)
    if len(answer) <= MAX_ANSWER_CHARS:
        return answer
    return (
        "요청 결과가 안전한 출력 길이를 초과합니다. "
        "검사 도구, node type 또는 특정 역량으로 범위를 좁혀 다시 질문해 주세요."
    )


def validate_grounded_answer(answer: str, context: GroundedAnswerContext) -> GroundedAnswerValidation:
    """Apply deterministic minimum grounding checks to a streamed final answer."""

    errors: list[str] = []
    stripped = answer.strip() if isinstance(answer, str) else ""
    if not stripped:
        errors.append("응답이 비어 있습니다.")
        return GroundedAnswerValidation(valid=False, errors=errors)
    if len(stripped) > MAX_ANSWER_CHARS:
        errors.append("응답이 허용된 길이를 초과했습니다.")
    for name, definition in context.exact_definitions.items():
        definition_is_linked = (
            definition in stripped
            if len(context.exact_definitions) == 1
            else _anchored_block_contains_text(
                stripped,
                name,
                definition,
                labels=context.exact_definitions,
            )
        )
        if not definition_is_linked:
            errors.append(
                f"{name}과 등록 정의의 연결이 누락되었거나 변경되었습니다."
            )
    if context.hierarchy_text and context.hierarchy_text not in stripped:
        errors.append("검증된 위계 구조가 누락되었거나 변경되었습니다.")
    for name in context.required_names:
        if name not in stripped:
            errors.append(f"필수 결과 사실 {name}이 응답에서 누락되었습니다.")
    item_labels = [
        fact.label
        for fact in context.facts
        if fact.fact_type == "item"
    ]
    if (
        len(item_labels) > 1
        and context.intent
        in {
            QueryIntent.ITEM_LOOKUP,
            QueryIntent.CATALOG_QUERY,
            QueryIntent.COMPARISON_QUERY,
        }
        and not _anchored_labels_in_order(
            stripped,
            item_labels,
            require_list_marker=True,
        )
    ):
        errors.append("조회 결과 항목의 등록 순서가 누락되었거나 바뀌었습니다.")
    allowed_names = set(context.allowed_names)
    for registered_name in context.registry_names:
        if registered_name not in allowed_names and _contains_unallowed_name(stripped, registered_name, allowed_names):
            errors.append(f"질의 결과에 없는 등록 항목 {registered_name}이 추가되었습니다.")
    number_values = [
        int(match.group())
        for match in re.finditer(
            r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])",
            stripped,
        )
        if not _is_list_ordinal(stripped, match)
    ]
    allowed_numbers = set(context.allowed_numbers)
    for number in number_values:
        if number not in allowed_numbers:
            errors.append(f"검증된 사실에 없는 수치 {number}이(가) 추가되었습니다.")
    if item_labels and context.intent in {
        QueryIntent.CATALOG_QUERY,
        QueryIntent.COMPARISON_QUERY,
    }:
        total = next(
            (
                int(fact.value)
                for fact in context.facts
                if fact.fact_type == "count" and fact.label == "total"
            ),
            None,
        )
        if total is not None and not _line_contains_total(stripped, total):
            errors.append("조회 결과의 전체 개수가 누락되었거나 바뀌었습니다.")
    if context.truncated:
        total = next(
            (
                int(fact.value)
                for fact in context.facts
                if fact.fact_type == "count" and fact.label == "total"
            ),
            None,
        )
        if total is not None and not _line_contains_total(stripped, total):
            errors.append("잘린 결과의 전체 개수가 누락되었습니다.")
        if not any(
            token in stripped
            for token in ("일부", "생략", "출력 길이", "범위를 좁")
        ):
            errors.append("결과가 일부만 표시되었다는 안내가 누락되었습니다.")
    aggregate_group_labels = [
        fact.label
        for fact in context.facts
        if fact.fact_type == "group_count" and fact.label != "전체"
    ]
    if len(aggregate_group_labels) > 1 and not _anchored_labels_in_order(
        stripped,
        aggregate_group_labels,
    ):
        errors.append("집계 그룹의 등록 순서가 누락되었거나 바뀌었습니다.")
    relation_group_labels = [
        fact.label
        for fact in context.facts
        if fact.fact_type == "relation_group"
    ]
    if len(relation_group_labels) > 1 and not _anchored_labels_in_order(
        stripped,
        relation_group_labels,
    ):
        errors.append("관계 대상의 요청 순서가 누락되었거나 바뀌었습니다.")
    for fact in context.facts:
        if fact.fact_type == "item" and isinstance(fact.value, Mapping):
            instrument_label = fact.value.get("instrument")
            if instrument_label is not None and not _anchored_line_contains_facts(
                stripped,
                fact.label,
                names=[str(instrument_label)],
                token_groups=[("검사 도구", "instrument")],
            ):
                errors.append(f"{fact.label}의 검사 도구 연결이 누락되었거나 바뀌었습니다.")
            node_type = fact.value.get("node_type")
            if node_type is not None and not _anchored_line_contains_facts(
                stripped,
                fact.label,
                names=[str(node_type)],
                token_groups=[("node type", "노드 유형", "항목 유형")],
            ):
                errors.append(f"{fact.label}의 node type 연결이 누락되었거나 바뀌었습니다.")
        elif fact.fact_type == "group_count":
            if fact.label == "전체":
                if int(fact.value) not in number_values:
                    errors.append("전체 집계 개수가 누락되었습니다.")
            elif not _anchored_line_contains_facts(
                stripped,
                fact.label,
                count=int(fact.value),
            ):
                errors.append(
                    f"집계 그룹 {fact.label}과 개수의 연결이 누락되었거나 바뀌었습니다."
                )
        elif fact.fact_type == "count" and context.intent == QueryIntent.AGGREGATE_QUERY:
            if int(fact.value) not in number_values:
                errors.append(f"집계 수치 {fact.value}이(가) 누락되었습니다.")
        elif fact.fact_type == "relation_group" and isinstance(fact.value, Mapping):
            relation_names = [str(name) for name in fact.value.get("item_names", [])]
            relation_count = int(fact.value.get("count", 0))
            relation = str(fact.value.get("relation", "related"))
            if not _anchored_line_contains_facts(
                stripped,
                fact.label,
                names=relation_names,
                count=relation_count if relation_count > 0 else None,
                token_groups=_relation_token_groups(relation),
                ordered_names=True,
            ):
                errors.append(
                    f"관계 대상 {fact.label}과 결과의 연결이 누락되었거나 바뀌었습니다."
                )
        elif fact.fact_type == "parent":
            token_groups: list[tuple[str, ...]] = [("부모", "상위")]
            if fact.value is None:
                token_groups.append(("없", "미등록"))
            if not _anchored_line_contains_facts(
                stripped,
                fact.label,
                names=[str(fact.value)] if fact.value is not None else [],
                token_groups=token_groups,
            ):
                errors.append(f"{fact.label}의 부모 사실 연결이 누락되었거나 바뀌었습니다.")
        elif fact.fact_type == "children":
            child_names = [str(name) for name in fact.value]
            token_groups = [("자식", "하위")]
            if not child_names:
                token_groups.append(("없", "미등록"))
            children_associated = _anchored_line_contains_facts(
                stripped,
                fact.label,
                names=child_names,
                token_groups=token_groups,
                ordered_names=True,
            )
            if not children_associated:
                errors.append(f"{fact.label}의 하위 사실 연결이 누락되었거나 바뀌었습니다.")
            else:
                stated_counts = _anchored_measure_counts(
                    stripped,
                    fact.label,
                    child_names,
                )
                if any(value != len(child_names) for value in stated_counts):
                    errors.append(f"{fact.label}의 직접 하위 항목 개수가 바뀌었습니다.")
        elif fact.fact_type == "path":
            if not _anchored_line_contains_facts(
                stripped,
                fact.label,
                names=[" > ".join(str(part) for part in fact.value)],
                token_groups=[("경로", "위계")],
            ):
                errors.append(f"{fact.label}의 경로 사실 연결이 누락되었거나 바뀌었습니다.")
        elif fact.fact_type == "analysis_included":
            truth_tokens = (
                ("예", "포함됨", "포함됩니다")
                if bool(fact.value)
                else ("제외", "아니", "포함되지")
            )
            if not _anchored_line_contains_facts(
                stripped,
                fact.label,
                token_groups=[("분석",), truth_tokens],
            ):
                errors.append(f"{fact.label}의 분석 사실 연결이 누락되었거나 바뀌었습니다.")
    return GroundedAnswerValidation(valid=not errors, errors=_unique(errors))


def _contains_unallowed_name(answer: str, candidate: str, allowed_names: set[str]) -> bool:
    """Ignore a shorter registered name only when it is inside an allowed one."""

    start = 0
    while True:
        index = answer.find(candidate, start)
        if index < 0:
            return False
        end = index + len(candidate)
        covered = False
        for allowed in allowed_names:
            if len(allowed) <= len(candidate):
                continue
            allowed_start = answer.find(allowed, max(0, index - len(allowed)))
            while allowed_start >= 0 and allowed_start <= index:
                if allowed_start + len(allowed) >= end:
                    covered = True
                    break
                allowed_start = answer.find(allowed, allowed_start + 1)
            if covered:
                break
        if not covered:
            return True
        start = end


def _is_list_ordinal(answer: str, match: re.Match[str]) -> bool:
    """Treat only a leading ``1.`` marker as formatting rather than a fact."""

    line_start = answer.rfind("\n", 0, match.start()) + 1
    prefix = answer[line_start : match.start()]
    return (
        not prefix.strip()
        and match.end() < len(answer)
        and answer[match.end()] == "."
    )


def _anchored_line_contains_facts(
    answer: str,
    label: str,
    *,
    names: Iterable[str] = (),
    count: int | None = None,
    token_groups: Iterable[Iterable[str]] = (),
    ordered_names: bool = False,
) -> bool:
    """Require grouped facts on a line explicitly anchored by their label."""

    anchor = re.compile(
        rf"^\s*(?:(?:[-*•])\s*)?(?:\d+[.)]\s*)?{re.escape(label)}"
    )
    for line in answer.splitlines():
        if not anchor.search(line):
            continue
        if ordered_names:
            name_cursor = 0
            ordered = True
            for name in names:
                name_index = line.find(name, name_cursor)
                if name_index < 0:
                    ordered = False
                    break
                name_cursor = name_index + len(name)
            if not ordered:
                continue
        elif any(name not in line for name in names):
            continue
        if any(
            not any(token in line for token in token_group)
            for token_group in token_groups
        ):
            continue
        if count is not None:
            values = [
                int(match.group())
                for match in re.finditer(
                    r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])",
                    line,
                )
                if not _is_list_ordinal(line, match)
            ]
            if count not in values:
                continue
        return True
    return False


def _relation_token_groups(relation: str) -> list[tuple[str, ...]]:
    structural = {
        "parent": [("부모", "직접 상위")],
        "ancestors": [("조상", "모든 상위", "상위 경로")],
        "children": [("자식", "직접 하위")],
        "descendants": [("후손", "모든 하위", "전체 하위")],
        "siblings": [("형제",)],
    }
    if relation in structural:
        return structural[relation]
    official = {
        "overall": "종합점수",
        "upper": "상위요인",
        "middle": "중위요인",
        "lower": "하위요인",
        "bottom": "최하위요인",
    }
    if relation in official:
        # Keep the tier and its target-relative "속한" marker contiguous so
        # ``하위요인`` cannot be satisfied by the longer ``최하위요인``.
        return [(f"속한 {official[relation]}",)]
    return []


def _anchored_labels_in_order(
    answer: str,
    labels: Iterable[str],
    *,
    require_list_marker: bool = False,
) -> bool:
    """Verify the order of item/group lines without using prose mentions."""

    cursor = 0
    marker = r"(?:\d+[.)]|[-*•])\s*" if require_list_marker else r"(?:(?:\d+[.)]|[-*•])\s*)?"
    for label in labels:
        pattern = re.compile(
            rf"^\s*{marker}{re.escape(label)}(?=$|[\s:：]|의)",
            re.MULTILINE,
        )
        match = pattern.search(answer, cursor)
        if match is None:
            return False
        cursor = match.end()
    return True


def _line_contains_total(answer: str, total: int) -> bool:
    for line in answer.splitlines():
        if not any(token in line for token in ("총", "전체")):
            continue
        values = [
            int(match.group())
            for match in re.finditer(
                r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])",
                line,
            )
            if not _is_list_ordinal(line, match)
        ]
        if total in values:
            return True
    return False


def _anchored_measure_counts(
    answer: str,
    label: str,
    child_names: Iterable[str],
) -> list[int]:
    """Read explicit Korean ``N개`` counts only from lines for one item."""

    anchor = re.compile(
        rf"^\s*(?:(?:[-*•])\s*)?(?:\d+[.)]\s*)?{re.escape(label)}"
    )
    values: list[int] = []
    expected_names = list(child_names)
    for line in answer.splitlines():
        if not anchor.search(line) or not any(
            token in line for token in ("자식", "하위")
        ):
            continue
        if expected_names:
            cursor = 0
            for name in expected_names:
                index = line.find(name, cursor)
                if index < 0:
                    break
                cursor = index + len(name)
            else:
                values.extend(
                    int(match.group(1))
                    for match in re.finditer(
                        r"(?<![A-Za-z0-9])(\d+)\s*개",
                        line,
                    )
                )
        elif any(token in line for token in ("없", "미등록")):
            values.extend(
                int(match.group(1))
                for match in re.finditer(
                    r"(?<![A-Za-z0-9])(\d+)\s*개",
                    line,
                )
            )
    return values


def _anchored_block_contains_text(
    answer: str,
    label: str,
    exact_text: str,
    *,
    labels: Iterable[str],
) -> bool:
    """Associate an exact, possibly multiline value with one item heading."""

    all_labels = list(labels)
    lines = answer.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    numbered_prefix = re.compile(r"^\s*\d+[.)]\s*")
    for index, line in enumerate(lines):
        if not re.match(
            rf"^\s*(?:(?:[-*•])\s*)?(?:\d+[.)]\s*)?{re.escape(label)}",
            line,
        ):
            continue
        numbered_heading = bool(numbered_prefix.match(line))
        end = len(answer)
        for next_index in range(index + 1, len(lines)):
            next_line = lines[next_index]
            if numbered_heading:
                is_boundary = bool(numbered_prefix.match(next_line)) and any(
                    re.match(
                        rf"^\s*\d+[.)]\s*{re.escape(other)}",
                        next_line,
                    )
                    for other in all_labels
                    if other != label
                )
            else:
                is_boundary = any(
                    re.match(
                        rf"^\s*(?:(?:[-*•])\s*)?{re.escape(other)}",
                        next_line,
                    )
                    for other in all_labels
                    if other != label
                )
            if is_boundary:
                end = offsets[next_index]
                break
        if exact_text in answer[offsets[index] : end]:
            return True
    return False


__all__ = [
    "GroundedAnswerContext",
    "GroundedAnswerValidation",
    "GroundedFact",
    "GroupBy",
    "HierarchyProfileValidation",
    "HierarchyTerminologyProfile",
    "HierarchyTier",
    "ItemField",
    "MAX_ANSWER_CHARS",
    "MAX_COMPARISON_ITEMS",
    "MAX_GROUNDED_FACT_CHARS",
    "MAX_RENDERED_ITEMS",
    "ParsedRegistryQuery",
    "PlanValidationResult",
    "QueryFilters",
    "QueryIntent",
    "QueryScope",
    "ResultKind",
    "RegistryQueryPlan",
    "RegistryQueryResult",
    "RelationType",
    "ResultGroup",
    "WRITTEN_PROFILE_ID",
    "build_grounded_answer_context",
    "detect_deterministic_query",
    "execute_registry_query",
    "node_type_term",
    "render_grounded_fallback",
    "validate_grounded_answer",
    "validate_hierarchy_terminology_profile",
    "validate_parsed_query",
]
