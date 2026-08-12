from __future__ import annotations

import competency_query
import pytest
from pydantic import ValidationError

from competency_query import (
    GroupBy,
    HierarchyTier,
    ItemField,
    MAX_ANSWER_CHARS,
    NormalizationIssueCode,
    NormalizationOption,
    NormalizationOutcome,
    ParsedRegistryQuery,
    QueryFilters,
    QueryIntent,
    QueryScope,
    RegistryQueryPlan,
    RegistryNormalizationIssue,
    RegistryNormalizationResult,
    RelationType,
    SemanticCandidateRequest,
    UnregisteredTargetResult,
    WRITTEN_PROFILE_ID,
    build_grounded_answer_context,
    detect_deterministic_query,
    execute_registry_query,
    node_type_term,
    normalize_registry_query,
    render_grounded_fallback,
    validate_grounded_answer,
    validate_hierarchy_terminology_profile,
    validate_parsed_query,
)
from competency_registry import RegistrySnapshot, build_registry_snapshot


SOURCE_HASH = "b" * 64


def _thaw(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _thaw(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(member) for member in value]
    return value


def _runtime_item(
    item_id: str,
    name: str,
    instrument: str,
    instrument_label: str,
    level: str,
    parent_id: str | None,
    path: list[str],
    *,
    aliases: list[str] | None = None,
    definition: str | None = None,
    analysis_included: bool = False,
) -> dict:
    return {
        "id": item_id,
        "name": name,
        "aliases": aliases or [],
        "instrument": instrument,
        "instrument_label": instrument_label,
        "level": level,
        "path": path,
        "parent_name": None,
        "parent_id": parent_id,
        "children": [],
        "children_ids": [],
        "definition": definition,
        "definition_status": "explicit" if definition is not None else "not_provided",
        "analysis_included": analysis_included,
        "notes": [],
        "source_section": "합성 테스트",
    }


def _snapshot(*, mutate=None) -> RegistrySnapshot:
    items: list[dict] = []

    def add(
        item_id: str,
        name: str,
        instrument: str,
        instrument_label: str,
        level: str,
        parent_id: str | None,
        path: list[str],
        *,
        aliases: list[str] | None = None,
    ) -> dict:
        item = _runtime_item(
            item_id,
            name,
            instrument,
            instrument_label,
            level,
            parent_id,
            path,
            aliases=aliases,
            definition=None if len(items) % 5 == 0 else f"{name}의 정확한 등록 정의입니다.",
            analysis_included=len(items) % 2 == 0,
        )
        items.append(item)
        if parent_id is not None:
            parent = next(existing for existing in items if existing["id"] == parent_id)
            parent["children_ids"].append(item_id)
            parent["children"].append(name)
            item["parent_name"] = parent["name"]
        return item

    overall = add(
        "w-overall",
        "역검 종합점수",
        "written_competency",
        "필기 역량검사",
        "overall",
        None,
        ["역검 종합점수"],
        aliases=["종합점수"],
    )
    middle_names = [
        ["전략성", "실행성", "성장성", "목표성"],
        ["협력성", "공감성", "소통성"],
        ["유연성", "회복성", "변화성"],
    ]
    upper_names = ["성과 예측", "관계 예측", "적응 예측"]
    serial = 0
    for upper_index, upper_name in enumerate(upper_names):
        upper_id = f"w-upper-{upper_index}"
        add(
            upper_id,
            upper_name,
            "written_competency",
            "필기 역량검사",
            "L1",
            overall["id"],
            ["역검 종합점수", upper_name],
        )
        for middle_index, middle_name in enumerate(middle_names[upper_index]):
            middle_id = f"w-middle-{upper_index}-{middle_index}"
            add(
                middle_id,
                middle_name,
                "written_competency",
                "필기 역량검사",
                "L2",
                upper_id,
                ["역검 종합점수", upper_name, middle_name],
            )
            lower_names = (
                ["기억력", "인지력", "분석력"]
                if middle_name == "전략성"
                else [f"{middle_name} 하위 {number}" for number in range(1, 4)]
            )
            for lower_index, lower_name in enumerate(lower_names):
                lower_id = f"w-lower-{serial}"
                serial += 1
                add(
                    lower_id,
                    lower_name,
                    "written_competency",
                    "필기 역량검사",
                    "L3",
                    middle_id,
                    ["역검 종합점수", upper_name, middle_name, lower_name],
                )
                if middle_name == "전략성":
                    for bottom_index in range(1, 4):
                        bottom_name = f"{lower_name} 세부 {bottom_index}"
                        add(
                            f"w-bottom-{lower_index}-{bottom_index}",
                            bottom_name,
                            "written_competency",
                            "필기 역량검사",
                            "L4",
                            lower_id,
                            ["역검 종합점수", upper_name, middle_name, lower_name, bottom_name],
                        )

    for factor_index, factor_name in enumerate(["표현능력", "답변태도", "대인매력"]):
        factor_id = f"v-factor-{factor_index}"
        add(
            factor_id,
            factor_name,
            "video_interview",
            "영상면접",
            "factor",
            None,
            ["영상면접", factor_name],
        )
        for item_index in range(1, 3):
            item_name = f"{factor_name} 세부항목 {item_index}"
            add(
                f"v-item-{factor_index}-{item_index}",
                item_name,
                "video_interview",
                "영상면접",
                "item",
                factor_id,
                ["영상면접", factor_name, item_name],
            )

    assert len(items) == 62
    if mutate is not None:
        mutate(items)
    document = {
        "schema_version": "1.0",
        "source": {"file": "synthetic.md", "sha256": SOURCE_HASH, "encoding": "utf-8"},
        "rules": {"test": "합성 테스트 규칙"},
        "validation": {"status": "passed", "counts": {"total": len(items)}},
        "items": items,
    }
    return build_registry_snapshot(
        {
            "id": 11,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(items),
        }
    )


def _plan(intent: QueryIntent, **updates) -> RegistryQueryPlan:
    return RegistryQueryPlan(intent=intent, **updates)


def _names(result, snapshot) -> list[str]:
    return [snapshot.id_lookup[item_id]["name"] for item_id in result.item_ids]


def _draft(
    intent_hint: str,
    *,
    target_mentions: list[str] | None = None,
    constraint_mentions: list[dict[str, str]] | None = None,
    semantic_description: str | None = None,
    reuse_previous_result: bool = False,
) -> dict:
    return {
        "intent_hint": intent_hint,
        "target_mentions": target_mentions or [],
        "constraint_mentions": constraint_mentions or [],
        "semantic_description": semantic_description,
        "reuse_previous_result": reuse_previous_result,
        "answer_mode": "registry_facts",
        "acknowledge_greeting": False,
        "out_of_scope_remainder": None,
    }


def test_fixture_is_graph_correct_and_profile_matches_current_contract() -> None:
    snapshot = _snapshot()
    profile = validate_hierarchy_terminology_profile(snapshot)

    assert profile.valid
    assert {tier: len(ids) for tier, ids in profile.tier_item_ids.items()} == {
        HierarchyTier.OVERALL: 1,
        HierarchyTier.UPPER: 3,
        HierarchyTier.MIDDLE: 10,
        HierarchyTier.LOWER: 30,
        HierarchyTier.BOTTOM: 9,
    }
    assert _names(
        execute_registry_query(
            _plan(
                QueryIntent.CATALOG_QUERY,
                instrument_ids=["written_competency"],
                hierarchy_tiers=[HierarchyTier.UPPER],
                terminology_profile_id=WRITTEN_PROFILE_ID,
            ),
            snapshot,
        ),
        snapshot,
    ) == ["성과 예측", "관계 예측", "적응 예측"]


def test_snapshot_id_lookup_has_shared_frozen_identity() -> None:
    snapshot = _snapshot()
    item = snapshot.id_lookup["w-overall"]
    assert item is snapshot.document["items"][0]
    assert item is snapshot.canonical_lookup["역검 종합점수"]
    assert item is snapshot.lookup["종합점수"]
    with pytest.raises(TypeError):
        item["name"] = "변조"  # type: ignore[index]


def test_catalog_filters_preserve_document_order_and_dynamic_node_types() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.CATALOG_QUERY,
        instrument_ids=["video_interview"],
        node_types=["item"],
    )
    result = execute_registry_query(plan, snapshot)

    assert _names(result, snapshot) == [
        "표현능력 세부항목 1",
        "표현능력 세부항목 2",
        "답변태도 세부항목 1",
        "답변태도 세부항목 2",
        "대인매력 세부항목 1",
        "대인매력 세부항목 2",
    ]
    assert all(node_type_term(snapshot.id_lookup[item_id]) == "세부항목" for item_id in result.item_ids)


def test_name_only_catalog_fallback_lists_every_selected_item_and_is_valid() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.CATALOG_QUERY,
        instrument_ids=["written_competency"],
        hierarchy_tiers=[HierarchyTier.UPPER],
        terminology_profile_id=WRITTEN_PROFILE_ID,
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert all(name in answer for name in ["성과 예측", "관계 예측", "적응 예측"])
    assert validate_grounded_answer(answer, context).valid
    assert not validate_grounded_answer(
        answer + " 이 목록의 실제 개수는 2개입니다.",
        context,
    ).valid
    swapped = answer.replace("1. 성과 예측", "1. __FIRST__")
    swapped = swapped.replace("2. 관계 예측", "1. 성과 예측")
    swapped = swapped.replace("1. __FIRST__", "2. 관계 예측")
    assert not validate_grounded_answer(swapped, context).valid


@pytest.mark.parametrize(
    ("filters", "predicate"),
    [
        (QueryFilters(root_only=True), lambda item, snapshot: item["parent_id"] is None),
        (QueryFilters(leaf_only=True), lambda item, snapshot: not item["children_ids"]),
        (QueryFilters(has_parent=True), lambda item, snapshot: item["parent_id"] is not None),
        (QueryFilters(has_children=True), lambda item, snapshot: bool(item["children_ids"])),
        (QueryFilters(analysis_included=True), lambda item, snapshot: item["analysis_included"]),
        (QueryFilters(definition_statuses=["not_provided"]), lambda item, snapshot: item["definition"] is None),
    ],
)
def test_catalog_boolean_and_status_filters(filters, predicate) -> None:
    snapshot = _snapshot()
    result = execute_registry_query(_plan(QueryIntent.CATALOG_QUERY, filters=filters), snapshot)
    assert result.item_ids
    assert all(predicate(snapshot.id_lookup[item_id], snapshot) for item_id in result.item_ids)


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        (RelationType.PARENT, ["전략성"]),
        (RelationType.ANCESTORS, ["역검 종합점수", "성과 예측", "전략성"]),
        (RelationType.CHILDREN, ["기억력 세부 1", "기억력 세부 2", "기억력 세부 3"]),
        (RelationType.DESCENDANTS, ["기억력 세부 1", "기억력 세부 2", "기억력 세부 3"]),
        (RelationType.SIBLINGS, ["인지력", "분석력"]),
    ],
)
def test_all_structural_relations(relation, expected) -> None:
    snapshot = _snapshot()
    target = snapshot.canonical_lookup["기억력"]
    if relation in {RelationType.CHILDREN, RelationType.DESCENDANTS}:
        target = snapshot.canonical_lookup["기억력"]
    result = execute_registry_query(
        _plan(QueryIntent.RELATION_QUERY, target_ids=[target["id"]], relation=relation),
        snapshot,
    )
    assert _names(result, snapshot) == expected


def test_descendants_max_depth_and_child_order() -> None:
    snapshot = _snapshot()
    result = execute_registry_query(
        _plan(
            QueryIntent.RELATION_QUERY,
            target_ids=[snapshot.canonical_lookup["전략성"]["id"]],
            relation=RelationType.DESCENDANTS,
            max_depth=1,
        ),
        snapshot,
    )
    assert _names(result, snapshot) == ["기억력", "인지력", "분석력"]


def test_root_siblings_use_same_instrument_and_document_order() -> None:
    snapshot = _snapshot()
    result = execute_registry_query(
        _plan(
            QueryIntent.RELATION_QUERY,
            target_ids=[snapshot.canonical_lookup["답변태도"]["id"]],
            relation=RelationType.SIBLINGS,
        ),
        snapshot,
    )
    assert _names(result, snapshot) == ["표현능력", "대인매력"]


def test_full_hierarchy_is_instrument_sectioned_and_virtual_label_not_a_node() -> None:
    snapshot = _snapshot()
    plan = _plan(QueryIntent.HIERARCHY_QUERY, instrument_ids=["video_interview"])
    result = execute_registry_query(plan, snapshot)
    context = build_grounded_answer_context(plan, result, snapshot)

    assert context.hierarchy_text is not None
    assert context.hierarchy_text.count("[영상면접]") == 1
    assert "├─ 표현능력" in context.hierarchy_text
    assert "│  ├─ 표현능력 세부항목 1" in context.hierarchy_text
    assert "└─ 대인매력" in context.hierarchy_text
    assert "─ 영상면접" not in context.hierarchy_text
    assert validate_grounded_answer(render_grounded_fallback(context), context).valid


def test_subtree_hierarchy_starts_at_target() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.HIERARCHY_QUERY,
        target_ids=[snapshot.canonical_lookup["전략성"]["id"]],
        scope=QueryScope.SUBTREE,
    )
    result = execute_registry_query(plan, snapshot)
    text = build_grounded_answer_context(plan, result, snapshot).hierarchy_text
    assert text is not None
    assert "└─ 전략성" in text
    assert "성과 예측" not in text
    assert text.index("기억력") < text.index("인지력") < text.index("분석력")


@pytest.mark.parametrize(
    ("group_by", "expected_keys"),
    [
        (GroupBy.NONE, {"total"}),
        (GroupBy.INSTRUMENT, {"written_competency", "video_interview", "total"}),
        (GroupBy.NODE_TYPE, {"overall", "L1", "L2", "L3", "L4", "factor", "item", "total"}),
        (GroupBy.DEPTH, {"0", "1", "2", "3", "4", "total"}),
        (GroupBy.DEFINITION_STATUS, {"explicit", "not_provided", "total"}),
        (GroupBy.ANALYSIS_INCLUDED, {"included", "excluded", "total"}),
        (
            GroupBy.HIERARCHY_TIER,
            {"overall", "upper", "middle", "lower", "bottom", "not_applicable", "total"},
        ),
    ],
)
def test_aggregate_groups_sum_to_dynamic_total(group_by, expected_keys) -> None:
    snapshot = _snapshot()
    result = execute_registry_query(_plan(QueryIntent.AGGREGATE_QUERY, group_by=group_by), snapshot)
    assert set(result.counts) == expected_keys
    assert result.counts["total"] == 62
    assert sum(group.count for group in result.groups) == 62


def test_official_tier_is_not_structural_leaf() -> None:
    snapshot = _snapshot()
    bottom = execute_registry_query(
        _plan(
            QueryIntent.CATALOG_QUERY,
            hierarchy_tiers=[HierarchyTier.BOTTOM],
            terminology_profile_id=WRITTEN_PROFILE_ID,
        ),
        snapshot,
    )
    leaves = execute_registry_query(
        _plan(QueryIntent.CATALOG_QUERY, filters=QueryFilters(leaf_only=True)),
        snapshot,
    )
    assert bottom.counts["total"] == 9
    assert leaves.counts["total"] > 9
    assert snapshot.canonical_lookup["실행성 하위 1"]["id"] in leaves.item_ids
    assert snapshot.canonical_lookup["실행성 하위 1"]["id"] not in bottom.item_ids


def test_video_never_receives_written_tier_filter() -> None:
    snapshot = _snapshot()
    parsed = ParsedRegistryQuery(
        intent=QueryIntent.CATALOG_QUERY,
        instrument_refs=["영상면접"],
        hierarchy_tiers=[HierarchyTier.UPPER],
    )
    validated = validate_parsed_query(parsed, snapshot)
    assert not validated.is_valid
    assert any("적용할 수 없습니다" in error for error in validated.errors)


def test_profile_mismatch_returns_clarification_instead_of_level_guess() -> None:
    def mutate(items):
        item = next(item for item in items if item["name"] == "성과 예측")
        item["level"] = "renamed-upper"

    snapshot = _snapshot(mutate=mutate)
    profile = validate_hierarchy_terminology_profile(snapshot)
    validated = validate_parsed_query(
        ParsedRegistryQuery(intent=QueryIntent.CATALOG_QUERY, hierarchy_tiers=[HierarchyTier.UPPER]),
        snapshot,
    )
    assert not profile.valid
    assert validated.plan is None
    assert validated.clarification


def test_parsed_query_canonicalizes_alias_and_deduplicates_stable_ids() -> None:
    snapshot = _snapshot()
    validated = validate_parsed_query(
        ParsedRegistryQuery(
            intent=QueryIntent.ITEM_LOOKUP,
            target_names=["종합점수", "역검 종합점수"],
        ),
        snapshot,
        user_question="종합점수",
    )
    assert validated.is_valid
    assert validated.plan is not None
    assert validated.plan.target_ids == ["w-overall"]
    assert validated.plan.user_question == "종합점수"


@pytest.mark.parametrize(
    "parsed, previous",
    [
        (ParsedRegistryQuery(intent=QueryIntent.ITEM_LOOKUP, target_names=["없는 이름"]), None),
        (ParsedRegistryQuery(intent=QueryIntent.CATALOG_QUERY, instrument_refs=["없는 검사"]), None),
        (ParsedRegistryQuery(intent=QueryIntent.CATALOG_QUERY, node_types=["없는 레벨"]), None),
        (ParsedRegistryQuery(intent=QueryIntent.RELATION_QUERY, relation=RelationType.PARENT), None),
        (ParsedRegistryQuery(intent=QueryIntent.COMPARISON_QUERY, target_names=["전략성"]), None),
        (ParsedRegistryQuery(intent=QueryIntent.HIERARCHY_QUERY, max_depth=0), None),
        (ParsedRegistryQuery(intent=QueryIntent.CATALOG_QUERY, root_only=True, has_parent=True), None),
        (ParsedRegistryQuery(intent=QueryIntent.CATALOG_QUERY, scope=QueryScope.PREVIOUS_RESULT), ["stale-id"]),
    ],
)
def test_invalid_or_conflicting_plans_are_rejected(parsed, previous) -> None:
    validated = validate_parsed_query(parsed, _snapshot(), previous)
    assert not validated.is_valid
    assert validated.errors or validated.clarification


@pytest.mark.parametrize(
    "parsed",
    [
        ParsedRegistryQuery(intent=QueryIntent.CATALOG_QUERY, relation=RelationType.PARENT),
        ParsedRegistryQuery(
            intent=QueryIntent.COMPARISON_QUERY,
            target_names=["전략성", "기억력"],
            related_tier=HierarchyTier.UPPER,
        ),
        ParsedRegistryQuery(intent=QueryIntent.CATALOG_QUERY, group_by=GroupBy.INSTRUMENT),
    ],
)
def test_intent_specific_fields_cannot_be_combined_with_other_intents(parsed) -> None:
    validated = validate_parsed_query(parsed, _snapshot())
    assert not validated.is_valid
    assert validated.errors


def test_written_tier_relation_rejects_mixed_written_and_video_targets() -> None:
    validated = validate_parsed_query(
        ParsedRegistryQuery(
            intent=QueryIntent.RELATION_QUERY,
            target_names=["전략성", "표현능력"],
            related_tier=HierarchyTier.UPPER,
        ),
        _snapshot(),
    )
    assert not validated.is_valid
    assert any("영상면접" in error for error in validated.errors)


def test_parser_model_forbids_extra_fields_and_plan_json_round_trip() -> None:
    with pytest.raises(ValidationError):
        ParsedRegistryQuery.model_validate({"intent": "help", "predicate": "run arbitrary code"})
    plan = _plan(
        QueryIntent.CATALOG_QUERY,
        hierarchy_tiers=[HierarchyTier.UPPER],
        filters=QueryFilters(analysis_included=True),
    )
    assert RegistryQueryPlan.model_validate_json(plan.model_dump_json()) == plan


@pytest.mark.parametrize(
    ("query", "intent", "detail"),
    [
        ("종합점수", QueryIntent.ITEM_LOOKUP, "w-overall"),
        ("전략성, 기억력", QueryIntent.ITEM_LOOKUP, "w-middle-0-0"),
        ("전체 역량 목록", QueryIntent.CATALOG_QUERY, None),
        ("전체 위계 구조", QueryIntent.HIERARCHY_QUERY, None),
        ("상위요인 목록", QueryIntent.CATALOG_QUERY, HierarchyTier.UPPER),
        ("최하위요인 개수", QueryIntent.AGGREGATE_QUERY, HierarchyTier.BOTTOM),
        ("영상면접 요인 목록", QueryIntent.CATALOG_QUERY, "factor"),
        ("영상면접 세부항목 목록", QueryIntent.CATALOG_QUERY, "item"),
        ("루트 항목 목록", QueryIntent.CATALOG_QUERY, "root"),
        ("말단 항목 목록", QueryIntent.CATALOG_QUERY, "leaf"),
        ("전체 역량 수", QueryIntent.AGGREGATE_QUERY, None),
        ("도움말", QueryIntent.HELP, None),
    ],
)
def test_deterministic_fast_paths(query, intent, detail) -> None:
    plan = detect_deterministic_query(query, _snapshot())
    assert plan is not None
    assert plan.intent == intent
    if isinstance(detail, HierarchyTier):
        assert plan.hierarchy_tiers == [detail]
    elif detail == "root":
        assert plan.filters.root_only
    elif detail == "leaf":
        assert plan.filters.leaf_only
    elif detail in {"factor", "item"}:
        assert plan.node_types == [detail]
    elif detail:
        assert detail in plan.target_ids


def test_natural_variant_is_left_for_structured_llm_parser() -> None:
    assert detect_deterministic_query("분석에 들어가는 항목들을 검사별로 묶어서 설명해 줘", _snapshot()) is None


def test_f01_f03_natural_tier_counts_share_one_canonical_plan_and_dynamic_total() -> None:
    snapshot = _snapshot()
    plans: list[RegistryQueryPlan] = []

    for raw_query in (
        "하위요인은 총 몇 개야?",
        "전체 하위요인의 개수가 궁금해",
        "하위 요인 수 알려줘",
    ):
        result = normalize_registry_query(
            raw_query=raw_query,
            # The raw, deterministic meaning is authoritative over a wrong
            # low-priority Gateway hint.
            draft=_draft(
                "catalog_query",
                constraint_mentions=[{"kind": "instrument", "text": "영상면접"}],
            ),
            snapshot=snapshot,
            previous_result_ids=[],
        )
        assert result.outcome == NormalizationOutcome.PLAN
        assert result.plan is not None
        plans.append(result.plan)

    canonical = [plan.model_dump(exclude={"user_question"}) for plan in plans]
    assert canonical[0] == canonical[1] == canonical[2]
    assert plans[0].intent == QueryIntent.AGGREGATE_QUERY
    assert plans[0].instrument_ids == ["written_competency"]
    assert plans[0].hierarchy_tiers == [HierarchyTier.LOWER]
    assert plans[0].scope == QueryScope.ALL
    assert execute_registry_query(plans[0], snapshot).counts["total"] == 30


def test_tier_count_space_underscore_and_hyphen_variants_keep_same_plan() -> None:
    snapshot = _snapshot()
    results = [
        normalize_registry_query(
            raw_query=raw_query,
            draft=_draft("aggregate_query"),
            snapshot=snapshot,
            previous_result_ids=[],
        )
        for raw_query in (
            "하위 요인은 총 몇 개야?",
            "하위_요인은 총 몇 개야?",
            "전체-하위-요인의 개수가 궁금해",
        )
    ]

    assert all(result.plan is not None for result in results)
    canonical = [result.plan.model_dump(exclude={"user_question"}) for result in results if result.plan]
    assert canonical[0] == canonical[1] == canonical[2]


def test_normalizer_exact_raw_target_beats_conflicting_draft_hint() -> None:
    snapshot = _snapshot()
    result = normalize_registry_query(
        raw_query="전략성의 정의를 알려줘",
        draft=_draft("semantic_search", target_mentions=["표현능력"]),
        snapshot=snapshot,
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.PLAN
    assert result.plan is not None
    assert result.plan.intent == QueryIntent.ITEM_LOOKUP
    assert result.plan.target_ids == [snapshot.canonical_lookup["전략성"]["id"]]
    assert result.plan.fields == [ItemField.DEFINITION]


def test_normalizer_resolves_alias_with_space_underscore_and_particle_variants() -> None:
    result = normalize_registry_query(
        raw_query="종합_점수는 뭐야?",
        draft=_draft("item_lookup", target_mentions=["종합 점수"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.PLAN
    assert result.plan is not None
    assert result.plan.target_ids == ["w-overall"]
    assert result.plan.hierarchy_tiers == []


def test_normalizer_unknown_definition_target_is_typed_unregistered_not_semantic_definition() -> None:
    result = normalize_registry_query(
        raw_query="정의감은 뭐야?",
        draft=_draft("item_lookup", target_mentions=["정의감"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.UNREGISTERED_TARGET
    assert result.unregistered_target is not None
    assert result.unregistered_target.code == NormalizationIssueCode.UNKNOWN_TARGET
    assert result.unregistered_target.target_mentions == ["정의감"]
    assert "정확한 역량명" in result.unregistered_target.question
    assert result.plan is None and result.semantic_request is None


def test_normalizer_unknown_target_with_behavior_description_requests_semantic_candidates() -> None:
    result = normalize_registry_query(
        raw_query="정의감처럼 갈등 때 다른 사람 말부터 듣는 행동과 가까운 역량을 찾아줘",
        draft=_draft(
            "semantic_search",
            target_mentions=["정의감"],
            semantic_description="갈등 때 다른 사람 말부터 듣는 행동",
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.SEMANTIC_CANDIDATES
    assert result.semantic_request == SemanticCandidateRequest(
        semantic_query="갈등 때 다른 사람 말부터 듣는 행동",
        target_mentions=["정의감"],
    )


def test_draft_semantic_description_without_raw_evidence_cannot_reclassify_unknown_definition() -> None:
    result = normalize_registry_query(
        raw_query="정의감은 뭐야?",
        draft=_draft(
            "semantic_search",
            target_mentions=["정의감"],
            semantic_description="갈등에서 공정하게 판단하는 행동",
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.UNREGISTERED_TARGET
    assert result.unregistered_target is not None
    assert result.unregistered_target.target_mentions == ["정의감"]
    assert result.semantic_request is None


def test_unknown_name_span_alone_is_not_behavioral_semantic_evidence() -> None:
    result = normalize_registry_query(
        raw_query="정의감은 뭐야?",
        draft=_draft(
            "semantic_search",
            target_mentions=["정의감"],
            semantic_description="정의감",
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.UNREGISTERED_TARGET
    assert result.semantic_request is None


def test_behavior_word_inside_unknown_definition_name_is_not_semantic_evidence() -> None:
    result = normalize_registry_query(
        raw_query="갈등관리의 정의를 알려줘",
        draft=_draft(
            "semantic_search",
            target_mentions=["갈등관리"],
            semantic_description="갈등관리",
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.UNREGISTERED_TARGET
    assert result.semantic_request is None


@pytest.mark.parametrize("raw_query", ["맡은 일을 끝까지 수행", "주도적으로 일함"])
def test_raw_behavior_fragments_remain_valid_semantic_candidate_requests(raw_query: str) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("semantic_search", semantic_description=raw_query),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.SEMANTIC_CANDIDATES
    assert result.semantic_request is not None
    assert result.semantic_request.semantic_query == raw_query


def test_behavior_question_ending_in_mwooya_is_not_mistaken_for_bare_definition() -> None:
    raw_query = "맡은 일을 꼼꼼하게 수행하는 역량은 뭐야?"
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft(
            "semantic_search",
            semantic_description="맡은 일을 꼼꼼하게 수행",
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.SEMANTIC_CANDIDATES
    assert result.semantic_request is not None
    assert result.semantic_request.semantic_query == "맡은 일을 꼼꼼하게 수행"


def test_nonsemantic_plan_does_not_store_untrusted_draft_semantic_description() -> None:
    result = normalize_registry_query(
        raw_query="전략성의 정의를 알려줘",
        draft=_draft(
            "item_lookup",
            target_mentions=["전략성"],
            semantic_description="날씨를 알려줘",
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.plan is not None
    assert result.plan.semantic_query is None


def test_normalized_label_collision_returns_public_ambiguous_target_choices() -> None:
    def mutate(items):
        next(item for item in items if item["name"] == "전략성")["aliases"].append("공통-표현")
        next(item for item in items if item["name"] == "실행성")["aliases"].append("공통 표현")

    result = normalize_registry_query(
        raw_query="공통 표현은 뭐야?",
        draft=_draft("item_lookup", target_mentions=["공통 표현"]),
        snapshot=_snapshot(mutate=mutate),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.AMBIGUOUS_TARGET
    assert {option.label for option in result.issue.options} == {"전략성", "실행성"}
    assert all("w-" not in option.label + option.description for option in result.issue.options)


def test_vague_structural_relation_returns_concrete_typed_choices() -> None:
    result = normalize_registry_query(
        raw_query="전략성의 상위 항목 알려줘",
        draft=_draft(
            "relation_query",
            target_mentions=["전략성"],
            constraint_mentions=[{"kind": "relation", "text": "상위 항목"}],
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.AMBIGUOUS_RELATION
    assert [option.label for option in result.issue.options] == ["직접 상위요인", "모든 상위요인"]
    assert "한 단계" in result.issue.question and "전체 경로" in result.issue.question


@pytest.mark.parametrize(
    ("raw_query", "relation"),
    [
        ("전략성의 상위요인을 알려줘", RelationType.PARENT),
        ("전략성의 하위요인을 알려줘", RelationType.CHILDREN),
        ("전략성의 모든 상위를 알려줘", RelationType.ANCESTORS),
        ("전략성의 모든 하위를 알려줘", RelationType.DESCENDANTS),
    ],
)
def test_target_relative_relation_phrases_auto_correct_only_when_unambiguous(raw_query, relation) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("relation_query", target_mentions=["전략성"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.PLAN
    assert result.plan is not None
    assert result.plan.relation == relation


def test_related_official_tier_phrase_is_normalized_separately_from_structure_relation() -> None:
    result = normalize_registry_query(
        raw_query="기억력 세부 1이 속한 중위요인을 알려줘",
        draft=_draft(
            "relation_query",
            target_mentions=["기억력 세부 1"],
            constraint_mentions=[{"kind": "hierarchy_tier", "text": "중위요인"}],
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.PLAN
    assert result.plan is not None
    assert result.plan.relation is None
    assert result.plan.related_tier == HierarchyTier.MIDDLE


def test_conflicting_parent_and_children_constraints_are_not_silently_dropped() -> None:
    result = normalize_registry_query(
        raw_query="전략성의 상위요인과 하위요인을 알려줘",
        draft=_draft("relation_query", target_mentions=["전략성"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS
    assert {option.label for option in result.issue.options} == {"직접 상위요인", "직접 하위요인"}


@pytest.mark.parametrize(
    ("raw_query", "expected_options"),
    [
        ("전략성의 상위와 하위를 알려줘", {"직접 상위요인", "직접 하위요인"}),
        ("전략성의 상위와 모든 하위를 알려줘", {"직접 상위요인", "모든 하위요인"}),
        ("전략성의 상위 항목과 하위 항목을 알려줘", {"직접 상위요인", "직접 하위요인"}),
    ],
)
def test_coordinated_upper_and_lower_relation_intent_is_never_partially_dropped(
    raw_query: str,
    expected_options: set[str],
) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("relation_query", target_mentions=["전략성"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS
    assert {option.label for option in result.issue.options} == expected_options


@pytest.mark.parametrize(
    "raw_query",
    ["전략성의 하위 구조를 보여줘", "전략성부터 시작하는 위계를 보여줘"],
)
def test_raw_subtree_wording_overrides_item_lookup_hint(raw_query: str) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("item_lookup", target_mentions=["전략성"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.plan is not None
    assert result.plan.intent == QueryIntent.HIERARCHY_QUERY
    assert result.plan.scope == QueryScope.SUBTREE
    assert result.plan.target_ids == ["w-middle-0-0"]


def test_missing_or_retired_previous_result_is_a_typed_issue() -> None:
    result = normalize_registry_query(
        raw_query="이전 결과에서 정의를 알려줘",
        draft=_draft("item_lookup", reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["retired-id"],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.MISSING_PREVIOUS_RESULT
    assert result.issue.options


def test_previous_follow_up_revalidates_stable_ids_without_treating_pronoun_as_name() -> None:
    result = normalize_registry_query(
        raw_query="그중 정의를 알려줘",
        draft=_draft("item_lookup", reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0", "retired-id"],
    )

    assert result.outcome == NormalizationOutcome.PLAN
    assert result.plan is not None
    assert result.plan.scope == QueryScope.PREVIOUS_RESULT
    assert result.plan.target_ids == ["w-middle-0-0"]
    assert result.plan.previous_result_ids == ["w-middle-0-0"]


def test_unknown_explicit_instrument_is_typed_and_lists_only_public_labels() -> None:
    result = normalize_registry_query(
        raw_query="성격검사의 역량 목록을 보여줘",
        draft=_draft(
            "catalog_query",
            constraint_mentions=[{"kind": "instrument", "text": "성격검사"}],
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.UNKNOWN_INSTRUMENT
    assert {option.label for option in result.issue.options} == {"필기 역량검사", "영상면접"}


def test_known_video_instrument_and_written_tier_are_a_constraint_conflict_not_unknown_instrument() -> None:
    result = normalize_registry_query(
        raw_query="영상면접의 상위요인 목록을 보여줘",
        draft=_draft(
            "catalog_query",
            constraint_mentions=[
                {"kind": "instrument", "text": "영상면접"},
                {"kind": "hierarchy_tier", "text": "상위요인"},
            ],
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS
    assert {option.label for option in result.issue.options} == {
        "필기 역량검사 단계",
        "선택한 검사 구조",
    }


def test_normalization_result_rejects_zero_or_multiple_payloads() -> None:
    with pytest.raises(ValidationError):
        RegistryNormalizationResult(outcome=NormalizationOutcome.PLAN)

    with pytest.raises(ValidationError):
        RegistryNormalizationResult(
            outcome=NormalizationOutcome.CLARIFICATION,
            plan=RegistryQueryPlan(intent=QueryIntent.CATALOG_QUERY),
            issue=RegistryNormalizationIssue(
                code=NormalizationIssueCode.MISSING_TARGET,
                question="대상을 알려 주세요.",
                options=[NormalizationOption(label="역량명", description="정식 이름")],
            ),
        )


def test_normalizer_accepts_gateway_like_model_without_importing_gateway_module() -> None:
    class GatewayDraftDouble:
        def model_dump(self, *, mode="python"):
            assert mode == "python"
            return _draft("item_lookup", target_mentions=["전략성"])

    result = normalize_registry_query(
        raw_query="전략성은 뭐야?",
        draft=GatewayDraftDouble(),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.PLAN
    assert result.plan is not None and result.plan.target_ids == ["w-middle-0-0"]


def test_public_normalizer_contract_is_exported_via_all() -> None:
    assert {
        "NormalizationIssueCode",
        "NormalizationOption",
        "NormalizationOutcome",
        "RegistryDraftProtocol",
        "RegistryNormalizationIssue",
        "RegistryNormalizationResult",
        "SemanticCandidateRequest",
        "UnregisteredTargetResult",
        "normalize_registry_query",
    } <= set(competency_query.__all__)


def test_raw_exact_target_cannot_be_replaced_by_draft_field_stopword() -> None:
    result = normalize_registry_query(
        raw_query="전략성의 정의를 알려줘",
        draft=_draft("item_lookup", target_mentions=["정의"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.PLAN
    assert result.plan is not None and result.plan.target_ids == ["w-middle-0-0"]


def test_korean_can_expression_does_not_turn_semantic_search_into_count_query() -> None:
    result = normalize_registry_query(
        raw_query="맡은 일을 끝까지 할 수 있는 역량을 찾아줘",
        draft=_draft(
            "semantic_search",
            semantic_description="맡은 일을 끝까지 하는 행동",
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.SEMANTIC_CANDIDATES
    assert result.semantic_request is not None


def test_raw_unknown_instrument_does_not_depend_on_gateway_constraint() -> None:
    result = normalize_registry_query(
        raw_query="성격검사의 역량 목록을 보여줘",
        draft=_draft("catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.UNKNOWN_INSTRUMENT


def test_wrong_gateway_constraint_kind_cannot_reinterpret_raw_tier_or_field() -> None:
    tier = normalize_registry_query(
        raw_query="중위요인 좀 목록으로 보여줘",
        draft=_draft(
            "catalog_query",
            constraint_mentions=[{"kind": "instrument", "text": "중위요인"}],
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )
    field = normalize_registry_query(
        raw_query="전략성의 정의를 알려줘",
        draft=_draft(
            "item_lookup",
            target_mentions=["전략성"],
            constraint_mentions=[{"kind": "node_type", "text": "정의"}],
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert tier.plan is not None and tier.plan.hierarchy_tiers == [HierarchyTier.MIDDLE]
    assert tier.plan.instrument_ids == ["written_competency"]
    assert field.plan is not None and field.plan.node_types == []
    assert field.plan.target_ids == ["w-middle-0-0"]


def test_raw_scope_conflict_is_typed_instead_of_first_match_wins() -> None:
    result = normalize_registry_query(
        raw_query="그중 전체 역량 목록을 보여줘",
        draft=_draft("catalog_query", reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0"],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None and result.issue.code == NormalizationIssueCode.AMBIGUOUS_SCOPE


def test_explicit_target_cannot_escape_previous_result_scope() -> None:
    result = normalize_registry_query(
        raw_query="이전 결과에서 실행성만 보여줘",
        draft=_draft("item_lookup", target_mentions=["실행성"], reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0"],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS
    assert "실행성" in result.issue.question


def test_target_plus_unagreed_official_tier_is_not_silently_broadened() -> None:
    result = normalize_registry_query(
        raw_query="전략성의 중위요인 목록을 보여줘",
        draft=_draft("catalog_query", target_mentions=["전략성"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None and result.issue.code == NormalizationIssueCode.AMBIGUOUS_RELATION


@pytest.mark.parametrize(
    ("raw_query", "instrument_id"),
    [
        ("필기 역량 목록", "written_competency"),
        ("영상 역량 목록", "video_interview"),
    ],
)
def test_raw_legacy_instrument_aliases_are_authoritative_without_draft_constraint(raw_query, instrument_id) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.plan is not None and result.plan.instrument_ids == [instrument_id]


def test_draft_reuse_flag_alone_cannot_silently_narrow_scope() -> None:
    result = normalize_registry_query(
        raw_query="역량 목록 좀 알려줄래",
        draft=_draft("catalog_query", reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0"],
    )

    assert result.plan is not None
    assert result.plan.scope == QueryScope.ALL
    assert result.plan.previous_result_ids == []


def test_raw_video_factor_and_analysis_filter_survive_missing_or_partial_draft() -> None:
    video = normalize_registry_query(
        raw_query="영상면접의 요인들을 보여줄래",
        draft=_draft("catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )
    analysis = normalize_registry_query(
        raw_query="분석에 포함된 항목 목록",
        draft=_draft(
            "catalog_query",
            constraint_mentions=[{"kind": "filter", "text": "분석에 포함"}],
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert video.plan is not None
    assert video.plan.instrument_ids == ["video_interview"]
    assert video.plan.node_types == ["factor"]
    assert analysis.plan is not None and analysis.plan.filters.analysis_included is True


@pytest.mark.parametrize(
    "raw_query",
    [
        "전략성 말고 실행성의 정의를 알려줘",
        "필기 말고 영상 역량 목록",
        "영상면접 제외하고 필기 역량 개수",
    ],
)
def test_explicit_target_or_instrument_negation_requires_typed_clarification(raw_query: str) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("item_lookup" if "정의" in raw_query else "catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS
    assert result.issue.options


@pytest.mark.parametrize(
    "raw_query",
    ["제외할 영상면접의 항목 목록", "전략성이 아닌 실행성의 정의를 알려줘"],
)
def test_preposed_or_copular_negation_is_not_silently_dropped(raw_query: str) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("item_lookup" if "정의" in raw_query else "catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS


def test_analysis_exclusion_with_instrument_and_plain_instrument_union_are_not_negation_conflicts() -> None:
    filtered = normalize_registry_query(
        raw_query="영상면접의 분석에서 제외된 항목 목록",
        draft=_draft("catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )
    union = normalize_registry_query(
        raw_query="필기와 영상 역량 목록",
        draft=_draft("catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert filtered.plan is not None
    assert filtered.plan.instrument_ids == ["video_interview"]
    assert filtered.plan.filters.analysis_included is False
    assert union.plan is not None
    assert union.plan.instrument_ids == ["written_competency", "video_interview"]


@pytest.mark.parametrize("raw_query", ["루트 항목 제외하고 목록", "말단 항목 말고 전체 목록"])
def test_unsupported_filter_negation_requires_typed_clarification(raw_query: str) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS


@pytest.mark.parametrize(
    "raw_query",
    [
        "분석에서 제외된 항목 목록",
        "분석에 포함되지 않은 항목 목록",
        "분석에 안 들어가는 항목 목록",
        "분석에 들어가지 않는 항목 목록",
        "분석 대상이 아닌 항목 목록",
        "분석에 사용 안 된 항목 목록",
    ],
)
def test_explicit_analysis_exclusion_is_a_false_filter(raw_query: str) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.plan is not None
    assert result.plan.filters.analysis_included is False


@pytest.mark.parametrize(
    ("raw_query", "relation"),
    [
        ("그중 부모를 알려줘", RelationType.PARENT),
        ("그중 직접 하위요인", RelationType.CHILDREN),
        ("그중 모든 하위요인", RelationType.DESCENDANTS),
    ],
)
def test_previous_result_relation_follow_up_uses_revalidated_active_ids(raw_query, relation) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("relation_query", reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0", "retired-id"],
    )

    assert result.plan is not None
    assert result.plan.scope == QueryScope.PREVIOUS_RESULT
    assert result.plan.target_ids == ["w-middle-0-0"]
    assert result.plan.relation == relation


def test_previous_relation_raw_scope_beats_draft_field_as_target() -> None:
    result = normalize_registry_query(
        raw_query="그중 부모를 알려줘",
        draft=_draft("relation_query", target_mentions=["부모"], reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0"],
    )

    assert result.plan is not None
    assert result.plan.target_ids == ["w-middle-0-0"]
    assert result.plan.relation == RelationType.PARENT


@pytest.mark.parametrize("pronoun", ["그 역량", "해당 역량"])
def test_previous_relation_pronoun_beats_wrong_draft_kind(pronoun: str) -> None:
    result = normalize_registry_query(
        raw_query=f"{pronoun}의 하위요인도 알려줘",
        draft=_draft(
            "item_lookup",
            constraint_mentions=[{"kind": "field", "text": "하위요인"}],
            reuse_previous_result=True,
        ),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0", "retired-id"],
    )

    assert result.plan is not None
    assert result.plan.scope == QueryScope.PREVIOUS_RESULT
    assert result.plan.target_ids == ["w-middle-0-0"]
    assert result.plan.relation == RelationType.CHILDREN


@pytest.mark.parametrize(
    "pronoun",
    ["그 역량", "해당 역량", "이 역량", "방금 본 역량", "앞서 나온 역량", "그것"],
)
def test_singular_previous_pronouns_reuse_one_active_stable_id(pronoun: str) -> None:
    result = normalize_registry_query(
        raw_query=f"{pronoun}의 정의를 알려줘",
        draft=_draft("item_lookup", reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0", "retired-id"],
    )

    assert result.plan is not None
    assert result.plan.scope == QueryScope.PREVIOUS_RESULT
    assert result.plan.target_ids == ["w-middle-0-0"]


def test_singular_previous_pronoun_with_multiple_active_items_is_ambiguous_target() -> None:
    result = normalize_registry_query(
        raw_query="해당 역량의 정의를 알려줘",
        draft=_draft("item_lookup", reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0", "w-middle-0-1"],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.AMBIGUOUS_TARGET
    assert {option.label for option in result.issue.options} == {"전략성", "실행성"}
    assert "w-middle" not in result.issue.model_dump_json()


@pytest.mark.parametrize("pronoun", ["그 역량들", "해당 역량들", "그것들"])
def test_plural_previous_pronouns_keep_all_active_items(pronoun: str) -> None:
    result = normalize_registry_query(
        raw_query=f"{pronoun}의 정의를 알려줘",
        draft=_draft("item_lookup", reuse_previous_result=True),
        snapshot=_snapshot(),
        previous_result_ids=["w-middle-0-0", "w-middle-0-1"],
    )

    assert result.plan is not None
    assert result.plan.target_ids == ["w-middle-0-0", "w-middle-0-1"]
    assert result.plan.scope == QueryScope.PREVIOUS_RESULT


def test_tier_fast_path_rejects_additional_tiers_negation_and_instrument_constraints() -> None:
    snapshot = _snapshot()
    two_tiers = normalize_registry_query(
        raw_query="상위요인과 하위요인은 총 몇 개야?",
        draft=_draft("aggregate_query"),
        snapshot=snapshot,
        previous_result_ids=[],
    )
    three_tiers = normalize_registry_query(
        raw_query="상위요인, 중위요인, 하위요인의 총 개수",
        draft=_draft("aggregate_query"),
        snapshot=snapshot,
        previous_result_ids=[],
    )
    negated = normalize_registry_query(
        raw_query="하위요인 말고 최하위요인은 몇 개야?",
        draft=_draft("aggregate_query"),
        snapshot=snapshot,
        previous_result_ids=[],
    )
    wrong_instrument = normalize_registry_query(
        raw_query="상위요인은 영상면접에서 총 몇 개야?",
        draft=_draft("aggregate_query"),
        snapshot=snapshot,
        previous_result_ids=[],
    )

    assert two_tiers.plan is not None
    assert two_tiers.plan.hierarchy_tiers == [HierarchyTier.UPPER, HierarchyTier.LOWER]
    assert three_tiers.plan is not None
    assert three_tiers.plan.hierarchy_tiers == [
        HierarchyTier.UPPER,
        HierarchyTier.MIDDLE,
        HierarchyTier.LOWER,
    ]
    assert negated.issue is not None
    assert negated.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS
    assert wrong_instrument.issue is not None
    assert wrong_instrument.issue.code == NormalizationIssueCode.CONFLICTING_CONSTRAINTS


def test_coordinated_known_targets_and_mixed_unknown_target_preserve_all_explicit_names() -> None:
    known = normalize_registry_query(
        raw_query="전략성과 실행성의 정의를 알려줘",
        draft=_draft("item_lookup", target_mentions=["전략성", "실행성"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )
    mixed = normalize_registry_query(
        raw_query="전략성, 정의감의 정의",
        draft=_draft("item_lookup", target_mentions=["전략성", "정의감"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert known.plan is not None
    assert known.plan.target_ids == ["w-middle-0-0", "w-middle-0-1"]
    assert mixed.outcome == NormalizationOutcome.UNREGISTERED_TARGET
    assert mixed.unregistered_target is not None
    assert mixed.unregistered_target.target_mentions == ["정의감"]


@pytest.mark.parametrize(
    "raw_query",
    [
        "현재 필기검사의 역량 목록",
        "이 필기검사에서 역량 목록",
        "우리 영상면접의 요인 목록",
        "각 검사에서 역량 수",
    ],
)
def test_instrument_determiners_and_generic_each_are_not_unknown(raw_query) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("aggregate_query" if " 수" in raw_query else "catalog_query"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.issue is None or result.issue.code != NormalizationIssueCode.UNKNOWN_INSTRUMENT
    if raw_query == "각 검사에서 역량 수":
        assert result.plan is not None and result.plan.group_by == GroupBy.INSTRUMENT


def test_dynamic_active_instrument_label_is_resolved_without_gateway_constraint() -> None:
    def mutate(items):
        for item in items:
            if item["instrument"] == "video_interview":
                item["instrument"] = "practical_assessment"
                item["instrument_label"] = "실기 평가"

    result = normalize_registry_query(
        raw_query="실기 평가 항목 목록",
        draft=_draft("catalog_query"),
        snapshot=_snapshot(mutate=mutate),
        previous_result_ids=[],
    )

    assert result.plan is not None
    assert result.plan.instrument_ids == ["practical_assessment"]


def test_invalid_hierarchy_profile_preserves_concrete_public_validation_reason() -> None:
    def mutate(items):
        next(item for item in items if item["name"] == "성과 예측")["level"] = "unexpected_level"

    result = normalize_registry_query(
        raw_query="상위요인 목록을 보여줘",
        draft=_draft("catalog_query"),
        snapshot=_snapshot(mutate=mutate),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.UNSUPPORTED_REGISTRY_COMBINATION
    assert "상위요인 구성" in result.issue.question
    assert "일치하지 않습니다" in result.issue.question
    assert WRITTEN_PROFILE_ID not in result.issue.question


def test_normalized_dynamic_instrument_collision_requires_scope_clarification() -> None:
    def mutate(items):
        for item in items:
            if item["instrument"] != "video_interview":
                continue
            if item["path"][1] == "표현능력":
                item["instrument"] = "practical-a"
                item["instrument_label"] = "실기-평가"
            else:
                item["instrument"] = "practical-b"
                item["instrument_label"] = "실기 평가"

    result = normalize_registry_query(
        raw_query="실기 평가 항목 목록",
        draft=_draft("catalog_query"),
        snapshot=_snapshot(mutate=mutate),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.AMBIGUOUS_SCOPE
    assert {option.label for option in result.issue.options} == {"실기-평가", "실기 평가"}
    assert "practical-a" not in result.issue.model_dump_json()
    assert "practical-b" not in result.issue.model_dump_json()


def test_newline_separated_exact_names_remain_a_fast_path() -> None:
    plan = detect_deterministic_query("전략성\n기억력", _snapshot())
    assert plan is not None
    assert plan.target_ids == ["w-middle-0-0", "w-lower-0"]


def test_related_tier_finds_closest_ancestor_and_ambiguous_descendants_clarify() -> None:
    snapshot = _snapshot()
    nearest = validate_parsed_query(
        ParsedRegistryQuery(
            intent=QueryIntent.RELATION_QUERY,
            target_names=["기억력 세부 1"],
            related_tier=HierarchyTier.MIDDLE,
        ),
        snapshot,
    )
    assert nearest.is_valid and nearest.plan is not None
    assert _names(execute_registry_query(nearest.plan, snapshot), snapshot) == ["전략성"]

    ambiguous = validate_parsed_query(
        ParsedRegistryQuery(
            intent=QueryIntent.RELATION_QUERY,
            target_names=["전략성"],
            related_tier=HierarchyTier.LOWER,
        ),
        snapshot,
    )
    assert not ambiguous.is_valid
    assert "여러 항목" in (ambiguous.clarification or "")

    multi_target_ambiguous = validate_parsed_query(
        ParsedRegistryQuery(
            intent=QueryIntent.RELATION_QUERY,
            target_names=["전략성", "기억력", "인지력"],
            related_tier=HierarchyTier.LOWER,
        ),
        snapshot,
    )
    assert not multi_target_ambiguous.is_valid
    assert "여러 항목" in (multi_target_ambiguous.clarification or "")


def test_related_lower_tier_cannot_be_mislabeled_as_bottom_tier() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.RELATION_QUERY,
        target_ids=[snapshot.canonical_lookup["기억력 세부 1"]["id"]],
        related_tier=HierarchyTier.LOWER,
        terminology_profile_id=WRITTEN_PROFILE_ID,
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert validate_grounded_answer(answer, context).valid
    mislabeled = answer.replace("속한 하위요인", "속한 최하위요인")
    assert not validate_grounded_answer(mislabeled, context).valid


def test_previous_result_filters_stale_ids_and_preserves_registry_order() -> None:
    snapshot = _snapshot()
    first = snapshot.canonical_lookup["전략성"]["id"]
    second = snapshot.canonical_lookup["성과 예측"]["id"]
    validated = validate_parsed_query(
        ParsedRegistryQuery(
            intent=QueryIntent.CATALOG_QUERY,
            reuse_previous_result=True,
        ),
        snapshot,
        [first, "stale-id", second],
    )
    assert validated.is_valid and validated.plan is not None
    result = execute_registry_query(validated.plan, snapshot)
    assert _names(result, snapshot) == ["성과 예측", "전략성"]


def test_previous_result_comparison_preserves_conversation_order() -> None:
    snapshot = _snapshot()
    previous = [
        snapshot.canonical_lookup["관계 예측"]["id"],
        "stale-id",
        snapshot.canonical_lookup["성과 예측"]["id"],
    ]
    validated = validate_parsed_query(
        ParsedRegistryQuery(
            intent=QueryIntent.COMPARISON_QUERY,
            reuse_previous_result=True,
        ),
        snapshot,
        previous,
    )
    assert validated.is_valid and validated.plan is not None
    assert _names(execute_registry_query(validated.plan, snapshot), snapshot) == [
        "관계 예측",
        "성과 예측",
    ]


def test_multi_target_relation_keeps_each_targets_results_separate() -> None:
    snapshot = _snapshot()
    target_ids = [
        snapshot.canonical_lookup["전략성"]["id"],
        snapshot.canonical_lookup["기억력"]["id"],
    ]
    plan = _plan(
        QueryIntent.RELATION_QUERY,
        target_ids=target_ids,
        relation=RelationType.CHILDREN,
    )
    result = execute_registry_query(plan, snapshot)
    answer = render_grounded_fallback(build_grounded_answer_context(plan, result, snapshot))

    assert [group.label for group in result.groups] == ["전략성", "기억력"]
    assert _names(result.groups[0], snapshot) == ["기억력", "인지력", "분석력"]
    assert _names(result.groups[1], snapshot) == [
        "기억력 세부 1",
        "기억력 세부 2",
        "기억력 세부 3",
    ]
    assert "전략성의 직접 하위 항목" in answer
    assert "기억력의 직접 하위 항목" in answer
    context = build_grounded_answer_context(plan, result, snapshot)
    assert validate_grounded_answer(answer, context).valid
    assert not validate_grounded_answer(
        answer.replace("직접 하위 항목", "형제 항목"),
        context,
    ).valid


def test_relation_requested_definitions_are_rendered_and_grounded() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.RELATION_QUERY,
        target_ids=[snapshot.canonical_lookup["기억력"]["id"]],
        relation=RelationType.CHILDREN,
        fields=[ItemField.DEFINITION],
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert all(definition in answer for definition in context.exact_definitions.values())
    assert validate_grounded_answer(answer, context).valid


def test_comparison_preserves_target_order_and_exact_definition() -> None:
    snapshot = _snapshot()
    target_ids = [snapshot.canonical_lookup[name]["id"] for name in ["관계 예측", "성과 예측"]]
    plan = _plan(
        QueryIntent.COMPARISON_QUERY,
        target_ids=target_ids,
        fields=[ItemField.DEFINITION, ItemField.PATH, ItemField.ANALYSIS_INCLUDED],
    )
    result = execute_registry_query(plan, snapshot)
    context = build_grounded_answer_context(plan, result, snapshot)
    fallback = render_grounded_fallback(context)

    assert _names(result, snapshot) == ["관계 예측", "성과 예측"]
    assert context.exact_definitions["관계 예측"] in fallback
    assert validate_grounded_answer(fallback, context).valid


def test_grounding_validator_rejects_definitions_swapped_between_items() -> None:
    snapshot = _snapshot()
    names = ["성과 예측", "관계 예측"]
    plan = _plan(
        QueryIntent.COMPARISON_QUERY,
        target_ids=[snapshot.canonical_lookup[name]["id"] for name in names],
        fields=[ItemField.DEFINITION],
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    first_definition = context.exact_definitions[names[0]]
    second_definition = context.exact_definitions[names[1]]
    swapped = (
        f"등록된 사실을 기준으로 총 2개 역량을 비교했습니다.\n"
        f"1. {names[0]}\n{names[0]}의 등록된 정의: {second_definition}\n"
        f"2. {names[1]}\n{names[1]}의 등록된 정의: {first_definition}"
    )
    assert not validate_grounded_answer(swapped, context).valid


def test_multiline_definition_is_preserved_and_validated_in_anchored_block() -> None:
    multiline = "정확한 첫 줄\n정확한 둘째 줄"

    def mutate(items) -> None:
        target = next(item for item in items if item["name"] == "성과 예측")
        target["definition"] = multiline
        target["definition_status"] = "explicit"

    snapshot = _snapshot(mutate=mutate)
    plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[snapshot.canonical_lookup["성과 예측"]["id"]],
        fields=[ItemField.DEFINITION],
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert multiline in answer
    assert validate_grounded_answer(answer, context).valid


def test_comparison_that_filters_to_one_item_returns_clarification() -> None:
    snapshot = _snapshot()
    validated = validate_parsed_query(
        ParsedRegistryQuery(
            intent=QueryIntent.COMPARISON_QUERY,
            target_names=["전략성", "기억력"],
            node_types=["L2"],
        ),
        snapshot,
    )
    assert validated.is_valid and validated.plan is not None
    result = execute_registry_query(validated.plan, snapshot)
    assert result.kind.value == "clarification"
    assert "두 개 미만" in (result.clarification or "")


def test_comparison_renders_and_validates_instrument_and_node_type_per_item() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.COMPARISON_QUERY,
        target_ids=[
            snapshot.canonical_lookup["성과 예측"]["id"],
            snapshot.canonical_lookup["표현능력"]["id"],
        ],
        fields=[ItemField.INSTRUMENT_LABEL, ItemField.NODE_TYPE],
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert "성과 예측의 검사 도구: 필기 역량검사" in answer
    assert "성과 예측의 node type: L1" in answer
    assert "표현능력의 검사 도구: 영상면접" in answer
    assert "표현능력의 node type: 요인" in answer
    assert validate_grounded_answer(answer, context).valid

    swapped = answer.replace(
        "성과 예측의 검사 도구: 필기 역량검사",
        "성과 예측의 검사 도구: 영상면접",
    ).replace(
        "표현능력의 검사 도구: 영상면접",
        "표현능력의 검사 도구: 필기 역량검사",
    )
    assert not validate_grounded_answer(swapped, context).valid


def test_missing_definition_uses_existing_not_provided_text_exactly() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=["w-overall"],
        fields=[ItemField.DEFINITION],
    )
    result = execute_registry_query(plan, snapshot)
    context = build_grounded_answer_context(plan, result, snapshot)
    assert context.exact_definitions["역검 종합점수"] == "독립적인 정의가 제공되어 있지 않음"
    assert validate_grounded_answer(render_grounded_fallback(context), context).valid


def test_grounding_validator_rejects_changed_definition_extra_item_number_and_missing_tree() -> None:
    snapshot = _snapshot()
    lookup_plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[snapshot.canonical_lookup["성과 예측"]["id"]],
        fields=[ItemField.DEFINITION],
    )
    lookup_context = build_grounded_answer_context(
        lookup_plan,
        execute_registry_query(lookup_plan, snapshot),
        snapshot,
    )
    assert not validate_grounded_answer("성과 예측의 다른 정의입니다. 적응 예측은 999개입니다.", lookup_context).valid

    hierarchy_plan = _plan(QueryIntent.HIERARCHY_QUERY, instrument_ids=["video_interview"])
    hierarchy_context = build_grounded_answer_context(
        hierarchy_plan,
        execute_registry_query(hierarchy_plan, snapshot),
        snapshot,
    )
    assert not validate_grounded_answer("영상면접의 구조입니다.", hierarchy_context).valid


def test_grounding_validator_rejects_false_small_number_and_incomplete_aggregate() -> None:
    snapshot = _snapshot()
    lookup_plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[snapshot.canonical_lookup["성과 예측"]["id"]],
        fields=[ItemField.DEFINITION],
    )
    lookup_context = build_grounded_answer_context(
        lookup_plan,
        execute_registry_query(lookup_plan, snapshot),
        snapshot,
    )
    valid_lookup = render_grounded_fallback(lookup_context)
    assert not validate_grounded_answer(valid_lookup + " 관련 항목은 3개입니다.", lookup_context).valid

    multi_plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[
            snapshot.canonical_lookup[name]["id"]
            for name in ["기억력", "인지력", "분석력"]
        ],
        fields=[ItemField.DEFINITION],
    )
    multi_context = build_grounded_answer_context(
        multi_plan,
        execute_registry_query(multi_plan, snapshot),
        snapshot,
    )
    multi_answer = render_grounded_fallback(multi_context)
    assert validate_grounded_answer(multi_answer, multi_context).valid
    assert not validate_grounded_answer(
        multi_answer + " 이와 별개로 관련 항목은 2개입니다.",
        multi_context,
    ).valid

    aggregate_plan = _plan(QueryIntent.AGGREGATE_QUERY, group_by=GroupBy.INSTRUMENT)
    aggregate_context = build_grounded_answer_context(
        aggregate_plan,
        execute_registry_query(aggregate_plan, snapshot),
        snapshot,
    )
    assert not validate_grounded_answer("현재 등록 항목은 총 62개입니다.", aggregate_context).valid
    swapped_counts = "현재 등록 항목은 총 62개입니다.\n- 필기 역량검사: 9개\n- 영상면접: 53개"
    assert not validate_grounded_answer(swapped_counts, aggregate_context).valid


def test_grounding_validator_rejects_swapped_multi_target_relation_groups() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.RELATION_QUERY,
        target_ids=[
            snapshot.canonical_lookup["성과 예측"]["id"],
            snapshot.canonical_lookup["관계 예측"]["id"],
        ],
        relation=RelationType.CHILDREN,
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    valid_answer = render_grounded_fallback(context)
    assert validate_grounded_answer(valid_answer, context).valid

    swapped = (
        "성과 예측의 직접 하위 항목은 총 3개입니다: 협력성, 공감성, 소통성\n"
        "관계 예측의 직접 하위 항목은 총 4개입니다: 전략성, 실행성, 성장성, 목표성"
    )
    assert not validate_grounded_answer(swapped, context).valid


def test_grounding_validator_rejects_missing_requested_relationship_facts() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[snapshot.canonical_lookup["기억력"]["id"]],
        fields=[
            ItemField.DEFINITION,
            ItemField.PARENT,
            ItemField.CHILDREN,
            ItemField.PATH,
            ItemField.ANALYSIS_INCLUDED,
        ],
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert validate_grounded_answer(answer, context).valid
    assert not validate_grounded_answer(
        answer.replace("기억력의 등록된 부모: 전략성", "기억력의 연결: 전략성"),
        context,
    ).valid
    assert not validate_grounded_answer(
        answer.replace("기억력의 등록된 직접 하위 항목:", "기억력의 연결 항목:"),
        context,
    ).valid
    assert not validate_grounded_answer(
        answer.replace("기억력의 등록 경로:", "기억력의 연결:"),
        context,
    ).valid
    assert not validate_grounded_answer(
        answer.replace("기억력의 분석 포함:", "기억력의 판정:"),
        context,
    ).valid

    true_plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[snapshot.canonical_lookup["전략성"]["id"]],
        fields=[ItemField.ANALYSIS_INCLUDED],
    )
    true_context = build_grounded_answer_context(
        true_plan,
        execute_registry_query(true_plan, snapshot),
        snapshot,
    )
    true_answer = render_grounded_fallback(true_context)
    assert validate_grounded_answer(true_answer, true_context).valid
    assert not validate_grounded_answer(
        true_answer.replace("전략성의 분석 포함: 예", "전략성의 분석 포함: 아니요"),
        true_context,
    ).valid


def test_grounding_validator_rejects_children_swapped_between_item_targets() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[
            snapshot.canonical_lookup["성과 예측"]["id"],
            snapshot.canonical_lookup["관계 예측"]["id"],
        ],
        fields=[ItemField.CHILDREN, ItemField.PATH, ItemField.ANALYSIS_INCLUDED],
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert validate_grounded_answer(answer, context).valid

    first = "성과 예측의 등록된 직접 하위 항목: 전략성, 실행성, 성장성, 목표성"
    second = "관계 예측의 등록된 직접 하위 항목: 협력성, 공감성, 소통성"
    swapped = answer.replace(first, "__FIRST_CHILDREN__")
    swapped = swapped.replace(
        second,
        "성과 예측의 등록된 직접 하위 항목: 협력성, 공감성, 소통성",
    )
    swapped = swapped.replace(
        "__FIRST_CHILDREN__",
        "관계 예측의 등록된 직접 하위 항목: 전략성, 실행성, 성장성, 목표성",
    )
    assert not validate_grounded_answer(swapped, context).valid

    swapped_counts = answer.replace(
        "성과 예측의 등록된 직접 하위 항목: ",
        "성과 예측의 등록된 직접 하위 항목: 3개, ",
    ).replace(
        "관계 예측의 등록된 직접 하위 항목: ",
        "관계 예측의 등록된 직접 하위 항목: 4개, ",
    )
    assert not validate_grounded_answer(swapped_counts, context).valid


def test_grounding_validator_rejects_reordered_children_path_and_relation_results() -> None:
    snapshot = _snapshot()
    item_plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[snapshot.canonical_lookup["성과 예측"]["id"]],
        fields=[ItemField.CHILDREN, ItemField.PATH],
    )
    item_context = build_grounded_answer_context(
        item_plan,
        execute_registry_query(item_plan, snapshot),
        snapshot,
    )
    item_answer = render_grounded_fallback(item_context)
    assert validate_grounded_answer(item_answer, item_context).valid
    reversed_children = item_answer.replace(
        "전략성, 실행성, 성장성, 목표성",
        "목표성, 성장성, 실행성, 전략성",
    )
    assert not validate_grounded_answer(reversed_children, item_context).valid
    reversed_path = item_answer.replace(
        "역검 종합점수 > 성과 예측",
        "성과 예측 > 역검 종합점수",
    )
    assert not validate_grounded_answer(reversed_path, item_context).valid

    relation_plan = _plan(
        QueryIntent.RELATION_QUERY,
        target_ids=[snapshot.canonical_lookup["기억력"]["id"]],
        relation=RelationType.ANCESTORS,
    )
    relation_context = build_grounded_answer_context(
        relation_plan,
        execute_registry_query(relation_plan, snapshot),
        snapshot,
    )
    relation_answer = render_grounded_fallback(relation_context)
    assert validate_grounded_answer(relation_answer, relation_context).valid
    reversed_relation = relation_answer.replace(
        "역검 종합점수, 성과 예측, 전략성",
        "전략성, 성과 예측, 역검 종합점수",
    )
    assert not validate_grounded_answer(reversed_relation, relation_context).valid


def test_definition_number_is_not_mistaken_for_requested_children_count() -> None:
    def mutate(items) -> None:
        target = next(item for item in items if item["name"] == "성과 예측")
        target["definition"] = "성과를 예측하는 7개 단서를 설명합니다."
        target["definition_status"] = "explicit"

    snapshot = _snapshot(mutate=mutate)
    plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[snapshot.canonical_lookup["성과 예측"]["id"]],
        fields=[ItemField.DEFINITION, ItemField.CHILDREN],
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert validate_grounded_answer(answer, context).valid


def test_empty_result_and_help_have_safe_fallbacks() -> None:
    snapshot = _snapshot()
    empty_plan = _plan(QueryIntent.CATALOG_QUERY, filters=QueryFilters(root_only=True, has_children=False))
    empty_context = build_grounded_answer_context(empty_plan, execute_registry_query(empty_plan, snapshot), snapshot)
    assert "찾지 못했습니다" in render_grounded_fallback(empty_context)

    help_plan = _plan(QueryIntent.HELP)
    help_context = build_grounded_answer_context(help_plan, execute_registry_query(help_plan, snapshot), snapshot)
    assert "부모" in render_grounded_fallback(help_context)
    assert validate_grounded_answer(render_grounded_fallback(help_context), help_context).valid


def test_ungrouped_aggregate_fallback_self_validates_without_literal_group_label() -> None:
    snapshot = _snapshot()
    plan = _plan(QueryIntent.AGGREGATE_QUERY, group_by=GroupBy.NONE)
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert "총 62개" in answer
    assert validate_grounded_answer(answer, context).valid


def test_relation_fallback_names_relation_target_and_empty_registered_parent() -> None:
    snapshot = _snapshot()
    root_id = snapshot.canonical_lookup["표현능력"]["id"]
    plan = _plan(
        QueryIntent.RELATION_QUERY,
        target_ids=[root_id],
        relation=RelationType.PARENT,
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert answer == "표현능력에는 등록된 직접 상위 항목이 없습니다."
    assert validate_grounded_answer(answer, context).valid


def test_item_lookup_fallback_uses_natural_labels_and_exact_definition() -> None:
    snapshot = _snapshot()
    plan = _plan(
        QueryIntent.ITEM_LOOKUP,
        target_ids=[snapshot.canonical_lookup["성과 예측"]["id"]],
        fields=[ItemField.DEFINITION, ItemField.CHILDREN, ItemField.PATH],
    )
    context = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot)
    answer = render_grounded_fallback(context)
    assert "성과 예측의 등록된 정의는 다음과 같습니다" in answer
    assert "등록된 직접 하위 항목" in answer
    assert "등록 경로" in answer
    assert validate_grounded_answer(answer, context).valid


def test_hierarchy_max_depth_applies_to_full_scope() -> None:
    snapshot = _snapshot()
    plan = _plan(QueryIntent.HIERARCHY_QUERY, instrument_ids=["written_competency"], max_depth=1)
    result = execute_registry_query(plan, snapshot)
    assert result.counts["total"] == 4
    assert _names(result, snapshot) == ["역검 종합점수", "성과 예측", "관계 예측", "적응 예측"]


def test_dynamic_size_and_output_truncation_without_fixed_validation_count() -> None:
    snapshot = _snapshot()
    document = _thaw(snapshot.document)
    for index in range(50):
        name = f"확장 루트 {index}"
        document["items"].append(
            _runtime_item(
                f"future-{index}",
                name,
                "future",
                "미래 검사",
                "future-root",
                None,
                [name],
                definition=f"{name} 정의",
            )
        )
    document["validation"] = {"status": "passed", "counts": {"total": len(document["items"])}}
    row = {
        "id": 12,
        "source_filename": "synthetic.md",
        "source_sha256": SOURCE_HASH,
        "schema_version": "1.0",
        "registry_json": document,
        "item_count": len(document["items"]),
    }
    expanded = build_registry_snapshot(row)
    result = execute_registry_query(_plan(QueryIntent.CATALOG_QUERY), expanded)
    assert result.counts["total"] == 112
    assert len(result.item_ids) == 100
    assert result.truncated
    context = build_grounded_answer_context(_plan(QueryIntent.CATALOG_QUERY), result, expanded)
    answer = render_grounded_fallback(context)
    assert validate_grounded_answer(answer, context).valid
    without_total = "\n".join(answer.splitlines()[1:])
    assert not validate_grounded_answer(without_total, context).valid
    without_truncation_notice = answer.replace(
        "\n결과가 길어 일부만 표시했습니다. 검사 도구나 항목으로 범위를 좁혀 주세요.",
        "",
    )
    assert not validate_grounded_answer(without_truncation_notice, context).valid


def test_detailed_output_is_limited_before_writer_context_and_stays_under_char_cap() -> None:
    snapshot = _snapshot()
    document = _thaw(snapshot.document)
    for item in document["items"]:
        item["definition"] = f"{item['name']} " + ("가" * 600)
        item["definition_status"] = "explicit"
    expanded = build_registry_snapshot(
        {
            "id": 14,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(document["items"]),
        }
    )
    plan = _plan(QueryIntent.CATALOG_QUERY, fields=[ItemField.DEFINITION])
    result = execute_registry_query(plan, expanded)
    context = build_grounded_answer_context(plan, result, expanded)
    answer = render_grounded_fallback(context)

    assert 0 < len(result.item_ids) < result.counts["total"]
    assert result.truncated
    assert len(answer) <= MAX_ANSWER_CHARS
    assert validate_grounded_answer(answer, context).valid


def test_aggregate_group_item_ids_are_limited_and_marked_truncated() -> None:
    snapshot = _snapshot()
    document = _thaw(snapshot.document)
    for index in range(50):
        name = f"확장 루트 {index}"
        document["items"].append(
            _runtime_item(
                f"future-{index}",
                name,
                "written_competency",
                "필기 역량검사",
                "future-root",
                None,
                [name],
                definition=f"{name} 정의",
            )
        )
    document["validation"] = {"status": "passed", "counts": {"total": len(document["items"])}}
    expanded = build_registry_snapshot(
        {
            "id": 13,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(document["items"]),
        }
    )
    result = execute_registry_query(_plan(QueryIntent.AGGREGATE_QUERY), expanded)
    assert result.counts["total"] == 112
    assert len(result.groups[0].item_ids) == 100
    assert result.truncated


def test_arbitrary_six_level_hierarchy_renders_without_level_hardcoding() -> None:
    snapshot = _snapshot()
    # Existing written hierarchy reaches five registered levels. The virtual
    # video prefix makes six visible path components without becoming a node.
    deepest = snapshot.canonical_lookup["기억력 세부 1"]
    assert len(deepest["path"]) == 5
    plan = _plan(
        QueryIntent.HIERARCHY_QUERY,
        target_ids=[snapshot.canonical_lookup["역검 종합점수"]["id"]],
        scope=QueryScope.SUBTREE,
    )
    text = build_grounded_answer_context(plan, execute_registry_query(plan, snapshot), snapshot).hierarchy_text
    assert text and "기억력 세부 1" in text


def test_six_registered_node_levels_render_without_level_hardcoding() -> None:
    snapshot = _snapshot()
    document = _thaw(snapshot.document)
    parent_id = None
    path: list[str] = []
    for depth in range(6):
        name = f"6단계 노드 {depth + 1}"
        item_id = f"six-level-{depth + 1}"
        path.append(name)
        item = _runtime_item(
            item_id,
            name,
            "six_level_instrument",
            "6단계 검사",
            f"depth-{depth}",
            parent_id,
            path.copy(),
            definition=f"{name} 정의",
        )
        document["items"].append(item)
        if parent_id is not None:
            parent = next(existing for existing in document["items"] if existing["id"] == parent_id)
            parent["children_ids"].append(item_id)
            parent["children"].append(name)
            item["parent_name"] = parent["name"]
        parent_id = item_id
    document["validation"] = {"status": "passed", "counts": {"total": len(document["items"])}}
    expanded = build_registry_snapshot(
        {
            "id": 15,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(document["items"]),
        }
    )
    plan = _plan(QueryIntent.HIERARCHY_QUERY, instrument_ids=["six_level_instrument"])
    context = build_grounded_answer_context(plan, execute_registry_query(plan, expanded), expanded)
    assert context.hierarchy_text is not None
    assert all(f"6단계 노드 {depth}" in context.hierarchy_text for depth in range(1, 7))


# ---------------------------------------------------------------------------
# Near-match suggestions for mentions that resolve to nothing
# ---------------------------------------------------------------------------

# One syllable swapped for a visually and phonetically close one, the shape of
# an ordinary Korean typo.
_TYPO_SYLLABLES = {
    "성": "셩",
    "력": "렵",
    "도": "됴",
    "구": "규",
    "측": "책",
    "수": "슈",
    "능": "늠",
    "매": "맹",
    "통": "툥",
    "화": "홰",
}


def _one_syllable_typo(name: str) -> str | None:
    for index, char in enumerate(name):
        if char in _TYPO_SYLLABLES:
            return name[:index] + _TYPO_SYLLABLES[char] + name[index + 1 :]
    return None


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("공감셩", "공감성"),
        ("분석렵", "분석력"),
        ("전략썽", "전략성"),
        ("전략", "전략성"),
    ],
)
def test_near_match_recovers_a_mistyped_registered_name(
    mention: str,
    expected: str,
) -> None:
    assert expected in competency_query._near_registered_names(mention, _snapshot())


@pytest.mark.parametrize(
    "mention",
    ["그릿", "역량 목록", "전체 역량 목록", "위계 구조", "오늘 날씨"],
)
def test_near_match_stays_silent_for_non_names(mention: str) -> None:
    assert competency_query._near_registered_names(mention, _snapshot()) == []


@pytest.mark.parametrize("mention", ["", "성", "력"])
def test_near_match_ignores_mentions_below_the_length_floor(mention: str) -> None:
    # A single syllable is close to far too much of the registry to be a signal.
    assert competency_query._near_registered_names(mention, _snapshot()) == []


def test_near_match_caps_the_candidate_count() -> None:
    # "공감성 하위" is close to three siblings plus their parent.
    candidates = competency_query._near_registered_names("공감성 하위", _snapshot())

    assert 0 < len(candidates) <= competency_query.MAX_NEAR_MATCH_CANDIDATES


def test_near_match_reports_canonical_names_and_deduplicates_by_item() -> None:
    snapshot = _snapshot()
    # "종합점수" is an alias of "역검 종합점수"; both labels point at one item.
    candidates = competency_query._near_registered_names("종합점슈", snapshot)

    assert "역검 종합점수" in candidates
    assert "종합점수" not in candidates
    assert len(candidates) == len(set(candidates))


def test_every_registered_name_survives_a_single_syllable_typo() -> None:
    """The registry has near-identical sibling names by design.

    They collide with each other far above the cutoff, which is harmless: a
    label that resolves exactly never reaches near-matching.  What has to hold
    is that a typo still recovers the name it was typed from.
    """

    snapshot = _snapshot()
    missed: list[tuple[str, str, list[str]]] = []
    checked = 0
    for name in snapshot.canonical_names:
        typo = _one_syllable_typo(name)
        if typo is None or typo in snapshot.lookup:
            continue
        checked += 1
        candidates = competency_query._near_registered_names(typo, snapshot)
        if name not in candidates:
            missed.append((typo, name, candidates))

    assert checked >= 50, "fixture should cover a production-sized registry"
    assert missed == []


def test_near_match_result_is_a_clarification_with_registered_options() -> None:
    result = competency_query._result_with_near_match(
        "공감셩", ["공감성"], ("near_match_suggestion",)
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.NEAR_MATCH_TARGET
    assert [option.label for option in result.issue.options] == ["공감성"]
    assert "공감셩" in result.issue.question
    assert result.applied_rule_ids == ["near_match_suggestion"]


def test_near_match_result_requires_at_least_one_candidate() -> None:
    with pytest.raises(ValueError):
        competency_query._result_with_near_match("공감셩", [], ())


def test_mistyped_name_in_the_raw_query_gets_a_suggestion() -> None:
    result = normalize_registry_query(
        raw_query="공감셩이 뭐야?",
        draft=_draft("item_lookup", target_mentions=["공감셩"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.NEAR_MATCH_TARGET
    assert "공감성" in [option.label for option in result.issue.options]
    assert "near_match_suggestion" in result.applied_rule_ids


@pytest.mark.parametrize(
    "raw_query",
    ["그릿이 뭐야?", "리더십이 뭐야?", "창의력의 정의를 알려줘"],
)
def test_unrelated_unknown_name_still_reports_unregistered(raw_query: str) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft("item_lookup"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.UNREGISTERED_TARGET


def test_unregistered_name_close_to_a_registered_one_is_offered_as_a_choice() -> None:
    """A near miss need not be a typo to be worth suggesting.

    "회복탄력성" is a real concept that is not in this registry, but it sits
    close to the registered "회복성".  Offering that as a choice the user can
    decline beats a dead end, so this is intended rather than a false positive.
    """

    result = normalize_registry_query(
        raw_query="회복탄력성의 정의를 알려줘",
        draft=_draft("item_lookup"),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.NEAR_MATCH_TARGET
    assert [option.label for option in result.issue.options] == ["회복성"]


def test_described_behaviour_outranks_a_spelling_suggestion() -> None:
    result = normalize_registry_query(
        raw_query="공감셩이 뭐야? 팀원의 감정을 살피는 행동을 말하는 거야",
        draft=_draft(
            "semantic_search",
            target_mentions=["공감셩"],
            semantic_description="팀원의 감정을 살피는 행동",
        ),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.SEMANTIC_CANDIDATES


# The entry model copies the user's noun phrase into target_mentions.  Each of
# these was rejected as an unregistered competency before the split between
# declared and incidental mentions.
_SCOPE_NOUN_COPIES = [
    ("역량 목록 좀 알려줘", "catalog_query", "역량 목록", QueryIntent.CATALOG_QUERY),
    ("역량 종류 좀 알려줘", "catalog_query", "역량 종류", QueryIntent.CATALOG_QUERY),
    ("전체 역량 목록 좀 알려줘", "catalog_query", "전체 역량 목록", QueryIntent.CATALOG_QUERY),
    ("등록된 역량 다 보여줘", "catalog_query", "등록된 역량", QueryIntent.CATALOG_QUERY),
    ("어떤 역량들이 있어?", "catalog_query", "역량들", QueryIntent.CATALOG_QUERY),
    ("역량 리스트 뽑아줘", "catalog_query", "역량 리스트", QueryIntent.CATALOG_QUERY),
    ("위계 구조 알려줘", "hierarchy_query", "위계 구조", QueryIntent.HIERARCHY_QUERY),
    ("역량 트리 보여줘", "hierarchy_query", "역량 트리", QueryIntent.HIERARCHY_QUERY),
    ("검사별 역량 수 알려줘", "aggregate_query", "검사별 역량 수", QueryIntent.AGGREGATE_QUERY),
    ("분석에 포함되는 역량만", "catalog_query", "분석에 포함되는 역량", QueryIntent.CATALOG_QUERY),
]


@pytest.mark.parametrize(
    ("raw_query", "intent_hint", "copied", "expected_intent"),
    _SCOPE_NOUN_COPIES,
    ids=[case[0] for case in _SCOPE_NOUN_COPIES],
)
def test_copied_scope_noun_does_not_become_an_unregistered_name(
    raw_query: str,
    intent_hint: str,
    copied: str,
    expected_intent: QueryIntent,
) -> None:
    result = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft(intent_hint, target_mentions=[copied]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.PLAN
    assert result.plan is not None
    assert result.plan.intent == expected_intent
    assert "draft_mention_dropped" in result.applied_rule_ids


@pytest.mark.parametrize(
    ("raw_query", "intent_hint", "copied", "expected_intent"),
    _SCOPE_NOUN_COPIES,
    ids=[case[0] for case in _SCOPE_NOUN_COPIES],
)
def test_scope_questions_are_unaffected_by_whether_the_draft_copies_a_noun(
    raw_query: str,
    intent_hint: str,
    copied: str,
    expected_intent: QueryIntent,
) -> None:
    snapshot = _snapshot()
    with_copy = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft(intent_hint, target_mentions=[copied]),
        snapshot=snapshot,
        previous_result_ids=[],
    )
    without_copy = normalize_registry_query(
        raw_query=raw_query,
        draft=_draft(intent_hint),
        snapshot=snapshot,
        previous_result_ids=[],
    )

    assert with_copy.plan == without_copy.plan


def test_incidental_draft_mention_close_to_a_name_is_offered_as_a_correction() -> None:
    # "공감셩 알려줘" carries no particle marking it as a target, so it lands in
    # the incidental bucket - but it is still worth a suggestion.
    result = normalize_registry_query(
        raw_query="공감셩 알려줘",
        draft=_draft("item_lookup", target_mentions=["공감셩"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.CLARIFICATION
    assert result.issue is not None
    assert result.issue.code == NormalizationIssueCode.NEAR_MATCH_TARGET
    assert "공감성" in [option.label for option in result.issue.options]


def test_explicitly_named_unknown_still_reports_unregistered_beside_a_known_name() -> None:
    # A name the user marked as a target keeps its authority even though it
    # arrived through the draft; only incidental mentions are droppable.
    result = normalize_registry_query(
        raw_query="전략성, 정의감의 정의",
        draft=_draft("item_lookup", target_mentions=["전략성", "정의감"]),
        snapshot=_snapshot(),
        previous_result_ids=[],
    )

    assert result.outcome == NormalizationOutcome.UNREGISTERED_TARGET
    assert result.unregistered_target is not None
    assert result.unregistered_target.target_mentions == ["정의감"]
    assert "draft_mention_dropped" not in result.applied_rule_ids
