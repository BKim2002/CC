from __future__ import annotations

import pytest
from pydantic import ValidationError

from competency_query import (
    GroupBy,
    HierarchyTier,
    ItemField,
    MAX_ANSWER_CHARS,
    ParsedRegistryQuery,
    QueryFilters,
    QueryIntent,
    QueryScope,
    RegistryQueryPlan,
    RelationType,
    WRITTEN_PROFILE_ID,
    build_grounded_answer_context,
    detect_deterministic_query,
    execute_registry_query,
    node_type_term,
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
        instrument_labels=["영상면접"],
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
        (ParsedRegistryQuery(intent=QueryIntent.CATALOG_QUERY, instrument_labels=["없는 검사"]), None),
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
