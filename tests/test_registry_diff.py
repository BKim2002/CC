from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts.registry_diff import (
    ACTIVE_ITEM_REMOVED,
    ACTIVE_ITEM_RETIRED,
    ANALYSIS_FLAG_CHANGED,
    INSTRUMENT_CHANGED,
    ITEM_MOVED,
    LOOKUP_LABEL_REMOVED,
    NODE_TYPE_CHANGED,
    RENAMED_WITHOUT_OLD_NAME_ALIAS,
    RETIRED_WITHOUT_REPLACEMENT,
    ROOT_CHANGED,
    ROOT_PATH_PREFIX_CHANGED,
    RegistryDiffError,
    diff_registries,
)


SHA256 = "a" * 64


def runtime_item(
    item_id: str,
    name: str,
    *,
    parent_id: str | None = None,
    level: str = "factor",
    definition: str | None = "synthetic definition",
    aliases: list[str] | None = None,
    analysis_included: bool = True,
    children_ids: list[str] | None = None,
    instrument: str = "written",
) -> dict[str, object]:
    return {
        "id": item_id,
        "name": name,
        "instrument": instrument,
        "level": level,
        "parent_id": parent_id,
        "children_ids": list(children_ids or []),
        "definition": definition,
        "aliases": list(aliases or []),
        "analysis_included": analysis_included,
    }


def runtime(*items: dict[str, object], source_hash: str = SHA256) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": {"sha256": source_hash},
        "items": list(items),
    }


def source_item(
    item_id: str,
    name: str,
    *,
    parent_id: str | None = None,
    node_type: str = "factor",
    definition: str | None = "synthetic definition",
    aliases: list[str] | None = None,
    analysis_included: bool = True,
    status: str = "active",
    replacement_id: str | None = None,
    order: int = 10,
    instrument: str = "written",
) -> dict[str, object]:
    return {
        "id": item_id,
        "name": name,
        "instrument": instrument,
        "node_type": node_type,
        "parent_id": parent_id,
        "order": order,
        "definition": definition,
        "aliases": list(aliases or []),
        "analysis_included": analysis_included,
        "status": status,
        "replacement_id": replacement_id,
    }


def test_identical_registries_have_empty_diff() -> None:
    root = runtime_item("root", "Root", level="root", analysis_included=False)
    child = runtime_item("child", "Child", parent_id="root")
    root["children_ids"] = ["child"]
    document = runtime(root, child)

    result = diff_registries(document, deepcopy(document), current_version_id=7)

    assert result.is_empty
    assert not result.has_changes
    assert not result.is_breaking
    assert all(count == 0 for count in result.counts.values())
    assert result.max_depth == (2, 2)
    assert result.current_version_id == 7
    assert result.candidate_source_sha256 == SHA256


@pytest.mark.parametrize(
    ("category", "mutate"),
    [
        ("renamed", lambda item: item.update(name="Renamed")),
        (
            "definition_changed",
            lambda item: item.update(definition="new private definition"),
        ),
        ("aliases_changed", lambda item: item.update(aliases=["Former name"])),
        (
            "analysis_flag_changed",
            lambda item: item.update(analysis_included=False),
        ),
        ("node_type_changed", lambda item: item.update(level="subfactor")),
    ],
)
def test_classifies_same_id_field_changes(category: str, mutate) -> None:
    before_item = runtime_item("stable-id", "Competency")
    after_item = deepcopy(before_item)
    mutate(after_item)

    result = diff_registries(runtime(before_item), runtime(after_item))

    assert len(getattr(result, category)) == 1
    assert getattr(result, category)[0].item_id == "stable-id"
    assert sum(result.counts.values()) == 1


def test_classifies_added_and_removed_by_id() -> None:
    current = runtime(runtime_item("removed-id", "Old"))
    candidate = runtime(runtime_item("added-id", "New"))

    result = diff_registries(current, candidate)

    assert [change.item_id for change in result.added] == ["added-id"]
    assert [change.item_id for change in result.removed] == ["removed-id"]
    assert result.removed[0].breaking_reasons == (ACTIVE_ITEM_REMOVED,)


def test_classifies_move_and_root_change_as_breaking() -> None:
    parent = runtime_item("parent", "Parent", level="root", analysis_included=False)
    before = runtime_item("item", "Item")
    after = runtime_item("item", "Item", parent_id="parent")

    result = diff_registries(runtime(before), runtime(parent, after))

    assert len(result.moved) == 1
    assert result.moved[0].breaking_reasons == (ITEM_MOVED, ROOT_CHANGED)
    assert result.max_depth == (1, 2)


def test_instrument_change_is_explicit_and_breaking() -> None:
    before = runtime_item("stable", "Competency", instrument="written")
    after = runtime_item("stable", "Competency", instrument="video")

    result = diff_registries(runtime(before), runtime(after))

    assert len(result.instrument_changed) == 1
    assert result.instrument_changed[0].breaking_reasons == (INSTRUMENT_CHANGED,)
    assert result.is_breaking


def test_virtual_root_path_prefix_change_is_breaking_without_parent_move() -> None:
    before = runtime_item("stable-root", "Root", level="root")
    after = deepcopy(before)
    before["path"] = ["Root"]
    after["path"] = ["Virtual instrument", "Root"]

    result = diff_registries(runtime(before), runtime(after))

    assert not result.moved
    assert len(result.root_path_prefix_changed) == 1
    assert result.root_path_prefix_changed[0].breaking_reasons == (
        ROOT_PATH_PREFIX_CHANGED,
    )
    assert result.is_breaking


def test_lookup_label_removal_and_unaliased_rename_are_breaking() -> None:
    alias_removed = diff_registries(
        runtime(runtime_item("stable", "Name", aliases=["Legacy label"])),
        runtime(runtime_item("stable", "Name")),
    )
    unaliased_rename = diff_registries(
        runtime(runtime_item("stable", "Old name")),
        runtime(runtime_item("stable", "New name")),
    )
    safe_rename = diff_registries(
        runtime(runtime_item("stable", "Old name")),
        runtime(runtime_item("stable", "New name", aliases=["Old name"])),
    )

    assert alias_removed.aliases_changed[0].breaking_reasons == (
        LOOKUP_LABEL_REMOVED,
    )
    assert unaliased_rename.renamed[0].breaking_reasons == (
        RENAMED_WITHOUT_OLD_NAME_ALIAS,
    )
    assert safe_rename.renamed[0].breaking_reasons == ()
    assert not safe_rename.is_breaking


def test_source_items_enable_status_and_replacement_classification() -> None:
    before_runtime = runtime(runtime_item("old", "Old"), runtime_item("new", "New"))
    after_runtime = runtime(runtime_item("new", "New"))
    before_source = [
        source_item("old", "Old", order=10),
        source_item("new", "New", order=20),
    ]
    after_source = [
        source_item(
            "old",
            "Old",
            status="retired",
            replacement_id="new",
            order=10,
        ),
        source_item("new", "New", order=20),
    ]

    result = diff_registries(
        before_runtime,
        after_runtime,
        current_source_items=before_source,
        candidate_source_items=after_source,
    )

    assert not result.removed
    assert len(result.status_changed) == 1
    assert result.status_changed[0].breaking_reasons == (ACTIVE_ITEM_RETIRED,)
    assert len(result.replacement_changed) == 1
    assert result.replacement_changed[0].details["after_replacement_id"] == "new"


def test_retire_without_replacement_has_both_breaking_reasons() -> None:
    before_runtime = runtime(runtime_item("old", "Old"))
    before_source = [source_item("old", "Old")]
    after_source = [source_item("old", "Old", status="retired")]

    result = diff_registries(
        before_runtime,
        runtime(),
        current_source_items=before_source,
        candidate_source_items=after_source,
    )

    assert result.status_changed[0].breaking_reasons == (
        ACTIVE_ITEM_RETIRED,
        RETIRED_WITHOUT_REPLACEMENT,
    )


def test_existing_retired_item_cannot_drop_replacement_nonbreaking() -> None:
    runtime_document = runtime(runtime_item("new", "New"))
    current_source = [
        source_item(
            "old",
            "Old",
            status="retired",
            replacement_id="new",
        ),
        source_item("new", "New"),
    ]
    candidate_source = [
        source_item("old", "Old", status="retired", replacement_id=None),
        source_item("new", "New"),
    ]

    result = diff_registries(
        runtime_document,
        deepcopy(runtime_document),
        current_source_items=current_source,
        candidate_source_items=candidate_source,
    )

    assert result.replacement_changed[0].breaking_reasons == (
        RETIRED_WITHOUT_REPLACEMENT,
    )
    assert result.is_breaking


def test_classifies_relative_sibling_and_root_order_changes() -> None:
    root = runtime_item(
        "root", "Root", level="root", analysis_included=False, children_ids=["a", "b"]
    )
    a = runtime_item("a", "A", parent_id="root")
    b = runtime_item("b", "B", parent_id="root")
    current = runtime(root, a, b)

    candidate_root = deepcopy(root)
    candidate_root["children_ids"] = ["b", "a"]
    candidate = runtime(candidate_root, a, b)

    result = diff_registries(current, candidate)

    assert [change.item_id for change in result.order_changed] == ["a", "b"]
    assert not result.is_breaking

    roots_before = runtime(runtime_item("r1", "R1"), runtime_item("r2", "R2"))
    roots_after = runtime(runtime_item("r2", "R2"), runtime_item("r1", "R1"))
    root_result = diff_registries(roots_before, roots_after)
    assert [change.item_id for change in root_result.order_changed] == ["r1", "r2"]


def test_numeric_source_order_is_compared_by_relative_position() -> None:
    current = runtime(runtime_item("a", "A"), runtime_item("b", "B"))
    same_relative_order = [
        source_item("a", "A", order=10),
        source_item("b", "B", order=20),
    ]
    reversed_order = [
        source_item("a", "A", order=20),
        source_item("b", "B", order=10),
    ]

    unchanged = diff_registries(
        current,
        deepcopy(current),
        candidate_source_items=same_relative_order,
    )
    changed = diff_registries(
        current,
        deepcopy(current),
        candidate_source_items=reversed_order,
    )

    assert not unchanged.order_changed
    assert [change.item_id for change in changed.order_changed] == ["a", "b"]


def test_addition_does_not_misreport_surviving_sibling_order() -> None:
    current = runtime(runtime_item("a", "A"), runtime_item("b", "B"))
    candidate = runtime(
        runtime_item("new", "New"),
        runtime_item("a", "A"),
        runtime_item("b", "B"),
    )

    result = diff_registries(current, candidate)

    assert len(result.added) == 1
    assert not result.order_changed


def test_breaking_summary_covers_default_policy_without_definition_or_secret() -> None:
    private_definition = "DO-NOT-PRINT definition content"
    current_items = [
        runtime_item("remove", "Remove", definition=private_definition),
        runtime_item("move", "Move"),
        runtime_item("type", "Type"),
        runtime_item("analysis", "Analysis"),
        runtime_item("retire", "Retire"),
        runtime_item("parent", "Parent", level="root", analysis_included=False),
    ]
    candidate_items = [
        runtime_item("move", "Move", parent_id="parent"),
        runtime_item("type", "Type", level="subfactor"),
        runtime_item("analysis", "Analysis", analysis_included=False),
        runtime_item("parent", "Parent", level="root", analysis_included=False),
    ]
    current_source = [
        source_item(
            item["id"],
            item["name"],
            node_type=item["level"],
            analysis_included=item["analysis_included"],
        )
        for item in current_items
    ]
    candidate_source = [
        source_item(
            item["id"],
            item["name"],
            parent_id=item["parent_id"],
            node_type=item["level"],
            analysis_included=item["analysis_included"],
        )
        for item in candidate_items
    ] + [source_item("retire", "Retire", status="retired")]
    candidate = runtime(*candidate_items, source_hash="postgresql://user:secret@host/db")

    result = diff_registries(
        runtime(*current_items),
        candidate,
        current_source_items=current_source,
        candidate_source_items=candidate_source,
        current_version_id=11,
    )
    summary_dict = result.to_summary_dict()
    rendered = json.dumps(summary_dict, ensure_ascii=False)
    summary_text = result.to_summary_text()

    assert result.is_breaking
    assert result.has_breaking_changes
    assert result.breaking_reason_counts == {
        ACTIVE_ITEM_REMOVED: 1,
        ACTIVE_ITEM_RETIRED: 1,
        ANALYSIS_FLAG_CHANGED: 1,
        ITEM_MOVED: 1,
        NODE_TYPE_CHANGED: 1,
        RETIRED_WITHOUT_REPLACEMENT: 1,
        ROOT_CHANGED: 1,
    }
    assert summary_dict["current_version"] == 11
    assert summary_dict["candidate_source_sha256"] is None
    assert private_definition not in rendered
    assert private_definition not in summary_text
    assert "postgresql://" not in rendered
    assert "secret" not in summary_text


def test_definition_change_summary_contains_no_definition_values() -> None:
    result = diff_registries(
        runtime(runtime_item("id", "Name", definition="old definition")),
        runtime(runtime_item("id", "Name", definition="new definition")),
    )

    change = result.definition_changed[0]
    assert change.details == {}
    rendered = json.dumps(result.summary_dict())
    assert "old definition" not in rendered
    assert "new definition" not in rendered


def test_alias_order_only_is_not_a_change() -> None:
    current = runtime(runtime_item("id", "Name", aliases=["B", "A"]))
    candidate = runtime(runtime_item("id", "Name", aliases=["A", "B"]))

    assert not diff_registries(current, candidate).aliases_changed


def test_duplicate_ids_are_rejected() -> None:
    duplicate = runtime(runtime_item("same", "One"), runtime_item("same", "Two"))

    with pytest.raises(RegistryDiffError, match="duplicate item id"):
        diff_registries(duplicate, runtime())


def test_source_only_retired_item_does_not_contribute_to_depth() -> None:
    active_root = runtime_item("active", "Active", level="root")
    retired_source = source_item(
        "retired",
        "Retired",
        parent_id="active",
        status="retired",
    )

    result = diff_registries(
        runtime(active_root),
        runtime(active_root),
        candidate_source_items=[source_item("active", "Active"), retired_source],
    )

    assert result.max_depth_after == 1
    assert [change.item_id for change in result.added] == ["retired"]
