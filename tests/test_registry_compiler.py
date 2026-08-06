from __future__ import annotations

from copy import deepcopy

import pytest

import scripts.registry_compiler as compiler
from competency_registry import RegistryValidationError
from competency_registry import validate_registry_document
from scripts.registry_compiler import (
    RegistryCompileError,
    compile_registry,
    validate_graph,
)
from scripts.registry_source_v2 import (
    SourceRegistry,
    parse_source_v2_text,
    render_source_v2,
)


SOURCE_HASH = "a" * 64


def _item(
    item_id: str,
    name: str,
    *,
    instrument: str = "written",
    node_type: str = "factor",
    parent_id: str | None = None,
    order: int = 10,
    definition: str | None = None,
    aliases: list[str] | None = None,
    analysis_included: bool = False,
    status: str = "active",
    replacement_id: str | None = None,
    notes: list[str] | None = None,
) -> dict:
    return {
        "id": item_id,
        "name": name,
        "instrument": instrument,
        "node_type": node_type,
        "parent_id": parent_id,
        "order": order,
        "definition": definition,
        "aliases": aliases or [],
        "analysis_included": analysis_included,
        "status": status,
        "replacement_id": replacement_id,
        "notes": notes or [],
    }


def _base_items() -> list[dict]:
    return [
        _item(
            "w-root",
            "종합",
            node_type="overall",
            order=10,
            definition="전체 요약",
        ),
        _item(
            "w-category",
            "범주",
            node_type="category",
            parent_id="w-root",
            order=10,
        ),
        _item(
            "w-late",
            "나중 요인",
            parent_id="w-category",
            order=20,
            definition="나중 정의",
        ),
        _item(
            "w-early",
            "먼저 요인",
            parent_id="w-category",
            order=10,
            definition="먼저 정의",
            aliases=["첫 요인"],
            analysis_included=True,
        ),
        # The video instrument intentionally has multiple roots.
        _item(
            "v-second-root",
            "영상 둘째 루트",
            instrument="video",
            order=20,
            definition="둘째 영상 정의",
        ),
        _item(
            "v-first-root",
            "영상 첫째 루트",
            instrument="video",
            order=10,
            definition="첫째 영상 정의",
        ),
    ]


def _source(
    items: list[dict] | None = None,
    *,
    instruments: list[dict] | None = None,
) -> SourceRegistry:
    return SourceRegistry.model_validate(
        {
            "source_schema_version": "2.0",
            "registry_schema_version": "1.0",
            "instruments": instruments
            or [
                {"id": "written", "label": "필기 검사"},
                {"id": "video", "label": "영상 면접"},
            ],
            "items": _base_items() if items is None else items,
        },
        strict=True,
    )


def _compile(source: SourceRegistry) -> dict:
    return compile_registry(
        source,
        source_filename="registry-v2.md",
        source_sha256=SOURCE_HASH,
    )


def _by_id(document: dict) -> dict[str, dict]:
    return {item["id"]: item for item in document["items"]}


def test_compile_builds_runtime_schema_and_multiple_roots() -> None:
    document = _compile(_source())
    items = _by_id(document)

    assert document["schema_version"] == "1.0"
    assert document["source"] == {
        "file": "registry-v2.md",
        "sha256": SOURCE_HASH,
        "encoding": "utf-8",
        "format": "markdown-v2",
        "source_schema_version": "2.0",
    }
    assert document["validation"]["counts"]["total"] == 6
    assert items["v-first-root"]["parent_id"] is None
    assert items["v-second-root"]["parent_id"] is None
    assert validate_registry_document(document) == document


def test_declared_instrument_without_active_items_is_allowed() -> None:
    source = _source(
        instruments=[
            {"id": "written", "label": "필기 검사"},
            {"id": "video", "label": "영상 면접"},
            {"id": "future", "label": "향후 검사"},
        ]
    )

    validate_graph(source)
    document = _compile(source)

    assert all(item["instrument"] != "future" for item in document["items"])


def test_compile_runs_final_runtime_schema_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_runtime_document(document: object) -> dict:
        raise RegistryValidationError("synthetic runtime rejection")

    monkeypatch.setattr(compiler, "validate_registry_document", reject_runtime_document)

    with pytest.raises(
        RegistryCompileError,
        match="failed runtime schema validation",
    ):
        _compile(_source())


def test_compile_derives_paths_parents_children_and_order() -> None:
    document = _compile(_source())
    items = _by_id(document)

    assert [item["id"] for item in document["items"]] == [
        "w-root",
        "w-category",
        "w-early",
        "w-late",
        "v-first-root",
        "v-second-root",
    ]
    assert items["w-category"]["children_ids"] == ["w-early", "w-late"]
    assert items["w-category"]["children"] == ["먼저 요인", "나중 요인"]
    assert items["w-early"]["path"] == ["종합", "범주", "먼저 요인"]
    assert items["w-early"]["parent_name"] == "범주"
    assert items["w-early"]["order"] == 10


def test_instrument_virtual_root_prefix_preserves_user_visible_path() -> None:
    source = _source(
        instruments=[
            {
                "id": "written",
                "label": "필기 검사",
                "root_path_prefix": ["가상 검사", "필기 영역"],
            },
            {"id": "video", "label": "영상 면접"},
        ]
    )

    items = _by_id(_compile(source))

    assert items["w-root"]["path"] == ["가상 검사", "필기 영역", "종합"]
    assert items["w-root"]["parent_id"] is None
    assert items["w-root"]["parent_name"] == "필기 영역"
    assert items["w-early"]["path"] == [
        "가상 검사",
        "필기 영역",
        "종합",
        "범주",
        "먼저 요인",
    ]


def test_compile_definition_status_rules() -> None:
    items = _by_id(_compile(_source()))

    assert items["w-root"]["definition_status"] == "summary"
    assert items["w-category"]["definition_status"] == "not_provided"
    assert items["w-early"]["definition_status"] == "explicit"


def test_rename_keeps_stable_id_and_old_name_can_be_alias() -> None:
    items = _base_items()
    target = next(item for item in items if item["id"] == "w-early")
    target["name"] = "선행 요인"
    target["aliases"] = ["먼저 요인"]

    compiled = _by_id(_compile(_source(items)))

    assert compiled["w-early"]["id"] == "w-early"
    assert compiled["w-early"]["name"] == "선행 요인"
    assert compiled["w-early"]["aliases"] == ["먼저 요인"]


def test_add_and_retire_change_dynamic_counts_and_children() -> None:
    items = _base_items()
    retired = next(item for item in items if item["id"] == "w-late")
    retired["status"] = "retired"
    retired["replacement_id"] = "w-early"
    items.append(
        _item(
            "w-new",
            "새 요인",
            parent_id="w-category",
            order=30,
            definition="새 정의",
        )
    )

    document = _compile(_source(items))
    compiled = _by_id(document)

    assert "w-late" not in compiled
    assert compiled["w-category"]["children_ids"] == ["w-early", "w-new"]
    assert document["validation"]["counts"]["total"] == 6
    assert document["validation"]["counts"]["retired"] == 1


def test_parent_move_recomputes_old_and_new_branches() -> None:
    items = _base_items()
    items.append(
        _item(
            "w-other-category",
            "다른 범주",
            node_type="category",
            parent_id="w-root",
            order=20,
        )
    )
    moved = next(item for item in items if item["id"] == "w-late")
    moved["parent_id"] = "w-other-category"

    compiled = _by_id(_compile(_source(items)))

    assert compiled["w-category"]["children_ids"] == ["w-early"]
    assert compiled["w-other-category"]["children_ids"] == ["w-late"]
    assert compiled["w-late"]["path"] == ["종합", "다른 범주", "나중 요인"]


@pytest.mark.parametrize("depth", [2, 5, 6])
def test_variable_hierarchy_depth(depth: int) -> None:
    items = [
        _item(
            f"node-{index}",
            f"노드 {index}",
            node_type="overall" if index == 0 else f"depth_{index + 1}",
            parent_id=None if index == 0 else f"node-{index - 1}",
            order=10,
            definition=f"노드 {index} 정의",
        )
        for index in range(depth)
    ]
    source = _source(
        items,
        instruments=[{"id": "written", "label": "필기 검사"}],
    )

    deepest = _by_id(_compile(source))[f"node-{depth - 1}"]

    assert len(deepest["path"]) == depth


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda items: items.append(deepcopy(items[0])),
            "Duplicate item id",
        ),
        (
            lambda items: items.__setitem__(
                1,
                {**items[1], "name": items[0]["name"]},
            ),
            "Duplicate active item name",
        ),
        (
            lambda items: items.__setitem__(
                3,
                {**items[3], "aliases": [items[0]["name"]]},
            ),
            "collision",
        ),
        (
            lambda items: items.__setitem__(
                3,
                {**items[3], "aliases": ["별칭", "별칭"]},
            ),
            "Duplicate alias",
        ),
    ],
)
def test_rejects_id_name_and_alias_collisions(mutate, message: str) -> None:
    items = _base_items()
    mutate(items)

    with pytest.raises(RegistryCompileError, match=message):
        validate_graph(_source(items))


def test_retired_alias_still_uses_global_source_namespace() -> None:
    items = _base_items()
    retired = next(item for item in items if item["id"] == "w-late")
    retired["status"] = "retired"
    retired["replacement_id"] = "w-early"
    retired["aliases"] = ["종합"]

    with pytest.raises(RegistryCompileError, match="Global name/alias collision"):
        validate_graph(_source(items))


def test_rejects_duplicate_instrument_id_and_unknown_instrument() -> None:
    source = _source(
        instruments=[
            {"id": "written", "label": "필기 검사"},
            {"id": "written", "label": "중복 필기 검사"},
            {"id": "video", "label": "영상 면접"},
        ]
    )
    with pytest.raises(RegistryCompileError, match="Duplicate instrument"):
        validate_graph(source)

    items = _base_items()
    items[0]["instrument"] = "missing"
    with pytest.raises(RegistryCompileError, match="unknown instrument"):
        validate_graph(_source(items))


@pytest.mark.parametrize(
    ("parent_id", "message"),
    [
        ("missing", "missing parent"),
        ("w-early", "own parent"),
        ("v-first-root", "Cross-instrument parent"),
    ],
)
def test_rejects_invalid_parent_reference(parent_id: str, message: str) -> None:
    items = _base_items()
    target = next(item for item in items if item["id"] == "w-early")
    target["parent_id"] = parent_id

    with pytest.raises(RegistryCompileError, match=message):
        validate_graph(_source(items))


def test_rejects_cycle() -> None:
    items = _base_items()
    root = next(item for item in items if item["id"] == "w-root")
    root["parent_id"] = "w-early"

    with pytest.raises(RegistryCompileError, match="cycle"):
        validate_graph(_source(items))


def test_rejects_active_child_of_retired_parent() -> None:
    items = _base_items()
    category = next(item for item in items if item["id"] == "w-category")
    category["status"] = "retired"

    with pytest.raises(RegistryCompileError, match="retired parent"):
        validate_graph(_source(items))


def test_rejects_duplicate_sibling_and_root_order() -> None:
    items = _base_items()
    late = next(item for item in items if item["id"] == "w-late")
    late["order"] = 10
    with pytest.raises(RegistryCompileError, match="Duplicate sibling order"):
        validate_graph(_source(items))

    items = _base_items()
    second_root = next(item for item in items if item["id"] == "v-second-root")
    second_root["order"] = 10
    with pytest.raises(RegistryCompileError, match="Duplicate sibling order"):
        validate_graph(_source(items))


def test_retired_item_may_keep_replaced_active_sibling_order() -> None:
    items = _base_items()
    retired = next(item for item in items if item["id"] == "w-late")
    retired["status"] = "retired"
    retired["replacement_id"] = "w-early"
    retired["order"] = 10

    validate_graph(_source(items))


@pytest.mark.parametrize(
    ("status", "replacement_id", "message"),
    [
        ("active", "w-late", "Active item"),
        ("retired", "missing", "missing replacement"),
        ("retired", "w-early", "must be active"),
        ("retired", "w-late", "replace itself"),
        ("retired", "v-first-root", "Cross-instrument replacement"),
    ],
)
def test_replacement_rules(
    status: str,
    replacement_id: str,
    message: str,
) -> None:
    items = _base_items()
    target = next(item for item in items if item["id"] == "w-late")
    target["status"] = status
    target["replacement_id"] = replacement_id
    if replacement_id == "w-early" and status == "retired":
        replacement = next(item for item in items if item["id"] == "w-early")
        replacement["status"] = "retired"

    with pytest.raises(RegistryCompileError, match=message):
        validate_graph(_source(items))


def test_render_parse_compile_semantic_round_trip() -> None:
    source = _source()

    parsed = parse_source_v2_text(render_source_v2(source))
    before = _compile(source)
    after = _compile(parsed)

    assert after == before
    assert validate_registry_document(after) == after


@pytest.mark.parametrize(
    ("filename", "source_hash", "message"),
    [
        (" ", SOURCE_HASH, "source_filename"),
        ("registry.md", "not-a-hash", "source_sha256"),
    ],
)
def test_compile_rejects_invalid_source_metadata(
    filename: str,
    source_hash: str,
    message: str,
) -> None:
    with pytest.raises(RegistryCompileError, match=message):
        compile_registry(
            _source(),
            source_filename=filename,
            source_sha256=source_hash,
        )
