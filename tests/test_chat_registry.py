"""chat.registry 이식 검증.

tests/test_competency_registry.py 에서 옮겨 왔다.  달라진 곳은 두 군데다 --
import 경로, 그리고 v2가 만들지 않는 name_catalog/semantic_catalog 단언 제거.
나머지 검증(문서 검증, 방어적 복사, 재귀 freeze, 로더)은 그대로다.

v1 테스트는 지우지 않는다.  REBUILD_PLAN 5절에 따라 v1 삭제는 맨 마지막이다.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from chat import registry as chat_registry
from chat.registry import (
    RegistryValidationError,
    build_registry_snapshot,
    load_active_registry,
    validate_registry_document,
)


SOURCE_HASH = "a" * 64


def _item(
    item_id: str,
    name: str,
    *,
    aliases: list[str] | None = None,
) -> dict:
    return {
        "id": item_id,
        "name": name,
        "aliases": aliases or [],
        "instrument": "synthetic",
        "instrument_label": "합성 검사",
        "level": "factor",
        "path": [name],
        "parent_name": None,
        "parent_id": None,
        "children": [],
        "children_ids": [],
        "definition": f"{name}의 정의",
        "definition_status": "explicit",
        "analysis_included": False,
        "notes": [],
        "source_section": "테스트",
    }


def _document() -> dict:
    return {
        "schema_version": "1.0",
        "source": {
            "file": "synthetic.md",
            "sha256": SOURCE_HASH,
            "encoding": "utf-8",
        },
        "rules": {"synthetic": "합성 레지스트리 규칙"},
        "validation": {"status": "passed", "counts": {"total": 2}},
        "items": [
            _item("synthetic:alpha", "역량 알파", aliases=["알파"]),
            _item("synthetic:beta", "역량 베타"),
        ],
    }


def _row() -> dict:
    document = _document()
    return {
        "id": 7,
        "source_filename": "synthetic.md",
        "source_sha256": SOURCE_HASH,
        "schema_version": "1.0",
        "registry_json": document,
        "item_count": len(document["items"]),
    }


def test_build_snapshot_creates_indexes() -> None:
    snapshot = build_registry_snapshot(_row())

    assert snapshot.version_id == 7
    assert snapshot.canonical_names == ("역량 베타", "역량 알파")
    assert snapshot.canonical_lookup["역량 알파"]["id"] == "synthetic:alpha"
    assert snapshot.lookup["알파"] is snapshot.lookup["역량 알파"]
    with pytest.raises(FrozenInstanceError):
        snapshot.version_id = 8  # type: ignore[misc]


def test_build_snapshot_defensively_copies_database_document() -> None:
    row = _row()
    snapshot = build_registry_snapshot(row)

    row["registry_json"]["items"][0]["name"] = "변경된 이름"

    assert snapshot.lookup["역량 알파"]["name"] == "역량 알파"


def test_snapshot_recursively_freezes_shared_registry_items() -> None:
    snapshot = build_registry_snapshot(_row())
    document_item = snapshot.document["items"][0]
    canonical_item = snapshot.canonical_lookup["역량 알파"]
    alias_item = snapshot.lookup["알파"]
    id_item = snapshot.id_lookup["synthetic:alpha"]

    assert document_item is canonical_item
    assert canonical_item is alias_item
    assert alias_item is id_item

    with pytest.raises(TypeError):
        canonical_item["definition"] = "변조"  # type: ignore[index]

    with pytest.raises(AttributeError):
        canonical_item["aliases"].append("새 별칭")


class _FakeCursor:
    def __init__(self, row: dict | None) -> None:
        self.row = row
        self.executed: tuple[str, tuple[int]] | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, sql: str, params: tuple[int]) -> None:
        self.executed = (sql, params)

    def fetchone(self) -> dict | None:
        return self.row


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self.fake_cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return self.fake_cursor


def test_load_active_registry_uses_bound_singleton_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _FakeCursor(_row())
    captured: dict[str, object] = {}

    def fake_connect(database_url: str, *, row_factory):
        captured["database_url"] = database_url
        captured["row_factory"] = row_factory
        return _FakeConnection(cursor)

    monkeypatch.setattr(chat_registry, "connect", fake_connect)

    snapshot = load_active_registry("postgresql://private-value")

    assert snapshot.version_id == 7
    assert captured["database_url"] == "postgresql://private-value"
    assert cursor.executed is not None
    assert cursor.executed[1] == (1,)


def test_load_active_registry_rejects_missing_active_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _FakeCursor(None)
    monkeypatch.setattr(
        chat_registry,
        "connect",
        lambda *args, **kwargs: _FakeConnection(cursor),
    )

    with pytest.raises(RuntimeError, match="활성 역량 레지스트리가 없습니다"):
        load_active_registry("postgresql://private-value")


def test_validate_rejects_non_object_document() -> None:
    with pytest.raises(RegistryValidationError, match="JSON object"):
        validate_registry_document([])


@pytest.mark.parametrize("items", [None, {}, "items"])
def test_validate_rejects_missing_or_non_list_items(items: object) -> None:
    document = _document()
    if items is None:
        del document["items"]
    else:
        document["items"] = items

    with pytest.raises(
        RegistryValidationError,
        match="필수 키|items는 JSON array",
    ):
        validate_registry_document(document)


def test_build_snapshot_rejects_source_hash_mismatch() -> None:
    row = _row()
    row["source_sha256"] = "b" * 64

    with pytest.raises(RegistryValidationError, match="source_sha256"):
        build_registry_snapshot(row)


def test_build_snapshot_rejects_source_filename_mismatch() -> None:
    row = _row()
    row["source_filename"] = "different.md"

    with pytest.raises(RegistryValidationError, match="source_filename"):
        build_registry_snapshot(row)


@pytest.mark.parametrize(
    "source_sha256",
    ["a" * 63, "g" * 64, "not-a-hash"],
)
def test_validate_rejects_non_hex_source_hash(source_sha256: str) -> None:
    document = _document()
    document["source"]["sha256"] = source_sha256

    with pytest.raises(RegistryValidationError, match="64자리 16진수"):
        validate_registry_document(document)


@pytest.mark.parametrize("source_key", ["file", "sha256", "encoding"])
def test_validate_requires_source_metadata(source_key: str) -> None:
    document = _document()
    del document["source"][source_key]

    with pytest.raises(RegistryValidationError, match=f"source.{source_key}"):
        validate_registry_document(document)


def test_validate_rejects_failed_source_validation() -> None:
    document = _document()
    document["validation"]["status"] = "failed"

    with pytest.raises(RegistryValidationError, match="status"):
        validate_registry_document(document)


@pytest.mark.parametrize("duplicate_kind", ["id", "name", "alias"])
def test_validate_rejects_duplicate_identifiers_and_names(
    duplicate_kind: str,
) -> None:
    document = _document()
    first, second = document["items"]

    if duplicate_kind == "id":
        second["id"] = first["id"]
    elif duplicate_kind == "name":
        second["name"] = first["name"]
    else:
        second["aliases"] = [first["aliases"][0]]

    with pytest.raises(RegistryValidationError, match="중복|충돌"):
        validate_registry_document(document)


def test_validate_rejects_alias_that_shadows_canonical_name() -> None:
    document = _document()
    document["items"][0]["aliases"] = [document["items"][1]["name"]]

    with pytest.raises(RegistryValidationError, match="충돌"):
        validate_registry_document(document)


def test_build_snapshot_rejects_item_count_mismatch() -> None:
    row = _row()
    row["item_count"] = 99

    with pytest.raises(RegistryValidationError, match="item_count"):
        build_registry_snapshot(row)


def test_build_snapshot_rejects_non_positive_item_count() -> None:
    row = _row()
    row["item_count"] = 0

    with pytest.raises(RegistryValidationError, match="1 이상"):
        build_registry_snapshot(row)


def test_validate_rejects_unsupported_schema_version() -> None:
    document = _document()
    document["schema_version"] = "2.0"

    with pytest.raises(RegistryValidationError, match="schema_version"):
        validate_registry_document(document)


def test_validate_rejects_missing_required_item_key() -> None:
    document = _document()
    del document["items"][0]["definition"]

    with pytest.raises(RegistryValidationError, match="필수 키"):
        validate_registry_document(document)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("instrument", None),
        ("definition", {"unexpected": "object"}),
        ("parent_id", ["unexpected"]),
        ("analysis_included", "false"),
        ("source_section", ""),
    ],
)
def test_validate_rejects_invalid_item_field_types(
    field: str,
    invalid_value: object,
) -> None:
    document = _document()
    document["items"][0][field] = invalid_value

    with pytest.raises(RegistryValidationError, match=field):
        validate_registry_document(document)


@pytest.mark.parametrize("top_level_key", ["source", "rules", "validation"])
def test_validate_requires_top_level_registry_sections(
    top_level_key: str,
) -> None:
    document = _document()
    del document[top_level_key]

    with pytest.raises(RegistryValidationError, match="필수 키"):
        validate_registry_document(document)


def test_validate_requires_rules_object() -> None:
    document = _document()
    document["rules"] = []

    with pytest.raises(RegistryValidationError, match="rules.*object"):
        validate_registry_document(document)


@pytest.mark.parametrize(
    "counts",
    [None, {}, {"total": True}, {"total": -1}, {"total": 99}],
)
def test_validate_rejects_invalid_validation_counts(counts: object) -> None:
    document = _document()
    if counts is None:
        del document["validation"]["counts"]
    else:
        document["validation"]["counts"] = counts

    with pytest.raises(RegistryValidationError, match="counts"):
        validate_registry_document(document)


def test_small_synthetic_registry_is_valid_without_62_items() -> None:
    document = _document()
    document["items"] = [deepcopy(document["items"][0])]
    document["validation"]["counts"]["total"] = 1
    row = _row()
    row["registry_json"] = document
    row["item_count"] = 1

    assert len(build_registry_snapshot(row).canonical_names) == 1


def test_snapshot_derives_id_lookup_and_reports_a_missing_stable_id() -> None:
    row = _row()
    snapshot = build_registry_snapshot(row)
    first = snapshot.document["items"][0]

    # The production path always supplies id_lookup; a directly constructed
    # snapshot derives it instead.
    assert snapshot.id_lookup[first["id"]]["name"] == first["name"]

    with pytest.raises(RegistryValidationError, match="stable ID"):
        chat_registry.RegistrySnapshot(
            version_id=snapshot.version_id,
            source_filename=snapshot.source_filename,
            source_sha256=snapshot.source_sha256,
            schema_version=snapshot.schema_version,
            document={"items": [{"name": "id 없는 항목"}]},
            lookup={},
            canonical_lookup={},
            canonical_names=(),
        )
