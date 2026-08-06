"""future-proof PostgreSQL registry upload와 activation guard 테스트."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from psycopg.types.json import Jsonb

from competency_registry import RegistryValidationError
from scripts import upload_competency_registry as uploader
from scripts.registry_source_v2 import SourceRegistry, render_source_v2


class FakeCursor:
    def __init__(
        self,
        fetch_rows: list[object | None],
        *,
        fail_when_sql_contains: str | None = None,
    ) -> None:
        self.fetch_rows = list(fetch_rows)
        self.fail_when_sql_contains = fail_when_sql_contains
        self.calls: list[tuple[str, object | None]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: str, params: object | None = None) -> None:
        self.calls.append((statement, params))
        if (
            self.fail_when_sql_contains is not None
            and self.fail_when_sql_contains in statement
        ):
            raise RuntimeError("activation failed with a private DB detail")

    def fetchone(self) -> object | None:
        if not self.fetch_rows:
            raise AssertionError("unexpected fetchone call")
        return self.fetch_rows.pop(0)


class FakeConnection:
    def __init__(
        self,
        fetch_rows: list[object | None],
        *,
        fail_when_sql_contains: str | None = None,
    ) -> None:
        self.cursor_instance = FakeCursor(
            fetch_rows,
            fail_when_sql_contains=fail_when_sql_contains,
        )
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closes += 1


def runtime_item(
    item_id: str,
    name: str,
    *,
    definition: str = "등록된 정의",
) -> dict[str, object]:
    return {
        "id": item_id,
        "name": name,
        "aliases": [],
        "instrument": "synthetic",
        "instrument_label": "합성 검사",
        "level": "root",
        "path": [name],
        "parent_name": None,
        "parent_id": None,
        "children": [],
        "children_ids": [],
        "definition": definition,
        "definition_status": "explicit",
        "analysis_included": False,
        "notes": [],
        "source_section": "테스트",
    }


def runtime_document(
    source_markdown: str,
    items: list[dict[str, object]],
    *,
    source_format: str | None = None,
) -> dict[str, Any]:
    source: dict[str, object] = {
        "file": "competency-source.md",
        "sha256": hashlib.sha256(source_markdown.encode("utf-8")).hexdigest(),
        "encoding": "utf-8",
    }
    if source_format is not None:
        source["format"] = source_format
    return {
        "schema_version": "1.0",
        "source": source,
        "rules": {"registry_policy": "synthetic test policy"},
        "validation": {
            "status": "passed",
            "counts": {"total": len(items)},
        },
        "items": items,
    }


def make_prepared(
    *,
    source_markdown: str = "# confidential source text",
    item_id: str = "stable:item",
    name: str = "등록 항목",
    definition: str = "candidate private definition",
) -> uploader.PreparedRegistry:
    document = runtime_document(
        source_markdown,
        [runtime_item(item_id, name, definition=definition)],
    )
    return uploader.PreparedRegistry(
        source_filename="competency-source.md",
        source_markdown=source_markdown,
        source_sha256=document["source"]["sha256"],
        schema_version="1.0",
        document=document,
        item_count=1,
    )


@pytest.fixture
def prepared_registry() -> uploader.PreparedRegistry:
    return make_prepared()


def active_row(
    version_id: int,
    document: dict[str, Any],
    source_markdown: str,
) -> tuple[object, ...]:
    return version_id, document, source_markdown


def existing_version_row(
    version_id: int,
    document: dict[str, Any],
    *,
    source_markdown: str = "# confidential source text",
    item_count: int | None = None,
) -> tuple[object, ...]:
    return (
        version_id,
        document["source"]["file"],
        source_markdown,
        document["source"]["sha256"],
        document["schema_version"],
        document,
        len(document["items"]) if item_count is None else item_count,
    )


def sql_calls(connection: FakeConnection) -> list[str]:
    return [statement for statement, _ in connection.cursor_instance.calls]


def sql_index(connection: FakeConnection, text: str) -> int:
    return next(
        index
        for index, statement in enumerate(sql_calls(connection))
        if text in statement
    )


def test_prepare_registry_uses_generic_runtime_validation_not_fixed_62(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.md"
    source_markdown = "# one-item registry"
    source_path.write_text(source_markdown, encoding="utf-8")
    document = runtime_document(
        source_markdown,
        [runtime_item("stable:one", "한 항목")],
    )
    seen_formats: list[str] = []

    def fake_build(path: Path, source_format: str) -> dict[str, Any]:
        seen_formats.append(source_format)
        return document

    monkeypatch.setattr(uploader, "build_registry", fake_build)

    prepared = uploader.prepare_registry(
        source_path,
        source_format="markdown-v1",
    )

    assert prepared.item_count == 1
    assert prepared.document == document
    assert seen_formats == ["markdown-v1"]
    assert not hasattr(uploader, "EXPECTED_COUNTS")
    assert not hasattr(uploader, "validate_items")


def test_prepare_registry_rejects_empty_and_runtime_invalid_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.md"
    source.write_text("source", encoding="utf-8")
    empty = runtime_document("source", [])
    monkeypatch.setattr(uploader, "build_registry", lambda *args, **kwargs: empty)
    with pytest.raises(uploader.RegistryError, match="비어 있지 않은"):
        uploader.prepare_registry(source)

    invalid = runtime_document("source", [runtime_item("id", "이름")])
    del invalid["items"][0]["path"]
    monkeypatch.setattr(uploader, "build_registry", lambda *args, **kwargs: invalid)
    with pytest.raises(RegistryValidationError, match="필수 키"):
        uploader.prepare_registry(source)


def test_prepare_registry_rechecks_exact_source_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.md"
    source.write_text("actual source", encoding="utf-8")
    wrong_document = runtime_document(
        "different source",
        [runtime_item("id", "이름")],
    )
    monkeypatch.setattr(
        uploader,
        "build_registry",
        lambda *args, **kwargs: wrong_document,
    )

    with pytest.raises(uploader.RegistryError, match="해시"):
        uploader.prepare_registry(source)


def test_prepare_v2_keeps_source_items_for_retirement_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-v2.md"
    source.write_text("v2 source", encoding="utf-8")
    document = runtime_document(
        "v2 source",
        [runtime_item("active:id", "활성")],
        source_format="markdown-v2",
    )
    source_items = ({"id": "retired:id", "status": "retired"},)
    monkeypatch.setattr(
        uploader,
        "build_registry",
        lambda *args, **kwargs: document,
    )
    monkeypatch.setattr(
        uploader,
        "_v2_source_items",
        lambda text: source_items,
    )

    prepared = uploader.prepare_registry(source, source_format="markdown-v2")

    assert prepared.source_format == "markdown-v2"
    assert prepared.source_items == source_items


def test_plain_dry_run_never_connects_to_database(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    seen_formats: list[str] = []

    def fake_prepare(
        source: Path,
        *,
        source_format: str,
    ) -> uploader.PreparedRegistry:
        seen_formats.append(source_format)
        return prepared_registry

    monkeypatch.setattr(uploader, "prepare_registry", fake_prepare)

    def forbidden_connect(url: str) -> None:
        raise AssertionError("plain dry-run must not connect")

    monkeypatch.setattr(uploader, "connect", forbidden_connect)
    result = uploader.upload_competency_registry(
        Path("not-read.md"),
        source_format="markdown-v2",
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.version_id is None
    assert result.item_count == 1
    assert result.registry_diff is None
    assert seen_formats == ["markdown-v2"]


def test_direct_script_v2_dry_run_uses_canonical_package_imports(
    tmp_path: Path,
) -> None:
    source = SourceRegistry.model_validate(
        {
            "source_schema_version": "2.0",
            "registry_schema_version": "1.0",
            "instruments": [{"id": "synthetic", "label": "합성 검사"}],
            "items": [
                {
                    "id": "synthetic:root",
                    "name": "합성 루트",
                    "instrument": "synthetic",
                    "node_type": "root",
                    "parent_id": None,
                    "order": 0,
                    "definition": "합성 정의",
                    "aliases": [],
                    "analysis_included": False,
                    "status": "active",
                    "replacement_id": None,
                    "notes": [],
                }
            ],
        },
        strict=True,
    )
    source_path = tmp_path / "source-v2.md"
    source_path.write_text(render_source_v2(source), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(uploader.PROJECT_ROOT / "scripts" / "upload_competency_registry.py"),
            "--source",
            str(source_path),
            "--source-format",
            "markdown-v2",
            "--dry-run",
        ],
        cwd=uploader.PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "item_count: 1" in completed.stdout


def test_dry_run_with_diff_uses_read_only_connection(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection(
        [
            active_row(
                7,
                prepared_registry.document,
                prepared_registry.source_markdown,
            )
        ]
    )
    monkeypatch.setattr(
        uploader,
        "prepare_registry",
        lambda *args, **kwargs: prepared_registry,
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_competency_registry(
        Path("not-read.md"),
        dry_run=True,
        diff_active=True,
        database_url="postgresql://local/registry",
    )

    assert result.dry_run is True
    assert result.registry_diff is not None
    assert result.registry_diff.is_empty
    assert any("SET TRANSACTION READ ONLY" in sql for sql in sql_calls(connection))
    assert not any("INSERT" in sql for sql in sql_calls(connection))
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closes == 1


def test_new_hash_inserts_dynamic_registry_with_jsonb(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection([None, (17,)])
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_prepared_registry(
        prepared_registry,
        database_url="postgresql://local/registry",
    )

    assert result.version_id == 17
    assert result.created is True
    assert result.item_count == 1
    assert connection.commits == 1
    insert_calls = [
        (statement, params)
        for statement, params in connection.cursor_instance.calls
        if "INSERT INTO competency_data.registry_versions" in statement
    ]
    assert len(insert_calls) == 1
    params = insert_calls[0][1]
    assert isinstance(params, tuple)
    assert isinstance(params[4], Jsonb)
    assert params[5] == 1


def test_existing_hash_is_idempotent_without_insert(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection(
        [existing_version_row(23, prepared_registry.document)]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_prepared_registry(
        prepared_registry,
        database_url="postgresql://local/registry",
    )

    assert result.version_id == 23
    assert result.created is False
    assert not any(
        "INSERT INTO competency_data.registry_versions" in statement
        for statement in sql_calls(connection)
    )
    assert connection.commits == 1


def test_existing_hash_ignores_only_local_source_basename(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    stored_document = deepcopy(prepared_registry.document)
    stored_document["source"]["file"] = "original-private-name.md"
    connection = FakeConnection(
        [existing_version_row(24, stored_document)]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_prepared_registry(
        prepared_registry,
        database_url="postgresql://local/registry",
    )

    assert result.version_id == 24
    assert result.created is False
    assert connection.commits == 1


def test_existing_same_hash_with_different_runtime_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    stored_document = deepcopy(prepared_registry.document)
    stored_document["items"][0]["id"] = "different-stored-id"
    connection = FakeConnection(
        [
            active_row(
                7,
                prepared_registry.document,
                prepared_registry.source_markdown,
            ),
            existing_version_row(23, stored_document),
        ]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    with pytest.raises(uploader.RegistryError, match="compiler 결과와 다릅니다"):
        uploader.upload_prepared_registry(
            prepared_registry,
            activate=True,
            expected_current_version=7,
            database_url="postgresql://local/registry",
        )

    assert not any(
        "INSERT INTO competency_data.registry_current" in statement
        for statement in sql_calls(connection)
    )
    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.parametrize("damage", ["item_count", "source_markdown"])
def test_existing_hash_rejects_damaged_db_metadata_before_activation(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
    damage: str,
) -> None:
    row_kwargs: dict[str, object] = {}
    if damage == "item_count":
        row_kwargs["item_count"] = 99
    else:
        row_kwargs["source_markdown"] = "tampered stored source"
    connection = FakeConnection(
        [
            active_row(
                7,
                prepared_registry.document,
                prepared_registry.source_markdown,
            ),
            existing_version_row(
                23,
                prepared_registry.document,
                **row_kwargs,
            ),
        ]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    with pytest.raises(uploader.RegistryError, match="metadata가 손상"):
        uploader.upload_prepared_registry(
            prepared_registry,
            activate=True,
            expected_current_version=7,
            database_url="postgresql://local/registry",
        )

    assert not any(
        "INSERT INTO competency_data.registry_current" in statement
        for statement in sql_calls(connection)
    )
    assert connection.rollbacks == 1


def test_inactive_upload_allows_breaking_diff(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    current_source = "# current private source"
    current_document = runtime_document(
        current_source,
        [runtime_item("removed:id", "삭제될 항목")],
    )
    connection = FakeConnection(
        [active_row(31, current_document, current_source), None, (41,)]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_prepared_registry(
        prepared_registry,
        diff_active=True,
        database_url="postgresql://local/registry",
    )

    assert result.registry_diff is not None
    assert result.registry_diff.is_breaking
    assert result.version_id == 41
    assert result.activated is False
    assert connection.commits == 1


def test_activation_locks_and_rechecks_before_candidate_lookup(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection(
        [
            active_row(
                31,
                prepared_registry.document,
                prepared_registry.source_markdown,
            ),
            existing_version_row(31, prepared_registry.document),
        ]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_prepared_registry(
        prepared_registry,
        activate=True,
        expected_current_version=31,
        database_url="postgresql://local/registry",
    )

    assert result.activated is True
    assert result.registry_diff is not None
    assert sql_index(connection, "LOCK TABLE") < sql_index(connection, "FOR UPDATE")
    assert sql_index(connection, "FOR UPDATE") < sql_index(connection, "WHERE source_sha256")
    assert sql_index(connection, "WHERE source_sha256") < sql_index(
        connection,
        "INSERT INTO competency_data.registry_current",
    )
    assert connection.commits == 1


def test_expected_current_version_mismatch_rolls_back_before_insert(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection(
        [
            active_row(
                31,
                prepared_registry.document,
                prepared_registry.source_markdown,
            )
        ]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    with pytest.raises(uploader.RegistryError, match="기대값"):
        uploader.upload_prepared_registry(
            prepared_registry,
            activate=True,
            expected_current_version=30,
            database_url="postgresql://local/registry",
        )

    assert not any("registry_versions (" in sql for sql in sql_calls(connection))
    assert not any("WHERE source_sha256" in sql for sql in sql_calls(connection))
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_breaking_activation_is_rejected_before_candidate_insert(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    current_source = "# current source"
    current_document = runtime_document(
        current_source,
        [runtime_item("removed:id", "기존 항목")],
    )
    connection = FakeConnection([active_row(31, current_document, current_source)])
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    with pytest.raises(uploader.RegistryError, match="allow-breaking"):
        uploader.upload_prepared_registry(
            prepared_registry,
            activate=True,
            expected_current_version=31,
            database_url="postgresql://local/registry",
        )

    assert not any("WHERE source_sha256" in sql for sql in sql_calls(connection))
    assert not any("registry_versions (" in sql for sql in sql_calls(connection))
    assert connection.rollbacks == 1


def test_allow_breaking_permits_insert_and_activation(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    current_source = "# current source"
    current_document = runtime_document(
        current_source,
        [runtime_item("removed:id", "기존 항목")],
    )
    connection = FakeConnection(
        [active_row(31, current_document, current_source), None, (47,)]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_prepared_registry(
        prepared_registry,
        activate=True,
        expected_current_version=31,
        allow_breaking=True,
        database_url="postgresql://local/registry",
    )

    assert result.version_id == 47
    assert result.created is True
    assert result.activated is True
    assert result.registry_diff is not None and result.registry_diff.is_breaking
    assert connection.commits == 1


def test_first_activation_explicitly_expects_none(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection([None, None, (17,)])
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_prepared_registry(
        prepared_registry,
        activate=True,
        expected_current_version=None,
        database_url="postgresql://local/registry",
    )

    assert result.version_id == 17
    assert result.activated is True
    assert connection.commits == 1


def test_activation_failure_rolls_back_new_version_insert(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection(
        [None, None, (47,)],
        fail_when_sql_contains="INSERT INTO competency_data.registry_current",
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    with pytest.raises(RuntimeError, match="activation failed"):
        uploader.upload_prepared_registry(
            prepared_registry,
            activate=True,
            expected_current_version=None,
            database_url="postgresql://local/registry",
        )

    assert any("registry_versions (" in sql for sql in sql_calls(connection))
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closes == 1


def test_active_v2_source_items_are_used_for_diff(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    active_source = "structured active source"
    active_document = runtime_document(
        active_source,
        [runtime_item("stable:item", "등록 항목")],
        source_format="markdown-v2",
    )
    active_source_items = (
        {
            "id": "retired:id",
            "name": "퇴역 항목",
            "instrument": "synthetic",
            "node_type": "root",
            "parent_id": None,
            "order": 1,
            "definition": "퇴역 정의",
            "aliases": [],
            "analysis_included": False,
            "status": "retired",
            "replacement_id": "stable:item",
            "notes": [],
        },
    )
    captured: dict[str, object] = {}
    connection = FakeConnection(
        [
            active_row(9, active_document, active_source),
            existing_version_row(23, prepared_registry.document),
        ]
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)
    monkeypatch.setattr(
        uploader,
        "_v2_source_items",
        lambda text: active_source_items,
    )
    real_diff = uploader.diff_registries

    def capture_diff(*args: object, **kwargs: object) -> Any:
        captured.update(kwargs)
        return real_diff(*args, **kwargs)

    monkeypatch.setattr(uploader, "diff_registries", capture_diff)

    uploader.upload_prepared_registry(
        prepared_registry,
        diff_active=True,
        database_url="postgresql://local/registry",
    )

    assert captured["current_source_items"] == active_source_items


@pytest.mark.parametrize(
    "argv",
    [
        ["--source", "source.md", "--activate"],
        [
            "--source",
            "source.md",
            "--activate",
            "--expected-current-version",
            "1",
            "--dry-run",
        ],
        ["--source", "source.md", "--expected-current-version", "1"],
        ["--source", "source.md", "--allow-breaking"],
    ],
)
def test_cli_rejects_unsafe_flag_combinations(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        uploader.parse_args(argv)


def test_cli_parses_source_format_and_expected_none() -> None:
    args = uploader.parse_args(
        [
            "--source",
            "source.md",
            "--source-format",
            "markdown-v2",
            "--activate",
            "--expected-current-version",
            "none",
            "--allow-breaking",
        ]
    )

    assert args.source_format == "markdown-v2"
    assert args.expected_current_version is None
    assert args.allow_breaking is True


def test_programmatic_activate_and_dry_run_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        uploader,
        "prepare_registry",
        lambda *args, **kwargs: pytest.fail("must fail before preparation"),
    )
    with pytest.raises(uploader.RegistryError, match="동시에"):
        uploader.upload_competency_registry(
            Path("source.md"),
            activate=True,
            dry_run=True,
            expected_current_version=1,
        )


@pytest.mark.parametrize("invalid_expected", [True, False, 0, -1, "1"])
def test_programmatic_expected_version_rejects_invalid_types(
    monkeypatch: pytest.MonkeyPatch,
    invalid_expected: object,
) -> None:
    monkeypatch.setattr(
        uploader,
        "prepare_registry",
        lambda *args, **kwargs: pytest.fail("must fail before preparation"),
    )

    with pytest.raises(uploader.RegistryError, match="양의 정수"):
        uploader.upload_competency_registry(
            Path("source.md"),
            activate=True,
            expected_current_version=invalid_expected,
        )


def test_parse_failure_happens_before_any_database_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "invalid.md"
    source_path.write_text("not a competency registry", encoding="utf-8")

    def fail_build(source: Path, source_format: str) -> dict[str, Any]:
        raise uploader.RegistryError("parse failed")

    monkeypatch.setattr(uploader, "build_registry", fail_build)
    monkeypatch.setattr(
        uploader,
        "connect",
        lambda url: pytest.fail("invalid data must not connect"),
    )

    with pytest.raises(uploader.RegistryError, match="parse failed"):
        uploader.upload_competency_registry(
            source_path,
            source_format="markdown-v2",
        )


def test_output_never_contains_secret_source_or_definitions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    current_source = "# private active source"
    current_definition = "active super secret definition"
    current_document = runtime_document(
        current_source,
        [
            runtime_item(
                "stable:item",
                "등록 항목",
                definition=current_definition,
            )
        ],
    )
    registry_diff = uploader.diff_registries(
        current_document,
        prepared_registry.document,
        current_version_id=3,
    )
    result = uploader.UploadResult(
        version_id=None,
        source_sha256=prepared_registry.source_sha256,
        item_count=1,
        created=False,
        activated=False,
        dry_run=True,
        registry_diff=registry_diff,
    )

    uploader.print_result(result)
    output = capsys.readouterr().out
    assert "definition_changed: 1" in output
    assert current_definition not in output
    assert prepared_registry.document["items"][0]["definition"] not in output
    assert current_source not in output
    assert prepared_registry.source_markdown not in output

    secret_url = "postgresql://user:password@private-host/database"

    def failing_main(argv: object = None) -> int:
        raise RuntimeError(secret_url + prepared_registry.source_markdown)

    monkeypatch.setattr(uploader, "main", failing_main)
    assert uploader._cli() == 1
    failure_output = capsys.readouterr()
    combined = failure_output.out + failure_output.err
    assert "password" not in combined
    assert "private-host" not in combined
    assert prepared_registry.source_markdown not in combined
