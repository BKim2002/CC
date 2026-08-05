"""역량 레지스트리 PostgreSQL 업로드 스크립트 테스트."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from psycopg.types.json import Jsonb

from scripts import upload_competency_registry as uploader


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


@pytest.fixture
def prepared_registry() -> uploader.PreparedRegistry:
    return uploader.PreparedRegistry(
        source_filename="competency-source.md",
        source_markdown="# confidential source text",
        source_sha256="a" * 64,
        schema_version="1.0",
        document={
            "schema_version": "1.0",
            "source": {"sha256": "a" * 64},
            "items": [{"id": "sample"}],
        },
        item_count=62,
    )


def sql_calls(connection: FakeConnection) -> list[str]:
    return [statement for statement, _ in connection.cursor_instance.calls]


def test_dry_run_never_connects_to_database(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    monkeypatch.setattr(
        uploader,
        "prepare_registry",
        lambda source: prepared_registry,
    )

    def forbidden_connect(url: str) -> None:
        raise AssertionError("dry-run must not connect")

    monkeypatch.setattr(uploader, "connect", forbidden_connect)

    result = uploader.upload_competency_registry(
        Path("not-read.md"),
        activate=True,
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.version_id is None
    assert result.item_count == 62
    assert result.activated is False


def test_new_hash_inserts_registry_with_jsonb(
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
    assert connection.commits == 1
    assert connection.rollbacks == 0
    insert_calls = [
        (statement, params)
        for statement, params in connection.cursor_instance.calls
        if "INSERT INTO competency_data.registry_versions" in statement
    ]
    assert len(insert_calls) == 1
    params = insert_calls[0][1]
    assert isinstance(params, tuple)
    assert isinstance(params[4], Jsonb)
    assert params[0] == prepared_registry.source_filename
    assert params[1] == prepared_registry.source_markdown
    assert params[2] == prepared_registry.source_sha256
    assert params[5] == 62


def test_existing_hash_is_idempotent_without_insert(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection([(23,)])
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


def test_activate_updates_current_pointer_in_same_transaction(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection([(31,)])
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    result = uploader.upload_prepared_registry(
        prepared_registry,
        activate=True,
        database_url="postgresql://local/registry",
    )

    activation_calls = [
        params
        for statement, params in connection.cursor_instance.calls
        if "INSERT INTO competency_data.registry_current" in statement
    ]
    assert activation_calls == [(31,)]
    assert result.activated is True
    assert connection.commits == 1


def test_parse_failure_happens_before_any_database_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "invalid.md"
    source_path.write_text("not a competency registry", encoding="utf-8")

    def fail_build(source: Path) -> dict[str, Any]:
        raise uploader.RegistryError("parse failed")

    monkeypatch.setattr(uploader, "build_registry", fail_build)

    def forbidden_connect(url: str) -> None:
        raise AssertionError("invalid data must not connect")

    monkeypatch.setattr(uploader, "connect", forbidden_connect)

    with pytest.raises(uploader.RegistryError, match="parse failed"):
        uploader.upload_competency_registry(source_path)


def test_activation_failure_rolls_back_new_version_insert(
    monkeypatch: pytest.MonkeyPatch,
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    connection = FakeConnection(
        [None, (47,)],
        fail_when_sql_contains="INSERT INTO competency_data.registry_current",
    )
    monkeypatch.setattr(uploader, "connect", lambda url: connection)

    with pytest.raises(RuntimeError, match="activation failed"):
        uploader.upload_prepared_registry(
            prepared_registry,
            activate=True,
            database_url="postgresql://local/registry",
        )

    assert any(
        "INSERT INTO competency_data.registry_versions" in statement
        for statement in sql_calls(connection)
    )
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closes == 1


def test_success_and_cli_failure_output_contains_no_secret_or_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prepared_registry: uploader.PreparedRegistry,
) -> None:
    secret_url = "postgresql://user:password@private-host/database"
    connection = FakeConnection([(59,)])
    monkeypatch.setenv("DATABASE_URL", secret_url)
    monkeypatch.setattr(uploader, "connect", lambda url: connection)
    monkeypatch.setattr(
        uploader,
        "prepare_registry",
        lambda source: prepared_registry,
    )

    assert uploader.main(["--source", "not-read.md", "--activate"]) == 0
    success_output = capsys.readouterr()
    combined = success_output.out + success_output.err
    assert "version_id: 59" in combined
    assert "source_sha256" in combined
    assert "item_count: 62" in combined
    assert "password" not in combined
    assert "private-host" not in combined
    assert prepared_registry.source_markdown not in combined

    def failing_main(argv: object = None) -> int:
        raise RuntimeError(secret_url + prepared_registry.source_markdown)

    monkeypatch.setattr(uploader, "main", failing_main)
    assert uploader._cli() == 1
    failure_output = capsys.readouterr()
    combined_failure = failure_output.out + failure_output.err
    assert "password" not in combined_failure
    assert "private-host" not in combined_failure
    assert prepared_registry.source_markdown not in combined_failure
