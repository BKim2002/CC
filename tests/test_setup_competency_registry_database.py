"""역량 레지스트리 PostgreSQL 스키마 준비 스크립트 테스트."""

from __future__ import annotations

import pytest

from scripts import setup_competency_registry_database as registry_setup


class FakeCursor:
    def __init__(self, *, fail_on: int | None = None) -> None:
        self.statements: list[str] = []
        self.fail_on = fail_on

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: str) -> None:
        self.statements.append(statement)
        if self.fail_on == len(self.statements):
            raise RuntimeError("fake DDL failure")


class FakeConnection:
    def __init__(self, *, fail_on: int | None = None) -> None:
        self.cursor_instance = FakeCursor(fail_on=fail_on)
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


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        registry_setup.get_database_url()


def test_setup_executes_all_ddl_and_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection()
    secret_url = "postgresql://user:password@private-host/database"
    monkeypatch.setattr(registry_setup, "connect", lambda url: connection)

    registry_setup.setup_competency_registry_database(secret_url)

    assert len(connection.cursor_instance.statements) == len(
        registry_setup.DDL_STATEMENTS
    )
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closes == 1
    sql = "\n".join(connection.cursor_instance.statements)
    assert "CREATE SCHEMA IF NOT EXISTS competency_data" in sql
    assert "registry_versions" in sql
    assert "registry_current" in sql
    assert "registry_versions_created_at_idx" in sql


def test_setup_is_idempotently_callable_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[FakeConnection] = []

    def fake_connect(url: str) -> FakeConnection:
        connection = FakeConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(registry_setup, "connect", fake_connect)

    registry_setup.setup_competency_registry_database(
        "postgresql://local/registry"
    )
    registry_setup.setup_competency_registry_database(
        "postgresql://local/registry"
    )

    assert len(connections) == 2
    assert all(connection.commits == 1 for connection in connections)
    assert all(connection.rollbacks == 0 for connection in connections)


def test_setup_rolls_back_the_transaction_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(fail_on=3)
    monkeypatch.setattr(registry_setup, "connect", lambda url: connection)

    with pytest.raises(RuntimeError, match="fake DDL failure"):
        registry_setup.setup_competency_registry_database(
            "postgresql://local/registry"
        )

    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closes == 1


def test_success_and_failure_output_never_prints_database_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_url = "postgresql://user:password@private-host/database"
    monkeypatch.setenv("DATABASE_URL", secret_url)
    monkeypatch.setattr(registry_setup, "connect", lambda url: FakeConnection())

    assert registry_setup.main() == 0
    success_output = capsys.readouterr()
    assert "완료" in success_output.out
    assert "password" not in success_output.out + success_output.err
    assert "private-host" not in success_output.out + success_output.err

    def failing_main() -> int:
        raise RuntimeError(secret_url)

    monkeypatch.setattr(registry_setup, "main", failing_main)
    assert registry_setup._cli() == 1
    failure_output = capsys.readouterr()
    assert "password" not in failure_output.out + failure_output.err
    assert "private-host" not in failure_output.out + failure_output.err
