"""PostgreSQL 체크포인트 준비 스크립트의 비밀값 없는 단위 테스트."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from scripts import setup_checkpoint_database as checkpoint_setup


def test_database_url_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        checkpoint_setup.get_database_url()


def test_setup_uses_configured_database_without_printing_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    class FakeSaver:
        def setup(self) -> None:
            calls["setup_called"] = True

    class FakePostgresSaver:
        @classmethod
        @contextmanager
        def from_conn_string(
            cls,
            database_url: str,
        ) -> Iterator[FakeSaver]:
            calls["database_url"] = database_url
            yield FakeSaver()

    secret_url = "postgresql://user:password@private-host/database"
    monkeypatch.setenv("DATABASE_URL", secret_url)
    monkeypatch.setattr(
        checkpoint_setup,
        "PostgresSaver",
        FakePostgresSaver,
    )

    assert checkpoint_setup.main() == 0
    captured = capsys.readouterr()

    assert calls == {
        "database_url": secret_url,
        "setup_called": True,
    }
    assert "완료" in captured.out
    assert "password" not in captured.out
    assert "private-host" not in captured.out
