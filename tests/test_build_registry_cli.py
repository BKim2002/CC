"""레지스트리 빌더의 파일 비의존 CLI 계약 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_registry as builder


@pytest.fixture
def generated_registry() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": {"sha256": "a" * 64},
        "validation": {
            "status": "passed",
            "counts": {
                "written_overall": 1,
                "written_L1": 3,
                "written_L2": 10,
                "written_L3": 30,
                "written_L4": 9,
                "video_factor": 3,
                "video_item": 6,
                "total": 62,
            },
        },
        "items": [],
    }


def _validation_item(
    item_id: str,
    name: str,
    instrument: str,
    *,
    aliases: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": item_id,
        "name": name,
        "aliases": aliases or [],
        "instrument": instrument,
        "level": "synthetic",
        "path": [name],
        "definition": f"{name} 정의",
        "definition_status": "explicit",
        "analysis_included": False,
        "parent_id": "synthetic-root",
        "children": [],
    }


def test_source_argument_is_required() -> None:
    with pytest.raises(SystemExit):
        builder.parse_args([])


def test_no_output_path_performs_validation_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    generated_registry: dict[str, object],
) -> None:
    source = tmp_path / "source.md"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "build_registry",
        lambda path: generated_registry,
    )

    assert builder.main(["--source", str(source)]) == 0
    assert "검증 완료" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == [source]


def test_output_and_explicit_check_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    generated_registry: dict[str, object],
) -> None:
    source = tmp_path / "source.md"
    output = tmp_path / "generated.json"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "build_registry",
        lambda path: generated_registry,
    )

    assert (
        builder.main(
            ["--source", str(source), "--output", str(output)]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == generated_registry
    assert (
        builder.main(
            ["--source", str(source), "--check", str(output)]
        )
        == 0
    )


def test_output_and_check_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    generated_registry: dict[str, object],
) -> None:
    source = tmp_path / "source.md"
    output = tmp_path / "generated.json"
    source.write_text("source", encoding="utf-8")
    output.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "build_registry",
        lambda path: generated_registry,
    )

    with pytest.raises(builder.RegistryError, match="동시에"):
        builder.main(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--check",
                str(output),
            ]
        )


@pytest.mark.parametrize(
    ("first", "second", "expected_message"),
    [
        (
            _validation_item("one", "공통 이름", "instrument-one"),
            _validation_item("two", "공통 이름", "instrument-two"),
            "중복된 정식 명칭",
        ),
        (
            _validation_item("one", "정식 이름", "instrument-one"),
            _validation_item(
                "two",
                "다른 이름",
                "instrument-two",
                aliases=["정식 이름"],
            ),
            "정식 명칭 또는 다른 별칭과 충돌",
        ),
        (
            _validation_item(
                "one",
                "첫 이름",
                "instrument-one",
                aliases=["공통 별칭"],
            ),
            _validation_item(
                "two",
                "둘 이름",
                "instrument-two",
                aliases=["공통 별칭"],
            ),
            "정식 명칭 또는 다른 별칭과 충돌",
        ),
    ],
)
def test_validate_items_uses_one_global_lookup_namespace(
    monkeypatch: pytest.MonkeyPatch,
    first: dict[str, object],
    second: dict[str, object],
    expected_message: str,
) -> None:
    monkeypatch.setattr(builder, "EXPECTED_COUNTS", {"total": 2})
    monkeypatch.setattr(builder, "EXPECTED_LEVEL_1_CHILDREN", {})
    monkeypatch.setattr(builder, "STRATEGY_LEVEL_3", ())

    with pytest.raises(builder.RegistryError, match=expected_message):
        builder.validate_items([first, second])  # type: ignore[list-item]
