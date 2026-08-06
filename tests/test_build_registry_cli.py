"""레지스트리 빌더의 파일 비의존 CLI 계약 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_registry as builder
from scripts import registry_source_v1 as legacy


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


def test_source_format_is_explicit_and_defaults_to_v1() -> None:
    default_args = builder.parse_args(["--source", "source.md"])
    v2_args = builder.parse_args(
        [
            "--source",
            "source.md",
            "--source-format",
            "markdown-v2",
        ]
    )

    assert default_args.source_format == "markdown-v1"
    assert v2_args.source_format == "markdown-v2"


def test_dispatcher_does_not_guess_source_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    generated_registry: dict[str, object],
) -> None:
    source = tmp_path / "looks-like-v2.md"
    source.write_text(
        "```yaml\nsource_schema: competency-registry/v2\n```",
        encoding="utf-8",
    )
    calls: list[str] = []

    def build_v1(path: Path) -> dict[str, object]:
        calls.append(f"v1:{path.name}")
        return generated_registry

    def build_v2(path: Path) -> dict[str, object]:
        calls.append(f"v2:{path.name}")
        return generated_registry

    monkeypatch.setattr(builder, "build_registry_v1", build_v1)
    monkeypatch.setattr(builder, "_build_registry_v2", build_v2)

    assert builder.build_registry(source) is generated_registry
    assert (
        builder.build_registry(source, source_format="markdown-v2")
        is generated_registry
    )
    assert calls == ["v1:looks-like-v2.md", "v2:looks-like-v2.md"]


def test_dispatcher_rejects_unknown_source_format(tmp_path: Path) -> None:
    with pytest.raises(builder.RegistryError, match="지원하지 않는"):
        builder.build_registry(
            tmp_path / "source.md",
            source_format="auto",
        )


def test_legacy_public_imports_are_reexported() -> None:
    assert builder.RegistryError is legacy.RegistryError
    assert builder.EXPECTED_COUNTS is legacy.EXPECTED_COUNTS
    assert builder.validate_items is legacy.validate_items


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
        lambda path, source_format="markdown-v1": generated_registry,
    )

    assert builder.main(["--source", str(source)]) == 0
    assert "검증 완료" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == [source]


def test_cli_count_output_does_not_assume_legacy_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source-v2.md"
    source.write_text("source", encoding="utf-8")
    registry = {
        "source": {"sha256": "b" * 64},
        "validation": {
            "status": "passed",
            "counts": {"active": 63, "retired": 2, "total": 65},
        },
        "items": [],
    }
    seen_formats: list[str] = []

    def fake_build(path: Path, source_format: str) -> dict[str, object]:
        seen_formats.append(source_format)
        return registry

    monkeypatch.setattr(builder, "build_registry", fake_build)

    assert (
        builder.main(
            [
                "--source",
                str(source),
                "--source-format",
                "markdown-v2",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "active 63, retired 2, total 65" in output
    assert seen_formats == ["markdown-v2"]


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
        lambda path, source_format="markdown-v1": generated_registry,
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
        lambda path, source_format="markdown-v1": generated_registry,
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


def test_direct_cli_failure_hides_source_and_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_value = "TOP_SECRET_SOURCE_CONTENT"

    def failing_main(argv: object = None) -> int:
        raise RuntimeError(private_value)

    monkeypatch.setattr(builder, "main", failing_main)

    assert builder._cli() == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "변환 또는 검증에 실패했습니다" in combined
    assert private_value not in combined


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
    monkeypatch.setattr(legacy, "EXPECTED_COUNTS", {"total": 2})
    monkeypatch.setattr(legacy, "EXPECTED_LEVEL_1_CHILDREN", {})
    monkeypatch.setattr(legacy, "STRATEGY_LEVEL_3", ())

    with pytest.raises(builder.RegistryError, match=expected_message):
        legacy.validate_items([first, second])  # type: ignore[list-item]
