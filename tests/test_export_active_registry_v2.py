"""활성 runtime 레지스트리의 V2 export와 round-trip 테스트."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import export_active_registry_v2 as exporter
from scripts.registry_source_v2 import parse_source_v2_text


@pytest.fixture
def runtime_document() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": {
            "file": "private-source.md",
            "sha256": "a" * 64,
            "encoding": "utf-8",
        },
        "rules": {"fixture": "synthetic runtime fixture"},
        "validation": {"status": "passed", "counts": {"total": 3}},
        "items": [
            {
                "id": "stable_root",
                "name": "검사 전체",
                "aliases": [],
                "instrument": "fixture",
                "instrument_label": "합성 검사",
                "level": "root",
                "path": ["합성 검사", "검사 전체"],
                "parent_name": "합성 검사",
                "parent_id": None,
                "children": ["두 번째", "첫 번째"],
                "children_ids": ["stable_b", "stable_a"],
                "definition": "전체 요약",
                "definition_status": "summary",
                "analysis_included": False,
                "notes": [],
                "source_section": "fixture",
            },
            {
                "id": "stable_a",
                "name": "첫 번째",
                "aliases": ["과거 이름"],
                "instrument": "fixture",
                "instrument_label": "합성 검사",
                "level": "factor",
                "path": ["합성 검사", "검사 전체", "첫 번째"],
                "parent_name": "검사 전체",
                "parent_id": "stable_root",
                "children": [],
                "children_ids": [],
                "definition": "정의에 | 기호와\n여러 줄이 있습니다.",
                "definition_status": "provided",
                "analysis_included": True,
                "notes": ["합성 메모"],
                "source_section": "fixture",
            },
            {
                "id": "stable_b",
                "name": "두 번째",
                "aliases": [],
                "instrument": "fixture",
                "instrument_label": "합성 검사",
                "level": "factor",
                "path": ["합성 검사", "검사 전체", "두 번째"],
                "parent_name": "검사 전체",
                "parent_id": "stable_root",
                "children": [],
                "children_ids": [],
                "definition": None,
                "definition_status": "not_provided",
                "analysis_included": True,
                "notes": [],
                "source_section": "fixture",
            },
        ],
    }


def test_source_conversion_and_compile_preserve_semantics(
    runtime_document: dict[str, object],
) -> None:
    source = exporter.source_from_runtime_document(runtime_document)

    assert [item.id for item in source.items] == [
        "stable_root",
        "stable_a",
        "stable_b",
    ]
    order_by_id = {item.id: item.order for item in source.items}
    assert order_by_id == {"stable_root": 0, "stable_a": 1, "stable_b": 0}
    assert source.instruments[0].root_path_prefix == ["합성 검사"]
    assert source.items[1].definition == "정의에 | 기호와\n여러 줄이 있습니다."

    markdown = exporter.render_source_v2(source)
    parsed = parse_source_v2_text(markdown)
    assert parsed == source
    compiled = exporter.compile_rendered_source(
        markdown,
        source_filename="exported.md",
    )
    projection = exporter.verify_round_trip(runtime_document, compiled)

    assert projection["canonical_count"] == 3
    assert projection["lookup_count"] == 4
    compiled_root = next(item for item in compiled["items"] if item["id"] == "stable_root")
    assert compiled_root["children_ids"] == ["stable_b", "stable_a"]
    assert compiled_root["path"] == ["합성 검사", "검사 전체"]
    assert compiled_root["parent_name"] == "합성 검사"


def test_export_writes_only_after_round_trip_and_prints_no_source_or_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    runtime_document: dict[str, object],
) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    output_dir = tmp_path / "private"
    output_dir.mkdir()
    output = output_dir / "competencies-v2.md"
    secret_url = "postgresql://user:secret-password@private-host/db"
    snapshot = SimpleNamespace(version_id=7, document=runtime_document)

    monkeypatch.setattr(exporter, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(exporter, "get_database_url", lambda: secret_url)
    monkeypatch.setattr(exporter, "load_active_registry", lambda url: snapshot)

    assert exporter.main(["--output", str(output)]) == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert output.is_file()
    assert parse_source_v2_text(output.read_text(encoding="utf-8")).items
    assert "version_id: 7" in combined
    assert "canonical_count: 3" in combined
    assert "secret-password" not in combined
    assert "private-host" not in combined
    assert "정의에 | 기호" not in combined


def test_repository_output_and_default_overwrite_are_rejected_before_db_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    outside = tmp_path / "already-there.md"
    outside.write_text("keep me", encoding="utf-8")

    monkeypatch.setattr(exporter, "PROJECT_ROOT", project_root)

    def forbidden_load(url: str) -> object:
        raise AssertionError("unsafe output must be rejected before DB read")

    monkeypatch.setattr(exporter, "load_active_registry", forbidden_load)

    with pytest.raises(exporter.RegistryExportError, match="저장소 밖"):
        exporter.export_active_registry_v2(
            project_root / "source.md",
            database_url="postgresql://local/db",
        )
    with pytest.raises(exporter.RegistryExportError, match="이미 있습니다"):
        exporter.export_active_registry_v2(
            outside,
            database_url="postgresql://local/db",
        )
    assert outside.read_text(encoding="utf-8") == "keep me"


def test_explicit_overwrite_replaces_exact_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_document: dict[str, object],
) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    output = tmp_path / "existing.md"
    output.write_text("old", encoding="utf-8")
    snapshot = SimpleNamespace(version_id=11, document=runtime_document)

    monkeypatch.setattr(exporter, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(exporter, "load_active_registry", lambda url: snapshot)

    result = exporter.export_active_registry_v2(
        output,
        overwrite=True,
        database_url="postgresql://local/db",
    )

    assert result.version_id == 11
    assert output.read_text(encoding="utf-8").startswith("# 역량 정의서")


def test_round_trip_difference_is_rejected(
    runtime_document: dict[str, object],
) -> None:
    changed = deepcopy(runtime_document)
    changed["items"][1]["definition"] = "달라진 정의"  # type: ignore[index]

    with pytest.raises(exporter.RegistryExportError, match="round-trip"):
        exporter.verify_round_trip(runtime_document, changed)


def test_round_trip_rejects_multi_root_order_change(
    runtime_document: dict[str, object],
) -> None:
    current = deepcopy(runtime_document)
    other_root = deepcopy(current["items"][0])  # type: ignore[index]
    other_root.update(
        {
            "id": "stable_other_root",
            "name": "다른 루트",
            "path": ["합성 검사", "다른 루트"],
            "children": [],
            "children_ids": [],
            "definition": "다른 루트 정의",
        }
    )
    current["items"].append(other_root)  # type: ignore[union-attr]
    candidate = deepcopy(current)
    candidate_items = candidate["items"]  # type: ignore[assignment]
    candidate["items"] = [candidate_items[-1], *candidate_items[:-1]]  # type: ignore[index]

    with pytest.raises(exporter.RegistryExportError, match="round-trip"):
        exporter.verify_round_trip(current, candidate)


def test_round_trip_normalizes_only_documented_bookkeeping_fields(
    runtime_document: dict[str, object],
) -> None:
    normalized = deepcopy(runtime_document)
    for item in normalized["items"]:  # type: ignore[union-attr]
        item["source_section"] = "V2 normalized section"
        item["definition_status"] = "explicit"
        item["order"] = 99
        item["status"] = "active"
        item["replacement_id"] = None

    assert exporter.verify_round_trip(runtime_document, normalized)
    assert "source_section" in exporter.INTENTIONAL_ROUND_TRIP_NORMALIZATIONS


def test_no_overwrite_race_preserves_competing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_document: dict[str, object],
) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    output = tmp_path / "raced.md"
    snapshot = SimpleNamespace(version_id=13, document=runtime_document)

    def load_and_create(url: str) -> object:
        output.write_text("created by another process", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(exporter, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(exporter, "load_active_registry", load_and_create)

    with pytest.raises(exporter.RegistryExportError, match="검증 중"):
        exporter.export_active_registry_v2(
            output,
            database_url="postgresql://local/db",
        )
    assert output.read_text(encoding="utf-8") == "created by another process"


def test_atomic_overwrite_failure_preserves_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.md"
    output.write_text("original", encoding="utf-8")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(exporter.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failure"):
        exporter._write_verified_output(
            output,
            "replacement",
            overwrite=True,
        )

    assert output.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_cli_failure_does_not_echo_private_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_text = "postgresql://user:password@host/db 정의 전문"

    def failing_main(argv: object = None) -> int:
        raise RuntimeError(private_text)

    monkeypatch.setattr(exporter, "main", failing_main)

    assert exporter._cli() == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "내보내기에 실패했습니다" in combined
    assert "password" not in combined
    assert "정의 전문" not in combined
