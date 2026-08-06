"""V2 compiler 결과와 기존 RegistrySnapshot 계약의 통합 회귀 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

import competency_interpreter
from competency_registry import build_registry_snapshot
from scripts import upload_competency_registry as uploader
from scripts.registry_compiler import compile_registry
from scripts.registry_source_v2 import SourceRegistry, render_source_v2


def test_v2_dynamic_deep_registry_loads_with_rename_alias_lookup() -> None:
    items: list[dict[str, object]] = []
    for index in range(6):
        is_last = index == 5
        items.append(
            {
                "id": f"stable-{index}",
                "name": "새 역량명" if is_last else f"단계 {index + 1}",
                "instrument": "synthetic",
                "node_type": "analysis_factor" if is_last else "category",
                "parent_id": None if index == 0 else f"stable-{index - 1}",
                "order": 10,
                "definition": f"합성 정의 {index + 1}",
                "aliases": ["이전 역량명"] if is_last else [],
                "analysis_included": is_last,
                "status": "active",
                "replacement_id": None,
                "notes": [],
            }
        )

    source = SourceRegistry.model_validate(
        {
            "source_schema_version": "2.0",
            "registry_schema_version": "1.0",
            "instruments": [{"id": "synthetic", "label": "합성 검사"}],
            "items": items,
        },
        strict=True,
    )
    document = compile_registry(
        source,
        source_filename="synthetic-v2.md",
        source_sha256="b" * 64,
    )
    snapshot = build_registry_snapshot(
        {
            "id": 12,
            "source_filename": "synthetic-v2.md",
            "source_sha256": "b" * 64,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": 6,
        }
    )

    assert snapshot.schema_version == "1.0"
    assert len(snapshot.canonical_names) == 6
    assert len(snapshot.lookup) == 7
    assert snapshot.lookup["이전 역량명"] is snapshot.lookup["새 역량명"]
    assert snapshot.lookup["새 역량명"]["id"] == "stable-5"
    assert snapshot.lookup["새 역량명"]["path"] == tuple(
        [f"단계 {index}" for index in range(1, 6)] + ["새 역량명"]
    )
    assert snapshot.document["source"]["format"] == "markdown-v2"


@pytest.mark.parametrize("item_count", [61, 63])
def test_v2_real_prepare_pipeline_has_no_fixed_62_policy(
    item_count: int,
    tmp_path: Path,
) -> None:
    items: list[dict[str, object]] = [
        {
            "id": "stable-root",
            "name": "합성 루트",
            "instrument": "synthetic",
            "node_type": "root",
            "parent_id": None,
            "order": 0,
            "definition": "합성 루트 정의",
            "aliases": [],
            "analysis_included": False,
            "status": "active",
            "replacement_id": None,
            "notes": [],
        }
    ]
    items.extend(
        {
            "id": f"stable-factor-{index}",
            "name": f"합성 요인 {index}",
            "instrument": "synthetic",
            "node_type": "analysis_factor",
            "parent_id": "stable-root",
            "order": index,
            "definition": f"합성 요인 {index} 정의",
            "aliases": [],
            "analysis_included": True,
            "status": "active",
            "replacement_id": None,
            "notes": [],
        }
        for index in range(1, item_count)
    )
    source = SourceRegistry.model_validate(
        {
            "source_schema_version": "2.0",
            "registry_schema_version": "1.0",
            "instruments": [{"id": "synthetic", "label": "합성 검사"}],
            "items": items,
        },
        strict=True,
    )
    source_path = tmp_path / f"registry-{item_count}.md"
    source_path.write_text(render_source_v2(source), encoding="utf-8")

    prepared = uploader.prepare_registry(
        source_path,
        source_format="markdown-v2",
    )
    snapshot = build_registry_snapshot(
        {
            "id": item_count,
            "source_filename": source_path.name,
            "source_sha256": prepared.source_sha256,
            "schema_version": prepared.schema_version,
            "registry_json": prepared.document,
            "item_count": prepared.item_count,
        }
    )

    assert prepared.item_count == item_count
    assert len(snapshot.canonical_names) == item_count


def test_new_v2_snapshot_ignores_stale_checkpoint_names_and_answers_deep_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SourceRegistry.model_validate(
        {
            "source_schema_version": "2.0",
            "registry_schema_version": "1.0",
            "instruments": [
                {
                    "id": "synthetic",
                    "label": "합성 검사",
                    "root_path_prefix": ["가상 검사 루트"],
                }
            ],
            "items": [
                {
                    "id": f"stable-{index}",
                    "name": f"깊이 {index + 1}",
                    "instrument": "synthetic",
                    "node_type": "factor",
                    "parent_id": None if index == 0 else f"stable-{index - 1}",
                    "order": 10,
                    "definition": f"깊이 {index + 1} 정의",
                    "aliases": [],
                    "analysis_included": index == 4,
                    "status": "active",
                    "replacement_id": None,
                    "notes": [],
                }
                for index in range(5)
            ],
        },
        strict=True,
    )
    document = compile_registry(
        source,
        source_filename="deep-v2.md",
        source_sha256="c" * 64,
    )
    snapshot = build_registry_snapshot(
        {
            "id": 20,
            "source_filename": "deep-v2.md",
            "source_sha256": "c" * 64,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": 5,
        }
    )
    monkeypatch.setattr(competency_interpreter, "_registry_snapshot", snapshot)

    # Old checkpoint state can contain names removed by a newer registry.
    assert competency_interpreter.validate_registry_names(
        ["삭제된 이전 후보", "깊이 5"]
    ) == ["깊이 5"]

    answer = competency_interpreter.produce_answer(
        {
            "matched_items": [document["items"][-1]],
            "requested_fields": ["path"],
        }
    )["messages"].content
    assert (
        "위계 구조: 가상 검사 루트 > 깊이 1 > 깊이 2 > 깊이 3 > 깊이 4 > 깊이 5"
        in answer
    )
