"""V2 compiler 결과와 RegistrySnapshot 계약의 통합 회귀 테스트.

v1 삭제와 함께 stale-checkpoint 테스트를 뺐다.  그 테스트는
``competency_interpreter.validate_registry_names``가 오래된 체크포인트에 남은
삭제된 역량명을 걸러내는지 확인했다.  v2에는 그 함수가 없고, 같은 위험을
grounding judge가 막는다 -- 답변이 단정한 이름을 **현재** 스냅샷과 대조하므로
레지스트리에서 사라진 이름은 그 자리에서 걸린다(REBUILD_PLAN 3.2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chat.registry import build_registry_snapshot
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
