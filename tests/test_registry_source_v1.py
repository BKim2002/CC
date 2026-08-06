"""V1 legacy adapter의 기존 출력·정책 계약 회귀 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_registry as dispatcher
from scripts import registry_source_v1 as legacy


def test_v1_keeps_name_based_id_generation_inside_legacy_adapter() -> None:
    assert (
        legacy.make_id(legacy.WRITTEN_INSTRUMENT, "L3", "문제해결력")
        == "written:l3:문제해결력"
    )
    assert (
        legacy.make_id(legacy.VIDEO_INSTRUMENT, "item", "의사표현")
        == "video:item:의사표현"
    )
    assert not hasattr(dispatcher, "make_id")


def test_v1_keeps_fixed_62_item_policy() -> None:
    assert legacy.EXPECTED_COUNTS["total"] == 62


def test_v1_registry_document_preserves_legacy_json_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "legacy-source.md"
    source_bytes = "# 기존 역량 정의\n".encode()
    source_path.write_bytes(source_bytes)
    items = [{"id": "legacy:item", "name": "기존 항목"}]
    counts = {
        "written_overall": 1,
        "written_L1": 3,
        "written_L2": 10,
        "written_L3": 30,
        "written_L4": 9,
        "video_factor": 3,
        "video_item": 6,
        "total": 62,
    }
    monkeypatch.setattr(legacy, "build_items", lambda text: items)
    monkeypatch.setattr(legacy, "validate_items", lambda value: counts)

    document = legacy.build_registry_v1(source_path)
    expected = {
        "schema_version": "1.0",
        "source": {
            "file": "legacy-source.md",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "encoding": "utf-8",
        },
        "rules": {
            "written_analysis_unit": "Level 3 하위요인 30개",
            "level_4": "정의와 데이터 컬럼은 보존하되 분석 변수에서는 제외",
            "video_interview": (
                "필기 역량검사와 별도의 도구이며 competency score_cols에서 제외"
            ),
        },
        "validation": {"status": "passed", "counts": counts},
        "items": items,
    }

    assert json.dumps(document, ensure_ascii=False, separators=(",", ":")) == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
    )
