"""chat.tools — 개수와 트리 순회.

두 가지를 지킨다.

1. **순회가 정확하다.**  `_descendant_ids`·`_ancestor_ids`·형제 판정은 v1에서
   옮겨 온 것이고, 어긋나면 조용히 틀린 수를 낸다.  이식 당시에는 v1 구현과
   직접 대조했으나, v1 삭제와 함께 명시적 기댓값으로 바꿨다 -- 대조 대상이
   사라졌고, 기댓값을 적어 두는 편이 무엇을 보장하는지도 분명하다.
2. **위계 이름이 도구 출력에 실린다.**  1단계에서 `g06`이 자식을 부모의
   티어 이름으로 불렀다(8회 중 7회).  이름을 모델이 추론하지 않게 하는 것이
   3단계의 목적이다.
"""

from __future__ import annotations

import pytest

from chat.registry import build_registry_snapshot
from chat.tools import (
    UnknownCompetencyError,
    _ancestor_ids,
    _descendant_ids,
    count_competencies,
    get_relations,
    tier_term,
)

SOURCE_HASH = "c" * 64


def _item(
    item_id: str,
    name: str,
    *,
    level: str,
    parent: str | None = None,
    children: list[str] | None = None,
    instrument: str = "written_competency",
    instrument_label: str = "필기 역량검사",
    analysis_included: bool = False,
) -> dict:
    return {
        "id": item_id,
        "name": name,
        "aliases": [],
        "instrument": instrument,
        "instrument_label": instrument_label,
        "level": level,
        "path": [name],
        "parent_name": parent.split(":")[-1] if parent else None,
        "parent_id": parent,
        "children": [child.split(":")[-1] for child in (children or [])],
        "children_ids": children or [],
        "definition": f"{name} 정의",
        "definition_status": "explicit",
        "analysis_included": analysis_included,
        "notes": [],
        "source_section": "테스트",
    }


def _document() -> dict:
    #  root(L1)
    #    mid(L2)               <- 직속 2, 후손 4  (g06 과 같은 모양)
    #      leaf_a(L3) leaf_b(L3)
    #        deep_a(L4) deep_b(L4)
    #    other(L2)
    #  video_root(factor)
    items = [
        _item("w:root", "루트", level="L1", children=["w:mid", "w:other"]),
        _item("w:mid", "중간", level="L2", parent="w:root", children=["w:leaf_a", "w:leaf_b"]),
        _item("w:other", "다른중간", level="L2", parent="w:root"),
        _item(
            "w:leaf_a",
            "잎A",
            level="L3",
            parent="w:mid",
            children=["w:deep_a"],
            analysis_included=True,
        ),
        _item(
            "w:leaf_b",
            "잎B",
            level="L3",
            parent="w:mid",
            children=["w:deep_b"],
            analysis_included=True,
        ),
        _item("w:deep_a", "깊은A", level="L4", parent="w:leaf_a"),
        _item("w:deep_b", "깊은B", level="L4", parent="w:leaf_b"),
        _item(
            "v:root",
            "영상요인",
            level="factor",
            instrument="video_interview",
            instrument_label="영상면접",
        ),
    ]
    return {
        "schema_version": "1.0",
        "source": {"file": "synthetic.md", "sha256": SOURCE_HASH, "encoding": "utf-8"},
        "rules": {"synthetic": "합성 레지스트리 규칙"},
        "validation": {"status": "passed", "counts": {"total": len(items)}},
        "items": items,
    }


def _row() -> dict:
    document = _document()
    return {
        "id": 3,
        "source_filename": "synthetic.md",
        "source_sha256": SOURCE_HASH,
        "schema_version": "1.0",
        "registry_json": document,
        "item_count": len(document["items"]),
    }


@pytest.fixture
def snapshot():
    return build_registry_snapshot(_row())


# ── 순회 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("target_id", "expected"),
    [
        # 깊이 우선이고, 형제보다 자손을 먼저 낸다.
        ("w:root", ["w:mid", "w:leaf_a", "w:deep_a", "w:leaf_b", "w:deep_b", "w:other"]),
        ("w:mid", ["w:leaf_a", "w:deep_a", "w:leaf_b", "w:deep_b"]),
        ("w:leaf_a", ["w:deep_a"]),
        ("w:deep_a", []),
        ("v:root", []),
    ],
)
def test_descendants_walk_the_whole_subtree(snapshot, target_id, expected) -> None:
    assert _descendant_ids(target_id, snapshot) == expected


@pytest.mark.parametrize(
    ("target_id", "expected"),
    [
        # 뿌리부터 가까운 순.
        ("w:root", []),
        ("w:mid", ["w:root"]),
        ("w:deep_a", ["w:root", "w:mid", "w:leaf_a"]),
        ("v:root", []),
    ],
)
def test_ancestors_run_from_the_root_downward(snapshot, target_id, expected) -> None:
    assert _ancestor_ids(target_id, snapshot) == expected


def test_descendants_are_deeper_than_children(snapshot) -> None:
    """`g06`이 시험하려던 구분. 직속만 세면 후손을 놓친다."""

    assert len(_descendant_ids("w:mid", snapshot)) == 4
    assert get_relations(snapshot, name="중간", relation="children")["count"] == 2


# ── 위계 이름 ────────────────────────────────────────────────────────


def test_relations_carry_the_tier_name_so_the_model_need_not_infer_it(snapshot) -> None:
    result = get_relations(snapshot, name="중간", relation="children")

    assert result["target"]["tier"] == "중위요인"
    assert [item["tier"] for item in result["items"]] == ["하위요인", "하위요인"]


def test_tier_terms_do_not_mix_written_and_video_vocabulary(snapshot) -> None:
    assert tier_term(snapshot.lookup["영상요인"]) == "평가요인"
    assert tier_term(snapshot.lookup["루트"]) == "상위요인"


# ── 직속/후손이 갈릴 때 알린다 ────────────────────────────────────────


def test_relations_flag_the_two_readings_when_they_differ(snapshot) -> None:
    result = get_relations(snapshot, name="중간", relation="children")

    assert result["also"] == {
        "direct_children": 2,
        "all_descendants": 4,
        "note": "직속 자식 수와 전체 후손 수가 다르다. 어느 쪽을 말하는지 밝혀야 한다.",
    }


def test_relations_stay_quiet_when_the_two_readings_agree(snapshot) -> None:
    """잎A는 직속 1, 후손 1이다. 같으면 밝힐 것이 없다."""

    assert "also" not in get_relations(snapshot, name="잎A", relation="children")


# ── 형제 ──────────────────────────────────────────────────────────────


def test_siblings_stay_inside_one_instrument(snapshot) -> None:
    """부모가 없는 필기 루트와 영상 루트가 서로 형제가 되면 안 된다."""

    written_roots = get_relations(snapshot, name="루트", relation="siblings")

    assert written_roots["count"] == 0


def test_siblings_share_a_parent(snapshot) -> None:
    result = get_relations(snapshot, name="중간", relation="siblings")

    assert [item["name"] for item in result["items"]] == ["다른중간"]


# ── 집계 ──────────────────────────────────────────────────────────────


def test_count_without_criteria_counts_everything(snapshot) -> None:
    assert count_competencies(snapshot)["count"] == 8


def test_count_echoes_the_criteria_for_later_verification(snapshot) -> None:
    """4단계 judge가 '이 숫자는 무엇을 센 것인가'를 대조할 수 있어야 한다."""

    result = count_competencies(snapshot, level="L3")

    assert result["count"] == 2
    assert result["criteria"] == {"level": "L3"}
    assert result["tier"] == "하위요인"


def test_count_filters_combine(snapshot) -> None:
    assert count_competencies(snapshot, instrument="영상면접")["count"] == 1
    assert count_competencies(snapshot, analysis_included=True)["count"] == 2
    assert count_competencies(snapshot, level="L3", analysis_included=True)["count"] == 2
    assert count_competencies(snapshot, level="L4", analysis_included=True)["count"] == 0


# ── 미등록 이름 ───────────────────────────────────────────────────────


def test_unknown_name_raises_rather_than_counting_zero(snapshot) -> None:
    """0을 돌려주면 '자식이 없다'와 '그런 역량이 없다'를 구분할 수 없다."""

    with pytest.raises(UnknownCompetencyError, match="등록된 이름이나 별칭이 아닙니다"):
        get_relations(snapshot, name="없는역량", relation="children")


def test_counting_flags_when_the_overall_score_is_mixed_in() -> None:
    """`역검 종합점수`는 개별 역량이 아니다.

    레지스트리가 종합점수를 다른 항목과 같은 포맷으로 담고 있어, 구분 없이
    한 수만 주면 답변이 "역량 62개"라고 말하게 된다.  실제 역량은 61개다 --
    10단계에서 확정한 제품 결정이며, 숫자를 박지 않고 level에서 유도한다.
    """

    document = _document()
    document["items"].append(_item("w:all", "역검 종합점수", level="overall"))
    document["validation"]["counts"]["total"] = len(document["items"])
    snapshot = build_registry_snapshot(
        {
            "id": 4,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(document["items"]),
        }
    )

    result = count_competencies(snapshot)

    assert result["count"] == 9
    assert result["breakdown"]["competencies"] == 8
    assert result["breakdown"]["overall_scores"] == 1
    assert "61" not in result["breakdown"]["note"]  # 숫자를 박지 않는다


def test_counting_stays_quiet_when_no_overall_score_is_involved(snapshot) -> None:
    assert "breakdown" not in count_competencies(snapshot, level="L3")
    assert "breakdown" not in count_competencies(snapshot)
