"""chat.judge — 답변을 분해하지 않는 검수 모델.

이전 구조는 "LLM은 주장을 옮기고 참·거짓은 코드가 정한다"를 지키려고 답변을
숫자 스팬과 원자 질의로 번역했다.  그 번역이 31회 실행에서 지적 11건을 냈고
거의 전부가 맞는 답을 버린 것이었다.  파생값(합·단계 수·부분 집계)을 원자
질의로 옮길 수 없다는 것이 원인이다.

**그래서 이 층은 판정 필드를 갖는다.**  코드가 확정하던 성질을 잃었고, 그
대가로 번역이 사라졌다.  REBUILD_PLAN 3.9가 이 층에도 적용된다 -- 참 양성을
댈 수 없으면 정교화하지 않고 제거한다.

여기서 지키는 것은 배선이다. 실제 판정 품질은 `evals/spike.py`가 잰다.
"""

from __future__ import annotations

import pytest

from chat.judge import (
    REVIEW_SCHEMA,
    GroundingVerdict,
    check_grounding,
    confirmed_counts,
    review_system_message,
)
from chat.registry import build_registry_snapshot

SOURCE_HASH = "e" * 64


def _item(item_id: str, name: str, *, level: str) -> dict:
    return {
        "id": item_id,
        "name": name,
        "aliases": [],
        "instrument": "written_competency",
        "instrument_label": "필기 역량검사",
        "level": level,
        "path": [name],
        "parent_name": None,
        "parent_id": None,
        "children": [],
        "children_ids": [],
        "definition": f"{name} 정의",
        "definition_status": "explicit",
        "analysis_included": level == "L3",
        "notes": [],
        "source_section": "테스트",
    }


@pytest.fixture
def snapshot():
    items = [
        _item("w:parent", "부모", level="L2"),
        _item("w:a", "잎A", level="L3"),
        _item("w:b", "잎B", level="L3"),
    ]
    return build_registry_snapshot(
        {
            "id": 9,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": {
                "schema_version": "1.0",
                "source": {
                    "file": "synthetic.md",
                    "sha256": SOURCE_HASH,
                    "encoding": "utf-8",
                },
                "rules": {"synthetic": "합성 레지스트리 규칙"},
                "validation": {"status": "passed", "counts": {"total": len(items)}},
                "items": items,
            },
            "item_count": len(items),
        }
    )


class _ScriptedReviewer:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.seen: list = []

    def with_structured_output(self, schema):  # noqa: ANN001
        self.schema = schema
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.seen = list(messages)
        return self._payload


# ── 배선 ─────────────────────────────────────────────────────────────


def test_a_grounded_review_passes(snapshot) -> None:
    reviewer = _ScriptedReviewer(
        {"grounded": True, "reason": "", "uses_registry": True}
    )

    verdict = check_grounding(reviewer, snapshot, "잎A는 하위요인입니다.")

    assert verdict.ok
    assert verdict.judge_called
    assert verdict.findings == []


def test_a_rejection_carries_the_reviewer_sentence(snapshot) -> None:
    """내부 라벨을 조립하지 않는다.

    이전 구조는 `{'level': 'L3'} 항목 수는 30인데…`를 재생성 힌트로 넘겼고,
    그 문자열이 프로덕션에서 사용자 답변으로 새어 나갔다.
    """

    reviewer = _ScriptedReviewer(
        {
            "grounded": False,
            "reason": "하위요인은 2개인데 답변은 7개라고 했다.",
            "uses_registry": True,
        }
    )

    verdict = check_grounding(reviewer, snapshot, "하위요인은 7개입니다.")

    assert not verdict.ok
    assert verdict.findings[0].detail == "하위요인은 2개인데 답변은 7개라고 했다."
    assert "하위요인은 2개인데" in verdict.retry_hint()
    assert "{" not in verdict.retry_hint()


def test_a_rejection_without_a_reason_still_says_something(snapshot) -> None:
    reviewer = _ScriptedReviewer(
        {"grounded": False, "reason": "  ", "uses_registry": True}
    )

    verdict = check_grounding(reviewer, snapshot, "무언가")

    assert not verdict.ok
    assert verdict.findings[0].detail.strip()


def test_uses_registry_flows_through_for_the_refusal_signal(snapshot) -> None:
    """항소심이 이 값으로 거절을 감지한다 (chat.scope)."""

    reviewer = _ScriptedReviewer(
        {"grounded": True, "reason": "", "uses_registry": False}
    )

    assert not check_grounding(reviewer, snapshot, "날씨는 답할 수 없습니다.").uses_registry


def test_blank_answer_skips_the_reviewer(snapshot) -> None:
    class _NeverCalled:
        def with_structured_output(self, schema):  # noqa: ANN001
            raise AssertionError("검사할 것이 없으면 부르지 않는다")

    assert check_grounding(_NeverCalled(), snapshot, "   ").ok


# ── 검수 모델이 보는 것 ───────────────────────────────────────────────


def test_the_reviewer_sees_the_answer_and_nothing_else_from_the_turn(snapshot) -> None:
    """조회 내역을 주면 답변 대신 그것을 읽는다는 것이 이전 구조에서 관측됐다."""

    reviewer = _ScriptedReviewer({"grounded": True, "reason": "", "uses_registry": True})

    check_grounding(reviewer, snapshot, "답변 본문", [object(), object()])

    assert reviewer.seen[-1].content == "답변 본문"


def test_confirmed_counts_are_supplied_so_the_reviewer_need_not_count(snapshot) -> None:
    """프로토타입에서 62개를 눈으로 세어 다르다고 답한 적이 있다."""

    counts = confirmed_counts(snapshot)

    assert "L3=2" in counts
    assert "필기 역량검사=3" in counts


def test_confirmed_counts_separate_competencies_from_the_overall_score(snapshot) -> None:
    """`역검 종합점수`는 개별 역량이 아니다.

    레지스트리가 종합점수를 다른 항목과 같은 포맷으로 담고 있어 구분 없이
    세면 62가 나온다.  실제 역량은 61개다 -- 10단계에서 확정한 제품 결정이며,
    숫자를 박지 않고 level 값에서 유도한다.
    """

    counts = confirmed_counts(_with_overall())

    assert "역량=3" in counts
    assert "종합점수=1" in counts
    assert "전체 항목(역량+종합점수)=4" in counts


def _with_overall():
    items = [
        _item("w:parent", "부모", level="L2"),
        _item("w:a", "잎A", level="L3"),
        _item("w:b", "잎B", level="L3"),
        _item("w:all", "역검 종합점수", level="overall"),
    ]
    return build_registry_snapshot(
        {
            "id": 10,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": {
                "schema_version": "1.0",
                "source": {
                    "file": "synthetic.md",
                    "sha256": SOURCE_HASH,
                    "encoding": "utf-8",
                },
                "rules": {"synthetic": "합성 레지스트리 규칙"},
                "validation": {"status": "passed", "counts": {"total": len(items)}},
                "items": items,
            },
            "item_count": len(items),
        }
    )


def test_the_registry_leads_the_review_prompt(snapshot) -> None:
    message = review_system_message(snapshot)

    assert message.startswith("<registry>")
    assert message.index("</registry>") < message.index("<counts>")


# ── 이 층이 무엇을 포기했는지 ─────────────────────────────────────────


def test_the_schema_now_carries_a_verdict_and_that_is_deliberate() -> None:
    """3.2의 "판정 필드를 두지 않는다"를 의도적으로 뒤집었다.

    그 규칙이 강제한 번역이 오탐 11건을 만들었고, 규칙 자체는 측정된 적이
    없었다.  대신 3.9를 이 층에 적용한다 -- 참 양성을 못 대면 제거한다.
    이 테스트는 그 맞바꿈을 기록으로 남긴다.
    """

    assert "grounded" in REVIEW_SCHEMA["properties"]
    assert REVIEW_SCHEMA["properties"]["grounded"]["type"] == "boolean"


def test_verdict_defaults_are_safe() -> None:
    """검수를 부르지 않은 턴이 통과로 보이되 거절로 오인되지 않아야 한다."""

    verdict = GroundingVerdict(ok=True)

    assert not verdict.judge_called
    assert not verdict.uses_registry
