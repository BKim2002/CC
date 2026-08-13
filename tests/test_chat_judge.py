"""chat.judge — 주장 추출은 judge, 판정은 코드.

이 파일이 지키는 계약 하나가 나머지를 지탱한다: **judge는 참·거짓을 말하지
않는다.**  스키마에 판정 필드가 없고, 코드가 레지스트리에서 직접 계산한다.
judge가 아무리 그럴듯하게 말해도 숫자가 틀리면 걸린다.

3단계에서 관측한 실패가 이 층의 표적이다 -- 모델이 도구를 부르고도 그
결과와 무관한 숫자를 답에 썼다.  도구 호출 유무는 근거가 아니다.
"""

from __future__ import annotations

import pytest

from chat.judge import (
    JUDGE_SCHEMA,
    check_grounding,
    extract_number_spans,
    verify,
)
from chat.registry import build_registry_snapshot

SOURCE_HASH = "e" * 64


def _item(item_id: str, name: str, *, level: str, children: list[str] | None = None) -> dict:
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
        "children": [child.split(":")[-1] for child in (children or [])],
        "children_ids": children or [],
        "definition": f"{name} 정의",
        "definition_status": "explicit",
        "analysis_included": level == "L3",
        "notes": [],
        "source_section": "테스트",
    }


@pytest.fixture
def snapshot():
    items = [
        _item("w:parent", "부모", level="L2", children=["w:a", "w:b"]),
        _item("w:a", "잎A", level="L3"),
        _item("w:b", "잎B", level="L3"),
    ]
    document = {
        "schema_version": "1.0",
        "source": {"file": "synthetic.md", "sha256": SOURCE_HASH, "encoding": "utf-8"},
        "rules": {"synthetic": "합성 레지스트리 규칙"},
        "validation": {"status": "passed", "counts": {"total": len(items)}},
        "items": items,
    }
    return build_registry_snapshot(
        {
            "id": 9,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(items),
        }
    )


def _number(index: int, claim: str, **fields) -> dict:
    payload = {
        "index": index,
        "claim": claim,
        "level": None,
        "instrument": None,
        "analysis_included": None,
        "name": None,
        "relation": None,
    }
    payload.update(fields)
    return payload


# ── 스팬 추출 ────────────────────────────────────────────────────────


def test_list_ordinals_are_not_treated_as_facts() -> None:
    spans = extract_number_spans("하위요인은 2개입니다.\n1. 잎A\n2. 잎B")

    assert [span.value for span in spans] == [2]


def test_extraction_keeps_ambiguous_numbers_for_the_judge() -> None:
    """여기서 걸러내려 들면 v1의 어휘 목록이 다시 자란다 (3.6)."""

    spans = extract_number_spans("2026년 기준 하위요인 2개, 총 3개 항목")

    assert [span.value for span in spans] == [2026, 2, 3]


def test_span_context_travels_with_the_number(snapshot) -> None:
    span = extract_number_spans("부모의 자식은 2개입니다.")[0]

    assert "자식" in span.context


# ── 코드가 판정한다 ──────────────────────────────────────────────────


def test_wrong_count_is_caught_however_confident_the_judge_sounds(snapshot) -> None:
    spans = extract_number_spans("하위요인은 총 7개입니다.")
    judged = {"numbers": [_number(0, "count", level="L3")], "asserted_names": []}

    verdict = verify(snapshot, spans, judged)

    assert not verdict.ok
    assert verdict.findings[0].kind == "wrong_number"
    assert verdict.findings[0].expected == 2
    assert verdict.findings[0].actual == 7


def test_correct_count_passes_and_is_remembered_as_a_fact(snapshot) -> None:
    spans = extract_number_spans("하위요인은 총 2개입니다.")
    judged = {"numbers": [_number(0, "count", level="L3")], "asserted_names": []}

    verdict = verify(snapshot, spans, judged)

    assert verdict.ok
    assert verdict.checked_numbers == 1
    assert 2 in verdict.facts.values()


def test_relation_count_is_verified_against_the_tree(snapshot) -> None:
    spans = extract_number_spans("부모 아래에는 5개가 있습니다.")
    judged = {
        "numbers": [_number(0, "relation", name="부모", relation="children")],
        "asserted_names": [],
    }

    verdict = verify(snapshot, spans, judged)

    assert not verdict.ok
    assert verdict.findings[0].expected == 2


def test_unsupported_number_fails_even_though_nothing_contradicts_it(snapshot) -> None:
    """근거를 댈 수 없는 수량은 통과시키지 않는다 (3.2의 hallucination 갈래)."""

    spans = extract_number_spans("평균 점수는 72점입니다.")
    judged = {"numbers": [_number(0, "unsupported")], "asserted_names": []}

    verdict = verify(snapshot, spans, judged)

    assert not verdict.ok
    assert verdict.findings[0].kind == "unsupported_number"


def test_formatting_numbers_are_allowed_through(snapshot) -> None:
    spans = extract_number_spans("2026년 자료입니다.")
    judged = {"numbers": [_number(0, "not_a_claim")], "asserted_names": []}

    assert verify(snapshot, spans, judged).ok


# ── judge가 빠뜨리는 경로를 막는다 ────────────────────────────────────


def test_partial_judgement_fails_the_whole_answer(snapshot) -> None:
    """불편한 숫자를 조용히 빠뜨리고 통과시키는 경로가 없어야 한다."""

    spans = extract_number_spans("하위요인 2개, 그리고 99개의 무언가")
    judged = {"numbers": [_number(0, "count", level="L3")], "asserted_names": []}

    verdict = verify(snapshot, spans, judged)

    assert not verdict.ok
    assert verdict.findings[0].kind == "incomplete_judgement"


def test_judge_schema_has_no_verdict_field() -> None:
    """``matches: bool`` 을 두는 순간 LLM이 판정자가 된다 (3.2)."""

    fields = set(JUDGE_SCHEMA["properties"]["numbers"]["items"]["properties"])

    assert not fields & {"matches", "correct", "ok", "valid", "verdict"}


# ── 이름 ─────────────────────────────────────────────────────────────


def test_unregistered_name_asserted_as_registered_is_caught(snapshot) -> None:
    judged = {"numbers": [], "asserted_names": ["잎A", "없는역량"]}

    verdict = verify(snapshot, [], judged)

    assert not verdict.ok
    assert verdict.findings[0].kind == "unregistered_name"
    assert "없는역량" in verdict.findings[0].detail


def test_registered_names_pass(snapshot) -> None:
    verdict = verify(snapshot, [], {"numbers": [], "asserted_names": ["부모", "잎A"]})

    assert verdict.ok
    assert verdict.checked_names == 2


# ── 재생성 힌트 ──────────────────────────────────────────────────────


def test_retry_hint_carries_the_true_value(snapshot) -> None:
    """같은 프롬프트로 다시 부르면 또 틀린다 (3.2)."""

    spans = extract_number_spans("하위요인은 총 7개입니다.")
    judged = {"numbers": [_number(0, "count", level="L3")], "asserted_names": []}

    hint = verify(snapshot, spans, judged).retry_hint()

    assert "2인데" in hint
    assert "확인된 사실" in hint


# ── 모델을 부르지 않아야 할 때 ────────────────────────────────────────


class _NeverCalled:
    def with_structured_output(self, schema):  # noqa: ANN001
        raise AssertionError("검사할 것이 없으면 judge를 부르지 않는다")


def test_empty_answer_skips_the_judge(snapshot) -> None:
    assert check_grounding(_NeverCalled(), snapshot, "   ").ok
