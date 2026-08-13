"""chat.turn — 작성, 검수, 복구.

지키는 계약 셋.

1. **검증 후 공개.**  검수를 통과하지 못한 문장은 사용자에게 가지 않는다.
2. **재생성은 참값을 들고 간다.**  같은 프롬프트로 다시 부르면 또 틀린다.
3. **폴백은 항상 맞다.**  도구 결과만 나열하므로 지어낼 여지가 없다.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chat.judge import Finding, GroundingVerdict
from chat.registry import build_registry_snapshot
from chat.turn import FALLBACK_EMPTY, render_fallback, run_turn

SOURCE_HASH = "f" * 64


@pytest.fixture
def snapshot():
    items = [
        {
            "id": "w:only",
            "name": "유일",
            "aliases": [],
            "instrument": "written_competency",
            "instrument_label": "필기 역량검사",
            "level": "L3",
            "path": ["유일"],
            "parent_name": None,
            "parent_id": None,
            "children": [],
            "children_ids": [],
            "definition": "유일 정의",
            "definition_status": "explicit",
            "analysis_included": True,
            "notes": [],
            "source_section": "테스트",
        }
    ]
    return build_registry_snapshot(
        {
            "id": 1,
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
                "validation": {"status": "passed", "counts": {"total": 1}},
                "items": items,
            },
            "item_count": 1,
        }
    )


class _ScriptedModel:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.seen: list[list] = []

    def bind_tools(self, tools):  # noqa: ANN001
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.seen.append(list(messages))
        return AIMessage(content=self._texts.pop(0))


class _ScriptedJudge:
    """정해진 판정을 순서대로 돌려준다. check_grounding 대신 주입한다."""

    def __init__(self, verdicts: list[GroundingVerdict]) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0

    def __call__(self, judge_model, snapshot, text, tool_calls=()):  # noqa: ANN001
        self.calls += 1
        self.seen_tool_calls = list(tool_calls)
        return self._verdicts.pop(0)


def _ok() -> GroundingVerdict:
    return GroundingVerdict(ok=True, judge_called=True)


def _bad(detail: str = "하위요인 수는 1인데 답변은 9라고 했다.") -> GroundingVerdict:
    return GroundingVerdict(
        ok=False,
        judge_called=True,
        findings=[Finding(kind="wrong_number", detail=detail, expected=1, actual=9)],
        facts={"L3 항목 수": 1},
    )


def test_a_passing_answer_is_returned_as_written(snapshot, monkeypatch) -> None:
    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_ok()]))
    model = _ScriptedModel(["하위요인은 1개입니다."])

    result = run_turn(model, object(), snapshot, [HumanMessage(content="몇 개?")])

    assert result.text == "하위요인은 1개입니다."
    assert result.attempts == 1
    assert not result.used_fallback


def test_a_failing_answer_never_reaches_the_user(snapshot, monkeypatch) -> None:
    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_bad(), _ok()]))
    model = _ScriptedModel(["하위요인은 9개입니다.", "하위요인은 1개입니다."])

    result = run_turn(model, object(), snapshot, [HumanMessage(content="몇 개?")])

    assert "9개" not in result.text
    assert result.text == "하위요인은 1개입니다."
    assert result.attempts == 2


def test_retry_carries_the_true_value_into_the_prompt(snapshot, monkeypatch) -> None:
    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_bad(), _ok()]))
    model = _ScriptedModel(["하위요인은 9개입니다.", "하위요인은 1개입니다."])

    run_turn(model, object(), snapshot, [HumanMessage(content="몇 개?")])

    retry_prompt = model.seen[1][-1].content
    assert "1인데" in retry_prompt
    assert "L3 항목 수: 1" in retry_prompt


def test_two_failures_fall_back_to_tool_results_only(snapshot, monkeypatch) -> None:
    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_bad(), _bad()]))
    model = _ScriptedModel(["9개입니다.", "여전히 9개입니다."])

    result = run_turn(model, object(), snapshot, [HumanMessage(content="몇 개?")])

    assert result.used_fallback
    assert "9개" not in result.text
    assert "L3 항목 수: 1" in result.text


def test_judge_is_called_once_per_attempt(snapshot, monkeypatch) -> None:
    judge = _ScriptedJudge([_bad(), _ok()])
    monkeypatch.setattr("chat.turn.check_grounding", judge)
    model = _ScriptedModel(["9개", "1개"])

    run_turn(model, object(), snapshot, [HumanMessage(content="몇 개?")])

    assert judge.calls == 2


def test_fallback_without_any_verified_fact_says_so(snapshot) -> None:
    assert render_fallback(None) == FALLBACK_EMPTY
    assert render_fallback(GroundingVerdict(ok=False)) == FALLBACK_EMPTY


# ── 항소심 배선 (5단계) ──────────────────────────────────────────────


def _refusal() -> GroundingVerdict:
    """등록 개체를 하나도 주장하지 않은, 통과한 답변 = 실질적 거절."""

    return GroundingVerdict(ok=True, judge_called=True, checked_names=0, checked_numbers=0)


def _grounded() -> GroundingVerdict:
    return GroundingVerdict(ok=True, judge_called=True, checked_names=1)


class _ScriptedAppealModel:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    def with_structured_output(self, schema):  # noqa: ANN001
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.calls += 1
        return self._payload


def test_a_grounded_answer_is_never_appealed(snapshot, monkeypatch) -> None:
    """통과한 답변은 다시 보지 않는다. 거짓 수용은 grounding이 받친다 (3.3)."""

    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_grounded()]))
    appeal_model = _ScriptedAppealModel({"lookup": "유일", "reason": "관련 있다."})

    result = run_turn(
        _ScriptedModel(["유일의 정의입니다."]),
        object(),
        snapshot,
        [HumanMessage(content="유일이 뭐야?")],
        appeal_model=appeal_model,
    )

    assert appeal_model.calls == 0
    assert result.appealed is None


def test_a_refusal_that_should_have_answered_is_overturned(snapshot, monkeypatch) -> None:
    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_refusal(), _grounded()]))
    model = _ScriptedModel(["답할 수 없습니다.", "유일의 정의입니다."])

    result = run_turn(
        model,
        object(),
        snapshot,
        [HumanMessage(content="그거 뭐야?")],
        appeal_model=_ScriptedAppealModel({"lookup": "유일", "reason": "지시대명사"}),
    )

    assert result.appealed is not None and result.appealed.overturned
    assert result.text == "유일의 정의입니다."


def test_an_upheld_refusal_is_left_alone(snapshot, monkeypatch) -> None:
    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_refusal()]))

    result = run_turn(
        _ScriptedModel(["날씨는 답할 수 없습니다."]),
        object(),
        snapshot,
        [HumanMessage(content="오늘 날씨는?")],
        appeal_model=_ScriptedAppealModel({"lookup": None, "reason": "레지스트리에 없다."}),
    )

    assert result.appealed is not None and not result.appealed.overturned
    assert result.text == "날씨는 답할 수 없습니다."


def test_a_rewrite_that_fails_grounding_keeps_the_refusal(snapshot, monkeypatch) -> None:
    """거절은 안전한 상태다. 근거 없는 답변으로 바꾸는 것은 개선이 아니다."""

    failed = GroundingVerdict(ok=False, judge_called=True)
    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_refusal(), failed]))

    result = run_turn(
        _ScriptedModel(["답할 수 없습니다.", "유일은 99개입니다."]),
        object(),
        snapshot,
        [HumanMessage(content="그거 뭐야?")],
        appeal_model=_ScriptedAppealModel({"lookup": "유일", "reason": "지시대명사"}),
    )

    assert result.text == "답할 수 없습니다."


def test_without_an_appeal_model_nothing_is_appealed(snapshot, monkeypatch) -> None:
    monkeypatch.setattr("chat.turn.check_grounding", _ScriptedJudge([_refusal()]))

    result = run_turn(
        _ScriptedModel(["답할 수 없습니다."]),
        object(),
        snapshot,
        [HumanMessage(content="오늘 날씨는?")],
    )

    assert result.appealed is None
