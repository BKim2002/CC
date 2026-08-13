"""chat.scope — 거절에만 거는 비대칭 항소심.

세 가지를 지킨다.

1. **거절 감지가 어휘에 기대지 않는다.**  "죄송"·"답할 수 없"을 세기 시작하면
   `scope_response.py`의 500줄이 다시 자란다(REBUILD_PLAN 3.6).  등록 개체를
   하나도 주장하지 않았다는 grounding 결과를 신호로 쓴다.
2. **항소심이 판정하지 않는다.**  무엇을 찾아야 하는지만 말하고, 그 이름이
   실제로 등록되어 있는지는 코드가 확인한다.  지어낸 이름으로는 뒤집히지
   않는다.
3. **통과한 답변은 다시 보지 않는다.**  거짓 수용은 grounding이 받치므로
   항소심은 거절에만 건다(3.3).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chat.judge import GroundingVerdict
from chat.registry import build_registry_snapshot
from chat.scope import _conversation_text, appeal, looks_like_refusal

SOURCE_HASH = "a1" * 32


@pytest.fixture
def snapshot():
    items = [
        {
            "id": "w:one",
            "name": "성실성",
            "aliases": ["성실"],
            "instrument": "written_competency",
            "instrument_label": "필기 역량검사",
            "level": "L2",
            "path": ["성실성"],
            "parent_name": None,
            "parent_id": None,
            "children": [],
            "children_ids": [],
            "definition": "성실성 정의",
            "definition_status": "explicit",
            "analysis_included": True,
            "notes": [],
            "source_section": "테스트",
        }
    ]
    return build_registry_snapshot(
        {
            "id": 2,
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


class _ScriptedAppeal:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.seen: list = []

    def with_structured_output(self, schema):  # noqa: ANN001
        self._schema = schema
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.seen = list(messages)
        return self._payload


# ── 거절 감지 ────────────────────────────────────────────────────────


def test_an_answer_asserting_nothing_registered_reads_as_a_refusal() -> None:
    assert looks_like_refusal(GroundingVerdict(ok=True, checked_names=0, checked_numbers=0))


def test_an_answer_that_asserted_a_registered_name_is_not_a_refusal() -> None:
    assert not looks_like_refusal(GroundingVerdict(ok=True, checked_names=1))


def test_an_answer_that_asserted_a_number_is_not_a_refusal() -> None:
    """b02처럼 거절하면서도 등록 정보를 준 답변은 항소 대상이 아니다."""

    assert not looks_like_refusal(GroundingVerdict(ok=True, checked_numbers=1))


def test_missing_verdict_does_not_trigger_an_appeal() -> None:
    assert not looks_like_refusal(None)


# ── 코드가 뒤집는다 ──────────────────────────────────────────────────


def test_a_registered_lookup_overturns_the_refusal(snapshot) -> None:
    model = _ScriptedAppeal({"lookup": "성실성", "reason": "등록된 역량이다."})

    result = appeal(model, snapshot, [], message_limit=6)

    assert result.overturned
    assert result.lookup == "성실성"


def test_an_alias_also_overturns(snapshot) -> None:
    model = _ScriptedAppeal({"lookup": "성실", "reason": "별칭이다."})

    assert appeal(model, snapshot, [], message_limit=6).overturned


def test_an_invented_name_does_not_overturn(snapshot) -> None:
    """항소심이 그럴듯한 이름을 지어내도 코드가 막는다."""

    model = _ScriptedAppeal({"lookup": "혁신성", "reason": "관련 있어 보인다."})

    result = appeal(model, snapshot, [], message_limit=6)

    assert not result.overturned
    assert result.lookup is None


def test_no_lookup_upholds_the_refusal(snapshot) -> None:
    model = _ScriptedAppeal({"lookup": None, "reason": "레지스트리에 없다."})

    assert not appeal(model, snapshot, [], message_limit=6).overturned


# ── 맥락은 원문으로, 그리고 짧게 ──────────────────────────────────────


def test_the_appeal_sees_raw_conversation_not_a_summary(snapshot) -> None:
    """요약을 주면 요약한 쪽의 오판이 전제로 딸려 들어간다 (3.8)."""

    model = _ScriptedAppeal({"lookup": None, "reason": "없다."})
    messages = [HumanMessage(content="성실성이 뭐야?"), AIMessage(content="정의입니다.")]

    appeal(model, snapshot, messages, message_limit=6)

    payload = model.seen[-1].content
    assert "사용자: 성실성이 뭐야?" in payload
    assert "챗봇: 정의입니다." in payload


def test_conversation_is_trimmed_to_the_limit() -> None:
    messages = [HumanMessage(content=f"질문{index}") for index in range(10)]

    text = _conversation_text(messages, 3)

    assert "질문9" in text
    assert "질문6" not in text


def test_system_messages_are_left_out_of_the_conversation(snapshot) -> None:
    from langchain_core.messages import SystemMessage

    text = _conversation_text(
        [SystemMessage(content="레지스트리 전문"), HumanMessage(content="안녕")], 6
    )

    assert "레지스트리 전문" not in text
    assert "사용자: 안녕" in text
