"""스코프 항소심 — 거절에만, 재판정이 아니라 구성 과제로.

관측된 실패는 한 방향뿐이다.  역량 관련 질문인데 거절되거나 되물어진다.
반대 방향(무관한 질문의 수용)은 값이 싸다 -- 레지스트리 근거가 없으니
grounding judge에서 걸린다.  **거짓 수용은 이미 다른 층이 막고, 거짓 거절만
무방비**다.  그래서 이 층은 심사위원이 아니라 항소심이다.

상관 오류를 깨는 것이 두 번째 설계 요점이다.  "이 거절이 맞았나?"라고 물으면
같은 모델이 같은 대화를 보고 같은 답을 한다.  대신 구성 과제를 준다.

    이 질문에 답하려면 레지스트리에서 무엇을 찾아야 하는가?

구체적인 조회 대상을 대면 거절이 틀린 것이고, 못 대면 맞은 것이다.  판정이
아니라 구성이므로 실패 방식이 달라진다.  그리고 **코드가 그 이름이 실제로
등록되어 있는지 확인한다** -- 항소심이 이름을 지어내면 뒤집히지 않는다.

REBUILD_PLAN 3.3 참고.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from chat.judge import GroundingVerdict
from chat.registry import RegistrySnapshot

APPEAL_SYSTEM_PROMPT = """\
너는 역량 레지스트리 챗봇이 답하지 못한 질문을 다시 본다.

판정하지 않는다. 다음 하나만 답한다.

    이 질문에 답하려면 레지스트리에서 무엇을 찾아야 하는가?

찾을 대상이 분명하면 `lookup`에 **등록된 역량 이름 하나**를 적는다. 등록
이름 목록이 주어지므로 거기 있는 이름만 적는다. 비슷해 보인다고 목록에 없는
이름을 만들지 않는다.

찾을 것이 없으면 `lookup`을 비운다. 날짜·시각·날씨·뉴스·일반 개념·코딩·개인
평가·채용/의료/법률/금융 판단은 레지스트리에 없다.

대화가 함께 주어진다. 지시대명사("그거", "아까 그것")가 앞선 발화의 역량을
가리키면 그 역량 이름을 적는다. 다만 앞에서 역량을 이야기했다는 사실만으로
무관한 질문을 관련 있다고 하지 않는다 — 지금 이 질문이 그 역량에 대해
묻고 있어야 한다.
"""

APPEAL_SCHEMA: dict[str, Any] = {
    "title": "scope_appeal",
    "description": "답하려면 무엇을 찾아야 하는지만 적는다. 판정하지 않는다.",
    "type": "object",
    "properties": {
        "lookup": {
            "type": ["string", "null"],
            "description": "찾아야 할 등록 역량 이름. 없으면 null.",
        },
        "reason": {"type": "string", "description": "한 문장."},
    },
    "required": ["lookup", "reason"],
    "additionalProperties": False,
}


@dataclass
class AppealResult:
    overturned: bool
    lookup: str | None = None
    reason: str = ""
    called: bool = False

    def retry_hint(self) -> str:
        return (
            f"이 질문은 등록된 역량 '{self.lookup}'과(와) 관련이 있다. "
            "범위 밖이라고 답하지 말고, 등록된 정보로 답한다."
        )


def looks_like_refusal(verdict: GroundingVerdict | None) -> bool:
    """등록 개체를 하나도 주장하지 않은 답변을 실질적 거절로 본다.

    어휘로 판별하지 않는 것이 핵심이다.  "죄송"·"answer할 수 없"을 세기
    시작하면 `scope_response.py`의 500줄이 다시 자란다(REBUILD_PLAN 3.6).
    grounding 결과는 스키마에서 유도되는 신호다 -- 답변이 등록된 이름이나
    수를 하나라도 단정했는가.

    인사와 기능 안내도 여기 걸린다.  해롭지 않다 -- 항소심이 "찾을 것 없음"을
    돌려주고 아무 일도 일어나지 않는다.  비용은 호출 한 번이다.
    """

    if verdict is None:
        return False
    return verdict.checked_names == 0 and verdict.checked_numbers == 0


def _conversation_text(messages: Sequence[Any], limit: int) -> str:
    """가공되지 않은 최근 대화 원문.

    입력 LLM이 요약한 맥락을 주면 그 오판이 전제로 딸려 들어가 상관 오류가
    뒷문으로 돌아온다(3.8).
    """

    lines: list[str] = []
    for message in list(messages)[-limit:]:
        role = getattr(message, "type", "")
        if role not in {"human", "ai"}:
            continue
        speaker = "사용자" if role == "human" else "챗봇"
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"{speaker}: {content.strip()}")
    return "\n".join(lines)


def appeal(
    appeal_model: Any,
    snapshot: RegistrySnapshot,
    messages: Sequence[Any],
    *,
    message_limit: int,
) -> AppealResult:
    """거절을 다시 본다. 뒤집을지는 코드가 정한다."""

    from langchain_core.messages import HumanMessage, SystemMessage

    payload = {
        "conversation": _conversation_text(messages, message_limit),
        "registered_names": list(snapshot.canonical_names),
    }
    structured = appeal_model.with_structured_output(APPEAL_SCHEMA)
    judged = structured.invoke(
        [
            SystemMessage(content=APPEAL_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    ) or {}

    lookup = (judged.get("lookup") or "").strip()
    reason = str(judged.get("reason") or "")

    # 항소심이 이름을 지어내면 뒤집지 않는다.  판정은 코드가 한다.
    if lookup and lookup in snapshot.lookup:
        return AppealResult(overturned=True, lookup=lookup, reason=reason, called=True)
    return AppealResult(overturned=False, lookup=None, reason=reason, called=True)
