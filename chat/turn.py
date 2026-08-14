"""한 턴 — 작성, 검수, 복구.

    작성 LLM → 검수 모델 → (어긋나면 재작성 1회) → 폴백

토큰은 1차 시도에서만 흘린다.  검수는 답변 전체를 읽으므로 접두사를 판정할
수 없고, 그래서 흘리는 순간 검수 전 텍스트가 화면에 보인다(3.5 개정).
재작성이 필요해지면 흘리지 않고 완성한 뒤 한 번에 교체한다.

확정되고 저장되는 것은 검수를 통과한 텍스트뿐이다.  화면에 잠시 보이는 것과
대화에 남는 것이 다를 수 있다는 뜻이고, 그 맞바꿈은 반려율이 낮다는 측정에
근거한다(REBUILD_PLAN 3.5).

재작성은 같은 프롬프트로 다시 부르지 않는다.  그러면 또 틀린다.  검수 모델이
무엇이 어긋났는지 문장으로 알려주고 그것을 지시로 넘긴다.  그래도 실패하면
지어내지 않고 못 만들었다고 말한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from chat.answer import AnswerResult, answer
from chat.judge import GroundingVerdict, check_grounding
from chat.models import MESSAGE_LIMIT, ModelRole
from chat.registry import RegistrySnapshot
from chat.scope import AppealResult, appeal, looks_like_refusal

FALLBACK_EMPTY = (
    "요청하신 내용을 등록된 정보로 확인하지 못했습니다. "
    "역량 이름을 알려주시면 등록된 정의와 위계를 안내해 드리겠습니다."
)


@dataclass
class TurnResult:
    text: str
    streamed: str = ""
    verdict: GroundingVerdict | None = None
    attempts: int = 0
    used_fallback: bool = False
    appealed: AppealResult | None = None
    answers: list[AnswerResult] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return round(sum(item.seconds for item in self.answers), 2)

    @property
    def usage(self) -> dict[str, int]:
        totals = {"input": 0, "output": 0, "cached": 0}
        for item in self.answers:
            for key, value in item.usage.items():
                totals[key] += value
        return totals


def _trimmed(messages: Sequence[Any]) -> list[Any]:
    """대화만 자르고 시스템 메시지는 남긴다.

    맨 앞을 통째로 자르면 레지스트리 전문이 사라진다.  ``answer_turn``이 이미
    히스토리를 상한까지 줄인 뒤 시스템 메시지와 질문을 붙이므로, 여기서 다시
    앞에서부터 자르면 상한을 넘긴 만큼 시스템 메시지가 먼저 밀려난다 --
    주고받기 여섯 번이면 레지스트리 없이 답하게 된다.
    """

    system = [item for item in messages if getattr(item, "type", "") == "system"]
    rest = [item for item in messages if getattr(item, "type", "") != "system"]
    return [*system, *rest[-MESSAGE_LIMIT[ModelRole.ANSWER] :]]


def render_fallback(verdict: GroundingVerdict | None) -> str:
    """두 번 실패하면 지어내지 않고 못 만들었다고 말한다.

    이전에는 검수가 확정한 사실을 나열했는데, 질문과 무관한 사실이 나가고
    내부 라벨이 그대로 노출되었다(``- {'level': 'L3'} 항목 수: 30``).  검수
    모델은 참값 목록을 만들지 않으므로 나열할 것도 없고, 질문에 답하지 못했다는
    사실을 그대로 말하는 편이 정직하다.
    """

    return FALLBACK_EMPTY


def run_turn(
    answer_model: Any,
    judge_model: Any,
    snapshot: RegistrySnapshot,
    messages: Sequence[Any],
    *,
    appeal_model: Any = None,
    max_attempts: int = 2,
    on_delta: Callable[[str], None] | None = None,
) -> TurnResult:
    """답변 하나를 만들고, 검수하고, 거절이면 한 번 항소한다.

    ``on_delta``는 **1차 시도에만** 붙는다.  재생성과 항소 재작성은 흘리지
    않는다 -- 다 쓰인 답이 통째로 다른 답으로 바뀌는 것을 보는 편이 한 줄
    정정보다 훨씬 혼란스럽다.  눈에 보이는 교체는 최대 한 번이다.

    흘려보낸 텍스트는 ``streamed``에 남는다.  최종본과 다르면 호출부가
    교체 신호를 보낼 수 있다.
    """

    from langchain_core.messages import HumanMessage

    conversation = _trimmed(messages)
    result = TurnResult(text="")

    for attempt in range(1, max_attempts + 1):
        streaming = on_delta if attempt == 1 else None
        written = answer(answer_model, snapshot, conversation, streaming)
        result.answers.append(written)
        result.attempts = attempt
        result.text = written.text
        if attempt == 1:
            result.streamed = written.text

        verdict = check_grounding(
            judge_model,
            snapshot,
            written.text,
            written.tool_calls,
        )
        result.verdict = verdict
        if verdict.ok:
            break

        if attempt < max_attempts:
            # 참값을 사실로 주입한다. 같은 프롬프트로 다시 부르면 또 틀린다.
            conversation = [*messages, HumanMessage(content=verdict.retry_hint())]
    else:
        result.used_fallback = True
        result.text = render_fallback(result.verdict)
        return result

    # 항소심은 **거절에만** 건다. 통과한 답변은 다시 보지 않는다 -- 거짓
    # 수용은 grounding이 받치고, 무방비인 것은 거짓 거절뿐이다 (3.3).
    if appeal_model is None or not looks_like_refusal(result.verdict):
        return result

    verdict_of_appeal = appeal(
        appeal_model,
        snapshot,
        messages,
        message_limit=MESSAGE_LIMIT[ModelRole.APPEAL],
    )
    result.appealed = verdict_of_appeal
    if not verdict_of_appeal.overturned:
        return result

    # 뒤집혔다. 무엇을 찾아야 하는지 들고 다시 쓴다.
    retried = answer(
        answer_model,
        snapshot,
        [*conversation, HumanMessage(content=verdict_of_appeal.retry_hint())],
    )
    result.answers.append(retried)
    result.attempts += 1

    retried_verdict = check_grounding(
        judge_model, snapshot, retried.text, retried.tool_calls
    )
    # 다시 쓴 답이 검수를 통과할 때만 채택한다. 실패하면 원래 거절을 남긴다 --
    # 거절은 안전한 상태이고, 근거 없는 답변으로 바꾸는 것은 개선이 아니다.
    if retried_verdict.ok:
        result.text = retried.text
        result.verdict = retried_verdict
    return result
