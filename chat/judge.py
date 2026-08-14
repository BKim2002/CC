"""검수 모델 — 답변을 분해하지 않고 레지스트리와 대조한다.

이전 구조는 답변을 코드가 판정할 수 있는 형태로 **번역**했다.  숫자 스팬을
뽑고, judge가 각 숫자를 ``count(level=…)`` 같은 원자 질의로 옮기고, 코드가 그
질의를 실행해 비교했다.  "LLM은 주장을 옮기고 참·거짓은 코드가 정한다"는
원칙을 지키려면 그 번역이 필요했다.

**그 번역이 위험의 원천이었다.**  31회 실행에서 나온 지적 11건 중 거의 전부가
검증 실패가 아니라 번역 실패다 -- 성실성의 자식 3을 L3 전체 30으로, 참인
53·9·62를 "확인 불가"로, 적응 예측 아래 L3 9개를 후손 12개로 옮겼다.  답변은
맞았고 옮기는 과정이 틀렸다.

원인은 하나다.  답변의 숫자는 대부분 **파생값**이다 -- 합(62 = 53 + 9), 단계
수("2단계 구조"), 부분 집계("적응 예측 아래 L3만 9개").  판정 어휘는 원자
질의뿐이었고, 파생값의 종류는 표현 방식과 마찬가지로 무한하다.  갈래를 하나씩
더하는 것은 이전 구현이 무너진 것과 같은 모양이라 그 길로 가지 않는다.

그래서 번역을 없앴다.  검수 모델이 레지스트리 전문과 답변을 그대로 읽는다.
파생값을 원자 질의로 옮길 필요가 없으므로 옮기다 틀릴 일도 없다.

**대가를 분명히 한다.**  이 층은 ``grounded`` 판정을 모델이 내린다.  코드가
확정하던 성질을 잃었고, 모델이 자기 답을 통과시킬 가능성이 열려 있다.  그래서
``docs/REBUILD_PLAN.md`` 3.9가 이 층에도 그대로 적용된다 -- 실행 기록에서
실제로 잡아낸 오류를 댈 수 없으면 정교화하지 않고 제거한다.

    python evals/spike.py --audit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from chat.prompt import render_registry
from chat.registry import RegistrySnapshot

REVIEW_INSTRUCTIONS = """\
너는 역량 레지스트리 챗봇의 답변을 검수한다. 위 <registry>가 유일한 사실
근거다. 답변이 레지스트리와 어긋나는 곳이 있는지만 본다.

어긋난다는 것은 다음 중 하나다.

- 등록되지 않은 이름을 등록된 역량인 것처럼 정의하거나 위계에 놓았다
- 숫자가 레지스트리에서 실제로 세어지는 값과 다르다
- 레지스트리에 없는 방법·절차·해석 기준·권고를 사실처럼 제시했다
- 정의나 관계를 레지스트리와 다르게 말했다

어긋나지 않는 것:

- 답변이 짧거나 불친절하거나 목록을 생략한 것. **표현은 검수 대상이 아니다.**
- 답변이 "등록되어 있지 않다"고 밝힌 이름. 올바른 처리다.
- 범위 밖 질문을 거절한 것.
- 레지스트리에 근거가 있고 답변이 그 근거를 정확히 옮긴 경우, 더 자세히 쓸 수
  있었다는 이유로 반려하지 않는다.

개수를 셀 때는 <counts>의 확정값을 쓴다. 거기 없는 개수만 <registry>의 부모·
자식 관계로 확인한다. 눈대중으로 세어 답변과 다르다고 하지 않는다.

`grounded`가 false이면 `reason`에 **무엇이 어떻게 다른지** 한 문장으로 적는다.
그 문장이 그대로 재작성 지시가 되므로, 참값을 알면 함께 적는다.

`uses_registry`는 답변이 등록된 역량 정보를 실제로 전달했는지다. 인사·거절·
되묻기처럼 등록 사실을 하나도 말하지 않았으면 false다.
"""

REVIEW_SCHEMA: dict[str, Any] = {
    "title": "grounding_review",
    "description": "답변이 레지스트리와 어긋나는지 본다.",
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "reason": {"type": "string"},
        "uses_registry": {"type": "boolean"},
    },
    "required": ["grounded", "reason", "uses_registry"],
    "additionalProperties": False,
}

# 검수 모델이 직접 세지 않게 확정값을 준다.  프로토타입에서 62개를 눈으로 세어
# 61이라 답한 적이 있는데, 공교롭게도 그 61은 '역검 종합점수'를 뺀 수라 오산이
# 아니라 정당한 구분이었다.  그 모호성은 별도 과제로 남기고, 여기서는 코드가
# 확정할 수 있는 값을 확정해 준다.
_COUNT_LEVELS = ("overall", "L1", "L2", "L3", "L4", "factor", "item")


@dataclass
class Finding:
    kind: str
    detail: str
    expected: Any = None
    actual: Any = None


@dataclass
class GroundingVerdict:
    ok: bool
    findings: list[Finding] = field(default_factory=list)
    uses_registry: bool = False
    judge_called: bool = False

    def retry_hint(self) -> str:
        """재생성 프롬프트에 넣을 지시.

        검수 모델의 문장을 그대로 쓴다.  이전 구조에서는 내부 라벨을 조립해
        넘겼고(``{'level': 'L3'} 항목 수는 30인데…``), 그 문자열이 프로덕션에서
        답변으로 새어 나갔다.
        """

        lines = [
            "이전 답변에서 아래가 레지스트리와 어긋난다. 그 부분만 고치고",
            "나머지는 유지한다. 원래 질문에 답하는 것이 우선이며 답의 범위를",
            "좁히지 않는다.",
            "",
        ]
        lines.extend(f"- {finding.detail}" for finding in self.findings)
        return "\n".join(lines)


def confirmed_counts(snapshot: RegistrySnapshot) -> str:
    """코드가 확정할 수 있는 개수를 검수 모델에 준다."""

    from chat.tools import count_competencies

    parts = [f"전체 항목={count_competencies(snapshot)['count']}"]
    parts.extend(
        f"{level}={count_competencies(snapshot, level=level)['count']}"
        for level in _COUNT_LEVELS
    )
    parts.extend(
        f"{label}={count_competencies(snapshot, instrument=label)['count']}"
        for label in sorted(
            {item["instrument_label"] for item in snapshot.document["items"]}
        )
    )
    return "<counts>\n" + " · ".join(parts) + "\n</counts>"


def review_system_message(snapshot: RegistrySnapshot) -> str:
    """레지스트리를 맨 앞에 둔다. 작성 쪽과 같은 이유로 캐시 접두가 안정된다."""

    return "\n\n".join(
        [render_registry(snapshot), confirmed_counts(snapshot), REVIEW_INSTRUCTIONS]
    )


def check_grounding(
    judge_model: Any,
    snapshot: RegistrySnapshot,
    answer_text: str,
    tool_calls: Sequence[Any] = (),
) -> GroundingVerdict:
    """답변 하나를 검수한다.

    ``tool_calls``는 받지만 쓰지 않는다.  호출부 계약을 유지하기 위한 것이고,
    검수 모델은 답변과 레지스트리만 본다 -- 조회 내역을 주면 답변 대신 그것을
    읽는다는 것이 이전 구조에서 이미 관측되었다.
    """

    from langchain_core.messages import HumanMessage, SystemMessage

    if not answer_text.strip():
        return GroundingVerdict(ok=True)

    structured = judge_model.with_structured_output(REVIEW_SCHEMA)
    reviewed = structured.invoke(
        [
            SystemMessage(content=review_system_message(snapshot)),
            HumanMessage(content=answer_text),
        ]
    ) or {}

    grounded = bool(reviewed.get("grounded", True))
    reason = str(reviewed.get("reason") or "").strip()
    verdict = GroundingVerdict(
        ok=grounded,
        uses_registry=bool(reviewed.get("uses_registry", False)),
        judge_called=True,
    )
    if not grounded:
        verdict.findings.append(
            Finding(
                kind="not_grounded",
                detail=reason or "레지스트리와 어긋나는 내용이 있다.",
            )
        )
    return verdict
