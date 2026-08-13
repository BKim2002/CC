"""grounding judge — 주장을 뽑는 것은 judge, 참을 정하는 것은 코드.

배선의 핵심은 **judge에게 판정을 시키지 않는 것**이다.  judge 출력 스키마에
``matches: bool`` 같은 필드를 두면 그 순간 LLM이 판정자가 되고, 도구를 보지
않고도 통과시킬 수 있다.  여기서 judge는 "이 숫자가 무엇을 주장하는가"만
말하고, 그 주장이 참인지는 코드가 레지스트리에서 확인한다.

    1. 코드: 답변에서 숫자 스팬을 뽑는다 (과검출 허용, 누락만 차단)
    2. 숫자도 이름도 없으면 judge를 부르지 않는다
    3. judge: 스팬마다 주장하는 질의를, 이름마다 등록 여부 단정을 반환
    4. 코드: 질의를 실행해 값을 비교하고 판정한다
    5. 코드: 스팬 개수와 judge 응답 개수가 다르면 자동 실패

REBUILD_PLAN 3.2 참고.

3단계에서 실제로 관측한 실패가 이 층의 존재 이유다 -- 모델이 도구를 부르고도
그 결과와 무관한 숫자를 답에 썼다.  도구 호출 유무는 근거가 되지 않는다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from chat.registry import RegistrySnapshot
from chat.tools import (
    VIDEO_TIER_TERMS,
    WRITTEN_TIER_TERMS,
    UnknownCompetencyError,
    count_competencies,
    get_relations,
)

# 과검출은 괜찮고 누락만 막으면 된다.  판정은 judge가 아니라 코드가 하므로
# 무해한 숫자가 섞여도 `not_a_claim`으로 걸러진다.
_NUMBER = re.compile(r"(?<![\d.,])(\d{1,4})(?![\d])")

# 서수 목록 표지(줄 앞의 ``1.``)만 서식으로 본다.  v1 `_is_list_ordinal` 이식.
_MARKDOWN_NOISE = re.compile(r"^#{1,6}\s")


@dataclass(frozen=True)
class NumberSpan:
    """답변에 등장한 숫자 하나와 그 주변."""

    index: int
    value: int
    context: str


@dataclass
class Finding:
    """코드가 내린 판정 하나."""

    kind: str
    detail: str
    expected: Any = None
    actual: Any = None


@dataclass
class GroundingVerdict:
    ok: bool
    findings: list[Finding] = field(default_factory=list)
    checked_numbers: int = 0
    checked_names: int = 0
    judge_called: bool = False
    facts: dict[str, Any] = field(default_factory=dict)

    def retry_hint(self) -> str:
        """재생성 프롬프트에 넣을 참값.

        같은 프롬프트로 다시 부르면 또 틀린다.  3.2가 "참값을 사실로 주입해
        한 번만 재생성한다"고 한 이유다.

        지목한 것만 고치라고 명시한다.  전면 재작성으로 읽히면 맞았던 부분까지
        버린다 -- 5단계에서 `g05`가 "등록 역량 62개"를 정확히 답하고도 재생성
        뒤에 "종합점수 1개"라는 좁고 무관한 답으로 흘러갔다.
        """

        lines = [
            "이전 답변에서 아래 항목만 잘못되었다. 그 부분만 고치고 나머지는",
            "유지한다. 원래 질문에 답하는 것이 우선이며, 답의 범위를 좁히지",
            "않는다.",
            "",
            "고칠 것:",
        ]
        for finding in self.findings:
            lines.append(f"- {finding.detail}")
        if self.facts:
            lines.append("")
            lines.append("확인된 사실 (그대로 쓴다):")
            for key, value in self.facts.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)


def _is_list_ordinal(text: str, match: re.Match[str]) -> bool:
    """줄 앞의 ``1.`` 표지만 서식으로 본다 (v1 `_is_list_ordinal` 이식)."""

    line_start = text.rfind("\n", 0, match.start()) + 1
    prefix = text[line_start : match.start()]
    return (
        not prefix.strip()
        and match.end() < len(text)
        and text[match.end()] in {".", ")"}
    )


def extract_number_spans(text: str, *, window: int = 45) -> list[NumberSpan]:
    """답변 속 숫자를 빠짐없이 뽑는다.

    서수 목록 표지만 제외한다.  나머지는 전부 judge에게 보내고, 사실 주장이
    아닌 것은 judge가 ``not_a_claim``으로 표시한다.  여기서 걸러내려 들면
    v1의 어휘 목록이 다시 자라기 시작한다 (REBUILD_PLAN 3.6).
    """

    spans: list[NumberSpan] = []
    for match in _NUMBER.finditer(text):
        if _is_list_ordinal(text, match):
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        if _MARKDOWN_NOISE.match(text[line_start : line_start + 8]):
            pass  # 제목 줄의 숫자도 주장일 수 있다. 남긴다.
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        spans.append(
            NumberSpan(
                index=len(spans),
                value=int(match.group(1)),
                context=text[start:end].replace("\n", " "),
            )
        )
    return spans


# judge도 위계 용어를 알아야 한다.  "하위요인"이 L3라는 것은 level 값에서
# 유도되지 않는 도메인 어휘라, 알려주지 않으면 judge가 엉뚱한 level로 옮기고
# 코드가 그 잘못된 질의를 성실히 실행한다 -- 4단계 첫 실행에서 "하위요인"이
# factor로 매핑되어 30이 0으로 판정되었다.  도구·프롬프트와 같은 사전에서
# 만들어 셋이 갈라질 수 없게 한다.
_TIER_LINES = "\n".join(
    [
        *(f"  {term} = level {level}" for level, term in WRITTEN_TIER_TERMS.items()),
        *(f"  {term} = level {level} (영상면접)" for level, term in VIDEO_TIER_TERMS.items()),
    ]
)

JUDGE_SYSTEM_PROMPT = f"""\
너는 역량 레지스트리 챗봇의 답변을 검수한다. 판정하지 않는다 — 답변이
무엇을 주장하는지만 구조화해서 옮긴다. 참인지 거짓인지는 다른 곳에서
확인한다.

위계 용어와 level 값의 대응:

{_TIER_LINES}

주어지는 것:

- 검수 대상 답변
- 답변에서 기계적으로 추출한 숫자 목록 (서식용 숫자가 섞여 있다)
- 답변을 쓸 때 실제로 조회한 내역 (없을 수도 있다)

조회 내역은 답변이 무엇에 대해 말하는지 고르는 데 쓴다. 앞선 대화를 볼 수
없으므로, "하위요인은 3가지입니다"처럼 대상이 생략된 문장은 조회 내역의
대상을 따른다. 조회했다는 사실 자체는 근거가 되지 않는다 — 조회한 것과
다른 수를 답에 썼을 수도 있고, 그 판정은 다른 곳에서 한다.

할 일 두 가지.

**1. 숫자마다 그 숫자가 주장하는 질의를 적는다.**

  claim 값:
  - "count"        전체 또는 조건에 맞는 등록 항목 수를 주장한다
  - "relation"     특정 역량의 부모·자식·조상·후손·형제 수를 주장한다
  - "levels"       위계가 몇 단계인지를 주장한다 ("2단계 구조")
  - "not_a_claim"  서식·순번·인용 등 레지스트리 사실이 아니다
  - "unsupported"  수량을 단정하지만 위 질의로 표현할 수 없다

  levels이면 instrument를 채운다. 검사를 특정하지 않았으면 비운다.

  count이면 level(overall/L1/L2/L3/L4/factor/item), instrument(필기 역량검사
  /영상면접), analysis_included 중 답변이 실제로 말한 조건만 채운다. 말하지
  않은 조건은 비운다.

  relation이면 name과 relation(parent/children/ancestors/descendants/siblings)
  을 채운다.

  **count와 relation을 가르는 기준은 하나다 — 특정 역량이 기준점인가.**
  "성실성의 하위요인은 3개", "전략성 아래에 3개"처럼 한 역량을 기준으로 센
  수는 위계 이름이 들어 있어도 relation이다. count는 기준점 없이 레지스트리
  전체 또는 한 위계·검사 전체를 셀 때만 쓴다. 대상이 문장에 없으면 조회
  내역의 대상을 쓴다.

  조건은 답변이 말한 것만 채우고, 말하지 않은 조건을 **추가하지 않는다.**
  조건이 하나도 없어도 count다 — 조건 없는 count는 레지스트리 전체 수다.

  예시:

    "등록된 역량은 총 62개"        → count, 조건 없음
    "필기 역량검사: 53개"          → count, instrument=필기 역량검사
    "하위요인은 30개"              → count, level=L3
    "필기 역량검사 상위요인 3개"    → count, level=L1, instrument=필기 역량검사
    "분석에 포함되는 역량 30개"     → count, analysis_included=true
    "성실성의 하위요인 3개"         → relation, name=성실성, children
    "전략성 아래 전체 후손 12개"    → relation, name=전략성, descendants
    "영상면접은 2단계 구조"         → levels, instrument=영상면접
    목록 표지 "1."                 → not_a_claim
    "평균 점수는 72점"             → unsupported

  검사 이름만으로 세는 것도, 조건 없이 전체를 세는 것도 count로 표현된다.
  이런 것을 unsupported로 보내지 않는다.

**2. 답변이 등록된 역량이라고 단정한 이름을 모두 적는다.**

  답변이 "등록되어 있지 않다"고 말한 이름은 넣지 않는다. 등록된 역량으로
  다루거나 정의·위계를 제시한 이름만 넣는다.

추측하지 않는다. 답변이 실제로 말한 것만 옮긴다.
"""

JUDGE_SCHEMA: dict[str, Any] = {
    # langchain의 with_structured_output은 raw JSON Schema를 함수로 바꾸므로
    # 최상위 title이 없으면 ValueError로 거부한다.
    "title": "grounding_claims",
    "description": "답변이 주장하는 것을 구조화해서 옮긴다. 판정하지 않는다.",
    "type": "object",
    "properties": {
        "numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "claim": {
                        "type": "string",
                        "enum": ["count", "relation", "levels", "not_a_claim", "unsupported"],
                    },
                    "level": {
                        "type": ["string", "null"],
                        "enum": ["overall", "L1", "L2", "L3", "L4", "factor", "item", None],
                    },
                    "instrument": {
                        "type": ["string", "null"],
                        "enum": ["필기 역량검사", "영상면접", None],
                    },
                    "analysis_included": {"type": ["boolean", "null"]},
                    "name": {"type": ["string", "null"]},
                    "relation": {
                        "type": ["string", "null"],
                        "enum": [
                            "parent",
                            "children",
                            "ancestors",
                            "descendants",
                            "siblings",
                            None,
                        ],
                    },
                },
                "required": [
                    "index",
                    "claim",
                    "level",
                    "instrument",
                    "analysis_included",
                    "name",
                    "relation",
                ],
                "additionalProperties": False,
            },
        },
        "asserted_names": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["numbers", "asserted_names"],
    "additionalProperties": False,
}


_RELATION_TERMS: Mapping[str, str] = {
    "parent": "상위",
    "ancestors": "조상",
    "children": "직속 하위",
    "descendants": "전체 하위",
    "siblings": "형제",
}


def _count_label(criteria: Mapping[str, Any]) -> str:
    """사람이 읽을 라벨을 만든다.

    폴백 답변과 재생성 힌트에 그대로 실리므로 dict repr을 쓰면 안 된다 --
    4단계 첫 실행에서 ``{'level': 'L3'} 항목 수: 30``이 사용자 화면까지
    나갔다.
    """

    parts: list[str] = []
    instrument = criteria.get("instrument")
    if instrument:
        parts.append(str(instrument))
    level = criteria.get("level")
    if level:
        term = WRITTEN_TIER_TERMS.get(level) or VIDEO_TIER_TERMS.get(level) or level
        parts.append(f"{term}({level})")
    included = criteria.get("analysis_included")
    if included is not None:
        parts.append("분석 포함" if included else "분석 제외")
    if not parts:
        return "등록 항목 수"
    return " ".join(parts) + " 수"


def _true_value(snapshot: RegistrySnapshot, mapping: Mapping[str, Any]) -> tuple[int | None, str]:
    """judge가 옮긴 주장을 레지스트리에서 실제로 계산한다."""

    claim = mapping.get("claim")
    if claim == "count":
        result = count_competencies(
            snapshot,
            level=mapping.get("level"),
            instrument=mapping.get("instrument"),
            analysis_included=mapping.get("analysis_included"),
        )
        return result["count"], _count_label(result["criteria"])
    if claim == "relation":
        name = mapping.get("name") or ""
        relation = mapping.get("relation") or "children"
        try:
            result = get_relations(snapshot, name=name, relation=relation)
        except UnknownCompetencyError:
            return None, f"'{name}'은(는) 등록되지 않은 이름"
        term = _RELATION_TERMS.get(str(relation), str(relation))
        return result["count"], f"{name}의 {term} 항목 수"
    if claim == "levels":
        # 위계 단계 수.  항목이 늘어도 자라지 않는 자료 형태이므로 3.6의
        # 경계 안쪽이다.  이 갈래가 없으면 "영상면접은 2단계 구조"처럼
        # 참인 문장이 unsupported로 걸려 답변 전체가 폴백으로 떨어진다.
        instrument = mapping.get("instrument")
        levels = {
            item["level"]
            for item in snapshot.document["items"]
            if instrument is None or item["instrument_label"] == instrument
        }
        label = f"{instrument} 위계 단계 수" if instrument else "위계 단계 수"
        return len(levels), label
    return None, ""


def verify(
    snapshot: RegistrySnapshot,
    spans: Sequence[NumberSpan],
    judged: Mapping[str, Any],
) -> GroundingVerdict:
    """judge가 옮긴 주장을 코드가 대조한다. 여기서만 참·거짓이 정해진다."""

    verdict = GroundingVerdict(ok=True, judge_called=True)
    numbers = list(judged.get("numbers", ()))

    # 5. 스팬 개수와 judge 응답 개수가 다르면 자동 실패.  judge가 불편한
    # 숫자를 조용히 빠뜨리는 경로를 막는다.
    if len(numbers) != len(spans):
        verdict.ok = False
        verdict.findings.append(
            Finding(
                kind="incomplete_judgement",
                detail=(
                    f"숫자 {len(spans)}개 중 {len(numbers)}개만 판정되었다. "
                    "전부 판정되어야 한다."
                ),
            )
        )
        return verdict

    by_index = {span.index: span for span in spans}
    for mapping in numbers:
        index = mapping.get("index")
        span = by_index.get(index)
        if span is None:
            verdict.ok = False
            verdict.findings.append(
                Finding(kind="unknown_span", detail=f"존재하지 않는 숫자 index {index}")
            )
            continue

        claim = mapping.get("claim")
        if claim == "not_a_claim":
            continue
        if claim == "unsupported":
            verdict.ok = False
            verdict.findings.append(
                Finding(
                    kind="unsupported_number",
                    detail=(
                        f"'{span.value}'을(를) 레지스트리로 확인할 수 없다: "
                        f"…{span.context}…"
                    ),
                    actual=span.value,
                )
            )
            continue

        verdict.checked_numbers += 1
        truth, label = _true_value(snapshot, mapping)
        if truth is None:
            verdict.ok = False
            verdict.findings.append(
                Finding(
                    kind="unverifiable_claim",
                    detail=f"'{span.value}'의 근거를 조회할 수 없다 ({label}).",
                    actual=span.value,
                )
            )
            continue
        if truth != span.value:
            verdict.ok = False
            verdict.findings.append(
                Finding(
                    kind="wrong_number",
                    detail=f"{label}은(는) {truth}인데 답변은 {span.value}이라고 했다.",
                    expected=truth,
                    actual=span.value,
                )
            )
            verdict.facts[label] = truth
        else:
            verdict.facts[label] = truth

    for name in judged.get("asserted_names", ()):
        cleaned = str(name).strip()
        if not cleaned:
            continue
        verdict.checked_names += 1
        if cleaned not in snapshot.lookup:
            verdict.ok = False
            verdict.findings.append(
                Finding(
                    kind="unregistered_name",
                    detail=f"'{cleaned}'은(는) 레지스트리에 없는데 등록된 역량처럼 다뤘다.",
                    actual=cleaned,
                )
            )

    return verdict


def check_grounding(
    judge_model: Any,
    snapshot: RegistrySnapshot,
    answer_text: str,
    tool_calls: Sequence[Any] = (),
) -> GroundingVerdict:
    """답변 하나를 검수한다. 검사할 것이 없으면 모델을 부르지 않는다.

    ``tool_calls``는 judge가 **대상을 고르는 데만** 쓴다.  대화를 주지 않기로
    한 3.8의 결정 때문에 "하위요인은 3가지입니다"처럼 대상이 생략된 문장에서
    judge가 엉뚱한 질의로 옮겼고(4단계 첫 실행의 `c01` 오탐), 조회 내역은
    대화 추론이 아니라 사실이므로 그 자리를 대신할 수 있다.

    조회 내역이 판정을 대신하지는 않는다.  코드는 judge가 옮긴 질의를 다시
    실행한다 -- 3단계에서 관측했듯 도구를 부르고도 무관한 수를 쓸 수 있다.
    """

    from langchain_core.messages import HumanMessage, SystemMessage

    spans = extract_number_spans(answer_text)
    if not spans and not answer_text.strip():
        return GroundingVerdict(ok=True)

    payload = {
        "answer": answer_text,
        "numbers": [
            {"index": span.index, "value": span.value, "context": span.context}
            for span in spans
        ],
        "lookups": [
            {
                "tool": call.name,
                "arguments": dict(call.arguments),
                "count": call.result.get("count"),
                "target": (call.result.get("target") or {}).get("name"),
            }
            for call in tool_calls
        ],
    }
    structured = judge_model.with_structured_output(JUDGE_SCHEMA)
    judged = structured.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    return verify(snapshot, spans, judged or {"numbers": [], "asserted_names": []})
