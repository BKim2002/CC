"""레지스트리를 프롬프트로 만드는 계층.

v1은 레지스트리를 두 개의 좁은 문자열로 요약해 모델에 보여 주었다 --
등록명 목록(`name_catalog`)과 의미 검색용 후보 목록(`semantic_catalog`).
어느 쪽도 전체가 아니었고, 그래서 질의 유형을 미리 정해 두어야 했다.

여기서는 62개 항목의 모든 필드를 그대로 싣는다.  6,100 토큰이면 충분하고,
블록이 매 턴 동일하므로 프롬프트 캐시가 걸린다(실측 적중률 96~98%).
docs/REBUILD_PLAN.md 2.2 참고.

레지스트리 블록을 항상 맨 앞에 두는 것은 캐시 접두를 안정시키기 위해서다.
시스템 프롬프트를 고치더라도 캐시가 깨지지 않도록 순서를 바꾸지 않는다.
"""

from __future__ import annotations

from chat.registry import RegistrySnapshot
from chat.tools import VIDEO_TIER_TERMS, WRITTEN_TIER_TERMS

# ─────────────────────────────────────────────────────────────────────
# 시스템 프롬프트
#
# 의도적으로 짧다.  넓게 시작해서 축소하는 것이 이번 재설계의 방향이고,
# 여기에 규칙을 미리 쌓으면 v1의 유형 목록이 프롬프트 안에서 되살아난다.
#
# 능력을 열거하지 않는다.  다만 열거를 금지하지도 않는다 --
# 1단계에서 모델이 스스로 열거했고 그 내용이 레지스트리에서 유도된
# 사실이었기 때문이다.  REBUILD_PLAN 3.4 참고.
#
# 처방 금지 줄(2.5단계에서 추가)은 코드 없이 b06을 해소했다.  같은 줄이
# b01·b03·b05·b07의 답변도 함께 정돈했다.  REBUILD_PLAN 3.9 참고.
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
너는 역량 레지스트리 전용 도우미다.

아래 <registry>에 등록된 역량 정보에 근거해서만 답한다. 이것이 사용자에게
하는 유일한 약속이며, 답할 수 있는 질문의 종류를 미리 정해두지 않는다.

지켜야 할 것:

- <registry>에 있는 사실만 말한다. 없는 역량 이름, 없는 정의, 없는 관계를
  만들지 않는다.
- 레지스트리에 없는 방법·절차·해석 기준·권고는 제시하지 않는다. 그런 것을
  물으면 등록되어 있지 않다고 알리고, 관련된 등록 정보만 제공한다.
- 사용자가 말한 이름이 <registry>에 없으면 그 사실을 알리고, 비슷한 등록
  이름이 있으면 후보로 제시한다. 일반 상식으로 정의하지 않는다.
- 역량과 무관한 질문(날짜, 시각, 날씨, 뉴스, 일반 개념, 코딩, 개인 평가,
  채용·의료·법률·금융 판단 등)에는 답하지 않는다. 답할 수 없다는 것을
  자연스럽게 알리고, 대신 무엇을 도울 수 있는지 짧게 안내한다.
- 인사와 감사에는 자연스럽게 응대한다.
- 사용자의 언어로 답한다.
- **개수를 말할 때는 도구로 확인한다.** 눈으로 세지 않는다.
- **특정 역량 하나의 부모·자식·조상·후손·형제를 말할 때도 도구로 확인한다.**
  도구가 돌려준 위계 이름(상위요인·중위요인·하위요인 등)을 그대로 쓰고,
  직접 고쳐 부르지 않는다.
- "○○ 아래에 몇 개"처럼 **한 역량을 기준으로 세는 질문은 관계 조회다.**
  전체를 세는 도구가 아니라 그 역량의 관계를 조회해 답한다.
- 전체 위계나 목록을 통째로 보여줄 때는 도구를 부르지 않는다. <registry>에
  이미 다 있으므로 항목마다 도구를 부르면 느려지기만 한다.
- 도구가 직속 자식 수와 전체 후손 수가 다르다고 알려주면, 어느 쪽을 말하는
  것인지 답에서 밝힌다.

해석이 둘 이상으로 갈리는 경우에만 되묻는다. 표현이 낯설다는 이유로
되묻지 않는다.
"""

# 상위/중위/하위요인은 level 값에서 유도되지 않는 도메인 어휘다.  빼면
# g01 계열이 "가정이 틀려서"가 아니라 "용어를 몰라서" 깨진다.  양쪽을
# 비교할 수 있게 build_system_message의 인자로 남겨 두었다.
#
# 용어를 여기에 다시 적지 않고 chat.tools 의 사전에서 만든다.  모델이 읽는
# 이름과 도구가 돌려주는 이름이 갈라지면 g06 같은 티어 오류를 도구로
# 고칠 수 없게 된다.
TIER_GLOSSARY = "\n".join(
    [
        "<tier_vocabulary>",
        "필기 역량검사의 위계는 level 필드로 표현되며 다음 용어로 불린다.",
        "",
        *(
            f"  {level:<8}= {term}"
            for level, term in WRITTEN_TIER_TERMS.items()
        ),
        "",
        "영상면접은 다음 두 단계를 쓴다.",
        "",
        *(
            f"  {level:<8}= {term}"
            for level, term in VIDEO_TIER_TERMS.items()
        ),
        "</tier_vocabulary>",
        "",
    ]
)


def render_registry(snapshot: RegistrySnapshot) -> str:
    """레지스트리 전문을 프롬프트 블록으로 만든다.

    한 항목이 한 문단이다.  필드는 전부 넣는다 -- 무엇이 실제로 필요한지
    모르는 상태에서 미리 줄이면 그 자체가 질의 유형을 정하는 일이 된다.
    """

    lines = [
        "<registry>",
        f"출처: {snapshot.source_filename}",
        f"항목 수: {len(snapshot.document['items'])}",
        "",
        "필드: name / aliases / id / level / 검사 / 부모 / 자식 / 분석포함 / 정의",
        "",
    ]

    for item in snapshot.document["items"]:
        head = f"[{item['name']}]"
        if item["aliases"]:
            head += f" (별칭: {', '.join(item['aliases'])})"
        lines.append(head)

        meta = [
            f"id={item['id']}",
            f"level={item['level']}",
            f"검사={item['instrument_label']}",
            f"부모={item['parent_name'] or '없음'}",
            f"자식={', '.join(item['children']) if item['children'] else '없음'}",
            f"분석포함={'예' if item['analysis_included'] else '아니오'}",
        ]
        lines.append("  " + " · ".join(meta))

        if item["definition"]:
            lines.append(f"  정의: {item['definition']}")
        else:
            lines.append(f"  정의: (없음 — definition_status={item['definition_status']})")

        if item["notes"]:
            lines.append(f"  비고: {' / '.join(item['notes'])}")
        lines.append("")

    lines.append("</registry>")
    return "\n".join(lines)


def build_system_message(
    snapshot: RegistrySnapshot,
    *,
    tier_glossary: bool = True,
) -> str:
    """레지스트리를 맨 앞에 두어 프롬프트 캐시 접두가 안정되게 만든다."""

    parts = [render_registry(snapshot)]
    if tier_glossary:
        parts.append(TIER_GLOSSARY)
    parts.append(SYSTEM_PROMPT)
    return "\n\n".join(parts)
