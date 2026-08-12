"""Pure validation and deterministic recovery for buffered Scope responses.

This module has no LangGraph or registry dependency.  It turns strict structured
drafts into public prose only after checking the capability boundary and the
topic-specific no-answer contract.
"""

from __future__ import annotations

import re
from typing import Any

from llm_gateway import MetaResponseDraft, OutOfScopeResponseDraft

SCOPE_CATEGORIES = frozenset(
    {
        "date_time",
        "weather",
        "news_current_events",
        "professional_advice",
        "personal_assessment",
        "employment_decision",
        "general_knowledge",
        "external_action",
        "unsafe",
        "other",
    }
)
SCOPE_MODES = frozenset(
    {"greeting", "thanks", "farewell", "bot_identity", "capability_help", "out_of_scope"}
)


def normalize_public_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def prefers_english(text: str) -> bool:
    return not re.search(r"[가-힣]", text) and bool(re.search(r"[A-Za-z]", text))


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?:\r?\n)+|(?<=[.!?。！？])\s+", text.strip())
        if part.strip()
    ]


def _clauses(text: str) -> list[str]:
    return [
        part.strip(" ,;")
        for part in re.split(
            r"(?<=[.!?。！？])\s+|"
            r"\s*(?:[,;:]|\b(?:but|however|although|though|yet|and)\b|"
            r"(?:하지만|그러나|다만|그렇지만|그런데|및|그리고|않고))\s*",
            text,
            flags=re.IGNORECASE,
        )
        if part.strip(" ,;")
    ] or [text.strip()]


def _ensure_sentence(text: str) -> str:
    normalized = normalize_public_text(text)
    if normalized and normalized[-1] not in ".!?。！？":
        normalized += "."
    return normalized


def category_label(category: str, *, english: bool) -> str:
    labels = {
        "date_time": ("날짜와 시각 요청", "date or time request"),
        "weather": ("날씨 요청", "weather request"),
        "news_current_events": ("최신 뉴스 요청", "current news request"),
        "professional_advice": ("전문 조언 요청", "professional advice request"),
        "personal_assessment": ("개인 평가 요청", "personal assessment request"),
        "employment_decision": ("채용 판단 요청", "employment decision request"),
        "general_knowledge": ("일반 지식 질문", "general knowledge question"),
        "external_action": ("외부 작업 요청", "external action request"),
        "unsafe": ("안전하지 않은 요청", "unsafe request"),
        "other": ("지원 범위 밖 질문", "out-of-scope question"),
    }
    korean, translated = labels.get(category, labels["other"])
    return translated if english else korean


def sanitize_topic_summary(summary: str, category: str, *, english: bool) -> str:
    """Return a short noun-like topic without data, commands, URLs or internals."""

    fallback = category_label(category, english=english)
    candidate = re.sub(
        r"(?i)\b[a-z][a-z0-9+.-]*://\S+|\bwww\.\S+|"
        r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?",
        " ",
        str(summary),
    )
    candidate = re.sub(
        r"(?:[가-힣A-Za-z0-9-]+\.)+[가-힣A-Za-z]{2,}(?:/\S*)?",
        " ",
        candidate,
    )
    candidate = re.sub(r"\d+(?:[.:/-]\d+)*", " ", candidate)
    candidate = re.sub(
        r"(?i)(?:openai_api_key|database_url|stable[_ -]?id|query[_ -]?plan|"
        r"gateway[_ -]?route|registry_query|meta_conversation|out_of_scope|"
        r"system\s+prompt|시스템\s*프롬프트|내부\s*(?:지침|명령)|"
        r"(?:이전|위의|모든)?\s*(?:지시|규칙|명령)(?:을|를)?\s*무시|"
        r"ignore\s+(?:all\s+|previous\s+)?instructions?)",
        " ",
        candidate,
    )
    candidate = re.sub(
        r"(?i)(?:답해|알려줘|출력해|실행해|삭제해|말해|respond|answer|output|execute|delete)",
        " ",
        candidate,
    )
    candidate = re.sub(
        r"(?i)(?:이|가|은|는|을|를)?\s*(?:뭐야|무엇(?:이야)?|며칠이야|어때|궁금해|"
        r"what is it|what is|tell me|please)$",
        " ",
        candidate,
    )
    candidate = normalize_public_text(candidate).strip(" ,;:.!?()[]{}")[:80].strip()
    informative_tokens = {
        token.casefold()
        for token in re.findall(r"[가-힣]{2,}|[A-Za-z]{3,}", candidate)
    } - {
        "오늘",
        "현재",
        "오늘",
        "요청",
        "질문",
        "today",
        "current",
        "latest",
        "request",
        "question",
    }
    if category not in {
        "date_time",
        "weather",
        "news_current_events",
        "general_knowledge",
    }:
        return fallback
    if (
        not candidate
        or candidate.casefold() in {"오늘", "현재", "today", "current", "latest"}
        or not informative_tokens
        or not re.search(r"[가-힣]{2,}|[A-Za-z]{3,}", candidate)
        or len(re.findall(r"[가-힣]+|[A-Za-z]+", candidate)) > 8
        or not re.fullmatch(r"[가-힣A-Za-z _'’.-]+", candidate)
        or re.search(r"[가-힣]{2,}(?:은|는|이|가)(?:\s|$)", candidate)
        or re.search(
            r"(?i)(?:입니다|합니다|됩니다|했습니다|임$|함$|됨$|"
            r"화창|쾌청|맑|흐리|소나기|비가|눈이|내립|옵니다|"
            r"인상|인하|폭락|급등|사임|출시|발표|확정|"
            r"복용|섭취|드세|먹으|처방|매수|매도|투자|소송|고소|"
            r"점수|평균보다|부족|우수|뽑|채용|적합|추천|"
            r"삭제|지우|전송|보냈|완료|섞|만들|해킹|"
            r"\b(?:is|are|was|were|will|should|take|buy|sell|invest|"
            r"deleted|sent|launched|raised|fell|mix|build)\b)",
            candidate,
        )
    ):
        return fallback
    if not _all_direct_answers_are_absent(
        acknowledgement=candidate,
        full_text=candidate,
    ):
        return fallback
    return candidate


_KOREAN_SUFFIXES = frozenset(
    {
        "",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "와",
        "과",
        "도",
        "만",
        "에",
        "에서",
        "에게",
        "로",
        "으로",
        "부터",
        "까지",
        "나",
        "이나",
        "거나",
        "된",
        "되는",
        "한",
        "하는",
        "할",
        "하고",
        "하며",
        "해서",
        "하려",
        "하려는",
        "하신",
        "신",
        "하셨군요",
        "하셨네요",
        "하시군요",
        "시군요",
        "으시군요",
        "는군요",
        "해하셨군요",
        "계시는군요",
        "했어요",
        "셨네요",
        "시면",
        "되어",
        "이라",
        "라",
        "했는지",
        "싶으시군요",
        "므로",
        "라서",
        "어서",
        "아서",
        "해",
        "하거나",
        "하지만",
        "세요",
        "아요",
        "어요",
        "입니다",
        "이에요",
        "예요",
        "합니다",
        "했습니다",
        "었다니",
        "습니다",
        "니다",
        "히",
        "하세요",
        "하지",
        "하지는",
        "않습니다",
        "않아요",
        "못합니다",
        "드립니다",
        "드릴",
        "드리지는",
        "드리지",
        "보실",
        "보세요",
        "볼",
        "실",
        "있습니다",
        "있어요",
        "없습니다",
        "없어요",
        "군요",
    }
)

_SCOPE_SAFE_ROOTS = frozenset(
    {
        "이",
        "에",
        "저",
        "제",
        "그",
        "수",
        "점",
        "것",
        "챗봇",
        "도우미",
        "등록",
        "역량",
        "레지스트리",
        "정보",
        "정의",
        "역량명",
        "하위요인",
        "목록",
        "위계",
        "관계",
        "부모",
        "자식",
        "조상",
        "후손",
        "형제",
        "집계",
        "개수",
        "비교",
        "검사",
        "유형",
        "필터",
        "행동",
        "기반",
        "후보",
        "활용",
        "제안",
        "질문",
        "요청",
        "내용",
        "특징",
        "주제",
        "궁금",
        "묻고",
        "묻",
        "찾고",
        "찾아볼",
        "찾아보",
        "찾기",
        "찾",
        "확인",
        "알고",
        "싶",
        "계시",
        "이해",
        "관해",
        # Pure connective with no content of its own, needed by the
        # "…에 관한 질문은 …" acknowledgement shape.
        "관한",
        "안내",
        "설명",
        "조회",
        "물어보",
        "알려",
        "도와",
        "지원",
        "제공",
        "있",
        "없",
        "않",
        "주",
        "대신",
        "특정",
        "관련",
        "바로",
        "함께",
        "이어",
        "다음",
        "범위",
        "한정",
        "중심",
        "다루",
        "해당",
        "직접",
        "답",
        "판단",
        "전문",
        "조언",
        "수행",
        "현재",
        "오늘",
        "날짜",
        "시각",
        "시간",
        "날씨",
        "최신",
        "뉴스",
        "소식",
        "일반",
        "지식",
        "개념",
        "의미",
        "개인",
        "평가",
        "채용",
        "직무",
        "외부",
        "작업",
        "파일",
        "문서",
        "이메일",
        "메일",
        "예약",
        "구매",
        "웹",
        "인터넷",
        "검색",
        "상식",
        "성격",
        "진단",
        "의료",
        "법률",
        "금융",
        "약",
        "복용",
        "처방",
        "주식",
        "매수",
        "매도",
        "투자",
        "삭제",
        "지우",
        "전송",
        "보내",
        "안전",
        "변화",
        "적응",
        "건강한",
        "협업",
        "업무",
        "관찰",
        "안녕",
        "반갑",
        "감사",
        "고맙",
        "도움",
        "되",
        "기쁩",
        "또",
        "뵙겠습니다",
        "가세요",
        "역할",
        "사례",
        "바탕",
        "관심",
        "원하시면",
        "드릴",
        "드리기",
        "어려워요",
        "말씀",
        "찾아드릴게요",
        "살펴볼",
        "다룰",
        "다루기",
        "있어",
        "자체",
        "못해요",
        "어떤",
        "계신지",
        "대해",
        "알려주시면",
    }
)

_ENGLISH_SCOPE_WORDS = frozenset(
    {
        "i",
        "m",
        "am",
        "a",
        "an",
        "the",
        "this",
        "chat",
        "chatbot",
        "assistant",
        "it",
        "its",
        "you",
        "your",
        "re",
        "s",
        "are",
        "is",
        "can",
        "cannot",
        "can't",
        "do",
        "does",
        "not",
        "only",
        "instead",
        "please",
        "about",
        "on",
        "to",
        "for",
        "of",
        "with",
        "and",
        "or",
        "so",
        "that",
        "which",
        "what",
        "when",
        "how",
        "understand",
        "asking",
        "asked",
        "ask",
        "question",
        "questions",
        "request",
        "looking",
        "interested",
        "want",
        "know",
        "check",
        "registered",
        "registry",
        "competency",
        "competencies",
        "skill",
        "skills",
        "information",
        "definition",
        "definitions",
        "hierarchy",
        "hierarchies",
        "relationship",
        "relationships",
        "parent",
        "child",
        "ancestor",
        "descendant",
        "sibling",
        "aggregate",
        "aggregates",
        "count",
        "counts",
        "comparison",
        "comparisons",
        "behavior",
        "based",
        "candidate",
        "candidates",
        "explain",
        "describe",
        "relate",
        "look",
        "find",
        "help",
        "provide",
        "support",
        "limited",
        "focused",
        "scope",
        "answer",
        "directly",
        "current",
        "live",
        "date",
        "time",
        "weather",
        "news",
        "general",
        "knowledge",
        "professional",
        "advice",
        "personal",
        "assessment",
        "employment",
        "decision",
        "external",
        "action",
        "unsafe",
        "today",
        "adapting",
        "change",
        "hello",
        "hi",
        "welcome",
        "thank",
        "thanks",
        "glad",
        "goodbye",
        "bye",
        "see",
        "return",
        "anytime",
        "meet",
        "good",
        "could",
        "would",
        "like",
        "feel",
        "free",
        "sounds",
        "curious",
        "role",
        "conditions",
        "try",
        "describing",
        "tell",
        "me",
        "identify",
        "relevant",
        "related",
        "itself",
        "my",
        "t",
        "forecast",
        "handles",
    }
)

_PUBLIC_SCOPE_CHARACTERS = re.compile(
    r"^[가-힣A-Za-z0-9\s.,!?;:'’\"“”()·-]+$"
)


def _controlled_scope_vocabulary(text: str, *, dynamic_terms: tuple[str, ...] = ()) -> bool:
    if not _PUBLIC_SCOPE_CHARACTERS.fullmatch(text):
        return False
    roots = set(_SCOPE_SAFE_ROOTS)
    dynamic_english: set[str] = set()
    for term in dynamic_terms:
        roots.update(re.findall(r"[가-힣]+", term))
        dynamic_english.update(
            token.casefold() for token in re.findall(r"[A-Za-z]+", term)
        )
    for token in re.findall(r"[가-힣]+|[A-Za-z]+", text):
        lowered = token.casefold()
        if re.fullmatch(r"[A-Za-z]+", token):
            if lowered not in _ENGLISH_SCOPE_WORDS and lowered not in dynamic_english:
                return False
            continue
        if not any(
            lowered == root or (
                lowered.startswith(root)
                and lowered[len(root) :] in _KOREAN_SUFFIXES
            )
            for root in roots
        ):
            return False
    return True


def _acknowledgement_shape_is_safe(
    text: str,
    *,
    safe_summary: str,
) -> bool:
    if len(_sentences(text)) != 1:
        return False
    normalized = normalize_public_text(text).rstrip(".!?。！？")
    korean_tails = (
        r"(?:을|를)?\s*확인하고\s*싶으시군요",
        r"(?:을|를)?\s*궁금해?하셨군요",
        r"(?:을|를)?\s*궁금하신\s*점을?\s*이해했습니다",
        r"(?:을|를)?\s*묻고\s*계시는군요",
        r"(?:을|를)?\s*찾고\s*계시는군요",
        r"(?:을|를)?\s*요청하셨군요",
        r"(?:을|를)?\s*요청하신\s*점을?\s*이해했습니다",
        r"(?:에\s*관해|에\s*대해)\s*궁금하신\s*점을?\s*이해했습니다",
        r"(?:에\s*관해|에\s*대해)\s*질문하신\s*점을?\s*이해했습니다",
        # Naming the limit up front reads better than restating the question.
        # It still answers nothing: the direct-answer and definition checks
        # below apply to this shape exactly as they do to the others.
        r"(?:에\s*관한|에\s*대한)\s*질문은\s*제가\s*알려드릴\s*수\s*없습니다",
        r".+(?:궁금하신|요청하신)\s*(?:것으로|점으로)\s*"
        r"이해(?:했습니다|했어요)",
        r".+(?:물어보셨네요|질문하셨네요)",
    )
    english_shapes = (
        r"(?:i\s+understand\s+that\s+)?you(?:'re|\s+are)\s+(?:asking|looking)\s+(?:for|about)\s+.+",
        r"i\s+understand\s+your\s+(?:question|request)\s+about\s+.+",
        r"you\s+(?:want|would\s+like)\s+to\s+(?:know|check)\s+.+",
        r"it\s+sounds\s+like\s+you\s+are\s+(?:curious|asking)\s+about\s+.+",
        r"i\s+(?:can't|cannot)\s+answer\s+questions?\s+about\s+.+",
    )
    shaped = any(
        re.fullmatch(rf".+?{tail}", normalized)
        for tail in korean_tails
    ) or any(re.fullmatch(shape, normalized, re.I) for shape in english_shapes)
    return shaped and _controlled_scope_vocabulary(
        normalized,
        dynamic_terms=(safe_summary,),
    )


def _boundary_shape_is_safe(text: str, *, safe_summary: str) -> bool:
    if len(_sentences(text)) != 1:
        return False
    normalized = normalize_public_text(text)
    starts_safely = bool(
        re.match(
            r"^(?:이\s*챗봇|저|제(?:가|\s*역할)|등록(?:된)?\s*역량|역량\s*챗봇)",
            normalized,
        )
        or re.match(
            r"^(?:this\s+(?:chat|chatbot|assistant)|i\s+|my\s+(?:scope|role))",
            normalized,
            re.I,
        )
    )
    return starts_safely and _controlled_scope_vocabulary(normalized)


def _redirect_shape_is_safe(text: str) -> bool:
    if len(_sentences(text)) != 1:
        return False
    normalized = normalize_public_text(text)
    starts_safely = bool(
        re.match(
            r"^(?:대신|원하시면|관심|어떤|등록(?:된)?\s*역량|특정\s*등록)",
            normalized,
        )
        or re.match(r"^(?:you\s+can|instead|feel\s+free|try)", normalized, re.I)
    )
    return starts_safely and _controlled_scope_vocabulary(normalized)


def _meta_response_shape_is_safe(text: str) -> bool:
    return (
        len(_sentences(text)) == 1
        and _controlled_scope_vocabulary(text)
        and definition_claim_is_absent(text)
    )


def contains_internal_information(text: str) -> bool:
    return any(
        re.search(pattern, text, re.I)
        for pattern in (
            r"postgres(?:ql)?://",
            r"\b(?:database_url|openai_api_key|api[_ -]?key|stable[_ -]?id|query[_ -]?plan)\b",
            r"\b(?:gateway_route|registry_query|meta_conversation|out_of_scope)\b",
            r"\b(?:system prompt|hidden instructions?|developer message)\b",
            r"(?:시스템\s*프롬프트|내부\s*(?:지침|프롬프트|명령)|숨겨진\s*(?:지침|명령))",
        )
    )


def _has_registry_reference(text: str) -> bool:
    lowered = text.casefold()
    return any(
        token in lowered
        for token in (
            "역량",
            "레지스트리",
            "competency",
            "competencies",
            "registered skill",
            "registered skills",
        )
    )


def is_registry_redirect(text: str) -> bool:
    lowered = text.casefold()
    return _has_registry_reference(text) and any(
        token in lowered
        for token in (
            "질문",
            "물어",
            "설명",
            "조회",
            "찾",
            "알려",
            "도와",
            "ask",
            "explain",
            "look up",
            "find",
            "identify",
            "살펴",
            "help",
        )
    )


def scope_boundary_is_valid(text: str) -> bool:
    lowered = text.casefold()
    return _has_registry_reference(text) and any(
        token in lowered
        for token in (
            "범위",
            "다루",
            "한정",
            "만 ",
            "만을",
            "지원하지",
            "제공하지",
            "확인하지",
            "답하지",
            "can't",
            "cannot",
            "limited",
            "focused",
            "only",
            "scope",
            "do not provide",
            "does not provide",
            "어려",
            "못",
        )
    )


def _response_language_matches(text: str, raw_query: str) -> bool:
    if prefers_english(raw_query):
        return not re.search(r"[가-힣]", text)
    if re.search(r"[가-힣]", raw_query):
        return bool(re.search(r"[가-힣]", text))
    return True


def _topic_is_reflected(acknowledgement: str, summary: str, category: str) -> bool:
    lowered = acknowledgement.casefold()
    if not any(
        token in lowered
        for token in (
            "질문",
            "요청",
            "궁금",
            "묻",
            "물어",
            "찾고",
            "알아보",
            "확인하고 싶",
            "확인하려",
            "asked",
            "asking",
            "question",
            "request",
            "looking for",
            "want to know",
            "would like to check",
            "checking",
            "interested in",
            "curious",
            "would like to know",
        )
    ):
        return False
    stop = {"오늘", "현재", "요청", "질문", "the", "about", "request", "question"}
    tokens = [
        token.casefold()
        for token in re.findall(r"[가-힣]{2,}|[A-Za-z]{3,}", summary)
        if token.casefold() not in stop
    ]
    category_tokens = {
        "date_time": ("날짜", "시각", "시간", "date", "time"),
        "weather": ("날씨", "weather"),
        "news_current_events": ("뉴스", "소식", "news", "event"),
        "professional_advice": ("조언", "의료", "법률", "금융", "advice"),
        "personal_assessment": ("평가", "점수", "assessment", "score"),
        "employment_decision": ("채용", "직무", "employment", "hiring", "job"),
        "general_knowledge": ("개념", "의미", "지식", "concept", "meaning"),
        "external_action": ("작업", "파일", "이메일", "action", "file", "email"),
        "unsafe": ("위험", "안전", "unsafe", "safety"),
        "other": ("요청", "질문", "request", "question"),
    }
    probes = tuple(tokens) + category_tokens.get(category, ())
    return any(probe in lowered for probe in probes)


def category_direct_answer_is_absent(
    category: str,
    *,
    acknowledgement: str,
    full_text: str,
) -> bool:
    if category in {"date_time", "weather", "professional_advice", "personal_assessment"}:
        if re.search(r"(?<![A-Za-z0-9])\d+(?:[.:/-]\d+)*(?![A-Za-z0-9])", full_text):
            return False
    patterns: dict[str, tuple[str, ...]] = {
        "date_time": (
            r"(?:오늘|현재)\s*(?:은|는|날짜|시각).{0,30}(?:월요일|화요일|수요일|목요일|금요일|토요일|일요일)",
            r"(?:오늘|현재).{0,30}(?:일|이|삼|사|오|육|칠|팔|구|십|한|두|세|네)+(?:월|일|시)(?:입니다|이에요|예요)",
            r"(?:오전|오후)\s*(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두)\s*시",
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:a\.?m\.?|p\.?m\.?|o'clock)\b",
        ),
        "weather": (
            r"(?:맑|화창|쾌청|흐림|흐리|소나기|장마|태풍|우박|비|눈|강수|기온|습도|바람|미세먼지)"
            r".{0,35}(?:입니다|합니다|옵니다|내립니다|내릴|찾아올|예정|예상|전망|듯합니다|높|낮|춥|덥)",
            r"\b(?:sunny|cloudy|clear skies|raining|snowing|stormy|showers?|will rain|"
            r"temperature is|expected to|forecast(?:s|ed)?\s+(?:sunny|rain|snow|cloud|"
            r"storm|temperature|weather))\b",
            r"(?:현재\s*)?(?:날씨|하늘)(?:은|는|이|가)?\s*"
            r"(?:비|눈|맑음|흐림|화창)(?:[.!?]|$)",
        ),
        "news_current_events": (
            r"(?:금리|정부|대통령|총리|기업|시장|은행|회사|삼성|국회|법원).{0,45}"
            r"(?:인상|인하|사임|당선|발표|승리|발생|출시|폭락|급등|확정|결정|체결|합병|취임|해임)",
            r"\b(?:central bank|government|president|prime minister|company|market|court|parliament)"
            r".{0,55}(?:raised|cut|resigned|won|announced|launched|fell|rose|confirmed|decided|merged)\b",
        ),
        "professional_advice": (
            r"복용|섭취|드세요|먹으세요|챙기세요|처방|투약|진료|소송|고소|매수|매도|투자|계약",
            r"\b(?:mg|ml|dosage|dose|take|consume|tablet|pill|buy|sell|invest|sue|lawsuit)\b",
        ),
        "personal_assessment": (
            r"(?:당신|사용자|지원자|후보|점수|수준|성향|역량|협업|리더십).{0,35}"
            r"(?:높|낮|부족|우수|뛰어|평균\s*(?:이하|보다\s*못)|잘합니다|못합니다|진단)",
            r"\b(?:your score is|your level is|below average|you lack|you are (?:high|low|excellent)|"
            r"you (?:perform|collaborate) (?:very )?well)\b",
        ),
        "employment_decision": (
            r"(?:지원자|후보|사람|이분).{0,30}(?:뽑|선발|고용|채용|합격|불합격|적합|추천)",
            r"(?:채용|합격|직무).{0,30}(?:해야|좋겠|유리|적합|추천)",
            r"\b(?:should hire|should choose|will pass|will be hired|good fit for (?:the|this) job|"
            r"recommend (?:this|the) (?:candidate|job))\b",
        ),
        "external_action": (
            r"(?:파일|문서|이메일|메일|메시지|예약|구매).{0,35}"
            r"(?:삭제했|지웠|지워\s*두|전송했|보냈|완료했|예약했|구매했|처리했)",
            r"(?:삭제했|지웠|지워\s*두|전송했|보냈|완료했).{0,35}"
            r"(?:파일|문서|이메일|메일|메시지|예약|구매)",
            r"\b(?:deleted|erased|sent|booked|purchased|completed)\b.{0,35}"
            r"\b(?:file|document|email|message|reservation|purchase)\b",
            r"(?:파일|문서|이메일|메일|메시지|예약|구매).{0,35}"
            r"(?:삭제하|삭제합|지우|전송하|전송합|보내|예약하|예약합|구매하|구매합)",
        ),
        "unsafe": (
            r"(?:폭발|폭발물|무기|침입|해킹|자해).{0,50}(?:재료|섞|만들|단계|절차|먼저|다음)",
            r"(?:재료|섞|만들|단계|절차|먼저|다음).{0,50}(?:폭발|무기|침입|해킹|자해)",
            r"\b(?:step by step|instructions?|first|next|mix|build).{0,55}"
            r"(?:weapon|explosive|hack|break in|self-harm)\b",
        ),
    }
    lowered = full_text.casefold()
    if any(re.search(pattern, lowered, re.I) for pattern in patterns.get(category, ())):
        return False
    ack_lowered = acknowledgement.casefold()
    if any(token in ack_lowered for token in ("정답은", "답은", "answer is")):
        return False
    if category == "general_knowledge" and any(
        re.search(pattern, full_text, re.I)
        for pattern in (
            r"(?:은|는|란)\s+.{2,100}(?:입니다|이다)[.!]?",
            r"\b(?:means|is an?|refers to|is defined as)\b",
        )
    ):
        return False
    return True


def _all_direct_answers_are_absent(*, acknowledgement: str, full_text: str) -> bool:
    return all(
        category_direct_answer_is_absent(
            category,
            acknowledgement=acknowledgement,
            full_text=full_text,
        )
        for category in SCOPE_CATEGORIES - {"other"}
    )


def definition_claim_is_absent(text: str) -> bool:
    """Reject registry-like or general-definition claims in scope prose."""

    patterns = (
        r"(?:^|[.!?\n]\s*)[가-힣A-Za-z][가-힣A-Za-z0-9 _-]{0,40}"
        r"(?:은|는|이란|란)\s+[^.!?\n]{1,120}"
        r"(?:능력|특성|태도|자세|뜻|의미)(?:입니다|이다|라고\s*합니다)",
        r"[가-힣A-Za-z][가-힣A-Za-z0-9 _-]{1,40}\s*[:(]\s*"
        r"[^.!?\n]{1,100}(?:능력|특성|태도|자세|뜻|의미|ability|trait|attitude)",
        r"[^.!?\n]{1,80}(?:을|를)\s+[가-힣A-Za-z][가-힣A-Za-z0-9 _-]{1,40}"
        r"(?:이|라)고?\s*부르",
        r"\b[A-Za-z][A-Za-z0-9 _-]{1,40}\s+(?:means|is defined as|"
        r"is the ability|is a trait|is an attitude|refers to)\b",
        r"\b(?:also called|known as)\s+[A-Za-z][A-Za-z0-9 _-]{1,40}\b",
        r"(?:^|[.!?]\s*)[^.!?\n]{1,60}(?:은|는|이란|란)\s+"
        r"[^.!?\n]{0,100}(?:행동|관계|역량|요인|항목|업무)(?:입니다|이다)",
    )
    return not any(re.search(pattern, text, re.I) for pattern in patterns)


def manifest_claims_are_safe(text: str) -> bool:
    unsupported_topics = (
        "날짜",
        "시각",
        "시간",
        "날씨",
        "뉴스",
        "개인 평가",
        "개인",
        "채용",
        "직무 판단",
        "직무 평가",
        "파일 작업",
        "파일",
        "문서",
        "메일",
        "이메일",
        "전문 조언",
        "의료",
        "법률",
        "금융",
        "약",
        "일반 개념",
        "일반 주제",
        "웹",
        "인터넷",
        "성격",
        "상식",
        "웹 검색",
        "인터넷 검색",
        "최신 뉴스",
        "날씨 확인",
        "일반 지식",
        "개념 설명",
        "개인 점수",
        "성격 분석",
        "채용 추천",
        "의료 조언",
        "법률 조언",
        "금융 조언",
        "파일 삭제",
        "이메일 전송",
        "예약",
        "구매",
        "인터넷에서",
        "온라인에서",
        "일반 상식",
        "성격 진단",
        "성격을 진단",
        "약 복용",
        "복용 방법",
        "건강 처방",
        "주식",
        "매수",
        "매도",
        "문서 삭제",
        "문서를 삭제",
        "문서를 지우",
        "메일 전송",
        "메일을 보내",
        "외부 작업",
        "외부",
        "web search",
        "latest news",
        "weather",
        "general knowledge",
        "personal score",
        "personality analysis",
        "hiring recommendation",
        "medical advice",
        "legal advice",
        "financial advice",
        "delete files",
        "send email",
        "booking",
        "purchase",
        "online search",
        "common knowledge",
        "diagnose personality",
        "medication instructions",
        "prescription",
        "stock picks",
        "erase documents",
        "send mail",
    )
    english_unsupported_patterns = (
        r"\bdate\b",
        r"\btime\b",
        r"\bnews\b",
        r"\bpersonal\s+assessment\b",
        r"\bemployment\s+decision\b",
        r"\bfile\s+action\b",
        r"\bexternal\s+action\b",
        r"\bprofessional\s+advice\b",
        r"\bgeneral\s+concept\b",
    )
    negatives = (
        "할 수 없",
        "하지 않",
        "않",
        "못",
        "어렵",
        "어려",
        "없",
        "지원하지",
        "제공하지",
        "확인하지",
        "can't",
        "cannot",
        "do not",
        "does not",
        "not support",
    )
    acknowledgement_tokens = (
        "질문",
        "요청",
        "궁금",
        "묻",
        "찾고",
        "이해",
        "asking",
        "question",
        "request",
        "looking for",
        "want to know",
    )
    affirmative_claim_patterns = (
        r"(?:제공|지원|수행|설명|평가|판단|진단|삭제|전송|예약|구매)"
        r"(?:합니다|해드립니다|하겠습니다|할게요|할\s*수\s*있|하는\b|가능)",
        r"(?:지워|보내|찾아)(?:드립니다|드릴게요|겠습니다)",
        r"\b(?:can|will)\s+(?:provide|support|perform|explain|assess|judge|"
        r"diagnose|delete|send|book|purchase|search|check)\b",
        r"\b(?:i|we|this\s+chatbot)\s+(?:provide|support|perform|explain|"
        r"assess|judge|diagnose|delete|send|book|purchase|search|check)\b",
    )
    for clause in _clauses(text):
        lowered = clause.casefold()
        topic_positions = [
            lowered.find(topic)
            for topic in unsupported_topics
            if topic in lowered
        ]
        topic_positions.extend(
            match.start()
            for pattern in english_unsupported_patterns
            if (match := re.search(pattern, lowered)) is not None
        )
        if not topic_positions:
            continue
        first_topic_position = min(topic_positions)
        affirmative_positions = [
            match.start()
            for pattern in affirmative_claim_patterns
            if (match := re.search(pattern, lowered)) is not None
        ]
        if any(position >= first_topic_position for position in affirmative_positions):
            return False
        if any(negative in lowered for negative in negatives):
            continue
        if any(token in lowered for token in acknowledgement_tokens):
            continue
        return False
    return True


def validate_scope_draft(
    raw_draft: Any,
    *,
    mode: str,
    category: str,
    summary: str,
    raw_query: str,
) -> str | None:
    """Validate a strict draft and return the only publishable final string."""

    if mode == "out_of_scope":
        try:
            draft = OutOfScopeResponseDraft.model_validate(raw_draft)
        except Exception:
            return None
        fields = (
            _ensure_sentence(draft.acknowledgement),
            _ensure_sentence(draft.scope_boundary),
            _ensure_sentence(draft.registry_redirect),
        )
        acknowledgement, boundary, redirect = fields
        answer = " ".join(fields)
        if category not in SCOPE_CATEGORIES:
            return None
        safe_summary = sanitize_topic_summary(
            summary,
            category,
            english=prefers_english(raw_query),
        )
        if (
            len(answer) > 1000
            or not 2 <= len(_sentences(answer)) <= 4
            or not _response_language_matches(answer, raw_query)
            or contains_internal_information(answer)
            or not _topic_is_reflected(acknowledgement, safe_summary, category)
            or not _acknowledgement_shape_is_safe(
                acknowledgement,
                safe_summary=safe_summary,
            )
            or not scope_boundary_is_valid(boundary)
            or not _boundary_shape_is_safe(boundary, safe_summary=safe_summary)
            or not is_registry_redirect(redirect)
            or not _redirect_shape_is_safe(redirect)
            or not manifest_claims_are_safe(f"{boundary} {redirect}")
            or not definition_claim_is_absent(answer)
            or not _all_direct_answers_are_absent(
                acknowledgement=acknowledgement,
                full_text=answer,
            )
        ):
            return None
        return answer

    if mode not in SCOPE_MODES - {"out_of_scope"}:
        return None
    try:
        draft = MetaResponseDraft.model_validate(raw_draft)
    except Exception:
        return None
    response = _ensure_sentence(draft.response)
    redirect = _ensure_sentence(draft.registry_redirect)
    answer = f"{response} {redirect}"
    lowered = response.casefold()
    required_tokens = {
        "greeting": ("안녕", "반갑", "hello", "hi", "welcome"),
        "thanks": ("감사", "고마", "도움", "thank", "glad"),
        "farewell": ("안녕", "다음", "또", "goodbye", "bye", "see you"),
        "bot_identity": ("챗봇", "도우미", "assistant", "chatbot"),
        "capability_help": ("역량", "레지스트리", "competency", "registry"),
    }[mode]
    if (
        len(answer) > 1000
        or not 1 <= len(_sentences(answer)) <= 4
        or not _response_language_matches(answer, raw_query)
        or not any(token in lowered for token in required_tokens)
        or contains_internal_information(answer)
        or not _meta_response_shape_is_safe(response)
        or not is_registry_redirect(redirect)
        or not _redirect_shape_is_safe(redirect)
        or not manifest_claims_are_safe(answer)
        or not definition_claim_is_absent(answer)
    ):
        return None
    if mode in {"bot_identity", "capability_help"} and not _has_registry_reference(response):
        return None
    return answer


def validate_registry_scope_note(text: str, *, category: str, summary: str) -> bool:
    """Validate the mixed-input suffix without weakening registry grounding."""

    candidate = normalize_public_text(text)
    if not candidate or len(candidate) > 400 or category not in SCOPE_CATEGORIES:
        return False
    if contains_internal_information(candidate) or not scope_boundary_is_valid(candidate):
        return False
    if not definition_claim_is_absent(candidate):
        return False
    safe_summary = sanitize_topic_summary(
        summary,
        category,
        english=prefers_english(candidate),
    )
    if not _topic_is_reflected(candidate, safe_summary, category):
        return False
    clauses = _clauses(candidate)
    if not clauses or not _controlled_scope_vocabulary(
        clauses[0],
        dynamic_terms=(safe_summary,),
    ):
        return False
    if any(not _controlled_scope_vocabulary(clause) for clause in clauses[1:]):
        return False
    return (
        manifest_claims_are_safe(candidate)
        and _all_direct_answers_are_absent(
            acknowledgement=candidate,
            full_text=candidate,
        )
    )


# The out-of-scope answer is a template, so a single fixed redirect would make
# every such turn read identically.  Each variant is held to the same
# ``_redirect_shape_is_safe`` and ``is_registry_redirect`` contract the writer
# had to meet, which ``test_scope_response`` asserts for all of them.
OUT_OF_SCOPE_REDIRECTS_KO: tuple[str, ...] = (
    "대신 등록 역량의 정의·위계·관계를 물어보거나 행동 특징으로 역량 후보를 찾아볼 수 있습니다",
    "대신 등록 역량의 정의나 부모·자식 관계를 조회할 수 있습니다",
    "원하시면 등록 역량 목록이나 검사 유형의 집계를 살펴볼 수 있습니다",
    "관심 있는 행동 특징을 설명해 주시면 관련된 등록 역량 후보를 찾아볼 수 있습니다",
)
OUT_OF_SCOPE_REDIRECTS_EN: tuple[str, ...] = (
    "You can instead ask for a registered competency definition, hierarchy, "
    "relationship, or behavior-based candidate",
    "You can ask how a registered competency and its parent or child relate",
    "Instead, you can ask for the registered competency hierarchy or a comparison",
    "You can describe a behavior to find related registered competencies",
)
REDIRECT_VARIANT_COUNT = len(OUT_OF_SCOPE_REDIRECTS_KO)


def out_of_scope_redirect(variant: int, *, english: bool) -> str:
    """Pick one redirect deterministically; the caller owns the variant index."""

    variants = OUT_OF_SCOPE_REDIRECTS_EN if english else OUT_OF_SCOPE_REDIRECTS_KO
    return variants[variant % len(variants)]


def out_of_scope_draft(
    *,
    category: str,
    summary: str,
    raw_query: str,
    variant: int = 0,
) -> OutOfScopeResponseDraft:
    """Build the out-of-scope answer parts without calling a model.

    ``sanitize_topic_summary`` is the only place model-derived text enters, and
    it reduces that text to a short noun phrase or a fixed category label.
    """

    english = prefers_english(raw_query)
    if category not in SCOPE_CATEGORIES:
        category = "other"
    safe_summary = sanitize_topic_summary(summary, category, english=english)
    if english:
        return OutOfScopeResponseDraft(
            acknowledgement=f"I can't answer questions about {safe_summary}",
            scope_boundary=(
                "This chat is limited to registered competency information"
            ),
            registry_redirect=out_of_scope_redirect(variant, english=True),
        )
    return OutOfScopeResponseDraft(
        acknowledgement=f"{safe_summary}에 관한 질문은 제가 알려드릴 수 없습니다",
        scope_boundary="이 챗봇은 등록된 역량 정보만 다루는 범위로 한정되어 있습니다",
        registry_redirect=out_of_scope_redirect(variant, english=False),
    )


def scope_template_answer(
    *,
    category: str,
    summary: str,
    raw_query: str,
    variant: int = 0,
) -> str:
    """Render the complete out-of-scope answer as one publishable string.

    This is a normal response path rather than a recovery path, so it is not
    re-validated at request time.  ``test_scope_response`` proves every
    category, language and variant satisfies ``validate_scope_draft`` instead.
    """

    draft = out_of_scope_draft(
        category=category,
        summary=summary,
        raw_query=raw_query,
        variant=variant,
    )
    return " ".join(
        (
            _ensure_sentence(draft.acknowledgement),
            _ensure_sentence(draft.scope_boundary),
            _ensure_sentence(draft.registry_redirect),
        )
    )


def scope_fallback_draft(
    *,
    mode: str,
    category: str,
    summary: str,
    raw_query: str,
    variant: int = 0,
) -> OutOfScopeResponseDraft | MetaResponseDraft:
    """Build the deterministic prose for a scope turn.

    Out-of-scope now uses this as its normal path; the meta modes still use it
    only after the Scope Writer is exhausted.
    """

    english = prefers_english(raw_query)
    if mode == "out_of_scope":
        return out_of_scope_draft(
            category=category,
            summary=summary,
            raw_query=raw_query,
            variant=variant,
        )

    if english:
        responses = {
            "greeting": "Hello, it's good to meet you",
            "thanks": "You're welcome; I'm glad I could help",
            "farewell": "Goodbye, and feel free to return anytime",
            "bot_identity": "I'm a chatbot focused on registered competency information",
            "capability_help": (
                "I can explain registered competency definitions, hierarchies, "
                "relationships, aggregates, comparisons, and behavior-based candidates"
            ),
        }
        redirect = "You can ask about a registered competency or describe a behavior to find candidates"
    else:
        responses = {
            "greeting": "안녕하세요, 반갑습니다",
            "thanks": "고맙습니다, 도움이 되었다니 기쁩니다",
            "farewell": "안녕히 가세요, 다음에 또 뵙겠습니다",
            "bot_identity": "저는 등록된 역량 정보를 안내하는 역량 챗봇입니다",
            "capability_help": (
                "등록 역량의 정의·위계·관계·집계·비교와 행동 기반 후보 찾기를 "
                "도와드릴 수 있습니다"
            ),
        }
        redirect = "등록된 역량명을 묻거나 행동 특징을 설명해 역량 후보를 찾아보세요"
    return MetaResponseDraft(
        response=responses.get(mode, responses["greeting"]),
        registry_redirect=redirect,
    )
