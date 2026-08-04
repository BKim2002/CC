# competency_interpreter.py에 자연어 입력 처리 기능을 단계적으로 추가
# 체크포인트 A: 규칙 기반 이름/의도 추출과 오타 후보 제시
# 체크포인트 B: OpenAI 구조화 출력을 이용한 자연어 질의 분류
# 체크포인트 C: 정의와 위계 정보를 이용한 의미 기반 후보 검색
# 체크포인트 D: LangGraph 체크포인터를 이용한 후속 질문 처리

import os
import re
import threading
from contextlib import ExitStack
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Literal, NotRequired

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from competency_registry import lookup

# 프로젝트 폴더의 .env가 있으면 OPENAI_API_KEY 등을 읽는다.
# 이미 운영체제 환경 변수에 값이 있으면 그 값을 덮어쓰지 않는다.
load_dotenv()

RequestedField = Literal[
    "definition",
    "children",
    "path",
]

Intent = Literal[
    "definition",
    "children",
    "hierarchy",
    "full_info",
    "mixed",
]

QueryType = Literal[
    "named_lookup",
    "semantic_search",
    "out_of_scope",
]

LlmRoute = Literal[
    "find_competencies",
    "find_semantic_candidates",
    "present_candidates",
    "handle_unknown",
    "handle_out_of_scope",
]

class ParsedNaturalLanguageQuery(BaseModel):
    """체크포인트 B에서 LLM이 반드시 맞춰 반환할 출력 형식."""

    query_type: QueryType
    competency_names: list[str] = Field(default_factory=list)
    requested_fields: list[RequestedField] = Field(default_factory=list)
    semantic_query: str | None = None

class SemanticSelection(BaseModel):
    """체크포인트 C에서 LLM이 반드시 맞춰 반환할 출력 형식."""

    candidate_names: list[str] = Field(
        default_factory=list,
        max_length=3,
    )

class CompetencyState(MessagesState):
    """그래프의 모든 노드가 공유하는 상태."""

    raw_query: NotRequired[str]
    intent: NotRequired[Intent]
    requested_fields: NotRequired[list[RequestedField]]
    resolved_names: NotRequired[list[str]]
    matched_items: NotRequired[list[dict]]
    candidate_names: NotRequired[list[str]]
    unknown_query: NotRequired[str]
    semantic_query: NotRequired[str]
    llm_route: NotRequired[LlmRoute]

    # 체크포인트 D에서 이전 턴의 대상을 기억하는 값이다.
    last_resolved_names: NotRequired[list[str]]

# ---------------------------------------------------------------------------
# 공통 레지스트리와 규칙 기반 도구: 체크포인트 A
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """검색 시 무시할 공백과 일부 기호를 제거한다."""

    return re.sub(r"[\s_\-·.]+", "", text).lower()

# lookup에는 정식 이름뿐 아니라 구역검 별칭도 들어 있다.
# 이 딕셔너리는 정식 역량명만 key로 사용하는 조회용 딕셔너리다.
CANONICAL_LOOKUP = {
    item["name"]: item
    for item in lookup.values()
}

CANONICAL_NAMES = sorted(CANONICAL_LOOKUP)

# 자연어 문장 안에서 정식 이름과 별칭을 모두 검색하기 위한 목록이다.
# 각 원소는 ("정규화한 검색 이름", 역량 정보) 형태의 tuple이다.
SEARCH_ENTRIES: list[tuple[str, dict]] = [
    (normalize_text(search_name), item)
    for search_name, item in lookup.items()
]

# 짧은 이름이 긴 이름의 일부로 포함될 수 있으므로 긴 이름부터 확인한다.
SEARCH_ENTRIES.sort(
    key=lambda pair: len(pair[0]),
    reverse=True,
)

FIELD_KEYWORDS: dict[RequestedField, tuple[str, ...]] = {
    "definition": ("정의", "뜻", "의미"),
    "children": (
        "하위요인",
        "하위 요인",
        "구성요인",
        "구성 요인",
    ),
    "path": (
        "위계",
        "상위요인",
        "상위 요인",
        "어디에 속",
        "소속",
    ),
}

DEFAULT_FIELDS: list[RequestedField] = [
    "definition",
    "children",
    "path",
]

STOP_PHRASES = (
    "하위요인",
    "하위 요인",
    "구성요인",
    "구성 요인",
    "상위요인",
    "상위 요인",
    "정의",
    "뜻",
    "의미",
    "위계",
    "어디에 속해",
    "어디에 속",
    "소속",
    "알려줘",
    "설명해줘",
    "설명",
    "뭐야",
    "무엇이야",
    "궁금해",
)

PARTICLE_PATTERN = re.compile(
    r"(으로|에서|에게|까지|부터|처럼|보다|"
    r"하고|이랑|랑|의|은|는|이|가|을|를|"
    r"와|과|도|에)$"
)

FOLLOW_UP_REFERENCES = (
    "그 역량",
    "이 역량",
    "그 요인",
    "이 요인",
    "그것",
    "그거",
    "그걸",
    "해당 역량",
)

AFFIRMATIVE_ANSWERS = {
    "네",
    "예",
    "응",
    "맞아",
    "맞아요",
    "맞습니다",
    "그래",
    "그거",
    "그걸로",
}

ORDINAL_PATTERNS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, (r"(?<!\d)1\s*(?:번|번째)(?!\d)", r"첫\s*번째", r"첫째")),
    (1, (r"(?<!\d)2\s*(?:번|번째)(?!\d)", r"두\s*번째", r"둘째")),
    (2, (r"(?<!\d)3\s*(?:번|번째)(?!\d)", r"세\s*번째", r"셋째")),
)

def extract_registered_names(query: str) -> list[str]:
    """자연어 문장에서 등록된 정식 역량명 또는 별칭을 찾는다."""

    normalized_query = normalize_text(query)
    hits: list[tuple[int, int, dict]] = []

    for normalized_name, item in SEARCH_ENTRIES:
        position = normalized_query.find(normalized_name)

        if position >= 0:
            hits.append((position, -len(normalized_name), item))

    # 사용자가 문장에서 언급한 순서를 유지한다.
    # 같은 위치에서는 더 긴 이름을 먼저 둔다.
    hits.sort(key=lambda hit: (hit[0], hit[1]))

    names: list[str] = []
    seen_ids: set[str] = set()

    for _, _, item in hits:
        if item["id"] in seen_ids:
            continue

        names.append(item["name"])
        seen_ids.add(item["id"])

    return names

def normalize_requested_fields(
    fields: list[str],
) -> list[RequestedField]:
    """허용된 필드만 중복 없이 남긴다."""

    allowed = {"definition", "children", "path"}
    normalized: list[RequestedField] = []

    for field in fields:
        if field not in allowed or field in normalized:
            continue

        normalized.append(field)  # type: ignore[arg-type]

    return normalized

def intent_from_fields(
    requested_fields: list[RequestedField],
) -> Intent:
    """요청 필드 목록을 기존 intent 값으로 변환한다."""

    if not requested_fields or requested_fields == DEFAULT_FIELDS:
        return "full_info"

    if len(requested_fields) > 1:
        return "mixed"

    intent_by_field: dict[RequestedField, Intent] = {
        "definition": "definition",
        "children": "children",
        "path": "hierarchy",
    }

    return intent_by_field[requested_fields[0]]

def has_explicit_field_request(query: str) -> bool:
    """사용자가 이번 문장에서 정보 종류를 직접 말했는지 확인한다."""

    return any(
        keyword in query
        for keywords in FIELD_KEYWORDS.values()
        for keyword in keywords
    )

def detect_intent(
    query: str,
) -> tuple[Intent, list[RequestedField]]:
    """질문에서 사용자가 원하는 정보의 종류를 판별한다."""

    requested_fields: list[RequestedField] = []

    for field, keywords in FIELD_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            requested_fields.append(field)

    # 정보 종류가 명시되지 않으면 기존 스크립트처럼 전체 정보를 출력한다.
    if not requested_fields:
        requested_fields = DEFAULT_FIELDS.copy()

    return intent_from_fields(requested_fields), requested_fields

def extract_candidate_tokens(query: str) -> list[str]:
    """오타 후보 검색에 사용할 핵심 단어를 질문에서 추출한다."""

    cleaned = query

    for phrase in STOP_PHRASES:
        cleaned = cleaned.replace(phrase, " ")

    cleaned = re.sub(r"[^0-9A-Za-z가-힣_\-]+", " ", cleaned)

    tokens: list[str] = []

    for raw_token in cleaned.split():
        token = raw_token
        previous = None

        # "성실셩의"처럼 단어 뒤에 붙은 한국어 조사를 제거한다.
        while token != previous:
            previous = token
            token = PARTICLE_PATTERN.sub("", token)

        if len(normalize_text(token)) >= 2:
            tokens.append(token)

    return tokens

def suggest_names(
    query: str,
    threshold: float = 0.65,
    limit: int = 3,
) -> list[str]:
    """질문의 핵심 단어와 유사한 정식 역량명을 후보로 반환한다."""

    scores: dict[str, float] = {}

    for token in extract_candidate_tokens(query):
        normalized_token = normalize_text(token)

        for name in CANONICAL_NAMES:
            score = SequenceMatcher(
                None,
                normalized_token,
                normalize_text(name),
            ).ratio()

            if score >= threshold:
                scores[name] = max(score, scores.get(name, 0.0))

    sorted_names = sorted(
        scores,
        key=lambda name: scores[name],
        reverse=True,
    )

    return sorted_names[:limit]

def validate_registry_names(names: list[str]) -> list[str]:
    """LLM이 반환한 이름을 레지스트리에서 다시 검증한다."""

    validated: list[str] = []
    seen_ids: set[str] = set()

    for raw_name in names:
        name = raw_name.strip()
        item = lookup.get(name) or CANONICAL_LOOKUP.get(name)

        if item is None or item["id"] in seen_ids:
            continue

        validated.append(item["name"])
        seen_ids.add(item["id"])

    return validated

# ---------------------------------------------------------------------------
# 대화 맥락 해석: 체크포인트 D
# ---------------------------------------------------------------------------

def ordinal_index(query: str) -> int | None:
    """'1번', '두 번째' 같은 후보 번호를 0부터 시작하는 값으로 바꾼다."""

    stripped = query.strip()

    if stripped in {"1", "2", "3"}:
        return int(stripped) - 1

    for index, patterns in ORDINAL_PATTERNS:
        if any(re.search(pattern, query) for pattern in patterns):
            return index

    return None

def resolve_follow_up(
    query: str,
    previous_candidates: list[str],
    previous_names: list[str],
) -> list[str]:
    """이전 후보/답변을 이용해 생략된 역량명을 복원한다."""

    if previous_candidates:
        index = ordinal_index(query)

        if index is not None and index < len(previous_candidates):
            return [previous_candidates[index]]

        normalized_query = normalize_text(query)
        is_affirmative = normalized_query in {
            normalize_text(answer)
            for answer in AFFIRMATIVE_ANSWERS
        }
        refers_to_one = any(
            phrase in query
            for phrase in FOLLOW_UP_REFERENCES
        )

        if len(previous_candidates) == 1 and (
            is_affirmative or refers_to_one
        ):
            return [previous_candidates[0]]

    if previous_names:
        refers_to_previous = any(
            phrase in query
            for phrase in FOLLOW_UP_REFERENCES
        )

        # "하위요인도 알려줘"처럼 정보 종류만 이어서 말한 경우도
        # 직전에 확정된 역량을 대상으로 본다.
        if refers_to_previous or has_explicit_field_request(query):
            return previous_names

    return []

# ---------------------------------------------------------------------------
# OpenAI 구조화 출력 준비: 체크포인트 B와 C
# ---------------------------------------------------------------------------

def openai_is_configured() -> bool:
    """API 키의 실제 값을 노출하지 않고 설정 여부만 확인한다."""

    return bool(os.getenv("OPENAI_API_KEY", "").strip())

def selected_model_name() -> str:
    """환경 변수로 모델을 바꿀 수 있게 하되 기본값을 제공한다."""

    return os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()

@lru_cache(maxsize=4)
def _query_parser_for(model_name: str):
    """질의 분류용 구조화 출력 모델을 필요할 때 한 번만 만든다."""

    model = ChatOpenAI(
        model=model_name,
        max_retries=1,
        timeout=30,
    )

    return model.with_structured_output(
        ParsedNaturalLanguageQuery,
        method="json_schema",
    )

@lru_cache(maxsize=4)
def _semantic_selector_for(model_name: str):
    """의미 기반 후보 선택용 구조화 출력 모델을 만든다."""

    model = ChatOpenAI(
        model=model_name,
        max_retries=1,
        timeout=30,
    )

    return model.with_structured_output(
        SemanticSelection,
        method="json_schema",
    )

REGISTRY_NAME_LIST = "\n".join(
    f"- {name}"
    for name in CANONICAL_NAMES
)

def build_registry_catalog() -> str:
    """의미 검색에 필요한 레지스트리 정보를 짧은 목록으로 만든다."""

    lines: list[str] = []

    for name in CANONICAL_NAMES:
        item = CANONICAL_LOOKUP[name]
        definition = (
            item.get("definition")
            or "독립적인 정의가 제공되어 있지 않음"
        )
        path = " > ".join(item.get("path", [])) or "위계 정보 없음"

        lines.append(
            f"- 이름: {name}\n"
            f"  정의: {definition}\n"
            f"  위계: {path}"
        )

    return "\n".join(lines)

REGISTRY_CATALOG = build_registry_catalog()

# ---------------------------------------------------------------------------
# LangGraph 노드
# ---------------------------------------------------------------------------

def interpret_query(state: CompetencyState) -> dict:
    """노드 1: 현재 질문을 규칙으로 먼저 해석하고 후속 질문을 복원한다."""

    query = str(state["messages"][-1].content).strip()

    previous_fields = list(state.get("requested_fields", []))
    previous_candidates = list(state.get("candidate_names", []))
    previous_names = list(state.get("last_resolved_names", []))

    intent, requested_fields = detect_intent(query)
    explicit_fields = has_explicit_field_request(query)
    resolved_names = extract_registered_names(query)
    continued_from_previous = False

    if not resolved_names:
        resolved_names = resolve_follow_up(
            query,
            previous_candidates,
            previous_names,
        )
        continued_from_previous = bool(resolved_names)
    elif previous_candidates:
        # 후보 목록 다음 턴에 사용자가 정확한 이름만 입력한 경우다.
        continued_from_previous = any(
            name in previous_candidates
            for name in resolved_names
        )

    # "1번"처럼 정보 종류가 생략된 후속 질문은 앞선 요청을 이어받는다.
    if (
        continued_from_previous
        and not explicit_fields
        and previous_fields
    ):
        requested_fields = normalize_requested_fields(previous_fields)

        if not requested_fields:
            requested_fields = DEFAULT_FIELDS.copy()

        intent = intent_from_fields(requested_fields)

    return {
        "raw_query": query,
        "intent": intent,
        "requested_fields": requested_fields,
        "resolved_names": resolved_names,
        "matched_items": [],
        "candidate_names": [],
        "unknown_query": "" if resolved_names else query,
        "semantic_query": "",
    }

def route_after_interpret(
    state: CompetencyState,
) -> Literal[
    "find_competencies",
    "llm_interpret_query",
    "suggest_candidates",
]:
    """정확한 이름, API 사용 가능 여부에 따라 다음 노드를 선택한다."""

    if state.get("resolved_names"):
        return "find_competencies"

    if openai_is_configured():
        return "llm_interpret_query"

    return "suggest_candidates"

def llm_interpret_query(state: CompetencyState) -> dict:
    """노드 2B: 구조화 출력으로 질문의 종류와 요청 필드를 분류한다."""

    query = state.get("raw_query", "")

    system_prompt = f"""
당신은 역량 레지스트리 질의를 분류하는 파서입니다.
답변 문장을 쓰지 말고 지정된 구조화 형식만 반환하세요.

규칙:
1. 사용자가 정식 역량명, 별칭, 또는 명백한 철자 오류를 말하면
   query_type은 named_lookup입니다.
2. 이름 대신 행동이나 의미를 설명하며 알맞은 역량을 찾으면
   query_type은 semantic_search입니다.
3. 역량과 관계없는 질문이면 query_type은 out_of_scope입니다.
4. competency_names에는 아래 목록에 있는 정식 이름만 넣습니다.
5. requested_fields는 definition, children, path 중 요청한 것만 넣습니다.
6. semantic_search이면 semantic_query에 사용자의 검색 의도를 짧게 적습니다.
7. 정의를 직접 만들거나 추측하지 마세요.

허용된 정식 역량명:
{REGISTRY_NAME_LIST}
""".strip()

    try:
        parsed = _query_parser_for(
            selected_model_name()
        ).invoke(
            [
                ("system", system_prompt),
                ("human", query),
            ]
        )
    except Exception:
        candidates = suggest_names(query)

        return {
            "candidate_names": candidates,
            "llm_route": (
                "present_candidates"
                if candidates
                else "handle_unknown"
            ),
        }

    parsed_fields = normalize_requested_fields(
        list(parsed.requested_fields)
    )

    if not parsed_fields:
        parsed_fields = list(
            state.get("requested_fields", DEFAULT_FIELDS)
        )

    if parsed.query_type == "named_lookup":
        names = validate_registry_names(
            parsed.competency_names
        )

        if names:
            return {
                "resolved_names": names,
                "requested_fields": parsed_fields,
                "intent": intent_from_fields(parsed_fields),
                "llm_route": "find_competencies",
            }

        candidates = suggest_names(query)

        if candidates:
            return {
                "candidate_names": candidates,
                "requested_fields": parsed_fields,
                "llm_route": "present_candidates",
            }
        
        semantic_query = (
            parsed.semantic_query
            or query
        ).strip()

        return {
            "semantic_query": semantic_query,
            "requested_fields": parsed_fields,
            "llm_route": "find_semantic_candidates",
        }

    if parsed.query_type == "semantic_search":
        semantic_query = (
            parsed.semantic_query
            or query
        ).strip()

        return {
            "semantic_query": semantic_query,
            "requested_fields": parsed_fields,
            "llm_route": "find_semantic_candidates",
        }
    
    # LLM이 범위 밖으로 분류했더라도 뚜렷한 오타 후보는 먼저 살린다.
    candidates = suggest_names(query)

    if candidates:
        return {
            "candidate_names": candidates,
            "requested_fields": parsed_fields,
            "llm_route": "present_candidates",
        }

    return {
        "llm_route": "handle_out_of_scope",
    }

def route_after_llm(state: CompetencyState) -> LlmRoute:
    """구조화된 LLM 판정값에 따라 다음 노드를 선택한다."""

    return state.get("llm_route", "handle_unknown")

def find_semantic_candidates(state: CompetencyState) -> dict:
    """노드 2C: 정의와 위계를 비교해 의미상 가까운 후보를 고른다."""

    semantic_query = (
        state.get("semantic_query")
        or state.get("raw_query", "")
    )

    system_prompt = f"""
당신은 사용자의 설명과 역량 레지스트리를 비교하는 검색기입니다.
아래 레지스트리에서 의미상 가장 가까운 정식 역량명을 최대 3개 고르세요.
확신할 후보가 없으면 빈 목록을 반환하세요.
반드시 목록에 있는 이름만 반환하고 정의를 새로 만들지 마세요.

역량 레지스트리:
{REGISTRY_CATALOG}
""".strip()

    try:
        selection = _semantic_selector_for(
            selected_model_name()
        ).invoke(
            [
                ("system", system_prompt),
                ("human", semantic_query),
            ]
        )
        candidates = validate_registry_names(
            selection.candidate_names
        )[:3]
    except Exception:
        candidates = suggest_names(
            state.get("raw_query", "")
        )

    return {
        "candidate_names": candidates,
    }

def route_after_semantic_search(
    state: CompetencyState,
) -> Literal[
    "present_candidates",
    "handle_unknown",
]:
    """의미 검색 결과가 있으면 확인을 받고, 없으면 재질문을 요청한다."""

    if state.get("candidate_names"):
        return "present_candidates"

    return "handle_unknown"

def find_competencies(state: CompetencyState) -> dict:
    """노드 2A: 확정된 정식 역량명으로 레지스트리를 조회한다."""

    matched_items: list[dict] = []
    seen_ids: set[str] = set()

    for name in state.get("resolved_names", []):
        item = CANONICAL_LOOKUP.get(name)

        if item is None or item["id"] in seen_ids:
            continue

        matched_items.append(item)
        seen_ids.add(item["id"])

    return {
        "matched_items": matched_items,
        "last_resolved_names": [
            item["name"]
            for item in matched_items
        ],
        "candidate_names": [],
    }

def suggest_candidates(state: CompetencyState) -> dict:
    """정확한 이름이 없을 때 철자가 비슷한 정식 역량명을 찾는다."""

    candidate_names = suggest_names(
        state.get("raw_query", "")
    )

    return {
        "candidate_names": candidate_names,
    }

def route_after_suggestions(
    state: CompetencyState,
) -> Literal[
    "present_candidates",
    "handle_unknown",
]:
    """유사 후보가 존재하는지에 따라 다음 노드를 결정한다."""

    if state.get("candidate_names"):
        return "present_candidates"

    return "handle_unknown"

def produce_answer(state: CompetencyState) -> dict:
    """확정된 레지스트리 항목에서 사용자가 요청한 정보만 답한다."""

    requested_fields = set(
        state.get("requested_fields", DEFAULT_FIELDS)
    )
    matched_items = state.get("matched_items", [])

    if not matched_items:
        return {
            "messages": AIMessage(
                content="확정된 역량을 레지스트리에서 찾지 못했습니다."
            )
        }

    answers: list[str] = []

    for item in matched_items:
        sentences = [f"역량명: {item['name']}"]

        if "definition" in requested_fields:
            definition = item.get("definition")
            definition_status = item.get("definition_status")

            if definition_status == "not_provided" or not definition:
                sentences.append(
                    "정의: 원본 문서에 독립적인 정의가 "
                    "제공되어 있지 않습니다."
                )
            else:
                sentences.append(f"정의: {definition}")

        if "children" in requested_fields:
            children = item.get("children", [])

            if children:
                sentences.append(
                    f"하위요인: {', '.join(children)}"
                )
            else:
                sentences.append("하위요인: 없습니다.")

        if "path" in requested_fields:
            path = " > ".join(item.get("path", []))

            if path:
                sentences.append(f"위계 구조: {path}")

        answers.append("\n".join(sentences))

    return {
        "messages": AIMessage(
            content="\n\n".join(answers)
        )
    }

def present_candidates(state: CompetencyState) -> dict:
    """오타 또는 의미 검색 후보를 보여주고 다음 턴의 확인을 기다린다."""

    lines = [
        "입력한 표현과 관련된 역량 후보를 찾았습니다."
    ]

    for number, name in enumerate(
        state.get("candidate_names", []),
        start=1,
    ):
        item = CANONICAL_LOOKUP[name]
        definition = (
            item.get("definition")
            or "독립적인 정의가 제공되어 있지 않음"
        )

        lines.append(
            f"{number}. {name}: {definition}"
        )

    lines.append(
        "원하는 후보의 번호(예: 1번) 또는 정확한 역량명을 입력해 주세요."
    )

    return {
        "messages": AIMessage(
            content="\n".join(lines)
        )
    }

def handle_unknown(state: CompetencyState) -> dict:
    """이름과 후보를 모두 찾지 못했음을 안내한다."""

    query = state.get("unknown_query", "")

    message = (
        "입력에서 등록된 역량이나 관련 후보를 "
        f"찾지 못했습니다: {query}\n"
        "역량명 또는 찾고 싶은 행동의 특징을 조금 더 구체적으로 말해 주세요."
    )

    return {
        "messages": AIMessage(content=message)
    }

def handle_out_of_scope(state: CompetencyState) -> dict:
    """역량 레지스트리와 관계없는 질문임을 안내한다."""

    return {
        "messages": AIMessage(
            content=(
                "이 챗봇은 역량의 정의, 하위요인, 위계 구조와 "
                "관련 역량 찾기 질문에 답합니다. "
                "역량에 관한 질문으로 다시 입력해 주세요."
            )
        )
    }

# 그래프 구성

builder = StateGraph(CompetencyState)

builder.add_node("interpret_query", interpret_query)
builder.add_node("llm_interpret_query", llm_interpret_query)
builder.add_node("find_semantic_candidates", find_semantic_candidates)
builder.add_node("find_competencies", find_competencies)
builder.add_node("suggest_candidates", suggest_candidates)
builder.add_node("produce_answer", produce_answer)
builder.add_node("present_candidates", present_candidates)
builder.add_node("handle_unknown", handle_unknown)
builder.add_node("handle_out_of_scope", handle_out_of_scope)

builder.add_edge(START, "interpret_query")
builder.add_conditional_edges(
    "interpret_query",
    route_after_interpret,
)
builder.add_conditional_edges(
    "llm_interpret_query",
    route_after_llm,
)
builder.add_conditional_edges(
    "find_semantic_candidates",
    route_after_semantic_search,
)
builder.add_conditional_edges(
    "suggest_candidates",
    route_after_suggestions,
)
builder.add_edge("find_competencies", "produce_answer")
builder.add_edge("produce_answer", END)
builder.add_edge("present_candidates", END)
builder.add_edge("handle_unknown", END)
builder.add_edge("handle_out_of_scope", END)

_runtime_lock = threading.RLock()
_runtime_stack: ExitStack | None = None
checkpointer: BaseCheckpointSaver | None = None
app: Any | None = None


def _get_database_url() -> str:
    """PostgreSQL 연결 문자열을 환경 변수에서 읽는다."""

    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL 환경 변수가 설정되지 않았습니다."
        )

    return database_url


def _open_checkpointer(
    runtime_stack: ExitStack,
) -> BaseCheckpointSaver:
    """런타임 동안 유지할 PostgreSQL 체크포인터를 연다."""

    # 테이블은 배포 전에 이미 만들어 두었으므로 setup()은 호출하지 않는다.
    return runtime_stack.enter_context(
        PostgresSaver.from_conn_string(_get_database_url())
    )


def initialize_competency_runtime() -> None:
    """그래프와 PostgreSQL 체크포인터를 프로세스당 한 번 준비한다."""

    global app, _runtime_stack, checkpointer

    with _runtime_lock:
        if app is not None:
            return

        runtime_stack = ExitStack()

        try:
            saver = _open_checkpointer(runtime_stack)
            compiled_app = builder.compile(
                checkpointer=saver
            )
        except Exception:
            runtime_stack.close()
            raise

        _runtime_stack = runtime_stack
        checkpointer = saver
        app = compiled_app


def close_competency_runtime() -> None:
    """웹 서버 또는 CLI 종료 시 PostgreSQL 연결을 안전하게 닫는다."""

    global app, _runtime_stack, checkpointer

    with _runtime_lock:
        runtime_stack = _runtime_stack

        # 이후 호출에서 다시 초기화할 수 있도록 참조를 먼저 비운다.
        app = None
        checkpointer = None
        _runtime_stack = None

        if runtime_stack is not None:
            runtime_stack.close()


def run_competency(
    question: str,
    thread_id: str,
) -> dict[str, Any]:
    """LangGraph를 실행하고 웹 API가 사용할 전체 상태를 반환한다."""

    with _runtime_lock:
        initialize_competency_runtime()
        assert app is not None

        return app.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            },
        )


def get_competency_state(
    thread_id: str,
) -> dict[str, Any]:
    """스레드의 최신 내부 상태를 읽되 새 메시지는 추가하지 않는다."""

    with _runtime_lock:
        initialize_competency_runtime()
        assert app is not None

        snapshot = app.get_state(
            {
                "configurable": {
                    "thread_id": thread_id,
                }
            }
        )

        if not snapshot.values:
            return {}

        return dict(snapshot.values)


def ask_competency(
    question: str,
    thread_id: str = "competency-chat-1",
) -> str:
    """같은 thread_id로 호출하면 앞선 후보와 역량을 기억한다."""

    result = run_competency(
        question,
        thread_id,
    )

    return str(result["messages"][-1].content)


if __name__ == "__main__":
    try:
        print(
            ask_competency(
                "환경긍정과 과활성의 정의를 알려줘."
            )
        )
    finally:
        close_competency_runtime()
