"""등록된 역량을 자연어로 조회하고 대화 맥락을 유지하는 LangGraph."""

import os
import re
import threading
from contextlib import ExitStack
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

QueryType = Literal[
    "named_lookup",
    "semantic_search",
    "out_of_scope",
]

LlmRoute = Literal[
    "find_competencies",
    "find_semantic_candidates",
    "handle_unknown",
    "handle_out_of_scope",
]

class ParsedNaturalLanguageQuery(BaseModel):
    """질의 분류 LLM이 반환할 구조화 형식."""

    query_type: QueryType
    competency_names: list[str] = Field(default_factory=list)
    requested_fields: list[RequestedField] = Field(default_factory=list)
    semantic_query: str | None = None
    reuse_previous_fields: bool = False

class SemanticSelection(BaseModel):
    """의미 검색 LLM이 반환할 구조화 형식."""

    candidate_names: list[str] = Field(
        default_factory=list,
        max_length=3,
    )

class CompetencyState(MessagesState):
    """그래프의 모든 노드가 공유하는 상태."""

    raw_query: NotRequired[str]
    requested_fields: NotRequired[list[RequestedField]]
    resolved_names: NotRequired[list[str]]
    matched_items: NotRequired[list[dict]]
    candidate_names: NotRequired[list[str]]
    semantic_query: NotRequired[str]
    llm_route: NotRequired[LlmRoute]

    # 이전 턴에서 확정된 대상을 후속 질문에 사용한다.
    last_resolved_names: NotRequired[list[str]]

# ---------------------------------------------------------------------------
# 공통 레지스트리와 정확 일치 도구
# ---------------------------------------------------------------------------

# lookup에는 정식 이름뿐 아니라 구역검 별칭도 들어 있다.
# 이 딕셔너리는 정식 역량명만 key로 사용하는 조회용 딕셔너리다.
CANONICAL_LOOKUP = {
    item["name"]: item
    for item in lookup.values()
}

CANONICAL_NAMES = sorted(CANONICAL_LOOKUP)

DEFAULT_FIELDS: list[RequestedField] = [
    "definition",
    "children",
    "path",
]

EXACT_NAME_SEPARATOR = re.compile(r"[,，\n]+")


def extract_exact_registered_names(query: str) -> list[str]:
    """입력 전체가 정확한 등록 이름 목록일 때만 정식 이름을 반환한다."""

    stripped = query.strip()

    if not stripped:
        return []

    raw_names = [
        part.strip()
        for part in EXACT_NAME_SEPARATOR.split(stripped)
    ]

    if not raw_names or any(not name for name in raw_names):
        return []

    resolved: list[str] = []
    seen_ids: set[str] = set()

    for raw_name in raw_names:
        item = lookup.get(raw_name)

        # 한 항목이라도 정확히 등록된 이름/별칭이 아니면 전체 문장을
        # 자연어 질의로 보고 LLM 파서에 맡긴다.
        if item is None:
            return []

        if item["id"] in seen_ids:
            continue

        resolved.append(item["name"])
        seen_ids.add(item["id"])

    return resolved

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
# OpenAI 구조화 출력 준비
# ---------------------------------------------------------------------------

def selected_model_name() -> str:
    """환경 변수로 모델을 바꿀 수 있게 하되 기본값을 제공한다."""

    return os.getenv("OPENAI_MODEL", "").strip() or "gpt-5.4-mini"

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
    (
        f"- {name}"
        + (
            f" (등록 별칭: {', '.join(CANONICAL_LOOKUP[name].get('aliases', []))})"
            if CANONICAL_LOOKUP[name].get("aliases")
            else ""
        )
    )
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
    """입력 전체가 정확한 등록 이름인지 여부만 Python으로 판별한다."""

    query = str(state["messages"][-1].content).strip()
    resolved_names = extract_exact_registered_names(query)

    updates: dict[str, Any] = {
        "raw_query": query,
        "resolved_names": resolved_names,
        "matched_items": [],
        "semantic_query": "",
    }

    if resolved_names:
        previous_candidates = validate_registry_names(
            list(state.get("candidate_names", []))
        )
        previous_fields = normalize_requested_fields(
            list(state.get("requested_fields", []))
        )
        selected_previous_candidate = any(
            name in previous_candidates
            for name in resolved_names
        )

        updates["requested_fields"] = (
            previous_fields
            if selected_previous_candidate and previous_fields
            else DEFAULT_FIELDS.copy()
        )
        updates["candidate_names"] = []

    return updates


def route_after_interpret(
    state: CompetencyState,
) -> Literal[
    "find_competencies",
    "llm_interpret_query",
]:
    """정확한 이름만 Python 경로로 보내고 나머지는 모두 LLM으로 보낸다."""

    if state.get("resolved_names"):
        return "find_competencies"

    return "llm_interpret_query"

def llm_interpret_query(state: CompetencyState) -> dict:
    """구조화 출력으로 질문의 종류와 요청 필드를 분류한다."""

    query = state.get("raw_query", "")
    previous_candidates = validate_registry_names(
        list(state.get("candidate_names", []))
    )
    previous_names = validate_registry_names(
        list(state.get("last_resolved_names", []))
    )
    previous_fields = normalize_requested_fields(
        list(state.get("requested_fields", []))
    )

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
7. 사용자가 앞선 후보의 번호나 이름을 선택하면 이전 후보 목록에서 정식 이름을
   찾아 competency_names에 넣고 named_lookup으로 분류합니다.
8. "그 역량", "그거"처럼 앞선 대상을 가리키면 이전 확정 역량을
   competency_names에 넣고 named_lookup으로 분류합니다.
9. 후속 질문에서 정보 종류를 새로 말하지 않았다면 reuse_previous_fields를
   true로 설정합니다. 그 외에는 false입니다.
10. 정의를 직접 만들거나 추측하지 마세요.

이전 대화 문맥:
- 이전 확정 역량: {previous_names or '없음'}
- 이전 후보 목록: {previous_candidates or '없음'}
- 이전 요청 필드: {previous_fields or '없음'}

허용된 정식 역량명과 등록 별칭:
{REGISTRY_NAME_LIST}
""".strip()

    parsed = _query_parser_for(
        selected_model_name()
    ).invoke(
        [
            ("system", system_prompt),
            ("human", query),
        ]
    )

    parsed_fields = normalize_requested_fields(
        list(parsed.requested_fields)
    )

    if not parsed_fields:
        parsed_fields = (
            previous_fields
            if parsed.reuse_previous_fields and previous_fields
            else DEFAULT_FIELDS.copy()
        )

    if parsed.query_type == "named_lookup":
        names = validate_registry_names(
            parsed.competency_names
        )

        if names:
            return {
                "resolved_names": names,
                "requested_fields": parsed_fields,
                "candidate_names": [],
                "llm_route": "find_competencies",
            }

        return {
            "requested_fields": parsed_fields,
            "candidate_names": [],
            "llm_route": "handle_unknown",
        }

    if parsed.query_type == "semantic_search":
        semantic_query = (
            parsed.semantic_query
            or query
        ).strip()

        return {
            "semantic_query": semantic_query,
            "requested_fields": parsed_fields,
            "candidate_names": [],
            "llm_route": "find_semantic_candidates",
        }

    return {
        "candidate_names": [],
        "llm_route": "handle_out_of_scope",
    }

def route_after_llm(state: CompetencyState) -> LlmRoute:
    """구조화된 LLM 판정값에 따라 다음 노드를 선택한다."""

    return state.get("llm_route", "handle_unknown")

def find_semantic_candidates(state: CompetencyState) -> dict:
    """정의와 위계를 비교해 의미상 가까운 후보를 고른다."""

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

    return {
        "candidate_names": candidates,
    }

def route_after_candidates(
    state: CompetencyState,
) -> Literal[
    "present_candidates",
    "handle_unknown",
]:
    """후보가 있으면 확인을 받고, 없으면 재질문을 요청한다."""

    if state.get("candidate_names"):
        return "present_candidates"

    return "handle_unknown"

def find_competencies(state: CompetencyState) -> dict:
    """확정된 정식 역량명으로 레지스트리를 조회한다."""

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

    query = state.get("raw_query", "")

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
    route_after_candidates,
)
builder.add_edge("find_competencies", "produce_answer")
builder.add_edge("produce_answer", END)
builder.add_edge("present_candidates", END)
builder.add_edge("handle_unknown", END)
builder.add_edge("handle_out_of_scope", END)

_runtime_lock = threading.RLock()
_runtime_stack: ExitStack | None = None
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

    # 테이블은 setup_checkpoint_database.py로 미리 준비한다.
    return runtime_stack.enter_context(
        PostgresSaver.from_conn_string(_get_database_url())
    )


def initialize_competency_runtime() -> None:
    """그래프와 PostgreSQL 체크포인터를 프로세스당 한 번 준비한다."""

    global app, _runtime_stack

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
        app = compiled_app


def close_competency_runtime() -> None:
    """웹 서버 또는 CLI 종료 시 PostgreSQL 연결을 안전하게 닫는다."""

    global app, _runtime_stack

    with _runtime_lock:
        runtime_stack = _runtime_stack

        # 이후 호출에서 다시 초기화할 수 있도록 참조를 먼저 비운다.
        app = None
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
