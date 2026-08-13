"""프로세스 수명과 스레드별 대화 보존.

v1의 그래프는 노드가 열둘이었고 런타임 코드가 250줄이었다 -- 비동기
체크포인터 어댑터, 스레드 락 테이블, ExitStack 수명 관리.  분기가 없는
v2에서는 그 대부분이 필요 없다.

여기 남는 것은 둘뿐이다.

* **레지스트리 스냅샷을 한 번 읽어 들고 있는다.**  매 턴 DB를 다시 읽으면
  프롬프트 캐시 접두가 흔들린다.
* **스레드별 대화를 PostgreSQL에 보존한다.**  LangGraph 체크포인터를 그대로
  쓴다 -- 스키마와 설치 스크립트(`scripts/setup_checkpoint_database.py`)가
  이미 있고 잘 동작한다(REBUILD_PLAN 5절 6단계).

그래프는 노드 하나다.  분기가 없으므로 그래프가 하는 일은 체크포인터에
메시지를 붙이는 것뿐이고, 실제 판단은 전부 `chat.turn.run_turn`에 있다.
"""

from __future__ import annotations

import os
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from chat.models import MESSAGE_LIMIT, ModelRole, create_model
from chat.prompt import build_system_message
from chat.registry import RegistrySnapshot, load_active_registry
from chat.turn import TurnResult, run_turn


@dataclass
class Runtime:
    snapshot: RegistrySnapshot
    system_message: str
    graph: Any
    answer_model: Any
    judge_model: Any
    appeal_model: Any
    _stack: ExitStack

    def close(self) -> None:
        self._stack.close()


_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")
    return database_url


def _build_graph(checkpointer: Any) -> Any:
    """대화를 붙이기만 하는 노드 하나짜리 그래프.

    답을 만드는 일은 그래프 밖에서 한다.  안에서 하려면 호출 가능한 것을
    상태에 실어야 하는데, 상태는 체크포인트로 직렬화되므로 함수를 넣을 수
    없다.  분기가 없는 이상 그래프가 판단을 담을 이유도 없다.
    """

    builder = StateGraph(MessagesState)
    builder.add_node("append", lambda _state: {})
    builder.add_edge(START, "append")
    builder.add_edge("append", END)
    return builder.compile(checkpointer=checkpointer)


def initialize(*, database_url: str | None = None) -> Runtime:
    """서버 기동 시 한 번. 이미 열려 있으면 그대로 돌려준다."""

    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        url = database_url or _database_url()
        stack = ExitStack()
        try:
            snapshot = load_active_registry(url)
            checkpointer = stack.enter_context(PostgresSaver.from_conn_string(url))
            checkpointer.setup()
            _runtime = Runtime(
                snapshot=snapshot,
                system_message=build_system_message(snapshot),
                graph=_build_graph(checkpointer),
                answer_model=create_model(ModelRole.ANSWER),
                judge_model=create_model(ModelRole.JUDGE),
                appeal_model=create_model(ModelRole.APPEAL),
                _stack=stack,
            )
        except Exception:
            stack.close()
            _runtime = None
            raise
        return _runtime


def close() -> None:
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.close()
            _runtime = None


def active() -> Runtime:
    if _runtime is None:
        raise RuntimeError("역량 챗봇 runtime이 초기화되지 않았습니다.")
    return _runtime


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def answer_turn(
    thread_id: str,
    question: str,
    *,
    on_stage: Callable[[str], None] | None = None,
) -> TurnResult:
    """한 턴을 실행하고 대화에 보존한다.

    ``on_stage``는 사용자에게 보일 진행 단계를 흘린다.  검증 후 공개라서
    답이 나오기까지 화면이 비어 있고(3.5), 그 사이를 채울 것이 필요하다.
    """

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    runtime = active()
    if on_stage is not None:
        on_stage("질문을 이해하는 중")

    asked = HumanMessage(content=question)
    history = conversation(thread_id)[-MESSAGE_LIMIT[ModelRole.ANSWER] :]
    prompt = [
        SystemMessage(content=runtime.system_message),
        *history,
        asked,
    ]

    if on_stage is not None:
        on_stage("답변을 작성하는 중")
    result = run_turn(
        runtime.answer_model,
        runtime.judge_model,
        runtime.snapshot,
        prompt,
        appeal_model=runtime.appeal_model,
    )

    # 질문과 답을 함께 붙인다.  답을 만들다 실패하면 질문도 남지 않아,
    # 다음 턴이 답 없는 질문을 맥락으로 읽는 일이 없다.
    runtime.graph.invoke(
        {"messages": [asked, AIMessage(content=result.text)]},
        _config(thread_id),
    )
    return result


def conversation(thread_id: str) -> list[Any]:
    """보존된 대화를 돌려준다. 없으면 빈 목록."""

    state = active().graph.get_state(_config(thread_id))
    values = getattr(state, "values", None) or {}
    return list(values.get("messages", ()))


@dataclass
class PublicMessage:
    role: str
    content: str


def public_messages(thread_id: str) -> Iterator[PublicMessage]:
    """시스템 메시지와 도구 왕복을 걸러 사용자에게 보일 것만 남긴다."""

    for message in conversation(thread_id):
        role = getattr(message, "type", "")
        if role not in {"human", "ai"}:
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            yield PublicMessage(
                role="user" if role == "human" else "assistant",
                content=content,
            )
