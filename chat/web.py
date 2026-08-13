"""FastAPI 서빙 — v1과 같은 계약.

경로·요청·응답·SSE 이벤트 이름을 v1 `web_api.py`와 동일하게 유지한다.
`static/chat.js`가 그 계약에 맞춰져 있고 잘 동작하므로 건드리지 않는다
(REBUILD_PLAN 5절 6단계).

달라지는 것은 하나뿐이다.  **`delta` 이벤트를 보내지 않는다.**  검증 후
공개이므로 검수를 통과하지 않은 토큰은 나가지 않는다(3.5).  대신 `status`로
진행 단계를 흘려 대기 중 화면이 비어 있지 않게 한다 -- 6절에 미결정으로
남아 있던 항목이다.  `chat.js`는 이미 `status`와 `done`을 처리하므로
프런트엔드 변경이 없다.

`candidates`도 계약 유지를 위해 남긴다.  v1은 의미 검색 후보를 별도 필드로
내려보냈지만 v2는 후보를 답변 본문에서 제시한다.  항상 빈 배열이다.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Mapping
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chat import runtime

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "static"
INDEX_PATH = STATIC_DIR / "index.html"

FAILURE_MESSAGE = "답변을 만드는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."

# v1 `_SSE_EVENT_NAMES`와 같은 집합에서 `delta`만 쓰지 않는다.
_STREAM_ERROR = {
    "code": "answer_generation_failed",
    "message": FAILURE_MESSAGE,
    "retryable": True,
}


class ChatRequest(BaseModel):
    thread_id: UUID
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    candidates: list[str] = Field(default_factory=list)


class ThreadResponse(BaseModel):
    thread_id: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class ConversationResponse(BaseModel):
    thread_id: str
    messages: list[ConversationMessage]
    candidates: list[str] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.initialize()
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(title="역량 해석 챗봇", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _frame(event: str, payload: Mapping[str, Any]) -> str:
    data = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


async def _stream(request: Request, thread_id: str, message: str) -> AsyncIterator[str]:
    """검증을 통과한 답변만 내보낸다. 그 전에는 진행 단계만 흘린다."""

    yield _frame("start", {"thread_id": thread_id})

    stages: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_stage(stage: str) -> None:
        loop.call_soon_threadsafe(stages.put_nowait, stage)

    task = loop.run_in_executor(
        None,
        lambda: runtime.answer_turn(thread_id, message, on_stage=on_stage),
    )

    while True:
        if await request.is_disconnected():
            task.cancel()
            return
        stage_task = asyncio.ensure_future(stages.get())
        done, _ = await asyncio.wait(
            {stage_task, task}, return_when=asyncio.FIRST_COMPLETED
        )
        if stage_task in done:
            yield _frame("status", {"stage": stage_task.result()})
            continue
        stage_task.cancel()
        break

    try:
        result = await task
    except Exception:
        LOGGER.error("역량 답변 생성 실패: thread_id=%s", thread_id)
        yield _frame("error", _STREAM_ERROR)
        return

    # 남은 단계 알림을 비운다. 답이 이미 있으므로 내보내지 않는다.
    while not stages.empty():
        stages.get_nowait()

    yield _frame("done", {"answer": result.text, "candidates": []})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/threads", response_model=ThreadResponse)
def create_thread() -> ThreadResponse:
    # thread_id는 대화 식별자일 뿐 사용자 인증 수단이 아니다.
    return ThreadResponse(thread_id=str(uuid4()))


@app.get("/api/threads/{thread_id}/messages", response_model=ConversationResponse)
def get_thread_messages(thread_id: UUID) -> ConversationResponse:
    thread_id_text = str(thread_id)
    try:
        messages = [
            ConversationMessage(role=item.role, content=item.content)
            for item in runtime.public_messages(thread_id_text)
        ]
    except Exception as error:
        LOGGER.error("대화 상태 조회 실패: thread_id=%s", thread_id_text)
        raise HTTPException(
            status_code=500, detail="대화 기록을 불러오지 못했습니다."
        ) from error
    return ConversationResponse(thread_id=thread_id_text, messages=messages)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    thread_id = str(request.thread_id)
    try:
        result = runtime.answer_turn(thread_id, request.message)
    except Exception as error:
        LOGGER.error("역량 답변 생성 실패: thread_id=%s", thread_id)
        raise HTTPException(status_code=500, detail=FAILURE_MESSAGE) from error
    return ChatResponse(thread_id=thread_id, answer=result.text)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    return StreamingResponse(
        _stream(http_request, str(request.thread_id), request.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
