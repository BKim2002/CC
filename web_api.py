"""역량 해석 LangGraph를 브라우저에 연결하는 FastAPI 서버."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

import competency_interpreter as competency_runtime
from competency_interpreter import (
    close_competency_runtime,
    get_competency_state,
    initialize_competency_runtime,
    run_competency,
)
from llm_gateway import FIXED_FAILURE_MESSAGE

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_PATH = STATIC_DIR / "index.html"
MAX_MESSAGE_LENGTH = 2_000

_SSE_EVENT_NAMES = {
    "start",
    "status",
    "delta",
    "replace",
    "done",
    "error",
}
_SAFE_STATUS_STAGES = {
    "질문을 이해하는 중",
    "레지스트리를 조회하는 중",
    "답변을 작성하는 중",
}
_GENERIC_STREAM_ERROR = {
    "code": "answer_generation_failed",
    "message": FIXED_FAILURE_MESSAGE,
    "retryable": True,
}


class ChatRequest(BaseModel):
    """브라우저에서 전달하는 한 번의 채팅 요청."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    message: str = Field(
        max_length=MAX_MESSAGE_LENGTH,
    )
    thread_id: UUID

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value:
            raise ValueError("질문을 입력해 주세요.")

        return value


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    candidates: list[str]


class ThreadResponse(BaseModel):
    thread_id: str


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationResponse(BaseModel):
    thread_id: str
    messages: list[ConversationMessage]
    candidates: list[str]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """서버 시작과 종료에 맞춰 PostgreSQL 연결을 관리한다."""

    initialize_competency_runtime()

    try:
        yield
    finally:
        close_competency_runtime()


app = FastAPI(
    title="역량 해석 챗봇",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


def _content_to_text(content: object) -> str:
    """LangChain 메시지에서 브라우저에 보여줄 텍스트만 꺼낸다."""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []

    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
            continue

        if not isinstance(block, dict):
            continue

        if block.get("type") not in {None, "text"}:
            continue

        text = block.get("text")

        if isinstance(text, str):
            text_parts.append(text)

    return "\n".join(text_parts)


def _public_messages(state: dict) -> list[ConversationMessage]:
    """HumanMessage와 AIMessage만 공개 응답 형식으로 변환한다."""

    messages: list[ConversationMessage] = []

    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            role: Literal["user", "assistant"] = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        content = _content_to_text(message.content)

        if not content:
            continue

        messages.append(
            ConversationMessage(
                role=role,
                content=content,
            )
        )

    return messages


def _public_candidates(state: dict) -> list[str]:
    """활성 레지스트리에서 재검증한 정식 후보명만 공개한다."""

    latest_message = next(
        (
            message
            for message in reversed(list(state.get("messages", [])))
            if isinstance(message, (HumanMessage, AIMessage))
        ),
        None,
    )
    if not isinstance(latest_message, AIMessage):
        return []

    return _candidate_values(state.get("candidate_names"))


def _last_assistant_answer(state: dict) -> str:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, AIMessage):
            continue

        content = _content_to_text(message.content)

        if content:
            return content

    raise RuntimeError("LangGraph 응답에서 챗봇 메시지를 찾지 못했습니다.")


def _sse_frame(event: str, payload: Mapping[str, Any]) -> str:
    """UTF-8 JSON 한 건을 브라우저가 읽는 SSE frame으로 직렬화한다."""

    data = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event}\ndata: {data}\n\n"


def _event_parts(raw_event: object) -> tuple[str, dict[str, Any]]:
    """runtime event의 사소한 표현 차이를 공개 event 형식으로 모은다."""

    if isinstance(raw_event, tuple) and len(raw_event) == 2:
        event, payload = raw_event
    elif isinstance(raw_event, Mapping):
        event = (
            raw_event.get("event")
            or raw_event.get("type")
            or raw_event.get("kind")
        )
        payload = raw_event.get("data", raw_event.get("payload"))

        if payload is None:
            payload = {
                key: value
                for key, value in raw_event.items()
                if key not in {"event", "type", "kind"}
            }
    else:
        event = getattr(raw_event, "event", None)
        payload = getattr(
            raw_event,
            "data",
            getattr(raw_event, "payload", None),
        )

    if not isinstance(event, str):
        return "", {}

    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")

    if not isinstance(payload, Mapping):
        scalar_fields = {
            "status": "stage",
            "delta": "text",
            "replace": "answer",
        }
        scalar_field = scalar_fields.get(event.strip().lower())

        if scalar_field and isinstance(payload, str):
            return event.strip().lower(), {scalar_field: payload}

        return event.strip().lower(), {}

    return event.strip().lower(), dict(payload)


def _safe_status_stage(value: object) -> str | None:
    """내부 node 이름 대신 세 가지 사용자용 진행 상태만 허용한다."""

    if not isinstance(value, str):
        return None

    stage = value.strip()

    if stage in _SAFE_STATUS_STAGES:
        return stage

    lowered = stage.casefold()

    if any(token in lowered for token in ("질문", "parse", "understand")):
        return "질문을 이해하는 중"

    if any(
        token in lowered
        for token in ("레지스트리", "registry", "execute", "lookup")
    ):
        return "레지스트리를 조회하는 중"

    if any(token in lowered for token in ("답변", "answer", "write")):
        return "답변을 작성하는 중"

    return None


def _candidate_values(value: object) -> list[str]:
    """후보를 활성 레지스트리의 정식 이름 최대 세 개로 제한한다."""

    if not isinstance(value, (list, tuple)):
        return []

    candidates: list[str] = []

    for candidate in value:
        if not isinstance(candidate, str):
            continue

        name = candidate.strip()

        if name and name not in candidates:
            candidates.append(name)

    try:
        return competency_runtime.validate_registry_names(candidates)[:3]
    except Exception:
        # 후보는 부가 정보다. runtime 상태가 없거나 손상된 후보가 전달되어도
        # 내부 레지스트리 오류를 공개하지 않고 빈 목록으로 축소한다.
        return []


def _public_stream_event(
    raw_event: object,
    thread_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """runtime 출력에서 공개 계약에 속한 필드만 골라낸다."""

    event, payload = _event_parts(raw_event)

    if event not in _SSE_EVENT_NAMES:
        return None

    if event == "start":
        return event, {"thread_id": thread_id}

    if event == "status":
        stage = _safe_status_stage(payload.get("stage"))
        return (event, {"stage": stage}) if stage else None

    if event == "delta":
        text = payload.get("text")
        return (event, {"text": text}) if isinstance(text, str) and text else None

    if event == "replace":
        answer = payload.get("answer")
        return (
            (event, {"answer": answer})
            if isinstance(answer, str) and answer
            else None
        )

    if event == "done":
        answer = payload.get("answer")

        if not isinstance(answer, str) or not answer:
            return None

        return event, {
            "thread_id": thread_id,
            "answer": answer,
            "candidates": _candidate_values(
                payload.get("candidates", payload.get("candidate_names"))
            ),
        }

    if event == "error":
        # Runtime의 예외 메시지나 내부 error code는 브라우저로 전달하지 않는다.
        return "error", dict(_GENERIC_STREAM_ERROR)

    # 공개 event 이름이 늘어나도 처리되지 않은 event가 조용히 오류로 바뀌지 않게
    # 한다. 계약에 없는 event는 오류가 아니라 무시 대상이다.
    return None


async def _chat_event_stream(
    request: Request,
    message: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """LangGraph stream을 취소 가능한 공개 SSE stream으로 변환한다."""

    runtime_stream: object | None = None
    terminal_event_sent = False

    yield _sse_frame("start", {"thread_id": thread_id})

    try:
        runtime_stream = competency_runtime.run_competency_stream(
            message,
            thread_id,
        )

        if inspect.isawaitable(runtime_stream):
            runtime_stream = await runtime_stream

        if not hasattr(runtime_stream, "__aiter__"):
            raise TypeError("run_competency_stream must return an async iterator")

        async for raw_event in runtime_stream:
            if await request.is_disconnected():
                return

            public_event = _public_stream_event(raw_event, thread_id)

            if public_event is None:
                continue

            event, payload = public_event

            # start는 web layer가 정확히 한 번 먼저 보낸다.
            if event == "start":
                continue

            yield _sse_frame(event, payload)

            if event in {"done", "error"}:
                terminal_event_sent = True
                return

        if not terminal_event_sent and not await request.is_disconnected():
            yield _sse_frame("error", _GENERIC_STREAM_ERROR)
    except asyncio.CancelledError:
        # StreamingResponse가 연결 종료를 알리면 model stream까지 취소한다.
        raise
    except Exception:
        # The original exception can contain a DB URL, prompt, or source text.
        # Public and server-log surfaces both keep only the safe operation label.
        LOGGER.error("역량 스트림 생성 실패: thread_id=%s", thread_id)

        if not terminal_event_sent and not await request.is_disconnected():
            yield _sse_frame("error", _GENERIC_STREAM_ERROR)
    finally:
        close_stream = getattr(runtime_stream, "aclose", None)

        if callable(close_stream):
            try:
                await close_stream()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.error("역량 스트림 종료 실패: thread_id=%s", thread_id)


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


@app.get(
    "/api/threads/{thread_id}/messages",
    response_model=ConversationResponse,
)
def get_thread_messages(
    thread_id: UUID,
) -> ConversationResponse:
    thread_id_text = str(thread_id)

    try:
        state = get_competency_state(thread_id_text)
    except Exception as error:
        LOGGER.error("대화 상태 조회 실패: thread_id=%s", thread_id_text)
        raise HTTPException(
            status_code=500,
            detail="대화 기록을 불러오지 못했습니다.",
        ) from error

    return ConversationResponse(
        thread_id=thread_id_text,
        messages=_public_messages(state),
        candidates=_public_candidates(state),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    thread_id = str(request.thread_id)

    try:
        state = run_competency(
            request.message,
            thread_id,
        )
        answer = _last_assistant_answer(state)
    except Exception as error:
        LOGGER.error("역량 답변 생성 실패: thread_id=%s", thread_id)
        raise HTTPException(
            status_code=500,
            detail=FIXED_FAILURE_MESSAGE,
        ) from error

    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        candidates=_public_candidates(state),
    )


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
) -> StreamingResponse:
    """기존 ChatRequest를 받아 검증된 공개 event만 SSE로 전달한다."""

    thread_id = str(request.thread_id)

    return StreamingResponse(
        _chat_event_stream(
            http_request,
            request.message,
            thread_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
