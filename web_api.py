"""역량 해석 LangGraph를 브라우저에 연결하는 FastAPI 서버."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from competency_interpreter import (
    close_competency_runtime,
    get_competency_state,
    initialize_competency_runtime,
    run_competency,
)

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_PATH = STATIC_DIR / "index.html"
MAX_MESSAGE_LENGTH = 2_000


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
    """후보 이름만 중복 없이 공개한다."""

    candidates: list[str] = []

    for candidate in state.get("candidate_names", []):
        if not isinstance(candidate, str):
            continue

        name = candidate.strip()

        if name and name not in candidates:
            candidates.append(name)

    return candidates


def _last_assistant_answer(state: dict) -> str:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, AIMessage):
            continue

        content = _content_to_text(message.content)

        if content:
            return content

    raise RuntimeError("LangGraph 응답에서 챗봇 메시지를 찾지 못했습니다.")


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
        LOGGER.exception(
            "대화 상태 조회 실패: thread_id=%s",
            thread_id_text,
        )
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
        LOGGER.exception(
            "역량 답변 생성 실패: thread_id=%s",
            thread_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "답변을 만드는 중 문제가 발생했습니다. "
                "잠시 후 다시 시도해 주세요."
            ),
        ) from error

    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        candidates=_public_candidates(state),
    )
