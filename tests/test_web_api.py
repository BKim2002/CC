"""FastAPI 웹 채팅과 LangGraph 상태 연동에 대한 결정적 테스트."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import competency_interpreter as interpreter
import web_api
from competency_registry import RegistrySnapshot, build_registry_snapshot
from llm_gateway import FIXED_FAILURE_MESSAGE
from web_api import app


class GatewayStub:
    """모든 웹 입력을 새 strict Gateway 계약으로 구조화한다."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def invoke(
        self,
        messages: list[tuple[str, str]],
    ) -> dict:
        self.calls.append("gateway")
        query = messages[-1][1]

        if query in {
            "성실성",
            "성실성의 정의를 알려줘.",
            "공감성의 정의를 알려줘.",
        }:
            name = "공감성" if "공감성" in query else "성실성"
            return self._registry_decision(
                intent_hint="item_lookup",
                target_mentions=[name],
                constraint_mentions=[{"kind": "field", "text": "정의"}],
            )

        if query == "그 역량의 하위요인도 알려줘.":
            return self._registry_decision(
                intent_hint="item_lookup",
                constraint_mentions=[{"kind": "field", "text": "하위요인"}],
                reuse_previous_result=True,
            )

        if query == "맡은 일을 꼼꼼하게 수행하는 역량은 뭐야?":
            return self._registry_decision(
                intent_hint="semantic_search",
                constraint_mentions=[{"kind": "field", "text": "정의"}],
                semantic_description="맡은 일을 꼼꼼하게 수행하는 행동",
            )

        if query == "안녕하세요":
            return {
                "route": "meta_conversation",
                "kind": "greeting",
            }

        if query == "서울 날씨 알려줘":
            return {
                "route": "out_of_scope",
                "topic": {
                    "category": "weather",
                    "summary": "서울 날씨",
                },
            }

        return {
            "route": "out_of_scope",
            "topic": {
                "category": "other",
                "summary": "지원 범위 밖 질문",
            },
        }

    @staticmethod
    def _registry_decision(
        *,
        intent_hint: str,
        target_mentions: list[str] | None = None,
        constraint_mentions: list[dict[str, str]] | None = None,
        semantic_description: str | None = None,
        reuse_previous_result: bool = False,
    ) -> dict:
        return {
            "route": "registry_query",
            "draft": {
                "intent_hint": intent_hint,
                "target_mentions": target_mentions or [],
                "constraint_mentions": constraint_mentions or [],
                "semantic_description": semantic_description,
                "reuse_previous_result": reuse_previous_result,
                "answer_mode": "registry_facts",
                "acknowledge_greeting": False,
                "out_of_scope_remainder": None,
            },
        }


class SemanticSelectorStub:
    """의미 검색 결과를 실제 OpenAI 호출 없이 고정한다."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def invoke(
        self,
        _: list[tuple[str, str]],
    ) -> interpreter.SemanticSelection:
        self.calls.append("semantic_selector")
        return interpreter.SemanticSelection(
            candidate_names=["성실성"],
            confidence="medium",
            auto_select=False,
        )


class RegistryWriterStub:
    """prompt의 검증 기준 답변을 그대로 stream하는 grounded writer다."""

    _REFERENCE_MARKER = (
        "검증 기준 답변(사용자에게 그대로 복사하는 Python fallback이 아니라 사실 기준):\n"
    )

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def astream(self, messages: list[tuple[str, str]]):
        self.calls.append("registry_writer")
        human_prompt = messages[-1][1]
        assert self._REFERENCE_MARKER in human_prompt
        answer = human_prompt.split(self._REFERENCE_MARKER, 1)[1].strip()
        midpoint = max(1, len(answer) // 2)
        yield AIMessageChunk(content=answer[:midpoint])
        yield AIMessageChunk(content=answer[midpoint:])


class ScopeWriterStub:
    """Scope Writer의 strict 구조화 응답을 실제 호출 형태로 반환한다."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.force_invalid = False
        self.inputs: list[list[tuple[str, str]]] = []

    async def ainvoke(self, messages: list[tuple[str, str]]) -> dict[str, str]:
        self.calls.append("scope_writer")
        self.inputs.append(messages)
        human_prompt = messages[-1][1]
        if self.force_invalid:
            return {
                "acknowledgement": "서울은 맑고 기온은 28도입니다",
                "scope_boundary": "이 챗봇은 등록된 역량 정보만 다룹니다",
                "registry_redirect": "대신 등록 역량의 정의를 물어보세요",
            }
        if "mode: greeting" in human_prompt:
            return {
                "response": "안녕하세요, 반갑습니다",
                "registry_redirect": "등록된 역량의 정의나 위계를 물어보세요",
            }
        return {
            "acknowledgement": "서울 날씨 요청을 확인하고 싶으시군요",
            "scope_boundary": (
                "이 챗봇은 등록된 역량 정보만 다루므로 날씨를 직접 "
                "확인하지 않습니다"
            ),
            "registry_redirect": "대신 등록 역량의 정의나 위계를 물어보세요",
        }


def make_synthetic_registry() -> RegistrySnapshot:
    """운영 DB와 분리된 parent/child 일관 합성 스냅샷이다."""

    def item(
        item_id: str,
        name: str,
        definition: str,
        *,
        parent_id: str | None = None,
        parent_name: str | None = None,
        children: list[str] | None = None,
        children_ids: list[str] | None = None,
    ) -> dict:
        return {
            "id": item_id,
            "name": name,
            "aliases": [],
            "instrument": "synthetic",
            "instrument_label": "합성 검사",
            "level": "factor" if parent_id is None else "item",
            "path": [name] if parent_name is None else [parent_name, name],
            "parent_name": parent_name,
            "parent_id": parent_id,
            "children": children or [],
            "children_ids": children_ids or [],
            "definition": definition,
            "definition_status": "provided",
            "analysis_included": False,
            "notes": [],
            "source_section": "테스트",
        }

    child_names = ["점검행동", "조절행동", "유지행동"]
    child_ids = ["checking", "regulating", "maintaining"]
    items = [
        item(
            "conscientiousness",
            "성실성",
            "맡은 일을 꼼꼼하고 성실하게 수행하는 특성",
            children=child_names,
            children_ids=child_ids,
        ),
        *[
            item(
                child_id,
                child_name,
                f"{child_name}의 합성 테스트 정의",
                parent_id="conscientiousness",
                parent_name="성실성",
            )
            for child_id, child_name in zip(child_ids, child_names, strict=True)
        ],
        item(
            "empathy",
            "공감성",
            "타인의 감정과 입장을 이해하는 특성",
        ),
    ]
    source_hash = "b" * 64
    document = {
        "schema_version": "1.0",
        "source": {
            "file": "synthetic.md",
            "sha256": source_hash,
            "encoding": "utf-8",
        },
        "rules": {"synthetic": "웹 API 합성 레지스트리"},
        "validation": {
            "status": "passed",
            "counts": {"total": len(items)},
        },
        "items": items,
    }

    return build_registry_snapshot(
        {
            "id": 1,
            "source_filename": "synthetic.md",
            "source_sha256": source_hash,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(items),
        }
    )


@pytest.fixture()
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """OpenAI 호출과 실제 PostgreSQL을 분리한 테스트 클라이언트."""

    interpreter.close_competency_runtime()

    test_database_path = tmp_path / "test_checkpoints.sqlite"
    registry_snapshot = make_synthetic_registry()
    model_calls: list[str] = []
    gateway_stub = GatewayStub(model_calls)
    semantic_selector_stub = SemanticSelectorStub(model_calls)
    registry_writer_stub = RegistryWriterStub(model_calls)
    scope_writer_stub = ScopeWriterStub(model_calls)

    def open_test_checkpointer(
        runtime_stack: ExitStack,
        _: str,
    ) -> SqliteSaver:
        connection = runtime_stack.enter_context(
            closing(
                sqlite3.connect(
                    str(test_database_path),
                    check_same_thread=False,
                )
            )
        )

        return SqliteSaver(connection)

    monkeypatch.setattr(
        interpreter,
        "_open_checkpointer",
        open_test_checkpointer,
    )
    monkeypatch.setattr(
        interpreter,
        "load_active_registry",
        lambda _: registry_snapshot,
    )
    monkeypatch.setattr(
        interpreter,
        "_gateway_for",
        lambda _: gateway_stub,
    )
    monkeypatch.setattr(
        interpreter,
        "_semantic_selector_for",
        lambda _: semantic_selector_stub,
    )
    monkeypatch.setattr(
        interpreter,
        "_registry_writer_for",
        lambda _: registry_writer_stub,
    )
    monkeypatch.setattr(
        interpreter,
        "_scope_writer_for",
        lambda _, __: scope_writer_stub,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")

    with TestClient(app) as test_client:
        test_client.model_calls = model_calls  # type: ignore[attr-defined]
        test_client.scope_writer = scope_writer_stub  # type: ignore[attr-defined]
        yield test_client

    interpreter.close_competency_runtime()


def create_thread(client: TestClient) -> str:
    response = client.post("/api/threads")

    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    assert str(UUID(thread_id)) == thread_id

    return thread_id


def parse_sse_events(response_text: str) -> list[tuple[str, dict]]:
    """테스트 응답의 완결된 SSE frame을 event/data 쌍으로 읽는다."""

    events: list[tuple[str, dict]] = []

    for frame in re.split(r"\r?\n\r?\n", response_text.strip()):
        event_name = "message"
        data_lines: list[str] = []

        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())

        if data_lines:
            events.append(
                (event_name, json.loads("\n".join(data_lines)))
            )

    return events


def test_static_files_and_health(client: TestClient) -> None:
    health_response = client.get("/api/health")
    index_response = client.get("/")
    css_response = client.get("/static/style.css")
    js_response = client.get("/static/chat.js")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    assert index_response.status_code == 200
    assert "id=\"chatForm\"" in index_response.text
    assert "id=\"messageList\"" in index_response.text
    assert "id=\"newThreadButton\"" in index_response.text

    assert css_response.status_code == 200
    assert ".message-row.user" in css_response.text
    assert ".message-row.assistant" in css_response.text

    assert js_response.status_code == 200
    assert "competency_chat_thread_id" in js_response.text
    assert ".textContent" in js_response.text
    assert "innerHTML" not in js_response.text
    assert 'fetch("/api/chat/stream"' in js_response.text
    assert "response.body.getReader()" in js_response.text
    assert 'new TextDecoder("utf-8")' in js_response.text
    assert "decoder.decode(value, { stream: true })" in js_response.text
    assert "AbortController" in js_response.text

    assert ".streaming-bubble::after" in css_response.text
    assert "prefers-reduced-motion" in css_response.text


def test_new_thread_has_empty_history(client: TestClient) -> None:
    thread_id = create_thread(client)

    response = client.get(
        f"/api/threads/{thread_id}/messages"
    )

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": thread_id,
        "messages": [],
        "candidates": [],
    }


def test_history_hides_stale_candidates_when_latest_turn_has_no_assistant(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = create_thread(client)
    monkeypatch.setattr(
        web_api,
        "get_competency_state",
        lambda _: {
            "messages": [
                HumanMessage(content="행동과 가까운 역량은?"),
                AIMessage(content="성실성을 후보로 찾았습니다."),
                HumanMessage(content="취소된 다음 질문"),
            ],
            "candidate_names": ["성실성"],
        },
    )

    response = client.get(f"/api/threads/{thread_id}/messages")

    assert response.status_code == 200
    assert response.json()["candidates"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {
            "message": "   ",
            "thread_id": "00000000-0000-4000-8000-000000000000",
        },
        {
            "message": "가" * 2001,
            "thread_id": "00000000-0000-4000-8000-000000000000",
        },
        {
            "message": "성실성을 알려줘.",
            "thread_id": "00000000-0000-4000-8000-000000000000",
            "unexpected": True,
        },
    ],
)
def test_chat_input_validation(
    client: TestClient,
    payload: dict,
) -> None:
    response = client.post("/api/chat", json=payload)

    assert response.status_code == 422

    if payload["message"].isspace():
        assert "질문을 입력해 주세요." in response.text


def test_invalid_thread_id_is_rejected(client: TestClient) -> None:
    chat_response = client.post(
        "/api/chat",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": "not-a-uuid",
        },
    )
    history_response = client.get(
        "/api/threads/not-a-uuid/messages"
    )

    assert chat_response.status_code == 422
    assert history_response.status_code == 422


def test_stream_input_validation_reuses_chat_request(
    client: TestClient,
) -> None:
    blank_response = client.post(
        "/api/chat/stream",
        json={
            "message": "   ",
            "thread_id": "00000000-0000-4000-8000-000000000000",
        },
    )
    invalid_id_response = client.post(
        "/api/chat/stream",
        json={
            "message": "성실성을 알려줘.",
            "thread_id": "not-a-uuid",
        },
    )

    assert blank_response.status_code == 422
    assert "질문을 입력해 주세요." in blank_response.text
    assert invalid_id_response.status_code == 422


def test_stream_contract_utf8_headers_and_public_field_filtering(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = create_thread(client)
    answer = '성실성은 "맡은 일"을\n꼼꼼히 수행하는 특성입니다.'

    async def fake_stream(query: str, actual_thread_id: str):
        assert query == "성실성의 정의를 알려줘."
        assert actual_thread_id == thread_id

        # web layer의 start와 중복되므로 공개 응답에서는 한 번만 보여야 한다.
        yield {
            "event": "start",
            "payload": {"thread_id": actual_thread_id},
        }
        yield {
            "type": "status",
            "data": {
                "stage": "parse_query",
                "internal_plan": "stable-secret-id",
            },
        }
        yield ("delta", {"text": '성실성은 "맡은 일"을\n'})
        yield {"event": "delta", "payload": {"text": "꼼꼼히 수행하는 특성입니다."}}
        yield {
            "kind": "done",
            "answer": answer,
            "candidates": ["성실성", "미등록 후보", "성실성", 123],
            "internal_result": "stable-secret-id",
        }

    monkeypatch.setattr(
        interpreter,
        "run_competency_stream",
        fake_stream,
        raising=False,
    )

    response = client.post(
        "/api/chat/stream",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": thread_id,
        },
    )
    events = parse_sse_events(response.text)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert [event for event, _ in events] == [
        "start",
        "status",
        "delta",
        "delta",
        "done",
    ]
    assert events[0][1] == {"thread_id": thread_id}
    assert events[1][1] == {"stage": "질문을 이해하는 중"}
    assert "".join(
        payload["text"]
        for event, payload in events
        if event == "delta"
    ) == answer
    assert events[-1][1] == {
        "thread_id": thread_id,
        "answer": answer,
        "candidates": ["성실성"],
    }
    assert "stable-secret-id" not in response.text
    # ensure_ascii=False로 한국어를 실제 UTF-8 SSE data에 보낸다.
    assert "성실성" in response.text
    assert "\\\"맡은 일\\\"" in response.text
    assert "\\n" in response.text


def test_public_candidates_are_active_canonical_names_limited_to_three(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = create_thread(client)

    async def fake_stream(_: str, __: str):
        yield {
            "event": "done",
            "payload": {
                "answer": "검증된 후보입니다.",
                "candidates": [
                    "성실성",
                    "공감성",
                    "점검행동",
                    "조절행동",
                    "등록되지 않은 이름",
                ],
            },
        }

    monkeypatch.setattr(
        interpreter,
        "run_competency_stream",
        fake_stream,
        raising=False,
    )

    response = client.post(
        "/api/chat/stream",
        json={"message": "후보를 찾아줘", "thread_id": thread_id},
    )
    events = parse_sse_events(response.text)

    assert events[-1] == (
        "done",
        {
            "thread_id": thread_id,
            "answer": "검증된 후보입니다.",
            "candidates": ["성실성", "공감성", "점검행동"],
        },
    )


def test_stream_replace_overwrites_partial_answer(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = create_thread(client)
    fallback = "검증된 결정적 대체 답변입니다."

    async def fake_stream(_: str, __: str):
        yield {"event": "delta", "payload": {"text": "잠정 답변"}}
        yield {"event": "replace", "payload": {"answer": fallback}}
        yield {
            "event": "done",
            "payload": {
                "thread_id": thread_id,
                "answer": fallback,
                "candidates": ["성실성"],
            },
        }

    monkeypatch.setattr(
        interpreter,
        "run_competency_stream",
        fake_stream,
        raising=False,
    )

    response = client.post(
        "/api/chat/stream",
        json={
            "message": "성실성",
            "thread_id": thread_id,
        },
    )
    events = parse_sse_events(response.text)

    assert [event for event, _ in events] == [
        "start",
        "delta",
        "replace",
        "done",
    ]
    assert events[-2][1] == {"answer": fallback}
    assert events[-1][1]["answer"] == fallback
    assert events[-1][1]["candidates"] == ["성실성"]


def test_stream_internal_error_is_generic_and_does_not_leak(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    thread_id = create_thread(client)
    caplog.set_level(logging.ERROR, logger="web_api")

    async def broken_stream(_: str, __: str):
        yield {"event": "delta", "payload": {"text": "잠정"}}
        raise RuntimeError(
            "Traceback: postgresql://admin:password@secret-host/database"
        )

    monkeypatch.setattr(
        interpreter,
        "run_competency_stream",
        broken_stream,
        raising=False,
    )

    response = client.post(
        "/api/chat/stream",
        json={
            "message": "성실성",
            "thread_id": thread_id,
        },
    )
    events = parse_sse_events(response.text)

    assert [event for event, _ in events] == ["start", "delta", "error"]
    assert events[-1][1] == {
        "code": "answer_generation_failed",
        "message": FIXED_FAILURE_MESSAGE,
        "retryable": True,
    }
    assert "Traceback" not in response.text
    assert "password" not in response.text
    assert "secret-host" not in response.text
    assert "password" not in caplog.text
    assert "secret-host" not in caplog.text
    assert "postgresql://" not in caplog.text


def test_disconnected_sse_client_closes_runtime_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    class DisconnectRequest:
        async def is_disconnected(self) -> bool:
            return True

    class RuntimeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return {"type": "delta", "text": "저장되면 안 되는 잠정 답변"}

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(
        web_api.competency_runtime,
        "run_competency_stream",
        lambda *_: RuntimeStream(),
    )

    async def collect() -> list[str]:
        return [
            frame
            async for frame in web_api._chat_event_stream(
                DisconnectRequest(),  # type: ignore[arg-type]
                "질문",
                "00000000-0000-4000-8000-000000000001",
            )
        ]

    frames = asyncio.run(collect())
    assert len(frames) == 1
    assert "event: start" in frames[0]
    assert "잠정 답변" not in frames[0]
    assert closed


def test_real_stream_done_matches_checkpoint_history_and_non_stream(
    client: TestClient,
) -> None:
    stream_thread_id = create_thread(client)
    response = client.post(
        "/api/chat/stream",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": stream_thread_id,
        },
    )
    events = parse_sse_events(response.text)

    assert response.status_code == 200
    assert events[0][0] == "start"
    assert events[-1][0] == "done"
    assert any(event == "delta" for event, _ in events)
    done = events[-1][1]
    registry_deltas = [
        payload["text"]
        for event, payload in events
        if event == "delta"
    ]
    # Registry Writer stub은 두 chunk를 생성하지만 검증 전에는 공개되지 않는다.
    assert registry_deltas == [done["answer"]]

    history = client.get(
        f"/api/threads/{stream_thread_id}/messages"
    ).json()
    assert [message["role"] for message in history["messages"]] == [
        "user",
        "assistant",
    ]
    assert history["messages"][-1]["content"] == done["answer"]
    assert history["candidates"] == done["candidates"]

    non_stream_thread_id = create_thread(client)
    non_stream = client.post(
        "/api/chat",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": non_stream_thread_id,
        },
    ).json()
    assert non_stream["answer"] == done["answer"]


def test_direct_registry_request_is_gateway_first_and_uses_two_model_calls(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    response = client.post(
        "/api/chat",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": thread_id,
        },
    )
    state = interpreter.get_competency_state(thread_id)

    assert response.status_code == 200
    assert client.model_calls == [  # type: ignore[attr-defined]
        "gateway",
        "registry_writer",
    ]
    assert state["llm_call_count"] == 2
    assert state["messages"][-1].content == response.json()["answer"]


def test_semantic_request_uses_gateway_selector_and_writer_within_three_calls(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    response = client.post(
        "/api/chat",
        json={
            "message": "맡은 일을 꼼꼼하게 수행하는 역량은 뭐야?",
            "thread_id": thread_id,
        },
    )
    state = interpreter.get_competency_state(thread_id)

    assert response.status_code == 200
    assert response.json()["candidates"] == ["성실성"]
    assert client.model_calls == [  # type: ignore[attr-defined]
        "gateway",
        "semantic_selector",
        "registry_writer",
    ]
    assert state["llm_call_count"] == 3


def test_meta_request_is_gateway_first_and_uses_buffered_scope_writer(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    response = client.post(
        "/api/chat/stream",
        json={"message": "안녕하세요", "thread_id": thread_id},
    )
    events = parse_sse_events(response.text)
    done = events[-1][1]
    state = interpreter.get_competency_state(thread_id)

    assert response.status_code == 200
    assert events[-1][0] == "done"
    assert [
        payload["text"]
        for event, payload in events
        if event == "delta"
    ] == [done["answer"]]
    assert done["answer"].startswith("안녕하세요")
    assert done["candidates"] == []
    assert client.model_calls == [  # type: ignore[attr-defined]
        "gateway",
        "scope_writer",
    ]
    assert state["llm_call_count"] == 2
    assert state["response_mode"] == "llm"
    assert state["messages"][-1].content == done["answer"]


def test_out_of_scope_template_sse_matches_non_stream_and_checkpoint_history(
    client: TestClient,
) -> None:
    # Out-of-scope answers come from a template, so the exact public string is
    # pinned here; the Gateway is the only model call in the turn.
    expected = (
        "서울 날씨에 관한 질문은 제가 알려드릴 수 없습니다. "
        "이 챗봇은 등록된 역량 정보만 다루는 범위로 한정되어 있습니다. "
        "대신 등록 역량의 정의·위계·관계를 물어보거나 행동 특징으로 역량 후보를 "
        "찾아볼 수 있습니다."
    )
    stream_thread_id = create_thread(client)

    response = client.post(
        "/api/chat/stream",
        json={"message": "서울 날씨 알려줘", "thread_id": stream_thread_id},
    )
    events = parse_sse_events(response.text)
    event_names = [event for event, _ in events]
    done = events[-1][1]
    deltas = [payload["text"] for event, payload in events if event == "delta"]
    state = interpreter.get_competency_state(stream_thread_id)
    history = client.get(f"/api/threads/{stream_thread_id}/messages").json()

    assert response.status_code == 200
    assert event_names[0] == "start"
    assert event_names[-1] == "done"
    assert "error" not in event_names
    assert deltas == [expected]
    assert done == {
        "thread_id": stream_thread_id,
        "answer": expected,
        "candidates": [],
    }
    assert state["response_mode"] == "scope_template"
    assert state["scope_writer_attempts"] == 0
    assert state["llm_call_count"] == 1
    assert state["messages"][-1].content == expected
    assert history["messages"][-1]["content"] == expected
    assert history["candidates"] == []
    assert client.model_calls == ["gateway"]  # type: ignore[attr-defined]

    non_stream_thread_id = create_thread(client)
    non_stream = client.post(
        "/api/chat",
        json={"message": "서울 날씨 알려줘", "thread_id": non_stream_thread_id},
    ).json()
    non_stream_history = client.get(
        f"/api/threads/{non_stream_thread_id}/messages"
    ).json()

    assert non_stream["answer"] == done["answer"]
    assert non_stream["candidates"] == []
    assert non_stream_history["messages"][-1]["content"] == done["answer"]


def test_scope_fallback_sse_is_safe_and_byte_identical_across_api_surfaces(
    client: TestClient,
) -> None:
    # Meta turns are the only ones that still generate, so this is where a
    # rejected draft must degrade to the deterministic fallback.
    scope_writer: ScopeWriterStub = client.scope_writer  # type: ignore[attr-defined]
    scope_writer.force_invalid = True
    expected = (
        "안녕하세요, 반갑습니다. "
        "등록된 역량명을 묻거나 행동 특징을 설명해 역량 후보를 찾아보세요."
    )
    stream_thread_id = create_thread(client)

    response = client.post(
        "/api/chat/stream",
        json={"message": "안녕하세요", "thread_id": stream_thread_id},
    )
    events = parse_sse_events(response.text)
    event_names = [event for event, _ in events]
    done = events[-1][1]
    deltas = [payload["text"] for event, payload in events if event == "delta"]
    state = interpreter.get_competency_state(stream_thread_id)
    history = client.get(f"/api/threads/{stream_thread_id}/messages").json()

    assert response.status_code == 200
    assert event_names[0] == "start"
    assert event_names[-1] == "done"
    assert "error" not in event_names
    assert deltas == [expected]
    assert done["answer"] == expected
    assert done["candidates"] == []
    public_answer_text = "\n".join([*deltas, done["answer"]])
    assert "맑" not in public_answer_text
    assert "28도" not in public_answer_text
    assert state["response_mode"] == "scope_fallback"
    assert state["scope_writer_failed"] is True
    assert state["scope_writer_attempts"] == 2
    assert state["llm_call_count"] == 3
    assert state["messages"][-1].content == expected
    assert history["messages"][-1]["content"] == expected
    assert history["candidates"] == []
    assert client.model_calls == [  # type: ignore[attr-defined]
        "gateway",
        "scope_writer",
        "scope_writer",
    ]

    non_stream_thread_id = create_thread(client)
    non_stream = client.post(
        "/api/chat",
        json={"message": "안녕하세요", "thread_id": non_stream_thread_id},
    ).json()
    non_stream_history = client.get(
        f"/api/threads/{non_stream_thread_id}/messages"
    ).json()

    assert non_stream["answer"] == done["answer"]
    assert non_stream["candidates"] == []
    assert non_stream_history["messages"][-1]["content"] == done["answer"]


def test_scope_delta_is_published_only_after_checkpoint_commit(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    async def collect_and_check_commit() -> list[dict]:
        events: list[dict] = []
        async for event in interpreter.run_competency_stream(
            "서울 날씨 알려줘",
            thread_id,
        ):
            if event.get("type") == "delta":
                committed = interpreter.get_competency_state(thread_id)
                assert committed["messages"][-1].content == event["text"]
            events.append(event)
        return events

    events = asyncio.run(collect_and_check_commit())

    assert [event["type"] for event in events][-1] == "done"
    assert [event for event in events if event["type"] == "error"] == []
    assert [event["text"] for event in events if event["type"] == "delta"] == [
        events[-1]["answer"]
    ]


def test_natural_language_query_and_conversation_restore(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    chat_response = client.post(
        "/api/chat",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": thread_id,
        },
    )

    assert chat_response.status_code == 200
    chat_data = chat_response.json()
    assert chat_data["thread_id"] == thread_id
    assert "성실성" in chat_data["answer"]
    assert "맡은 일을 꼼꼼하고 성실하게 수행하는 특성" in chat_data["answer"]
    assert "역량명:" not in chat_data["answer"]
    assert chat_data["candidates"] == []

    history_response = client.get(
        f"/api/threads/{thread_id}/messages"
    )
    history_data = history_response.json()

    assert history_response.status_code == 200
    assert [
        message["role"]
        for message in history_data["messages"]
    ] == ["user", "assistant"]
    assert history_data["messages"][0]["content"] == (
        "성실성의 정의를 알려줘."
    )
    assert "성실성" in history_data["messages"][1]["content"]


def test_same_thread_follow_up_uses_previous_competency(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    first_response = client.post(
        "/api/chat",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": thread_id,
        },
    )
    second_response = client.post(
        "/api/chat",
        json={
            "message": "그 역량의 하위요인도 알려줘.",
            "thread_id": thread_id,
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    answer = second_response.json()["answer"]
    assert "성실성" in answer
    assert all(name in answer for name in ["점검행동", "조절행동", "유지행동"])


def test_different_threads_do_not_share_context(
    client: TestClient,
) -> None:
    first_thread_id = create_thread(client)
    second_thread_id = create_thread(client)

    first_response = client.post(
        "/api/chat",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": first_thread_id,
        },
    )
    second_response = client.post(
        "/api/chat",
        json={
            "message": "그 역량의 하위요인도 알려줘.",
            "thread_id": second_thread_id,
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert "점검행동" not in second_response.json()["answer"]


def test_candidate_state_and_candidate_selection(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    candidate_response = client.post(
        "/api/chat",
        json={
            "message": "맡은 일을 꼼꼼하게 수행하는 역량은 뭐야?",
            "thread_id": thread_id,
        },
    )

    assert candidate_response.status_code == 200
    assert candidate_response.json()["candidates"] == ["성실성"]

    candidate_history_response = client.get(
        f"/api/threads/{thread_id}/messages"
    )

    assert candidate_history_response.status_code == 200
    assert candidate_history_response.json()["candidates"] == ["성실성"]

    selection_response = client.post(
        "/api/chat",
        json={
            "message": "성실성",
            "thread_id": thread_id,
        },
    )

    assert selection_response.status_code == 200
    selection_data = selection_response.json()
    assert "성실성" in selection_data["answer"]
    assert "맡은 일을 꼼꼼하고 성실하게 수행하는 특성" in selection_data["answer"]
    assert "역량명:" not in selection_data["answer"]
    assert selection_data["candidates"] == []


def test_checkpoint_survives_runtime_reinitialization(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)
    chat_response = client.post(
        "/api/chat",
        json={
            "message": "공감성의 정의를 알려줘.",
            "thread_id": thread_id,
        },
    )

    assert chat_response.status_code == 200

    # 서버 프로세스 재시작과 동일하게 연결을 닫은 뒤 다시 읽는다.
    interpreter.close_competency_runtime()

    history_response = client.get(
        f"/api/threads/{thread_id}/messages"
    )

    assert history_response.status_code == 200
    assert "공감성" in history_response.text
    assert "타인의 감정과 입장을 이해하는 특성" in history_response.text


def test_existing_ask_competency_api_remains_compatible(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    answer = interpreter.ask_competency(
        "성실성의 정의를 알려줘.",
        thread_id=thread_id,
    )

    assert "성실성" in answer
    assert "맡은 일을 꼼꼼하고 성실하게 수행하는 특성" in answer


def test_public_assets_do_not_contain_server_secrets(
    client: TestClient,
) -> None:
    public_text = "\n".join(
        [
            client.get("/").text,
            client.get("/static/style.css").text,
            client.get("/static/chat.js").text,
        ]
    )

    forbidden_values = (
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "postgresql://",
        "C:\\Users\\ksy0823",
    )

    assert all(
        value not in public_text
        for value in forbidden_values
    )


def test_internal_error_response_does_not_leak_details(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = create_thread(client)

    def raise_internal_error(_: str, __: str) -> dict:
        raise RuntimeError(
            "Traceback: postgresql://admin:password@secret-host/database"
        )

    monkeypatch.setattr(web_api, "run_competency", raise_internal_error)

    response = client.post(
        "/api/chat",
        json={
            "message": "성실성의 정의를 알려줘.",
            "thread_id": thread_id,
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": FIXED_FAILURE_MESSAGE}
    assert "Traceback" not in response.text
    assert "password" not in response.text
    assert "secret-host" not in response.text


def test_unhandled_public_event_name_is_ignored_rather_than_reported_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Growing the public contract must not silently turn a new event into a
    # user-visible failure; only a real "error" event maps to the error frame.
    monkeypatch.setattr(
        web_api,
        "_SSE_EVENT_NAMES",
        web_api._SSE_EVENT_NAMES | {"progress"},
    )

    assert web_api._public_stream_event({"event": "progress"}, "thread") is None

    event, payload = web_api._public_stream_event({"event": "error"}, "thread")

    assert event == "error"
    assert payload == web_api._GENERIC_STREAM_ERROR
