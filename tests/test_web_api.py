"""FastAPI 웹 채팅과 LangGraph 상태 연동에 대한 결정적 테스트."""

from __future__ import annotations

import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

import competency_interpreter_v1 as interpreter
import web_api
from web_api import app


@pytest.fixture()
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """OpenAI 호출과 실제 PostgreSQL을 분리한 테스트 클라이언트."""

    interpreter.close_competency_runtime()

    test_database_path = tmp_path / "test_checkpoints.sqlite"

    def open_test_checkpointer(
        runtime_stack: ExitStack,
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
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with TestClient(app) as test_client:
        yield test_client

    interpreter.close_competency_runtime()


def create_thread(client: TestClient) -> str:
    response = client.post("/api/threads")

    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    assert str(UUID(thread_id)) == thread_id

    return thread_id


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


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        (
            {
                "message": "   ",
                "thread_id": "00000000-0000-4000-8000-000000000000",
            },
            "/api/chat",
        ),
        (
            {
                "message": "가" * 2001,
                "thread_id": "00000000-0000-4000-8000-000000000000",
            },
            "/api/chat",
        ),
        (
            {
                "message": "성실성을 알려줘.",
                "thread_id": "00000000-0000-4000-8000-000000000000",
                "unexpected": True,
            },
            "/api/chat",
        ),
    ],
)
def test_chat_input_validation(
    client: TestClient,
    payload: dict,
    path: str,
) -> None:
    response = client.post(path, json=payload)

    assert response.status_code == 422


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


def test_exact_query_and_conversation_restore(
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
    assert "역량명: 성실성" in chat_data["answer"]
    assert "정의:" in chat_data["answer"]
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
    assert "역량명: 성실성" in history_data["messages"][1]["content"]


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
    assert "역량명: 성실성" in answer
    assert "하위요인: 점검행동, 조절행동, 유지행동" in answer


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
            "message": "성실셩의 정의를 알려줘.",
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
    assert "역량명: 성실성" in selection_data["answer"]
    assert "정의:" in selection_data["answer"]
    assert "하위요인:" not in selection_data["answer"]
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
    assert "역량명: 공감성" in history_response.text


def test_existing_ask_competency_api_remains_compatible(
    client: TestClient,
) -> None:
    thread_id = create_thread(client)

    answer = interpreter.ask_competency(
        "성실성의 정의를 알려줘.",
        thread_id=thread_id,
    )

    assert "역량명: 성실성" in answer


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
        "competency_checkpoints.sqlite",
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
            "Traceback: C:\\secret\\competency_checkpoints.sqlite"
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
    assert response.json() == {
        "detail": (
            "답변을 만드는 중 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )
    }
    assert "Traceback" not in response.text
    assert "competency_checkpoints.sqlite" not in response.text
