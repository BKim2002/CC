"""chat.web — v1과 같은 계약, 검증 후 공개.

`static/chat.js`가 v1 계약에 맞춰져 있고 잘 동작한다.  경로·필드·SSE 이벤트
이름이 그대로인지 확인한다 -- 하나라도 어긋나면 프런트엔드가 조용히 깨진다.

`delta`를 보내지 않는 것만 v1과 다르다.  검수를 통과하지 않은 토큰은 나가지
않는다(REBUILD_PLAN 3.5).
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from chat import web
from chat.runtime import PublicMessage
from chat.turn import TurnResult


@pytest.fixture
def client(monkeypatch):
    """runtime을 통째로 대체한다. 이 계층의 계약만 본다."""

    monkeypatch.setattr(web.runtime, "initialize", lambda: None)
    monkeypatch.setattr(web.runtime, "close", lambda: None)
    with TestClient(web.app) as test_client:
        yield test_client


def _turn(text: str) -> TurnResult:
    return TurnResult(text=text)


# ── 계약 ─────────────────────────────────────────────────────────────


def test_health_and_thread_creation(client) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}

    thread_id = client.post("/api/threads").json()["thread_id"]

    assert len(thread_id) == 36


def test_chat_returns_the_v1_response_shape(client, monkeypatch) -> None:
    monkeypatch.setattr(
        web.runtime, "answer_turn", lambda *args, **kwargs: _turn("성실성은 …입니다.")
    )
    thread_id = str(uuid4())

    payload = client.post(
        "/api/chat", json={"thread_id": thread_id, "message": "성실성이 뭐야?"}
    ).json()

    assert payload == {
        "thread_id": thread_id,
        "answer": "성실성은 …입니다.",
        "candidates": [],
    }


def test_history_returns_only_user_and_assistant_turns(client, monkeypatch) -> None:
    monkeypatch.setattr(
        web.runtime,
        "public_messages",
        lambda _thread_id: iter(
            [PublicMessage(role="user", content="안녕"), PublicMessage(role="assistant", content="안녕하세요")]
        ),
    )
    thread_id = str(uuid4())

    payload = client.get(f"/api/threads/{thread_id}/messages").json()

    assert payload["messages"] == [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "안녕하세요"},
    ]
    assert payload["candidates"] == []


def test_a_failing_turn_becomes_a_fixed_message_not_a_traceback(client, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("DB 연결 문자열 postgres://user:secret@host")

    monkeypatch.setattr(web.runtime, "answer_turn", boom)

    response = client.post(
        "/api/chat", json={"thread_id": str(uuid4()), "message": "안녕"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == web.FAILURE_MESSAGE
    assert "secret" not in response.text


def test_a_blank_message_is_rejected_before_reaching_the_model(client) -> None:
    response = client.post("/api/chat", json={"thread_id": str(uuid4()), "message": ""})

    assert response.status_code == 422


# ── SSE ──────────────────────────────────────────────────────────────


def _events(body: str) -> list[tuple[str, dict]]:
    frames = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        frames.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return frames


def test_stream_never_emits_delta(client, monkeypatch) -> None:
    """검수를 통과하지 않은 토큰은 나가지 않는다 (3.5)."""

    monkeypatch.setattr(
        web.runtime, "answer_turn", lambda *args, **kwargs: _turn("검증된 답변")
    )

    body = client.post(
        "/api/chat/stream", json={"thread_id": str(uuid4()), "message": "질문"}
    ).text
    names = [name for name, _ in _events(body)]

    assert "delta" not in names
    assert names[0] == "start"
    assert names[-1] == "done"


def test_stream_reports_progress_stages_while_waiting(client, monkeypatch) -> None:
    """대기 중 화면이 비어 있지 않게 한다 (6절 미결정이던 항목)."""

    def answering(thread_id, message, *, on_stage=None):
        if on_stage is not None:
            on_stage("질문을 이해하는 중")
            on_stage("레지스트리를 조회하는 중")
        return _turn("답변")

    monkeypatch.setattr(web.runtime, "answer_turn", answering)

    body = client.post(
        "/api/chat/stream", json={"thread_id": str(uuid4()), "message": "질문"}
    ).text
    stages = [payload["stage"] for name, payload in _events(body) if name == "status"]

    assert "질문을 이해하는 중" in stages


def test_stream_final_answer_matches_the_nonstream_answer(client, monkeypatch) -> None:
    monkeypatch.setattr(
        web.runtime, "answer_turn", lambda *args, **kwargs: _turn("같은 답변")
    )
    request = {"thread_id": str(uuid4()), "message": "질문"}

    plain = client.post("/api/chat", json=request).json()["answer"]
    streamed = dict(_events(client.post("/api/chat/stream", json=request).text))["done"]

    assert streamed["answer"] == plain


def test_stream_failure_becomes_an_error_event(client, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("내부 오류 상세")

    monkeypatch.setattr(web.runtime, "answer_turn", boom)

    body = client.post(
        "/api/chat/stream", json={"thread_id": str(uuid4()), "message": "질문"}
    ).text
    events = dict(_events(body))

    assert events["error"]["code"] == "answer_generation_failed"
    assert "내부 오류 상세" not in body


def test_startup_logs_the_resolved_model_per_role(caplog) -> None:
    """배포된 설정이 낡은 것을 로그 한 줄로 알 수 있어야 한다.

    `OPENAI_MODEL` 하나가 네 역할을 모두 덮으므로 그 값이 낡으면 전부 조용히
    낡는다.  이 줄이 없어서 프로덕션 진단에 재현과 모델 조합 비교가 필요했다.
    """

    import logging

    from chat.runtime import log_selected_models

    with caplog.at_level(logging.INFO, logger="chat.runtime"):
        log_selected_models()

    logged = caplog.text
    assert "answer=" in logged
    assert "judge=" in logged
