"""chat.answer — 도구 호출 루프.

실제 모델을 부르지 않는다. 기존 테스트 규약대로 외부 API 호출은 eval에만
둔다(evals/spike.py).  여기서 지키는 것은 배선이다.

가장 중요한 계약은 **도구 호출 기록이 결과에 남는다**는 것이다. 4단계
grounding judge는 답변 속 숫자가 무엇을 주장하는지 판정하고 코드가 그것을
도구 결과와 대조한다. 기록이 없으면 대조할 대상이 없다(REBUILD_PLAN 3.2).
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chat.answer import MAX_TOOL_ROUNDS, answer, execute_tool
from chat.registry import build_registry_snapshot

SOURCE_HASH = "d" * 64


def _item(item_id: str, name: str, *, level: str, children: list[str] | None = None) -> dict:
    return {
        "id": item_id,
        "name": name,
        "aliases": [],
        "instrument": "written_competency",
        "instrument_label": "필기 역량검사",
        "level": level,
        "path": [name],
        "parent_name": None,
        "parent_id": None,
        "children": [child.split(":")[-1] for child in (children or [])],
        "children_ids": children or [],
        "definition": f"{name} 정의",
        "definition_status": "explicit",
        "analysis_included": True,
        "notes": [],
        "source_section": "테스트",
    }


@pytest.fixture
def snapshot():
    items = [
        _item("w:parent", "부모", level="L2", children=["w:child"]),
        _item("w:child", "자식", level="L3"),
    ]
    document = {
        "schema_version": "1.0",
        "source": {"file": "synthetic.md", "sha256": SOURCE_HASH, "encoding": "utf-8"},
        "rules": {"synthetic": "합성 레지스트리 규칙"},
        "validation": {"status": "passed", "counts": {"total": len(items)}},
        "items": items,
    }
    return build_registry_snapshot(
        {
            "id": 5,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(items),
        }
    )


class _ScriptedModel:
    """정해진 순서로 응답하는 가짜 모델. 받은 대화도 기록한다."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.seen: list[list] = []

    def bind_tools(self, tools):  # noqa: ANN001 - langchain 인터페이스
        self.bound_tools = tools
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.seen.append(list(messages))
        return self._responses.pop(0)


def _tool_request(call_id: str, name: str, arguments: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": arguments, "type": "tool_call"}],
    )


def test_tool_result_is_fed_back_and_recorded(snapshot) -> None:
    model = _ScriptedModel(
        [
            _tool_request("c1", "count_competencies", {"level": "L3"}),
            AIMessage(content="하위요인은 1개입니다."),
        ]
    )

    result = answer(model, snapshot, [HumanMessage(content="하위요인 몇 개야?")])

    assert result.text == "하위요인은 1개입니다."
    assert result.model_rounds == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "count_competencies"
    assert result.tool_calls[0].result["count"] == 1

    # 두 번째 호출에 도구 결과가 실려 있어야 한다.
    fed_back = json.loads(model.seen[1][-1].content)
    assert fed_back["count"] == 1
    assert fed_back["criteria"] == {"level": "L3"}


def test_answer_without_tool_calls_returns_directly(snapshot) -> None:
    model = _ScriptedModel([AIMessage(content="안녕하세요.")])

    result = answer(model, snapshot, [HumanMessage(content="안녕")])

    assert result.text == "안녕하세요."
    assert result.model_rounds == 1
    assert result.tool_calls == []


def test_unregistered_name_becomes_a_tool_result_not_an_exception(snapshot) -> None:
    """모델이 그 사실을 읽고 사용자에게 알려야 한다. 예외로 끊으면 그 경로가 없다."""

    model = _ScriptedModel(
        [
            _tool_request("c1", "get_relations", {"name": "없는역량", "relation": "children"}),
            AIMessage(content="'없는역량'은 등록되어 있지 않습니다."),
        ]
    )

    result = answer(model, snapshot, [HumanMessage(content="없는역량 하위는?")])

    assert result.tool_calls[0].result["error"] == "unregistered_name"
    assert "등록되어 있지 않습니다" in result.text


def test_runaway_tool_loop_stops_with_a_visible_message(snapshot) -> None:
    """빈 답을 정상으로 흘려보내면 그 턴이 조용히 사라진다."""

    model = _ScriptedModel(
        [
            _tool_request(f"c{index}", "count_competencies", {})
            for index in range(MAX_TOOL_ROUNDS)
        ]
    )

    result = answer(model, snapshot, [HumanMessage(content="몇 개야?")])

    assert result.text
    assert "중단" in result.text
    assert result.model_rounds == MAX_TOOL_ROUNDS


def test_both_tools_are_offered_to_the_model(snapshot) -> None:
    model = _ScriptedModel([AIMessage(content="네.")])

    answer(model, snapshot, [HumanMessage(content="안녕")])

    offered = {tool["function"]["name"] for tool in model.bound_tools}
    assert offered == {"count_competencies", "get_relations"}


def test_execute_tool_rejects_an_unknown_tool_name(snapshot) -> None:
    assert execute_tool(snapshot, "delete_everything", {})["error"] == "unknown_tool"
