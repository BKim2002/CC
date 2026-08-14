"""한 번의 답변 — 도구 호출 루프.

v1의 그래프는 노드가 열둘이었다.  여기는 하나다: 모델을 부르고, 도구를
요청하면 실행해서 돌려주고, 다시 부른다.  질의 유형을 미리 정하지 않으므로
분기가 없다.

``ToolCall`` 기록을 결과에 남기는 것이 4단계의 전제다.  grounding judge는
답변 속 숫자가 무엇을 주장하는지 판정하고, **코드가 그 주장을 도구 결과와
대조**한다.  도구가 무엇을 돌려주었는지 남지 않으면 대조할 대상이 없다.
REBUILD_PLAN 3.2 참고.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from chat.registry import RegistrySnapshot
from chat.tools import (
    UnknownCompetencyError,
    count_competencies,
    get_relations,
)

# 도구는 순수 함수이고 프로세스 안에서 돈다.  왕복이 모델 호출뿐이라
# 상한이 낮아도 된다.  넘으면 폭주로 보고 멈춘다.
MAX_TOOL_ROUNDS = 4

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "count_competencies",
            "description": (
                "조건에 맞는 등록 역량의 수를 센다. 개수를 말하기 전에 반드시 "
                "호출한다. 조건을 주지 않으면 전체를 센다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["overall", "L1", "L2", "L3", "L4", "factor", "item"],
                        "description": "이 위계만 센다. 하위요인은 L3, 최하위요인은 L4다.",
                    },
                    "instrument": {
                        "type": "string",
                        "enum": ["필기 역량검사", "영상면접"],
                        "description": "이 검사만 센다.",
                    },
                    "analysis_included": {
                        "type": "boolean",
                        "description": (
                            "분석 변수 포함 여부로 거른다. 사용자가 분석 포함을 "
                            "명시적으로 물었을 때만 쓴다. 생략하면 거르지 않는다."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_relations",
            "description": (
                "한 역량의 구조적 관계를 조회한다. 부모·자식·조상·후손·형제를 "
                "말하기 전에 반드시 호출한다. 각 항목의 위계 이름을 함께 "
                "돌려주므로 그 이름을 그대로 쓴다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "등록된 역량 이름 또는 별칭.",
                    },
                    "relation": {
                        "type": "string",
                        "enum": [
                            "parent",
                            "children",
                            "ancestors",
                            "descendants",
                            "siblings",
                        ],
                    },
                },
                "required": ["name", "relation"],
                "additionalProperties": False,
            },
        },
    },
)


@dataclass
class ToolCall:
    """한 번의 도구 호출. 4단계 judge가 대조할 원본이다."""

    name: str
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]


@dataclass
class AnswerResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model_rounds: int = 0
    seconds: float = 0.0
    usage: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0, "cached": 0})


def execute_tool(
    snapshot: RegistrySnapshot,
    name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """도구 하나를 실행한다.

    미등록 이름은 예외가 아니라 결과로 돌려준다.  모델이 그 사실을 읽고
    사용자에게 "등록되어 있지 않다"고 답해야 하며, 호출 실패로 끊으면
    그 경로가 사라진다.
    """

    if name == "count_competencies":
        return count_competencies(
            snapshot,
            level=arguments.get("level"),
            instrument=arguments.get("instrument"),
            analysis_included=arguments.get("analysis_included"),
        )
    if name == "get_relations":
        try:
            return get_relations(
                snapshot,
                name=str(arguments.get("name", "")),
                relation=arguments.get("relation", "children"),
            )
        except UnknownCompetencyError as error:
            return {"error": "unregistered_name", "message": str(error)}
    return {"error": "unknown_tool", "message": f"{name}은(는) 없는 도구입니다."}


def message_text(content: Any) -> str:
    """응답 본문에서 사람에게 보일 텍스트만 뽑는다.

    Responses API는 content를 블록 리스트로 돌려준다 -- 추론 블록과 텍스트
    블록이 섞여 있고, 추론 블록에는 ``encrypted_content``가 들어 있다.
    ``str()``을 그대로 쓰면 그 전부가 답변으로 새어 나간다.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _call(bound: Any, conversation: Sequence[Any], on_delta: Any) -> Any:
    """한 라운드를 호출한다. 흘릴 곳이 있으면 흘리면서 합친다.

    청크를 더해 얻은 메시지는 ``invoke``의 결과와 같은 모양이라 도구 호출과
    사용량이 그대로 들어 있다.  덕분에 아래 루프가 두 경로를 구분하지 않는다.
    """

    if on_delta is None:
        return bound.invoke(list(conversation))

    merged = None
    for chunk in bound.stream(list(conversation)):
        merged = chunk if merged is None else merged + chunk
        piece = message_text(chunk.content)
        if piece:
            on_delta(piece)
    return merged


def _usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None) or {}
    return {
        "input": int(metadata.get("input_tokens", 0)),
        "output": int(metadata.get("output_tokens", 0)),
        "cached": int((metadata.get("input_token_details") or {}).get("cache_read", 0)),
    }


def answer(
    model: Any,
    snapshot: RegistrySnapshot,
    messages: Sequence[Any],
    on_delta: Callable[[str], None] | None = None,
) -> AnswerResult:
    """도구가 붙은 모델을 답이 나올 때까지 부른다.

    ``model``은 도구가 바인딩되지 않은 상태로 받는다 -- 바인딩을 여기서
    하므로 호출자가 도구 목록을 알 필요가 없다.

    ``on_delta``가 있으면 토큰을 흘린다.  도구 호출만 하는 라운드는 텍스트가
    없으므로 아무것도 나가지 않고, 답을 쓰는 마지막 라운드만 흘러간다.
    """

    from langchain_core.messages import ToolMessage

    bound = model.bind_tools(list(TOOL_SCHEMAS))
    conversation = list(messages)
    result = AnswerResult(text="")
    started = time.perf_counter()

    for _ in range(MAX_TOOL_ROUNDS):
        response = _call(bound, conversation, on_delta)
        result.model_rounds += 1
        for key, value in _usage(response).items():
            result.usage[key] += value

        requested = getattr(response, "tool_calls", None) or []
        if not requested:
            result.text = message_text(response.content)
            break

        conversation.append(response)
        for call in requested:
            arguments = call.get("args") or {}
            tool_result = execute_tool(snapshot, call["name"], arguments)
            result.tool_calls.append(
                ToolCall(name=call["name"], arguments=arguments, result=tool_result)
            )
            conversation.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id=call["id"],
                )
            )
    else:
        # 상한까지 도구만 부르고 답을 내지 않았다.  빈 답을 정상으로
        # 흘려보내면 그 턴이 조용히 사라지므로 사실대로 남긴다.
        result.text = (
            "답변을 만들지 못했습니다. 도구 호출이 반복되어 중단했습니다."
        )

    result.seconds = round(time.perf_counter() - started, 2)
    return result
