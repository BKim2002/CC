"""정확 일치 라우팅과 LLM 파서에 대한 핵심 회귀 테스트."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

import competency_interpreter
from competency_interpreter import (
    DEFAULT_FIELDS,
    ParsedNaturalLanguageQuery,
    extract_exact_registered_names,
    interpret_query,
    llm_interpret_query,
    route_after_interpret,
    selected_model_name,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("수인지력", ["수인지력"]),
        ("표현능력_의사표현", ["의사표현"]),
        ("대인매력_매력도", ["매력도"]),
        ("인지력, 수인지력", ["인지력", "수인지력"]),
        ("인지력\n수인지력", ["인지력", "수인지력"]),
        ("수인지력의 정의를 알려줘.", []),
        ("인지력과 수인지력", []),
        ("성실셩", []),
    ],
)
def test_only_exact_registered_names_use_python_lookup(
    query: str,
    expected: list[str],
) -> None:
    assert extract_exact_registered_names(query) == expected


def test_exact_name_uses_python_route_and_default_fields() -> None:
    state = interpret_query(
        {
            "messages": [HumanMessage(content="책임성")],
        }
    )

    assert state["resolved_names"] == ["책임성"]
    assert state["requested_fields"] == DEFAULT_FIELDS
    assert route_after_interpret(state) == "find_competencies"


def test_natural_language_uses_llm_even_with_registered_name() -> None:
    state = interpret_query(
        {
            "messages": [
                HumanMessage(content="책임성의 뜻만 알려줘")
            ],
        }
    )

    assert state["resolved_names"] == []
    assert route_after_interpret(state) == "llm_interpret_query"


class StubParser:
    def __init__(
        self,
        result: ParsedNaturalLanguageQuery | Exception,
    ) -> None:
        self.result = result

    def invoke(self, _: object) -> ParsedNaturalLanguageQuery:
        if isinstance(self.result, Exception):
            raise self.result

        return self.result


def test_llm_parser_returns_python_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = StubParser(
        ParsedNaturalLanguageQuery(
            query_type="named_lookup",
            competency_names=["책임성"],
            requested_fields=["definition"],
        )
    )
    monkeypatch.setattr(
        competency_interpreter,
        "_query_parser_for",
        lambda _: parser,
    )

    result = llm_interpret_query(
        {
            "raw_query": "책임성의 뜻만 알려줘",
            "candidate_names": ["성취추구"],
        }
    )

    assert result == {
        "resolved_names": ["책임성"],
        "requested_fields": ["definition"],
        "candidate_names": [],
        "llm_route": "find_competencies",
    }


def test_llm_parser_failure_has_no_rule_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = StubParser(RuntimeError("LLM unavailable"))
    monkeypatch.setattr(
        competency_interpreter,
        "_query_parser_for",
        lambda _: parser,
    )

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        llm_interpret_query(
            {"raw_query": "책임성의 뜻만 알려줘"}
        )


def test_blank_model_setting_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    assert selected_model_name() == "gpt-5.4-mini"
