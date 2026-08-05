"""정확 일치 라우팅과 LLM 파서에 대한 핵심 회귀 테스트."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

import competency_interpreter
from competency_registry import RegistrySnapshot, build_registry_snapshot
from competency_interpreter import (
    DEFAULT_FIELDS,
    ParsedNaturalLanguageQuery,
    extract_exact_registered_names,
    interpret_query,
    llm_interpret_query,
    route_after_interpret,
    selected_model_name,
)


def make_synthetic_registry(version_id: int = 1) -> RegistrySnapshot:
    """운영 JSON 파일 없이 노드를 검증하는 작은 레지스트리다."""

    items = [
        {
            "id": "numeric_cognition",
            "name": "수인지력",
            "aliases": [],
            "definition": "수를 인지하는 능력",
            "definition_status": "provided",
            "children": [],
            "path": ["인지력", "수인지력"],
        },
        {
            "id": "expression",
            "name": "의사표현",
            "aliases": ["표현능력_의사표현"],
            "definition": "의사를 명확히 표현하는 능력",
            "definition_status": "provided",
            "children": [],
            "path": ["표현능력", "의사표현"],
        },
        {
            "id": "attractiveness",
            "name": "매력도",
            "aliases": ["대인매력_매력도"],
            "definition": "호감을 주는 대인 특성",
            "definition_status": "provided",
            "children": [],
            "path": ["대인매력", "매력도"],
        },
        {
            "id": "cognition",
            "name": "인지력",
            "aliases": [],
            "definition": "정보를 이해하는 능력",
            "definition_status": "provided",
            "children": ["수인지력"],
            "path": ["인지력"],
        },
        {
            "id": "responsibility",
            "name": "책임성",
            "aliases": [],
            "definition": "맡은 일을 책임지고 수행하는 특성",
            "definition_status": "provided",
            "children": [],
            "path": ["성실성", "책임성"],
        },
        {
            "id": "achievement",
            "name": "성취추구",
            "aliases": [],
            "definition": "높은 성과를 추구하는 특성",
            "definition_status": "provided",
            "children": [],
            "path": ["성취추구"],
        },
    ]
    for item in items:
        path = item["path"]
        item.update(
            {
                "instrument": "synthetic",
                "instrument_label": "합성 검사",
                "level": "factor",
                "parent_name": path[-2] if len(path) > 1 else None,
                "parent_id": None,
                "children_ids": [],
                "analysis_included": False,
                "notes": [],
                "source_section": "테스트",
            }
        )

    source_hash = f"{version_id:064x}"
    document = {
        "schema_version": "1.0",
        "source": {
            "file": "synthetic.md",
            "sha256": source_hash,
            "encoding": "utf-8",
        },
        "rules": {"synthetic": "합성 레지스트리 규칙"},
        "validation": {
            "status": "passed",
            "counts": {"total": len(items)},
        },
        "items": items,
    }

    return build_registry_snapshot(
        {
            "id": version_id,
            "source_filename": "synthetic.md",
            "source_sha256": source_hash,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": len(items),
        }
    )


@pytest.fixture(autouse=True)
def synthetic_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> RegistrySnapshot:
    competency_interpreter.close_competency_runtime()
    snapshot = make_synthetic_registry()
    monkeypatch.setattr(
        competency_interpreter,
        "_registry_snapshot",
        snapshot,
    )

    yield snapshot

    competency_interpreter.close_competency_runtime()


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


def test_runtime_initializes_registry_before_checkpointer(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    events: list[str] = []
    compiled_app = object()

    def load_registry(database_url: str) -> RegistrySnapshot:
        assert database_url == "postgresql://test"
        events.append("registry")
        return synthetic_registry

    def open_checkpointer(
        _: object,
        database_url: str,
    ) -> object:
        assert database_url == "postgresql://test"
        assert competency_interpreter._registry_snapshot is synthetic_registry
        events.append("checkpointer")
        return object()

    class BuilderStub:
        def compile(self, *, checkpointer: object) -> object:
            assert checkpointer is not None
            events.append("compile")
            return compiled_app

    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(
        competency_interpreter,
        "load_active_registry",
        load_registry,
    )
    monkeypatch.setattr(
        competency_interpreter,
        "_open_checkpointer",
        open_checkpointer,
    )
    monkeypatch.setattr(
        competency_interpreter,
        "builder",
        BuilderStub(),
    )

    competency_interpreter.initialize_competency_runtime()

    assert events == ["registry", "checkpointer", "compile"]
    assert competency_interpreter._registry_snapshot is synthetic_registry
    assert competency_interpreter.app is compiled_app


def test_runtime_failure_rolls_back_registry_and_resources(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    events: list[str] = []

    def open_checkpointer(
        runtime_stack: object,
        _: str,
    ) -> object:
        runtime_stack.callback(events.append, "closed")
        return object()

    class FailingBuilder:
        def compile(self, *, checkpointer: object) -> object:
            assert checkpointer is not None
            raise RuntimeError("compile failed")

    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(
        competency_interpreter,
        "load_active_registry",
        lambda _: synthetic_registry,
    )
    monkeypatch.setattr(
        competency_interpreter,
        "_open_checkpointer",
        open_checkpointer,
    )
    monkeypatch.setattr(
        competency_interpreter,
        "builder",
        FailingBuilder(),
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        competency_interpreter.initialize_competency_runtime()

    assert events == ["closed"]
    assert competency_interpreter._registry_snapshot is None
    assert competency_interpreter._runtime_stack is None
    assert competency_interpreter.app is None


def test_close_clears_registry_snapshot(
    synthetic_registry: RegistrySnapshot,
) -> None:
    assert competency_interpreter._registry_snapshot is synthetic_registry

    competency_interpreter.close_competency_runtime()

    assert competency_interpreter._registry_snapshot is None


def test_registry_helpers_fail_clearly_before_initialization() -> None:
    competency_interpreter.close_competency_runtime()

    with pytest.raises(
        RuntimeError,
        match="역량 레지스트리가 초기화되지 않았습니다",
    ):
        extract_exact_registered_names("책임성")


def test_matched_state_is_mutable_json_copy_of_frozen_snapshot(
    synthetic_registry: RegistrySnapshot,
) -> None:
    result = competency_interpreter.find_competencies(
        {"resolved_names": ["책임성"]}
    )
    matched_item = result["matched_items"][0]

    # LangGraph/PostgresSaver가 직렬화할 수 있는 일반 dict/list다.
    json.dumps(result["matched_items"], ensure_ascii=False)
    assert isinstance(matched_item, dict)
    assert isinstance(matched_item["children"], list)

    matched_item["definition"] = "상태에서만 변경"
    matched_item["path"].append("추가 경로")

    frozen_item = synthetic_registry.canonical_lookup["책임성"]
    assert frozen_item["definition"] == "맡은 일을 책임지고 수행하는 특성"
    assert frozen_item["path"] == ("성실성", "책임성")


def test_repeated_requests_load_registry_only_once(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    load_calls: list[str] = []

    class CompiledAppStub:
        def invoke(self, state: dict, *, config: dict) -> dict:
            assert config["configurable"]["thread_id"]
            return state

    class BuilderStub:
        def compile(self, *, checkpointer: object) -> CompiledAppStub:
            assert checkpointer is not None
            return CompiledAppStub()

    def load_registry(database_url: str) -> RegistrySnapshot:
        load_calls.append(database_url)
        return synthetic_registry

    competency_interpreter.close_competency_runtime()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(
        competency_interpreter,
        "load_active_registry",
        load_registry,
    )
    monkeypatch.setattr(
        competency_interpreter,
        "_open_checkpointer",
        lambda *_: object(),
    )
    monkeypatch.setattr(competency_interpreter, "builder", BuilderStub())

    competency_interpreter.run_competency("첫 질문", "thread-1")
    competency_interpreter.run_competency("둘째 질문", "thread-2")

    assert load_calls == ["postgresql://test"]


def test_close_then_reinitialize_loads_new_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_snapshot = make_synthetic_registry(version_id=1)
    second_snapshot = make_synthetic_registry(version_id=2)
    snapshots = iter([first_snapshot, second_snapshot])
    loaded_version_ids: list[int] = []

    class BuilderStub:
        def compile(self, *, checkpointer: object) -> object:
            assert checkpointer is not None
            return object()

    def load_registry(_: str) -> RegistrySnapshot:
        snapshot = next(snapshots)
        loaded_version_ids.append(snapshot.version_id)
        return snapshot

    competency_interpreter.close_competency_runtime()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(
        competency_interpreter,
        "load_active_registry",
        load_registry,
    )
    monkeypatch.setattr(
        competency_interpreter,
        "_open_checkpointer",
        lambda *_: object(),
    )
    monkeypatch.setattr(competency_interpreter, "builder", BuilderStub())

    competency_interpreter.initialize_competency_runtime()
    assert competency_interpreter._registry_snapshot is first_snapshot

    competency_interpreter.close_competency_runtime()
    competency_interpreter.initialize_competency_runtime()

    assert competency_interpreter._registry_snapshot is second_snapshot
    assert loaded_version_ids == [1, 2]
