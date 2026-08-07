"""정확 일치, generalized query graph, streaming의 핵심 회귀 테스트."""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

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

    assert result["resolved_names"] == ["책임성"]
    assert result["requested_fields"] == ["definition"]
    assert result["candidate_names"] == []
    assert result["llm_route"] == "find_competencies"
    assert result["query_plan"]["target_ids"] == ["responsibility"]


def test_llm_parser_failure_returns_safe_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = StubParser(RuntimeError("LLM unavailable"))
    monkeypatch.setattr(
        competency_interpreter,
        "_query_parser_for",
        lambda _: parser,
    )

    result = llm_interpret_query(
        {"raw_query": "책임성의 뜻만 알려줘"}
    )

    assert result["llm_route"] == "clarify_query"
    assert "LLM unavailable" not in result["clarification_prompt"]


def test_blank_model_setting_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    assert selected_model_name() == "gpt-5.6-luna"


def test_model_setting_override_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")

    assert selected_model_name() == "custom-model"


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


def test_close_defers_resource_shutdown_until_active_async_request_releases(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    closed: list[str] = []
    runtime_stack = ExitStack()
    runtime_stack.callback(closed.append, "closed")
    compiled_app = object()

    monkeypatch.setattr(competency_interpreter, "app", compiled_app)
    monkeypatch.setattr(competency_interpreter, "_runtime_stack", runtime_stack)
    monkeypatch.setattr(competency_interpreter, "_registry_snapshot", synthetic_registry)
    monkeypatch.setattr(competency_interpreter, "_runtime_active_users", 0)
    monkeypatch.setattr(competency_interpreter, "_runtime_close_pending", False)
    monkeypatch.setattr(competency_interpreter, "initialize_competency_runtime", lambda: None)

    async def scenario() -> None:
        async with competency_interpreter._async_runtime_execution() as leased_app:
            assert leased_app is compiled_app
            competency_interpreter.close_competency_runtime()
            assert closed == []
            assert competency_interpreter.app is compiled_app
            assert competency_interpreter._registry_snapshot is synthetic_registry
            with pytest.raises(RuntimeError, match="사용할 수 없습니다"):
                async with competency_interpreter._async_runtime_execution():
                    pass

        assert closed == ["closed"]
        assert competency_interpreter.app is None
        assert competency_interpreter._registry_snapshot is None
        assert competency_interpreter._runtime_stack is None

    asyncio.run(scenario())


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


def test_exact_graph_path_uses_stable_id_plan_without_registry_item_state() -> None:
    interpreted = interpret_query(
        {"messages": [HumanMessage(content="책임성")]}
    )

    assert interpreted["query_plan"]["target_ids"] == ["responsibility"]
    executed = competency_interpreter.find_competencies(interpreted)

    assert executed["result_ids"] == ["responsibility"]
    assert executed["matched_items"] == []
    json.dumps(executed["query_result"], ensure_ascii=False)
    assert "definition" not in executed["query_result"]


def test_exact_whitespace_collision_still_uses_stable_id_plan(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    items = [
        competency_interpreter._mutable_json_copy(item)
        for item in synthetic_registry.document["items"]
    ]
    items[0]["name"] = "A B"
    items[1]["aliases"] = ["A  B"]
    canonical_lookup = {item["name"]: item for item in items}
    lookup = dict(canonical_lookup)
    for item in items:
        for alias in item["aliases"]:
            lookup[alias] = item
    collision_snapshot = RegistrySnapshot(
        version_id=99,
        source_filename="collision.md",
        source_sha256="f" * 64,
        schema_version="1.0",
        document={"items": items},
        lookup=lookup,
        canonical_lookup=canonical_lookup,
        canonical_names=tuple(canonical_lookup),
        name_catalog="",
        semantic_catalog="",
    )
    monkeypatch.setattr(
        competency_interpreter,
        "_registry_snapshot",
        collision_snapshot,
    )

    interpreted = interpret_query(
        {"messages": [HumanMessage(content="A B")]}
    )
    executed = competency_interpreter.find_competencies(interpreted)

    assert interpreted["query_plan"]["target_ids"] == [items[0]["id"]]
    assert executed["matched_items"] == []
    assert "definition" not in executed["query_result"]


def test_invalid_collision_registry_never_checkpoints_legacy_item_dict(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_registry: RegistrySnapshot,
) -> None:
    items = [
        competency_interpreter._mutable_json_copy(item)
        for item in synthetic_registry.document["items"]
    ]
    items[0]["name"] = "A B"
    items[1]["aliases"] = ["A  B"]
    # The snapshot schema currently tolerates this malformed derived catalog,
    # but query validation must not fall through to the legacy item-dict node.
    items[1]["instrument_label"] = "불일치 검사명"
    canonical_lookup = {item["name"]: item for item in items}
    lookup = dict(canonical_lookup)
    for item in items:
        for alias in item["aliases"]:
            lookup[alias] = item
    collision_snapshot = RegistrySnapshot(
        version_id=100,
        source_filename="invalid-collision.md",
        source_sha256="e" * 64,
        schema_version="1.0",
        document={"items": items},
        lookup=lookup,
        canonical_lookup=canonical_lookup,
        canonical_names=tuple(canonical_lookup),
        name_catalog="",
        semantic_catalog="",
    )
    monkeypatch.setattr(competency_interpreter, "_registry_snapshot", collision_snapshot)

    interpreted = interpret_query({"messages": [HumanMessage(content="A B")]})
    assert interpreted["query_plan"] == {}
    assert interpreted["matched_items"] == []
    assert route_after_interpret(interpreted) == "clarify_query"

    compiled = competency_interpreter.builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "invalid-collision"}}
    final_state = compiled.invoke(
        {"messages": [HumanMessage(content="A B")]},
        config=config,
    )
    saved = compiled.get_state(config).values
    assert final_state["matched_items"] == []
    assert saved["matched_items"] == []
    assert "definition" not in str(saved.get("query_plan", {}))


def test_stale_id_context_does_not_fall_back_to_same_name() -> None:
    prompt = competency_interpreter._parser_prompt(
        {
            "last_result_ids": ["deleted-id"],
            "last_resolved_names": ["책임성"],
        }
    )
    legacy_prompt = competency_interpreter._parser_prompt(
        {"last_resolved_names": ["책임성"]}
    )

    assert "- 이전 결과 이름(최대 20개): 없음" in prompt
    assert "- 이전 결과 이름(최대 20개): ['책임성']" in legacy_prompt


def test_new_turn_resets_transient_query_state_but_keeps_last_context() -> None:
    updates = interpret_query(
        {
            "messages": [HumanMessage(content="모호한 새 질문")],
            "query_plan": {"old": True},
            "query_result": {"old": True},
            "result_ids": ["old"],
            "matched_items": [{"definition": "old"}],
            "last_query_plan": {"intent": "catalog_query"},
            "last_result_ids": ["responsibility", "stale-id"],
        }
    )

    assert updates["query_plan"] == {}
    assert updates["query_result"] == {}
    assert updates["result_ids"] == []
    assert updates["matched_items"] == []
    # LangGraph merge에서 반환하지 않은 last_* 값은 그대로 유지된다.
    assert "last_query_plan" not in updates
    assert "last_result_ids" not in updates


class StubStreamingWriter:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    async def astream(self, _: object):
        for text in self.chunks:
            yield AIMessageChunk(content=text)


def _exact_plan_and_result() -> tuple[dict, dict, list[str]]:
    interpreted = interpret_query(
        {"messages": [HumanMessage(content="책임성")]}
    )
    executed = competency_interpreter.find_competencies(interpreted)
    return (
        interpreted["query_plan"],
        executed["query_result"],
        executed["result_ids"],
    )


def test_streamed_writer_checkpoints_only_validated_final_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, query_result, result_ids = _exact_plan_and_result()
    context = competency_interpreter.build_grounded_answer_context(
        competency_interpreter.RegistryQueryPlan.model_validate(plan),
        competency_interpreter.RegistryQueryResult.model_validate(query_result),
        competency_interpreter._require_registry(),
    )
    expected = competency_interpreter.render_grounded_fallback(context)
    midpoint = len(expected) // 2
    events: list[dict] = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        competency_interpreter,
        "_response_writer_for",
        lambda _: StubStreamingWriter([expected[:midpoint], expected[midpoint:]]),
    )
    monkeypatch.setattr(
        competency_interpreter,
        "get_stream_writer",
        lambda: events.append,
    )

    update = asyncio.run(
        competency_interpreter.write_streamed_answer(
            {
                "raw_query": "책임성",
                "query_plan": plan,
                "query_result": query_result,
                "result_ids": result_ids,
            }
        )
    )

    assert update["messages"] == AIMessage(content=expected)
    assert update["response_mode"] == "llm"
    assert "query_result" not in update
    assert "matched_items" not in update
    assert "".join(
        event["text"] for event in events if event["type"] == "delta"
    ) == expected
    assert not any(event["type"] == "replace" for event in events)


def test_streamed_writer_replaces_trimmed_whitespace_to_match_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, query_result, result_ids = _exact_plan_and_result()
    context = competency_interpreter.build_grounded_answer_context(
        competency_interpreter.RegistryQueryPlan.model_validate(plan),
        competency_interpreter.RegistryQueryResult.model_validate(query_result),
        competency_interpreter._require_registry(),
    )
    expected = competency_interpreter.render_grounded_fallback(context)
    events: list[dict] = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        competency_interpreter,
        "_response_writer_for",
        lambda _: StubStreamingWriter(["\n", expected, " "]),
    )
    monkeypatch.setattr(
        competency_interpreter,
        "get_stream_writer",
        lambda: events.append,
    )

    update = asyncio.run(
        competency_interpreter.write_streamed_answer(
            {
                "raw_query": "책임성",
                "query_plan": plan,
                "query_result": query_result,
                "result_ids": result_ids,
            }
        )
    )

    assert str(update["messages"].content) == expected
    assert "".join(
        event["text"] for event in events if event["type"] == "delta"
    ) == f"\n{expected} "
    assert events[-1] == {"type": "replace", "answer": expected}


def test_invalid_partial_writer_is_replaced_and_never_checkpointed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, query_result, result_ids = _exact_plan_and_result()
    events: list[dict] = []
    partial = "책임성에 관한 검증되지 않은 수치 999"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        competency_interpreter,
        "_response_writer_for",
        lambda _: StubStreamingWriter([partial]),
    )
    monkeypatch.setattr(
        competency_interpreter,
        "get_stream_writer",
        lambda: events.append,
    )

    update = asyncio.run(
        competency_interpreter.write_streamed_answer(
            {
                "raw_query": "책임성",
                "query_plan": plan,
                "query_result": query_result,
                "result_ids": result_ids,
            }
        )
    )

    assert update["response_mode"] == "fallback"
    assert str(update["messages"].content) != partial
    assert events[-1] == {
        "type": "replace",
        "answer": str(update["messages"].content),
    }
    assert partial not in str(update["messages"].content)


def test_run_stream_done_matches_checkpoint_and_contains_no_partial_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    compiled = competency_interpreter.builder.compile(checkpointer=checkpointer)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(competency_interpreter, "app", compiled)
    monkeypatch.setattr(
        competency_interpreter,
        "initialize_competency_runtime",
        lambda: None,
    )

    async def collect() -> list[dict]:
        return [
            event
            async for event in competency_interpreter.run_competency_stream(
                "책임성",
                "stream-thread",
            )
        ]

    events = asyncio.run(collect())
    done = events[-1]
    assert events[0] == {"type": "start", "thread_id": "stream-thread"}
    assert done["type"] == "done"
    assert any(event["type"] == "delta" for event in events)

    saved = compiled.get_state(
        {"configurable": {"thread_id": "stream-thread"}}
    ).values
    assistant_messages = [
        message
        for message in saved["messages"]
        if isinstance(message, AIMessage)
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == done["answer"]
    assert saved["last_result_ids"] == ["responsibility"]
    assert saved["matched_items"] == []
    json.dumps(saved["query_plan"], ensure_ascii=False)
    json.dumps(saved["query_result"], ensure_ascii=False)


def test_stream_final_nodes_emit_answer_delta_before_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = "안전한 도움말 답변"

    class FinalOnlyApp:
        async def astream(self, *_: object, **__: object):
            yield (
                "values",
                {
                    "messages": [
                        HumanMessage(content="도움말"),
                        AIMessage(content=answer),
                    ],
                    "candidate_names": [],
                },
            )

    monkeypatch.setattr(competency_interpreter, "app", FinalOnlyApp())
    monkeypatch.setattr(
        competency_interpreter,
        "initialize_competency_runtime",
        lambda: None,
    )

    async def collect() -> list[dict]:
        return [
            event
            async for event in competency_interpreter.run_competency_stream(
                "도움말",
                "help-thread",
            )
        ]

    events = asyncio.run(collect())
    assert events[-2] == {"type": "delta", "text": answer}
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == answer


def test_cancelled_model_stream_does_not_checkpoint_partial_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    compiled = competency_interpreter.builder.compile(checkpointer=checkpointer)
    release_writer = asyncio.Event()

    class BlockingWriter:
        async def astream(self, _: object):
            yield AIMessageChunk(content="잠정 부분 답변")
            await release_writer.wait()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        competency_interpreter,
        "_response_writer_for",
        lambda _: BlockingWriter(),
    )
    monkeypatch.setattr(competency_interpreter, "app", compiled)
    monkeypatch.setattr(
        competency_interpreter,
        "initialize_competency_runtime",
        lambda: None,
    )

    async def cancel_after_partial() -> None:
        stream = competency_interpreter.run_competency_stream(
            "책임성",
            "cancel-thread",
        )
        async for event in stream:
            if event.get("type") == "delta":
                assert event["text"] == "잠정 부분 답변"
                await stream.aclose()
                return
        raise AssertionError("partial delta를 받지 못했습니다.")

    asyncio.run(cancel_after_partial())
    saved = compiled.get_state(
        {"configurable": {"thread_id": "cancel-thread"}}
    ).values
    assert not any(
        isinstance(message, AIMessage)
        for message in saved.get("messages", [])
    )
    assert "잠정 부분 답변" not in str(saved)


def test_same_thread_streams_are_serialized_without_blocking_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0

    class SlowFinalApp:
        async def astream(self, *_: object, **__: object):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            yield (
                "values",
                {
                    "messages": [AIMessage(content="완료")],
                    "candidate_names": [],
                },
            )

    monkeypatch.setattr(competency_interpreter, "app", SlowFinalApp())
    monkeypatch.setattr(
        competency_interpreter,
        "initialize_competency_runtime",
        lambda: None,
    )

    async def collect(thread_id: str) -> list[dict]:
        return [
            event
            async for event in competency_interpreter.run_competency_stream(
                "질문",
                thread_id,
            )
        ]

    async def run_pair() -> tuple[list[dict], list[dict]]:
        first, second = await asyncio.gather(
            collect("same-thread"),
            collect("same-thread"),
        )
        return first, second

    first, second = asyncio.run(run_pair())
    assert max_active == 1
    assert first[-1]["type"] == "done"
    assert second[-1]["type"] == "done"


def test_cancelled_lock_waiter_does_not_block_next_same_thread_request() -> None:
    async def scenario() -> None:
        first = competency_interpreter._async_thread_execution("cancel-waiter")
        await first.__aenter__()

        async def wait_for_lock() -> None:
            async with competency_interpreter._async_thread_execution(
                "cancel-waiter"
            ):
                return

        waiter = asyncio.create_task(wait_for_lock())
        await asyncio.sleep(0.03)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await first.__aexit__(None, None, None)

        async with asyncio.timeout(0.2):
            async with competency_interpreter._async_thread_execution(
                "cancel-waiter"
            ):
                pass

    asyncio.run(scenario())
    assert "cancel-waiter" not in competency_interpreter._thread_locks


def test_different_thread_streams_can_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0

    class SlowFinalApp:
        async def astream(self, *_: object, **__: object):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            yield (
                "values",
                {
                    "messages": [AIMessage(content="완료")],
                    "candidate_names": [],
                },
            )

    monkeypatch.setattr(competency_interpreter, "app", SlowFinalApp())
    monkeypatch.setattr(
        competency_interpreter,
        "initialize_competency_runtime",
        lambda: None,
    )

    async def collect(thread_id: str) -> list[dict]:
        return [
            event
            async for event in competency_interpreter.run_competency_stream(
                "질문",
                thread_id,
            )
        ]

    async def run_pair() -> None:
        await asyncio.gather(collect("thread-a"), collect("thread-b"))

    asyncio.run(run_pair())
    assert max_active == 2
