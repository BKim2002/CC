from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.registry_source_v2 import (
    SourceRegistry,
    SourceRegistryError,
    parse_source_v2,
    parse_source_v2_text,
    render_source_v2,
)


def _payload() -> dict:
    return {
        "source_schema_version": "2.0",
        "registry_schema_version": "1.0",
        "instruments": [
            {"id": "written", "label": "필기 역량검사"},
        ],
        "items": [
            {
                "id": "competency_0017",
                "name": "성실성",
                "instrument": "written",
                "node_type": "analysis_factor",
                "parent_id": None,
                "order": 10,
                "definition": "맡은 일을 꼼꼼하게 수행한다.",
                "aliases": [],
                "analysis_included": True,
                "status": "active",
                "replacement_id": None,
                "notes": [],
            }
        ],
    }


def _markdown(payload: dict | None = None, *, before: str = "# 문서") -> str:
    data = _payload() if payload is None else payload
    yaml_text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    return (
        f"{before}\n\n"
        "```competency-registry-yaml\n"
        f"{yaml_text}"
        "```\n"
    )


def test_parse_exactly_one_registry_fence() -> None:
    source = parse_source_v2_text(_markdown())

    assert source.source_schema_version == "2.0"
    assert source.items[0].id == "competency_0017"


@pytest.mark.parametrize(
    "text, count",
    [
        ("# no registry block\n", "0"),
        (_markdown() + "\n" + _markdown(), "2"),
    ],
)
def test_parse_requires_exactly_one_registry_fence(text: str, count: str) -> None:
    with pytest.raises(SourceRegistryError, match=rf"exactly one.*{count}"):
        parse_source_v2_text(text)


def test_parse_rejects_unclosed_or_empty_fence() -> None:
    with pytest.raises(SourceRegistryError, match="not closed"):
        parse_source_v2_text("```competency-registry-yaml\nitems: []\n")

    with pytest.raises(SourceRegistryError, match="empty"):
        parse_source_v2_text("```competency-registry-yaml\n```\n")


def test_parse_rejects_invalid_yaml() -> None:
    text = "```competency-registry-yaml\nitems: [\n```\n"

    with pytest.raises(SourceRegistryError, match="Invalid registry YAML"):
        parse_source_v2_text(text)


def test_parse_rejects_duplicate_yaml_key_at_any_depth() -> None:
    text = _markdown().replace(
        "  name: 성실성\n",
        "  name: 성실성\n  name: 중복 이름\n",
    )

    with pytest.raises(SourceRegistryError, match="duplicate key"):
        parse_source_v2_text(text)


def test_parse_rejects_unknown_field() -> None:
    payload = _payload()
    payload["unexpected"] = "not allowed"

    with pytest.raises(SourceRegistryError, match="extra_forbidden"):
        parse_source_v2_text(_markdown(payload))


def test_schema_error_does_not_echo_source_content() -> None:
    confidential = "CONFIDENTIAL-DEFINITION-CONTENT"
    payload = _payload()
    payload["items"][0]["definition"] = {"invalid": confidential}

    with pytest.raises(SourceRegistryError) as captured:
        parse_source_v2_text(_markdown(payload))

    assert confidential not in str(captured.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_schema_version", "3.0"),
        ("registry_schema_version", "2.0"),
    ],
)
def test_parse_rejects_unsupported_versions(field: str, value: str) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(SourceRegistryError, match=field):
        parse_source_v2_text(_markdown(payload))


@pytest.mark.parametrize(
    ("field", "coercible_value"),
    [
        ("id", 17),
        ("order", "10"),
        ("analysis_included", "true"),
    ],
)
def test_parse_disables_pydantic_type_coercion(
    field: str,
    coercible_value: object,
) -> None:
    payload = _payload()
    payload["items"][0][field] = coercible_value

    with pytest.raises(SourceRegistryError, match=field):
        parse_source_v2_text(_markdown(payload))


def test_item_id_is_required_and_never_inferred_from_name() -> None:
    payload = _payload()
    del payload["items"][0]["id"]

    with pytest.raises(SourceRegistryError, match="items.0.id"):
        parse_source_v2_text(_markdown(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("definition", " \n\t"),
        ("notes", ["   "]),
    ],
)
def test_parse_rejects_whitespace_only_content(
    field: str,
    value: object,
) -> None:
    payload = _payload()
    payload["items"][0][field] = value

    with pytest.raises(SourceRegistryError, match="whitespace-only"):
        parse_source_v2_text(_markdown(payload))


def test_instrument_root_path_prefix_is_strict_and_optional() -> None:
    source = parse_source_v2_text(_markdown())
    assert source.instruments[0].root_path_prefix == []

    payload = _payload()
    payload["instruments"][0]["root_path_prefix"] = ["가상 검사 루트"]
    parsed = parse_source_v2_text(_markdown(payload))
    assert parsed.instruments[0].root_path_prefix == ["가상 검사 루트"]

    payload["instruments"][0]["root_path_prefix"] = ["   "]
    with pytest.raises(SourceRegistryError, match="whitespace-only"):
        parse_source_v2_text(_markdown(payload))


def test_outside_markdown_does_not_change_parsed_source() -> None:
    first = parse_source_v2_text(_markdown(before="# 첫 제목\n| 임의 | 표 |"))
    second = parse_source_v2_text(
        _markdown(before="# 완전히 다른 제목\n\n일반 설명도 달라졌다.")
    )

    assert first == second


def test_definition_pipe_and_line_breaks_are_preserved() -> None:
    definition = "첫 줄에는 | 기호가 있다.\n둘째 줄도 그대로다.\n"
    payload = _payload()
    payload["items"][0]["definition"] = definition

    source = parse_source_v2_text(_markdown(payload))

    assert source.items[0].definition == definition


def test_parse_path_rejects_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "registry.md"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(SourceRegistryError, match="UTF-8"):
        parse_source_v2(path)


def test_render_round_trip_preserves_multiline_strings() -> None:
    payload = _payload()
    payload["items"][0]["definition"] = "첫 줄\n둘째 줄\n"
    payload["items"][0]["notes"] = ["메모 첫 줄\n메모 둘째 줄"]
    source = SourceRegistry.model_validate(payload, strict=True)

    rendered = render_source_v2(
        source,
        title="새 문서",
        preamble="이 설명은 파싱되지 않는다.",
    )

    assert rendered.count("```competency-registry-yaml") == 1
    assert parse_source_v2_text(rendered) == source


@pytest.mark.parametrize(
    ("title", "preamble", "message"),
    [
        (
            "제목\n```competency-registry-yaml",
            None,
            "single line",
        ),
        (
            "제목",
            "설명\n```competency-registry-yaml\nitems: []\n```",
            "must not contain",
        ),
    ],
)
def test_renderer_cannot_create_a_second_registry_fence(
    title: str,
    preamble: str | None,
    message: str,
) -> None:
    source = SourceRegistry.model_validate(_payload(), strict=True)

    with pytest.raises(SourceRegistryError, match=message):
        render_source_v2(source, title=title, preamble=preamble)


def test_yaml_error_does_not_retain_private_source_value() -> None:
    private_value = "TOP_SECRET_SOURCE_CONTENT"
    text = _markdown().replace(
        "  name: 성실성\n",
        f"  name: 성실성\n  name: {private_value}\n",
    )

    with pytest.raises(SourceRegistryError) as captured:
        parse_source_v2_text(text)

    assert private_value not in str(captured.value)
    assert captured.value.__cause__ is None
