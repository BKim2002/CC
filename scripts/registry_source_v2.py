"""Strict parser and renderer for competency registry Markdown source V2.

The surrounding Markdown is documentation only.  Registry data is read from
exactly one ``competency-registry-yaml`` fenced block and validated without
Pydantic type coercion.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


SOURCE_SCHEMA_VERSION = "2.0"
REGISTRY_SCHEMA_VERSION = "1.0"
FENCE_INFO_STRING = "competency-registry-yaml"


class SourceRegistryError(ValueError):
    """Raised when a structured Markdown V2 source is invalid."""


def _require_non_blank(value: str) -> str:
    """Reject blank strings without normalizing meaningful source content."""

    if not value.strip():
        raise ValueError("must not be empty or whitespace-only")
    return value


NonBlankString = Annotated[str, AfterValidator(_require_non_blank)]


class _StrictSourceModel(BaseModel):
    """Base configuration shared by every source-schema model."""

    model_config = ConfigDict(extra="forbid", strict=True)


class SourceInstrument(_StrictSourceModel):
    """A declared assessment instrument."""

    id: NonBlankString
    label: NonBlankString
    # Some legacy registries expose an instrument-level virtual ancestor in
    # item paths without storing it as a real item/parent ID.  Keeping that
    # prefix on the instrument preserves the user-visible path while the item
    # graph remains a single-parent tree of stable IDs.
    root_path_prefix: list[NonBlankString] = Field(default_factory=list)


class SourceItem(_StrictSourceModel):
    """One stable-ID source item before hierarchy fields are compiled."""

    id: NonBlankString
    name: NonBlankString
    instrument: NonBlankString
    node_type: NonBlankString
    parent_id: NonBlankString | None
    order: int
    definition: NonBlankString | None
    aliases: list[NonBlankString]
    analysis_included: bool
    status: Literal["active", "retired"]
    replacement_id: NonBlankString | None
    notes: list[NonBlankString]


class SourceRegistry(_StrictSourceModel):
    """Validated source schema embedded in structured Markdown V2."""

    source_schema_version: Literal["2.0"]
    registry_schema_version: Literal["1.0"]
    instruments: list[SourceInstrument] = Field(min_length=1)
    items: list[SourceItem] = Field(min_length=1)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate keys at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc

        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key ({key!r})",
                key_node.start_mark,
            )

        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class _LiteralSafeDumper(yaml.SafeDumper):
    """Safe dumper that keeps multiline source strings readable."""


def _represent_string(
    dumper: _LiteralSafeDumper,
    value: str,
) -> yaml.nodes.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str",
        value,
        style=style,
    )


_LiteralSafeDumper.add_representer(str, _represent_string)


_OPENING_FENCE = re.compile(
    rf"^[ ]{{0,3}}```{re.escape(FENCE_INFO_STRING)}[\t ]*$"
)
_CLOSING_FENCE = re.compile(r"^[ ]{0,3}```[\t ]*$")


def _extract_yaml_block(markdown_text: str) -> str:
    """Extract the sole registry fence without interpreting other Markdown."""

    lines = markdown_text.splitlines(keepends=True)
    opening_indexes = [
        index
        for index, line in enumerate(lines)
        if _OPENING_FENCE.fullmatch(line.rstrip("\r\n")) is not None
    ]

    if len(opening_indexes) != 1:
        raise SourceRegistryError(
            "Markdown source must contain exactly one "
            f"```{FENCE_INFO_STRING} fenced block; found "
            f"{len(opening_indexes)}."
        )

    opening_index = opening_indexes[0]
    closing_index: int | None = None
    for index in range(opening_index + 1, len(lines)):
        if _CLOSING_FENCE.fullmatch(lines[index].rstrip("\r\n")) is not None:
            closing_index = index
            break

    if closing_index is None:
        raise SourceRegistryError(
            f"The ```{FENCE_INFO_STRING} fenced block is not closed."
        )

    yaml_text = "".join(lines[opening_index + 1 : closing_index])
    if not yaml_text.strip():
        raise SourceRegistryError(
            f"The ```{FENCE_INFO_STRING} fenced block is empty."
        )
    return yaml_text


def parse_source_v2_text(text: str) -> SourceRegistry:
    """Parse and strictly validate structured Markdown V2 text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    yaml_text = _extract_yaml_block(text)
    try:
        payload = yaml.load(yaml_text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError:
        # PyYAML diagnostics include source lines and scalar values.  Registry
        # definitions may be private, so expose only a content-free category.
        raise SourceRegistryError(
            "Invalid registry YAML (duplicate key or syntax error)."
        ) from None

    if not isinstance(payload, dict):
        raise SourceRegistryError(
            "The registry YAML document must be a mapping/object."
        )

    try:
        return SourceRegistry.model_validate(payload, strict=True)
    except ValidationError as exc:
        # Pydantic's default string includes ``input_value``.  Avoid copying
        # definitions or other source content into CLI/log error messages.
        diagnostics = []
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(member) for member in error["loc"])
            diagnostics.append(f"{location}: {error['msg']} ({error['type']})")
        raise SourceRegistryError(
            "Invalid source schema V2: " + "; ".join(diagnostics)
        ) from None


def parse_source_v2(path: Path) -> SourceRegistry:
    """Read a UTF-8 Markdown file and parse its source-schema V2 block."""

    source_path = Path(path)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise SourceRegistryError(
            f"Unable to read registry source: {source_path}"
        ) from exc

    try:
        markdown_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceRegistryError(
            f"Registry source is not valid UTF-8: {source_path}"
        ) from exc

    return parse_source_v2_text(markdown_text)


def render_source_v2(
    source: SourceRegistry,
    *,
    title: str = "역량 정의서",
    preamble: str | None = None,
) -> str:
    """Render a validated source as Markdown with exactly one V2 YAML fence."""

    if not isinstance(source, SourceRegistry):
        raise TypeError("source must be a SourceRegistry")
    if not isinstance(title, str) or not title.strip():
        raise SourceRegistryError("title must be a non-blank string")
    if "\n" in title or "\r" in title:
        raise SourceRegistryError("title must be a single line")
    if preamble is not None and not isinstance(preamble, str):
        raise TypeError("preamble must be a string or None")
    if preamble is not None and any(
        _OPENING_FENCE.fullmatch(line) is not None
        for line in preamble.splitlines()
    ):
        raise SourceRegistryError(
            "preamble must not contain a competency registry fence"
        )

    payload = source.model_dump(mode="json")
    yaml_text = yaml.dump(
        payload,
        Dumper=_LiteralSafeDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    )

    sections = [f"# {title.strip()}"]
    if preamble is not None and preamble.strip():
        sections.append(preamble.rstrip())
    sections.append(
        f"```{FENCE_INFO_STRING}\n{yaml_text.rstrip()}\n```"
    )
    return "\n\n".join(sections) + "\n"


__all__ = [
    "FENCE_INFO_STRING",
    "REGISTRY_SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "SourceInstrument",
    "SourceItem",
    "SourceRegistry",
    "SourceRegistryError",
    "parse_source_v2",
    "parse_source_v2_text",
    "render_source_v2",
]
