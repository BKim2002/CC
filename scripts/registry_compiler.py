"""Generic single-parent tree validator and runtime registry compiler."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
from typing import Any

try:  # Support both package imports and direct execution from ``scripts``.
    from scripts.registry_source_v2 import SourceItem, SourceRegistry
except ModuleNotFoundError:  # pragma: no cover - direct-script compatibility
    from registry_source_v2 import SourceItem, SourceRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - direct CLI path
    sys.path.insert(0, str(PROJECT_ROOT))

from competency_registry import (  # noqa: E402 - path supports direct scripts
    RegistryValidationError,
    validate_registry_document,
)


class RegistryCompileError(ValueError):
    """Raised when source items cannot form a valid runtime registry tree."""


_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def _raise(message: str) -> None:
    raise RegistryCompileError(message)


def _index_source(
    source: SourceRegistry,
) -> tuple[dict[str, str], dict[str, SourceItem]]:
    instruments: dict[str, str] = {}
    for instrument in source.instruments:
        if instrument.id in instruments:
            _raise(f"Duplicate instrument id: {instrument.id}")
        instruments[instrument.id] = instrument.label

    items: dict[str, SourceItem] = {}
    for item in source.items:
        if item.id in items:
            _raise(f"Duplicate item id: {item.id}")
        items[item.id] = item
        if item.instrument not in instruments:
            _raise(
                f"Item {item.id} references unknown instrument: "
                f"{item.instrument}"
            )

    return instruments, items


def _validate_lookup_namespace(items: dict[str, SourceItem]) -> None:
    """Validate the global lookup namespace used by the active snapshot."""

    active_items = [item for item in items.values() if item.status == "active"]
    active_name_owners: dict[str, str] = {}

    # Canonical-name uniqueness is a runtime rule and therefore applies to
    # active items.  Retired canonical names may repeat because they are audit
    # records and never enter the runtime lookup.
    for item in active_items:
        owner = active_name_owners.get(item.name)
        if owner is not None:
            _raise(
                f"Duplicate active item name {item.name!r}: {owner}, {item.id}"
            )
        active_name_owners[item.name] = item.id

    # Aliases are still source-wide identifiers.  Rejecting collisions against
    # every canonical name and every other alias prevents a retired record from
    # making a future reactivation ambiguous.
    canonical_name_owners: dict[str, list[str]] = defaultdict(list)
    for item in items.values():
        canonical_name_owners[item.name].append(item.id)
    alias_owners: dict[str, str] = {}

    for item in source_order(items):
        aliases_seen: set[str] = set()
        for alias in item.aliases:
            if alias in aliases_seen:
                _raise(f"Duplicate alias {alias!r} on item {item.id}")
            aliases_seen.add(alias)

            if alias == item.name:
                _raise(
                    f"Alias {alias!r} duplicates the canonical name on item "
                    f"{item.id}"
                )

            canonical_owners = canonical_name_owners.get(alias, [])
            if canonical_owners:
                _raise(
                    f"Global name/alias collision {alias!r}: "
                    f"{', '.join(canonical_owners)}, {item.id}"
                )
            owner = alias_owners.get(alias)
            if owner is not None:
                _raise(f"Global duplicate alias {alias!r}: {owner}, {item.id}")
            alias_owners[alias] = item.id


def source_order(items: dict[str, SourceItem]) -> list[SourceItem]:
    """Return source items in their model order without exposing mutation."""

    # ``dict`` preserves insertion order and the index is built from the source
    # list.  Keeping this helper explicit makes validation diagnostics stable.
    return list(items.values())


def _validate_parent_relations(items: dict[str, SourceItem]) -> None:
    for item in items.values():
        if item.parent_id is None:
            continue

        if item.parent_id == item.id:
            _raise(f"Item {item.id} cannot be its own parent")

        parent = items.get(item.parent_id)
        if parent is None:
            _raise(
                f"Item {item.id} references missing parent: {item.parent_id}"
            )
        if parent.instrument != item.instrument:
            _raise(
                f"Cross-instrument parent is not allowed: {item.id} -> "
                f"{parent.id}"
            )
        if item.status == "active" and parent.status != "active":
            _raise(
                f"Active item {item.id} cannot reference retired parent "
                f"{parent.id}"
            )


def _validate_cycles(items: dict[str, SourceItem]) -> None:
    # 0 = unseen, 1 = visiting, 2 = complete.
    colors: dict[str, int] = {}
    stack: list[str] = []

    def visit(item_id: str) -> None:
        color = colors.get(item_id, 0)
        if color == 2:
            return
        if color == 1:
            cycle_start = stack.index(item_id)
            cycle = stack[cycle_start:] + [item_id]
            _raise("Hierarchy cycle detected: " + " -> ".join(cycle))

        colors[item_id] = 1
        stack.append(item_id)
        parent_id = items[item_id].parent_id
        if parent_id is not None:
            visit(parent_id)
        stack.pop()
        colors[item_id] = 2

    for item_id in items:
        visit(item_id)


def _validate_active_orders_and_reachability(
    instruments: dict[str, str],
    items: dict[str, SourceItem],
) -> None:
    active = {
        item_id: item
        for item_id, item in items.items()
        if item.status == "active"
    }
    if not active:
        _raise("A runtime registry must contain at least one active item")

    sibling_orders: dict[tuple[str, str | None], dict[int, str]] = defaultdict(dict)
    children: dict[str, list[str]] = defaultdict(list)
    roots_by_instrument: dict[str, list[str]] = defaultdict(list)

    for item in active.values():
        sibling_group = (item.instrument, item.parent_id)
        prior = sibling_orders[sibling_group].get(item.order)
        if prior is not None:
            parent_label = item.parent_id if item.parent_id is not None else "<root>"
            _raise(
                f"Duplicate sibling order {item.order} under {parent_label}: "
                f"{prior}, {item.id}"
            )
        sibling_orders[sibling_group][item.order] = item.id

        if item.parent_id is None:
            roots_by_instrument[item.instrument].append(item.id)
        else:
            children[item.parent_id].append(item.id)

    active_instruments = {item.instrument for item in active.values()}
    for instrument_id in active_instruments:
        if not roots_by_instrument[instrument_id]:
            _raise(f"Instrument has no active root: {instrument_id}")

    reached: set[str] = set()
    pending = [
        root_id
        for instrument_id in instruments
        for root_id in roots_by_instrument[instrument_id]
    ]
    while pending:
        item_id = pending.pop()
        if item_id in reached:
            # A single parent plus unique IDs makes this a duplicate edge.
            _raise(f"Duplicate hierarchy edge reaches item: {item_id}")
        reached.add(item_id)
        pending.extend(children[item_id])

    disconnected = sorted(set(active).difference(reached))
    if disconnected:
        _raise(
            "Active items are not reachable from an instrument root: "
            + ", ".join(disconnected)
        )


def _validate_replacements(items: dict[str, SourceItem]) -> None:
    for item in items.values():
        replacement_id = item.replacement_id
        if item.status == "active" and replacement_id is not None:
            _raise(f"Active item {item.id} cannot declare replacement_id")
        if replacement_id is None:
            continue
        if replacement_id == item.id:
            _raise(f"Retired item {item.id} cannot replace itself")

        replacement = items.get(replacement_id)
        if replacement is None:
            _raise(
                f"Item {item.id} references missing replacement: "
                f"{replacement_id}"
            )
        if replacement.status != "active":
            _raise(
                f"Replacement for {item.id} must be active: {replacement_id}"
            )
        if replacement.instrument != item.instrument:
            _raise(
                f"Cross-instrument replacement is not allowed: {item.id} -> "
                f"{replacement_id}"
            )


def validate_graph(source: SourceRegistry) -> None:
    """Validate generic rules for a single-parent, multi-root forest.

    Each instrument may have multiple roots.  All active items must form a
    reachable tree/forest within their own instrument; retired records remain
    available to diff tooling but are omitted from the runtime registry.
    """

    if not isinstance(source, SourceRegistry):
        raise TypeError("source must be a SourceRegistry")

    instruments, items = _index_source(source)
    _validate_lookup_namespace(items)
    _validate_parent_relations(items)
    _validate_cycles(items)
    _validate_active_orders_and_reachability(instruments, items)
    _validate_replacements(items)


def _definition_status(item: SourceItem) -> str:
    if item.definition is None:
        return "not_provided"
    if item.node_type == "overall":
        return "summary"
    return "explicit"


def compile_registry(
    source: SourceRegistry,
    *,
    source_filename: str,
    source_sha256: str,
    source_format: str = "markdown-v2",
) -> dict[str, Any]:
    """Compile source V2 into the existing runtime registry schema 1.0."""

    if not isinstance(source_filename, str) or not source_filename.strip():
        _raise("source_filename must be a non-blank string")
    if not isinstance(source_sha256, str) or (
        _SHA256_PATTERN.fullmatch(source_sha256) is None
    ):
        _raise("source_sha256 must be a 64-character hexadecimal string")
    if source_format != "markdown-v2":
        _raise("source_format must be 'markdown-v2' for source schema V2")

    validate_graph(source)
    instrument_labels, all_items = _index_source(source)
    instrument_sources = {
        instrument.id: instrument for instrument in source.instruments
    }
    active = {
        item_id: item
        for item_id, item in all_items.items()
        if item.status == "active"
    }

    children: dict[str, list[SourceItem]] = defaultdict(list)
    roots: dict[str, list[SourceItem]] = defaultdict(list)
    for item in active.values():
        if item.parent_id is None:
            roots[item.instrument].append(item)
        else:
            children[item.parent_id].append(item)

    def sort_items(values: list[SourceItem]) -> list[SourceItem]:
        return sorted(values, key=lambda value: (value.order, value.id))

    for child_items in children.values():
        child_items.sort(key=lambda value: (value.order, value.id))
    for root_items in roots.values():
        root_items.sort(key=lambda value: (value.order, value.id))

    paths: dict[str, list[str]] = {}
    ordered_source_items: list[SourceItem] = []

    def walk(item: SourceItem, ancestor_names: list[str]) -> None:
        path = [*ancestor_names, item.name]
        paths[item.id] = path
        ordered_source_items.append(item)
        for child in children[item.id]:
            walk(child, path)

    for instrument in source.instruments:
        for root in roots[instrument.id]:
            walk(root, list(instrument.root_path_prefix))

    runtime_items: list[dict[str, Any]] = []
    for item in ordered_source_items:
        parent = active.get(item.parent_id) if item.parent_id is not None else None
        instrument = instrument_sources[item.instrument]
        parent_name = (
            parent.name
            if parent is not None
            else (
                instrument.root_path_prefix[-1]
                if instrument.root_path_prefix
                else None
            )
        )
        child_items = children[item.id]
        runtime_items.append(
            {
                "id": item.id,
                "name": item.name,
                "aliases": list(item.aliases),
                "instrument": item.instrument,
                "instrument_label": instrument_labels[item.instrument],
                "level": item.node_type,
                "path": paths[item.id],
                "parent_name": parent_name,
                "parent_id": item.parent_id,
                "children": [child.name for child in child_items],
                "children_ids": [child.id for child in child_items],
                "definition": item.definition,
                "definition_status": _definition_status(item),
                "analysis_included": item.analysis_included,
                "notes": list(item.notes),
                "source_section": instrument_labels[item.instrument],
                # Runtime schema 1.0 permits additional fields.  Retaining
                # these source semantics enables lossless future diff/export.
                "order": item.order,
                "status": item.status,
                "replacement_id": item.replacement_id,
            }
        )

    instrument_counts = Counter(item.instrument for item in ordered_source_items)
    node_type_counts = Counter(item.node_type for item in ordered_source_items)
    counts: dict[str, int] = {
        "total": len(runtime_items),
        "active": len(runtime_items),
        "retired": len(source.items) - len(runtime_items),
    }
    counts.update(
        {
            f"instrument:{instrument_id}": instrument_counts[instrument_id]
            for instrument_id in sorted(instrument_counts)
        }
    )
    counts.update(
        {
            f"node_type:{node_type}": node_type_counts[node_type]
            for node_type in sorted(node_type_counts)
        }
    )

    document = {
        "schema_version": source.registry_schema_version,
        "source": {
            "file": source_filename,
            "sha256": source_sha256.lower(),
            "encoding": "utf-8",
            "format": source_format,
            "source_schema_version": source.source_schema_version,
        },
        "rules": {
            "hierarchy": (
                "Single-parent trees with one or more roots per instrument; "
                "path and children are derived from parent_id and an optional "
                "instrument root_path_prefix."
            ),
            "active_items": (
                "Only source items with status active are included at runtime."
            ),
            "stable_ids": (
                "Item ids are permanent identifiers and are never derived "
                "from names by this compiler."
            ),
        },
        "validation": {
            "status": "passed",
            "counts": counts,
        },
        "items": runtime_items,
    }
    try:
        return validate_registry_document(document)
    except RegistryValidationError as exc:
        raise RegistryCompileError(
            "Compiled registry failed runtime schema validation"
        ) from exc


__all__ = [
    "RegistryCompileError",
    "compile_registry",
    "validate_graph",
]
