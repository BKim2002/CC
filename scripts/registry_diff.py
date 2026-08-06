"""Safe, deterministic diffs for compiled competency registries.

The runtime registry contains active items only.  A structured V2 source can
also contain retired items and source-only fields such as ``status``,
``replacement_id`` and ``order``.  ``diff_registries`` accepts both views and
compares their union by stable item ID.

This module deliberately has no database or filesystem dependencies.  It also
never retains definition text in public change objects, so its summaries are
safe to print in command-line upload workflows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any


CHANGE_CATEGORIES = (
    "added",
    "removed",
    "renamed",
    "moved",
    "instrument_changed",
    "root_path_prefix_changed",
    "definition_changed",
    "aliases_changed",
    "analysis_flag_changed",
    "node_type_changed",
    "status_changed",
    "replacement_changed",
    "order_changed",
)

ACTIVE_ITEM_REMOVED = "active_item_removed"
ACTIVE_ITEM_RETIRED = "active_item_retired"
ITEM_MOVED = "item_moved"
NODE_TYPE_CHANGED = "node_type_changed"
ANALYSIS_FLAG_CHANGED = "analysis_flag_changed"
ROOT_CHANGED = "root_changed"
RETIRED_WITHOUT_REPLACEMENT = "retired_without_replacement"
INSTRUMENT_CHANGED = "instrument_changed"
LOOKUP_LABEL_REMOVED = "lookup_label_removed"
RENAMED_WITHOUT_OLD_NAME_ALIAS = "renamed_without_old_name_alias"
ROOT_PATH_PREFIX_CHANGED = "root_path_prefix_changed"

_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_MISSING = object()


class RegistryDiffError(ValueError):
    """Raised when a registry cannot be compared safely by stable ID."""


@dataclass(frozen=True)
class RegistryChange:
    """One safe, category-specific change.

    ``details`` contains only identifiers, status values, booleans, positions,
    and counts selected by this module.  In particular, definition contents
    and alias contents are never copied into a change.
    """

    category: str
    item_id: str
    before_name: str | None = None
    after_name: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    breaking_reasons: tuple[str, ...] = ()

    @property
    def is_breaking(self) -> bool:
        return bool(self.breaking_reasons)

    def to_summary_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "item_id": self.item_id,
            "before_name": self.before_name,
            "after_name": self.after_name,
        }
        if self.details:
            result["details"] = dict(self.details)
        if self.breaking_reasons:
            result["breaking_reasons"] = list(self.breaking_reasons)
        return result


@dataclass(frozen=True)
class RegistryDiff:
    """ID-based registry diff with a conservative default breaking policy."""

    added: tuple[RegistryChange, ...] = ()
    removed: tuple[RegistryChange, ...] = ()
    renamed: tuple[RegistryChange, ...] = ()
    moved: tuple[RegistryChange, ...] = ()
    instrument_changed: tuple[RegistryChange, ...] = ()
    root_path_prefix_changed: tuple[RegistryChange, ...] = ()
    definition_changed: tuple[RegistryChange, ...] = ()
    aliases_changed: tuple[RegistryChange, ...] = ()
    analysis_flag_changed: tuple[RegistryChange, ...] = ()
    node_type_changed: tuple[RegistryChange, ...] = ()
    status_changed: tuple[RegistryChange, ...] = ()
    replacement_changed: tuple[RegistryChange, ...] = ()
    order_changed: tuple[RegistryChange, ...] = ()
    max_depth_before: int = 0
    max_depth_after: int = 0
    current_version_id: int | None = None
    candidate_source_sha256: str | None = None

    @property
    def all_changes(self) -> tuple[RegistryChange, ...]:
        return tuple(
            change
            for category in CHANGE_CATEGORIES
            for change in getattr(self, category)
        )

    @property
    def breaking_changes(self) -> tuple[RegistryChange, ...]:
        return tuple(change for change in self.all_changes if change.is_breaking)

    @property
    def has_changes(self) -> bool:
        return bool(self.all_changes)

    @property
    def is_empty(self) -> bool:
        return not self.has_changes

    @property
    def is_breaking(self) -> bool:
        return bool(self.breaking_changes)

    @property
    def has_breaking_changes(self) -> bool:
        """Compatibility-friendly spelling for activation guards."""

        return self.is_breaking

    @property
    def counts(self) -> dict[str, int]:
        return {category: len(getattr(self, category)) for category in CHANGE_CATEGORIES}

    @property
    def breaking_reason_counts(self) -> dict[str, int]:
        reasons = Counter(
            reason
            for change in self.breaking_changes
            for reason in change.breaking_reasons
        )
        return dict(sorted(reasons.items()))

    @property
    def max_depth(self) -> tuple[int, int]:
        return self.max_depth_before, self.max_depth_after

    def to_summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary containing no definition text."""

        return {
            "current_version": self.current_version_id,
            "candidate_source_sha256": self.candidate_source_sha256,
            "counts": self.counts,
            "max_depth": {
                "before": self.max_depth_before,
                "after": self.max_depth_after,
            },
            "breaking": {
                "is_breaking": self.is_breaking,
                "change_count": len(self.breaking_changes),
                "reason_counts": self.breaking_reason_counts,
            },
            "changes": {
                category: [
                    change.to_summary_dict() for change in getattr(self, category)
                ]
                for category in CHANGE_CATEGORIES
            },
        }

    def summary_dict(self) -> dict[str, object]:
        return self.to_summary_dict()

    def to_summary_text(self) -> str:
        """Return a concise, safe text report suitable for stdout."""

        version = "none" if self.current_version_id is None else str(self.current_version_id)
        source_hash = self.candidate_source_sha256 or "unavailable"
        lines = [
            f"current_version: {version}",
            f"candidate_source_sha256: {source_hash}",
        ]
        lines.extend(f"{category}: {self.counts[category]}" for category in CHANGE_CATEGORIES)
        lines.extend(
            (
                f"max_depth: {self.max_depth_before} -> {self.max_depth_after}",
                f"breaking: {'yes' if self.is_breaking else 'no'}",
            )
        )
        for reason, count in self.breaking_reason_counts.items():
            lines.append(f"breaking_reason.{reason}: {count}")
        return "\n".join(lines)

    def summary_text(self) -> str:
        return self.to_summary_text()


@dataclass
class _NormalizedItem:
    item_id: str
    name: str
    instrument: str | None
    parent_id: str | None
    path: tuple[str, ...]
    node_type: str | None
    definition: object
    aliases: tuple[str, ...]
    analysis_included: bool | None
    status: str
    replacement_id: str | None
    raw_order: int | None
    input_index: int
    runtime_sibling_position: int | None = None
    order_rank: int | None = None


def _runtime_items(document: Mapping[str, object], label: str) -> list[Mapping[str, object]]:
    items = document.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise RegistryDiffError(f"{label}.items must be a sequence of mappings")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise RegistryDiffError(f"{label}.items[{index}] must be a mapping")
        result.append(item)
    return result


def _source_items(
    items: Sequence[Mapping[str, object]] | None,
    label: str,
) -> list[Mapping[str, object]]:
    if items is None:
        return []
    if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
        raise RegistryDiffError(f"{label} must be a sequence of mappings")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise RegistryDiffError(f"{label}[{index}] must be a mapping")
        result.append(item)
    return result


def _stable_id(item: Mapping[str, object], label: str) -> str:
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise RegistryDiffError(f"{label}.id must be a non-empty string")
    return item_id


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RegistryDiffError(f"{label} must be a string or null")
    return value


def _aliases(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RegistryDiffError(f"{label} must be a sequence of strings")
    aliases: list[str] = []
    for index, alias in enumerate(value):
        if not isinstance(alias, str):
            raise RegistryDiffError(f"{label}[{index}] must be a string")
        aliases.append(alias)
    # Alias order has no lookup semantics; duplicates are rejected upstream.
    return tuple(sorted(set(aliases)))


def _path(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RegistryDiffError(f"{label} must be a sequence of strings")
    members: list[str] = []
    for index, member in enumerate(value):
        if not isinstance(member, str):
            raise RegistryDiffError(f"{label}[{index}] must be a string")
        members.append(member)
    return tuple(members)


def _runtime_child_positions(
    runtime_items: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    positions: dict[str, int] = {}
    for item_index, item in enumerate(runtime_items):
        children = item.get("children_ids")
        if not isinstance(children, Sequence) or isinstance(
            children, (str, bytes, bytearray)
        ):
            continue
        for position, child_id in enumerate(children):
            if isinstance(child_id, str):
                positions.setdefault(child_id, position)
    return positions


def _normalize_registry(
    runtime: Mapping[str, object],
    source: Sequence[Mapping[str, object]] | None,
    label: str,
) -> dict[str, _NormalizedItem]:
    if not isinstance(runtime, Mapping):
        raise RegistryDiffError(f"{label} runtime must be a mapping")
    runtime_items = _runtime_items(runtime, label)
    source_items = _source_items(source, f"{label}_source_items")
    child_positions = _runtime_child_positions(runtime_items)

    combined: dict[str, dict[str, object]] = {}
    index_by_id: dict[str, int] = {}
    for index, item in enumerate(runtime_items):
        item_id = _stable_id(item, f"{label}.items[{index}]")
        if item_id in combined:
            raise RegistryDiffError(f"duplicate item id in {label} runtime: {item_id}")
        values = dict(item)
        values.setdefault("status", "active")
        values.setdefault("replacement_id", None)
        combined[item_id] = values
        index_by_id[item_id] = index

    seen_source: set[str] = set()
    source_offset = len(runtime_items)
    for source_index, item in enumerate(source_items):
        item_id = _stable_id(item, f"{label}_source_items[{source_index}]")
        if item_id in seen_source:
            raise RegistryDiffError(f"duplicate item id in {label} source: {item_id}")
        seen_source.add(item_id)
        if item_id in combined:
            # Source-only fields are authoritative; runtime-only derived fields
            # remain available when the source does not repeat them.
            combined[item_id].update(item)
        else:
            combined[item_id] = dict(item)
            index_by_id[item_id] = source_offset + source_index

    normalized: dict[str, _NormalizedItem] = {}
    for item_id, item in combined.items():
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RegistryDiffError(f"{label} item {item_id!r} has no valid name")
        parent_id = _optional_string(item.get("parent_id"), f"{label}.{item_id}.parent_id")
        instrument = _optional_string(
            item.get("instrument"), f"{label}.{item_id}.instrument"
        )
        node_type_value = item.get("node_type", item.get("level"))
        node_type = _optional_string(
            node_type_value, f"{label}.{item_id}.node_type"
        )
        status = item.get("status", "active")
        if not isinstance(status, str) or not status:
            raise RegistryDiffError(f"{label}.{item_id}.status must be a non-empty string")
        replacement_id = _optional_string(
            item.get("replacement_id"), f"{label}.{item_id}.replacement_id"
        )
        raw_order_value = item.get("order")
        if raw_order_value is not None and (
            not isinstance(raw_order_value, int) or isinstance(raw_order_value, bool)
        ):
            raise RegistryDiffError(f"{label}.{item_id}.order must be an integer or null")
        analysis = item.get("analysis_included")
        if analysis is not None and not isinstance(analysis, bool):
            raise RegistryDiffError(
                f"{label}.{item_id}.analysis_included must be a boolean or null"
            )
        normalized[item_id] = _NormalizedItem(
            item_id=item_id,
            name=name,
            instrument=instrument,
            parent_id=parent_id,
            path=_path(item.get("path"), f"{label}.{item_id}.path"),
            node_type=node_type,
            definition=item.get("definition", _MISSING),
            aliases=_aliases(item.get("aliases", ()), f"{label}.{item_id}.aliases"),
            analysis_included=analysis,
            status=status,
            replacement_id=replacement_id,
            raw_order=raw_order_value,
            input_index=index_by_id[item_id],
            runtime_sibling_position=child_positions.get(item_id),
        )

    _assign_order_ranks(normalized)
    return normalized


def _assign_order_ranks(items: Mapping[str, _NormalizedItem]) -> None:
    groups: dict[tuple[str | None, str | None], list[_NormalizedItem]] = defaultdict(list)
    for item in items.values():
        if item.status == "active":
            groups[(item.instrument, item.parent_id)].append(item)

    for siblings in groups.values():
        def sort_key(item: _NormalizedItem) -> tuple[int, int]:
            if item.raw_order is not None:
                return item.raw_order, item.input_index
            if item.runtime_sibling_position is not None:
                return item.runtime_sibling_position, item.input_index
            return item.input_index, item.input_index

        siblings.sort(key=sort_key)
        for rank, item in enumerate(siblings):
            item.order_rank = rank


def _max_depth(items: Mapping[str, _NormalizedItem]) -> int:
    active = {item_id: item for item_id, item in items.items() if item.status == "active"}
    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(item_id: str) -> int:
        if item_id in depths:
            return depths[item_id]
        if item_id in visiting:
            raise RegistryDiffError(f"cycle encountered while calculating depth at {item_id}")
        visiting.add(item_id)
        parent_id = active[item_id].parent_id
        value = 1 if parent_id not in active else depth(parent_id) + 1
        visiting.remove(item_id)
        depths[item_id] = value
        return value

    return max((depth(item_id) for item_id in active), default=0)


def _order_changes(
    current: Mapping[str, _NormalizedItem],
    candidate: Mapping[str, _NormalizedItem],
) -> dict[str, tuple[int, int]]:
    """Compare relative order among surviving siblings.

    Added/removed siblings can shift absolute positions without reordering any
    existing item.  Filtering each group to IDs present on both sides avoids
    reporting those harmless shifts as ``order_changed``.
    """

    eligible: dict[tuple[str | None, str | None], list[str]] = defaultdict(list)
    for item_id in current.keys() & candidate.keys():
        before = current[item_id]
        after = candidate[item_id]
        if (
            before.status == "active"
            and after.status == "active"
            and before.instrument == after.instrument
            and before.parent_id == after.parent_id
        ):
            eligible[(before.instrument, before.parent_id)].append(item_id)

    changes: dict[str, tuple[int, int]] = {}
    for item_ids in eligible.values():
        before_order = sorted(item_ids, key=lambda item_id: current[item_id].order_rank or 0)
        after_order = sorted(item_ids, key=lambda item_id: candidate[item_id].order_rank or 0)
        before_positions = {item_id: index for index, item_id in enumerate(before_order)}
        after_positions = {item_id: index for index, item_id in enumerate(after_order)}
        for item_id in item_ids:
            before_position = before_positions[item_id]
            after_position = after_positions[item_id]
            if before_position != after_position:
                changes[item_id] = (before_position, after_position)
    return changes


def _candidate_hash(candidate_runtime: Mapping[str, object]) -> str | None:
    source = candidate_runtime.get("source")
    if not isinstance(source, Mapping):
        return None
    value = source.get("sha256")
    # Never echo an arbitrary source metadata string: only a syntactically valid
    # digest is safe and useful in a CLI report.
    if isinstance(value, str) and _HEX_SHA256.fullmatch(value):
        return value.lower()
    return None


def _change(
    category: str,
    item_id: str,
    before: _NormalizedItem | None,
    after: _NormalizedItem | None,
    *,
    details: Mapping[str, object] | None = None,
    breaking_reasons: Sequence[str] = (),
) -> RegistryChange:
    return RegistryChange(
        category=category,
        item_id=item_id,
        before_name=None if before is None else before.name,
        after_name=None if after is None else after.name,
        details={} if details is None else dict(details),
        breaking_reasons=tuple(dict.fromkeys(breaking_reasons)),
    )


def diff_registries(
    current_runtime: Mapping[str, object],
    candidate_runtime: Mapping[str, object],
    *,
    current_source_items: Sequence[Mapping[str, object]] | None = None,
    candidate_source_items: Sequence[Mapping[str, object]] | None = None,
    current_version_id: int | None = None,
) -> RegistryDiff:
    """Compare two registries by stable ID.

    Runtime documents are normalized as active with no replacement.  When
    source items are supplied, they augment runtime items and provide source
    status, replacement, node type, and order.  Definitions are compared but
    never copied into the returned diff.
    """

    current = _normalize_registry(
        current_runtime, current_source_items, "current_runtime"
    )
    candidate = _normalize_registry(
        candidate_runtime, candidate_source_items, "candidate_runtime"
    )
    categories: dict[str, list[RegistryChange]] = {
        category: [] for category in CHANGE_CATEGORIES
    }

    current_ids = set(current)
    candidate_ids = set(candidate)

    for item_id in sorted(candidate_ids - current_ids):
        after = candidate[item_id]
        categories["added"].append(
            _change("added", item_id, None, after, details={"status": after.status})
        )

    for item_id in sorted(current_ids - candidate_ids):
        before = current[item_id]
        reasons = (ACTIVE_ITEM_REMOVED,) if before.status == "active" else ()
        categories["removed"].append(
            _change(
                "removed",
                item_id,
                before,
                None,
                details={"status": before.status},
                breaking_reasons=reasons,
            )
        )

    order_changes = _order_changes(current, candidate)
    for item_id in sorted(current_ids & candidate_ids):
        before = current[item_id]
        after = candidate[item_id]

        if before.name != after.name:
            old_name_preserved = before.name in after.aliases
            categories["renamed"].append(
                _change(
                    "renamed",
                    item_id,
                    before,
                    after,
                    details={
                        "old_name_preserved_as_alias": old_name_preserved,
                    },
                    breaking_reasons=(
                        ()
                        if old_name_preserved
                        else (RENAMED_WITHOUT_OLD_NAME_ALIAS,)
                    ),
                )
            )

        if before.parent_id != after.parent_id:
            reasons = [ITEM_MOVED]
            if (before.parent_id is None) != (after.parent_id is None):
                reasons.append(ROOT_CHANGED)
            categories["moved"].append(
                _change(
                    "moved",
                    item_id,
                    before,
                    after,
                    details={
                        "before_parent_id": before.parent_id,
                        "after_parent_id": after.parent_id,
                    },
                    breaking_reasons=reasons,
                )
            )

        if before.instrument != after.instrument:
            categories["instrument_changed"].append(
                _change(
                    "instrument_changed",
                    item_id,
                    before,
                    after,
                    details={
                        "before_instrument": before.instrument,
                        "after_instrument": after.instrument,
                    },
                    breaking_reasons=(INSTRUMENT_CHANGED,),
                )
            )

        before_root_prefix = before.path[:-1] if before.path else ()
        after_root_prefix = after.path[:-1] if after.path else ()
        if (
            before.status == "active"
            and after.status == "active"
            and before.parent_id is None
            and after.parent_id is None
            and before_root_prefix != after_root_prefix
        ):
            categories["root_path_prefix_changed"].append(
                _change(
                    "root_path_prefix_changed",
                    item_id,
                    before,
                    after,
                    details={
                        "before_prefix_depth": len(before_root_prefix),
                        "after_prefix_depth": len(after_root_prefix),
                    },
                    breaking_reasons=(ROOT_PATH_PREFIX_CHANGED,),
                )
            )

        if before.definition != after.definition:
            categories["definition_changed"].append(
                _change("definition_changed", item_id, before, after)
            )

        if before.aliases != after.aliases:
            before_labels = {before.name, *before.aliases}
            after_labels = {after.name, *after.aliases}
            removed_label_count = len(before_labels.difference(after_labels))
            categories["aliases_changed"].append(
                _change(
                    "aliases_changed",
                    item_id,
                    before,
                    after,
                    details={
                        "before_count": len(before.aliases),
                        "after_count": len(after.aliases),
                        "removed_lookup_label_count": removed_label_count,
                    },
                    breaking_reasons=(
                        (LOOKUP_LABEL_REMOVED,)
                        if removed_label_count
                        else ()
                    ),
                )
            )

        if before.analysis_included != after.analysis_included:
            categories["analysis_flag_changed"].append(
                _change(
                    "analysis_flag_changed",
                    item_id,
                    before,
                    after,
                    details={
                        "before": before.analysis_included,
                        "after": after.analysis_included,
                    },
                    breaking_reasons=(ANALYSIS_FLAG_CHANGED,),
                )
            )

        if before.node_type != after.node_type:
            categories["node_type_changed"].append(
                _change(
                    "node_type_changed",
                    item_id,
                    before,
                    after,
                    details={
                        "before_node_type": before.node_type,
                        "after_node_type": after.node_type,
                    },
                    breaking_reasons=(NODE_TYPE_CHANGED,),
                )
            )

        if before.status != after.status:
            reasons: list[str] = []
            if before.status == "active" and after.status == "retired":
                reasons.append(ACTIVE_ITEM_RETIRED)
                if after.replacement_id is None:
                    reasons.append(RETIRED_WITHOUT_REPLACEMENT)
            categories["status_changed"].append(
                _change(
                    "status_changed",
                    item_id,
                    before,
                    after,
                    details={
                        "before_status": before.status,
                        "after_status": after.status,
                    },
                    breaking_reasons=reasons,
                )
            )

        if before.replacement_id != after.replacement_id:
            replacement_reasons = (
                (RETIRED_WITHOUT_REPLACEMENT,)
                if after.status == "retired" and after.replacement_id is None
                else ()
            )
            categories["replacement_changed"].append(
                _change(
                    "replacement_changed",
                    item_id,
                    before,
                    after,
                    details={
                        "before_replacement_id": before.replacement_id,
                        "after_replacement_id": after.replacement_id,
                    },
                    breaking_reasons=replacement_reasons,
                )
            )

        if item_id in order_changes:
            before_position, after_position = order_changes[item_id]
            categories["order_changed"].append(
                _change(
                    "order_changed",
                    item_id,
                    before,
                    after,
                    details={
                        "before_position": before_position,
                        "after_position": after_position,
                    },
                )
            )

    return RegistryDiff(
        **{
            category: tuple(changes)
            for category, changes in categories.items()
        },
        max_depth_before=_max_depth(current),
        max_depth_after=_max_depth(candidate),
        current_version_id=current_version_id,
        candidate_source_sha256=_candidate_hash(candidate_runtime),
    )


__all__ = [
    "ACTIVE_ITEM_REMOVED",
    "ACTIVE_ITEM_RETIRED",
    "ANALYSIS_FLAG_CHANGED",
    "CHANGE_CATEGORIES",
    "ITEM_MOVED",
    "INSTRUMENT_CHANGED",
    "LOOKUP_LABEL_REMOVED",
    "NODE_TYPE_CHANGED",
    "RENAMED_WITHOUT_OLD_NAME_ALIAS",
    "RETIRED_WITHOUT_REPLACEMENT",
    "ROOT_PATH_PREFIX_CHANGED",
    "ROOT_CHANGED",
    "RegistryChange",
    "RegistryDiff",
    "RegistryDiffError",
    "diff_registries",
]
