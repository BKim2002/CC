"""PostgreSQL-backed competency registry loading and validation.

This module deliberately performs no file or database I/O at import time.  The
application calls :func:`load_active_registry` during runtime initialization
and keeps the returned snapshot in memory.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from types import MappingProxyType
from typing import Any, Mapping

from psycopg import connect
from psycopg.rows import dict_row


ACTIVE_REGISTRY_SQL = """
SELECT
    versions.id,
    versions.source_filename,
    versions.source_sha256,
    versions.schema_version,
    versions.registry_json,
    versions.item_count
FROM competency_data.registry_current AS current
JOIN competency_data.registry_versions AS versions
    ON versions.id = current.version_id
WHERE current.singleton = %s;
"""


REQUIRED_ITEM_KEYS = frozenset(
    {
        "id",
        "name",
        "aliases",
        "instrument",
        "instrument_label",
        "level",
        "path",
        "parent_name",
        "parent_id",
        "children",
        "children_ids",
        "definition",
        "definition_status",
        "analysis_included",
        "notes",
        "source_section",
    }
)

REQUIRED_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "rules",
        "validation",
        "items",
    }
)

SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class RegistryValidationError(ValueError):
    """Raised when a stored registry cannot safely be used at runtime."""


def _derive_id_lookup(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index items by stable ID for a snapshot built without an explicit map.

    ``build_registry_snapshot`` already validates every item, so this path only
    runs for directly constructed snapshots.  A missing ID is reported as a
    registry error instead of surfacing as a bare ``KeyError`` from a
    comprehension.
    """

    derived: dict[str, Mapping[str, Any]] = {}
    for item in document.get("items", ()):
        item_id = item.get("id") if isinstance(item, Mapping) else None
        if not item_id:
            raise RegistryValidationError("레지스트리 항목에 stable ID가 없습니다.")
        derived[str(item_id)] = item
    return derived


@dataclass(frozen=True)
class RegistrySnapshot:
    """One validated, active competency-registry version held in memory."""

    version_id: int
    source_filename: str
    source_sha256: str
    schema_version: str
    document: Mapping[str, Any]
    lookup: Mapping[str, Mapping[str, Any]]
    canonical_lookup: Mapping[str, Mapping[str, Any]]
    canonical_names: tuple[str, ...]
    name_catalog: str
    semantic_catalog: str
    id_lookup: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Recursively freeze every registry view while preserving aliases."""

        # A single memo is shared across every view.  Consequently an item
        # referenced by document, id_lookup, canonical_lookup, and an alias
        # remains the exact same frozen object rather than independent copies.
        id_lookup = self.id_lookup or _derive_id_lookup(self.document)
        memo: dict[int, object] = {}
        object.__setattr__(
            self,
            "document",
            _deep_freeze(self.document, memo),
        )
        object.__setattr__(
            self,
            "lookup",
            _deep_freeze(self.lookup, memo),
        )
        object.__setattr__(
            self,
            "id_lookup",
            _deep_freeze(id_lookup, memo),
        )
        object.__setattr__(
            self,
            "canonical_lookup",
            _deep_freeze(self.canonical_lookup, memo),
        )
        object.__setattr__(self, "canonical_names", tuple(self.canonical_names))


def _deep_freeze(value: Any, memo: dict[int, object]) -> Any:
    """Return a recursively read-only representation of JSON-like data."""

    if isinstance(value, Mapping):
        cached = memo.get(id(value))
        if cached is not None:
            return cached

        frozen = MappingProxyType(
            {
                key: _deep_freeze(member, memo)
                for key, member in value.items()
            }
        )
        memo[id(value)] = frozen
        return frozen

    if isinstance(value, (list, tuple)):
        cached = memo.get(id(value))
        if cached is not None:
            return cached

        frozen = tuple(_deep_freeze(member, memo) for member in value)
        memo[id(value)] = frozen
        return frozen

    return value


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{label}는 JSON object여야 합니다.")
    return value


def _require_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{label}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _require_nullable_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, label)


def _require_boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RegistryValidationError(f"{label}는 boolean이어야 합니다.")
    return value


def _require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise RegistryValidationError(f"{label}는 list여야 합니다.")
    for index, member in enumerate(value):
        _require_non_empty_string(member, f"{label}[{index}]")
    return value


def validate_registry_document(document: object) -> dict[str, Any]:
    """Validate and defensively copy a registry JSON document.

    The production-specific expectation of 62 items belongs to the upload
    validator.  Runtime validation intentionally accepts a smaller valid
    document so unit tests and future schema-compatible registries can use the
    same loader.
    """

    source_document = _require_mapping(document, "레지스트리 최상위 값")
    validated = deepcopy(dict(source_document))

    missing_document_keys = sorted(
        REQUIRED_DOCUMENT_KEYS.difference(validated)
    )
    if missing_document_keys:
        raise RegistryValidationError(
            "레지스트리 최상위 값에 필수 키가 없습니다: "
            + ", ".join(missing_document_keys)
        )

    if validated.get("schema_version") != "1.0":
        raise RegistryValidationError(
            "지원하지 않는 레지스트리 schema_version입니다. 기대값: 1.0"
        )

    source = _require_mapping(validated.get("source"), "source")
    source_filename = _require_non_empty_string(
        source.get("file"),
        "source.file",
    )
    source_sha256 = _require_non_empty_string(
        source.get("sha256"),
        "source.sha256",
    )
    _require_non_empty_string(source.get("encoding"), "source.encoding")
    if SHA256_PATTERN.fullmatch(source_sha256) is None:
        raise RegistryValidationError(
            "source.sha256은 64자리 16진수 문자열이어야 합니다."
        )

    rules = _require_mapping(validated.get("rules"), "rules")
    for rule_name, rule_description in rules.items():
        _require_non_empty_string(rule_name, "rules key")
        _require_non_empty_string(
            rule_description,
            f"rules.{rule_name}",
        )

    validation = _require_mapping(validated.get("validation"), "validation")
    if validation.get("status") != "passed":
        raise RegistryValidationError(
            "레지스트리 validation.status가 passed가 아닙니다."
        )
    counts = _require_mapping(validation.get("counts"), "validation.counts")
    if "total" not in counts:
        raise RegistryValidationError(
            "validation.counts에 total이 없습니다."
        )
    for count_name, count_value in counts.items():
        _require_non_empty_string(count_name, "validation.counts key")
        if (
            isinstance(count_value, bool)
            or not isinstance(count_value, int)
            or count_value < 0
        ):
            raise RegistryValidationError(
                f"validation.counts.{count_name}는 0 이상의 정수여야 합니다."
            )

    items = validated.get("items")
    if not isinstance(items, list):
        raise RegistryValidationError("items는 JSON array여야 합니다.")

    ids: set[str] = set()
    names: set[str] = set()

    for index, raw_item in enumerate(items):
        item = _require_mapping(raw_item, f"items[{index}]")
        missing_keys = sorted(REQUIRED_ITEM_KEYS.difference(item))
        if missing_keys:
            raise RegistryValidationError(
                f"items[{index}]에 필수 키가 없습니다: {', '.join(missing_keys)}"
            )

        item_id = _require_non_empty_string(item["id"], f"items[{index}].id")
        name = _require_non_empty_string(item["name"], f"items[{index}].name")

        if item_id in ids:
            raise RegistryValidationError(f"중복된 항목 ID가 있습니다: {item_id}")
        if name in names:
            raise RegistryValidationError(f"중복된 정식 이름이 있습니다: {name}")
        ids.add(item_id)
        names.add(name)

        _require_string_list(item["aliases"], f"items[{index}].aliases")

        for string_key in (
            "instrument",
            "instrument_label",
            "level",
            "definition_status",
            "source_section",
        ):
            _require_non_empty_string(
                item[string_key],
                f"items[{index}].{string_key}",
            )

        for nullable_string_key in (
            "parent_name",
            "parent_id",
            "definition",
        ):
            _require_nullable_string(
                item[nullable_string_key],
                f"items[{index}].{nullable_string_key}",
            )

        _require_boolean(
            item["analysis_included"],
            f"items[{index}].analysis_included",
        )

        for list_key in ("path", "children", "children_ids", "notes"):
            _require_string_list(item[list_key], f"items[{index}].{list_key}")

    if counts["total"] != len(items):
        raise RegistryValidationError(
            "validation.counts.total과 items의 실제 항목 수가 일치하지 않습니다."
        )

    # Treat canonical names and aliases as one shared lookup namespace.  This
    # catches an alias that shadows another item's canonical name as well as an
    # alias repeated across two items.
    label_owners = {
        item["name"]: item["id"]
        for item in items
    }

    for index, item in enumerate(items):
        aliases_seen: set[str] = set()
        for alias in item["aliases"]:
            if alias in aliases_seen:
                raise RegistryValidationError(
                    f"한 항목 안에 중복된 별칭이 있습니다: {alias}"
                )
            aliases_seen.add(alias)

            owner_id = label_owners.get(alias)
            if owner_id is not None:
                raise RegistryValidationError(
                    f"별칭이 여러 항목 또는 정식 이름과 충돌합니다: {alias}"
                )
            label_owners[alias] = item["id"]

    return validated


def _build_name_catalog(
    canonical_names: tuple[str, ...],
    canonical_lookup: Mapping[str, Mapping[str, Any]],
) -> str:
    lines: list[str] = []
    for name in canonical_names:
        aliases = canonical_lookup[name]["aliases"]
        alias_description = (
            f" (등록 별칭: {', '.join(aliases)})"
            if aliases
            else ""
        )
        lines.append(f"- {name}{alias_description}")
    return "\n".join(lines)


def _build_semantic_catalog(
    canonical_names: tuple[str, ...],
    canonical_lookup: Mapping[str, Mapping[str, Any]],
) -> str:
    lines: list[str] = []
    for name in canonical_names:
        item = canonical_lookup[name]
        definition = item.get("definition") or "독립적인 정의가 제공되어 있지 않음"
        path = " > ".join(item["path"]) or "위계 정보 없음"
        lines.append(
            f"- 이름: {name}\n"
            f"  정의: {definition}\n"
            f"  위계: {path}"
        )
    return "\n".join(lines)


def build_registry_snapshot(row: Mapping[str, Any]) -> RegistrySnapshot:
    """Build a validated in-memory snapshot from one database result row."""

    if not isinstance(row, Mapping):
        raise RegistryValidationError("활성 레지스트리 DB 행이 올바른 mapping이 아닙니다.")

    required_row_keys = {
        "id",
        "source_filename",
        "source_sha256",
        "schema_version",
        "registry_json",
        "item_count",
    }
    missing_row_keys = sorted(required_row_keys.difference(row))
    if missing_row_keys:
        raise RegistryValidationError(
            "활성 레지스트리 DB 행에 필수 열이 없습니다: "
            + ", ".join(missing_row_keys)
        )

    version_id = row["id"]
    if isinstance(version_id, bool) or not isinstance(version_id, int):
        raise RegistryValidationError("활성 레지스트리 version id가 정수가 아닙니다.")

    source_filename = _require_non_empty_string(
        row["source_filename"],
        "DB source_filename",
    )
    source_sha256 = _require_non_empty_string(
        row["source_sha256"],
        "DB source_sha256",
    )
    schema_version = _require_non_empty_string(
        row["schema_version"],
        "DB schema_version",
    )

    item_count = row["item_count"]
    if isinstance(item_count, bool) or not isinstance(item_count, int):
        raise RegistryValidationError("DB item_count가 정수가 아닙니다.")
    if item_count < 1:
        raise RegistryValidationError("DB item_count는 1 이상이어야 합니다.")

    document = validate_registry_document(row["registry_json"])
    if schema_version != document["schema_version"]:
        raise RegistryValidationError(
            "DB schema_version과 registry_json schema_version이 일치하지 않습니다."
        )
    if source_sha256 != document["source"]["sha256"]:
        raise RegistryValidationError(
            "DB source_sha256과 registry_json source.sha256이 일치하지 않습니다."
        )
    if source_filename != document["source"]["file"]:
        raise RegistryValidationError(
            "DB source_filename과 registry_json source.file이 일치하지 않습니다."
        )

    items = document["items"]
    if len(items) != item_count:
        raise RegistryValidationError(
            "DB item_count와 registry_json의 실제 항목 수가 일치하지 않습니다."
        )

    id_lookup = {item["id"]: item for item in items}
    canonical_lookup = {item["name"]: item for item in items}
    lookup = dict(canonical_lookup)
    for item in items:
        for alias in item["aliases"]:
            lookup[alias] = item

    canonical_names = tuple(sorted(canonical_lookup))
    return RegistrySnapshot(
        version_id=version_id,
        source_filename=source_filename,
        source_sha256=source_sha256,
        schema_version=schema_version,
        document=document,
        lookup=lookup,
        canonical_lookup=canonical_lookup,
        canonical_names=canonical_names,
        name_catalog=_build_name_catalog(canonical_names, canonical_lookup),
        semantic_catalog=_build_semantic_catalog(canonical_names, canonical_lookup),
        id_lookup=id_lookup,
    )


def load_active_registry(database_url: str) -> RegistrySnapshot:
    """Load, validate, and close the connection for the active DB version."""

    if not isinstance(database_url, str) or not database_url.strip():
        raise RuntimeError("DATABASE_URL이 설정되지 않았습니다.")

    with connect(database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(ACTIVE_REGISTRY_SQL, (1,))
            row = cursor.fetchone()

    if row is None:
        raise RuntimeError(
            "활성 역량 레지스트리가 없습니다. "
            "setup 및 upload 스크립트를 먼저 실행하세요."
        )

    return build_registry_snapshot(row)
