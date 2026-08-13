"""검증된 역량 레지스트리를 안전하게 비교·업로드·활성화한다."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from psycopg import connect
from psycopg.types.json import Jsonb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not __package__ and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_registry import RegistryError, build_registry
from scripts.registry_diff import RegistryDiff, diff_registries
from scripts.registry_source_v2 import parse_source_v2_text

from chat.registry import validate_registry_document


load_dotenv(PROJECT_ROOT / ".env")

SOURCE_FORMATS = ("markdown-v1", "markdown-v2")
_EXPECTED_CURRENT_UNSET = object()


FIND_VERSION_SQL = """
SELECT
    id,
    source_filename,
    source_markdown,
    source_sha256,
    schema_version,
    registry_json,
    item_count
FROM competency_data.registry_versions
WHERE source_sha256 = %s
"""

INSERT_VERSION_SQL = """
INSERT INTO competency_data.registry_versions (
    source_filename,
    source_markdown,
    source_sha256,
    schema_version,
    registry_json,
    item_count
)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (source_sha256) DO NOTHING
RETURNING id
"""

LOCK_CURRENT_SQL = """
LOCK TABLE competency_data.registry_current IN SHARE ROW EXCLUSIVE MODE
"""

READ_ACTIVE_SQL = """
SELECT
    current_registry.version_id,
    versions.registry_json,
    versions.source_markdown
FROM competency_data.registry_current AS current_registry
JOIN competency_data.registry_versions AS versions
    ON versions.id = current_registry.version_id
WHERE current_registry.singleton = 1
"""

READ_ACTIVE_FOR_UPDATE_SQL = READ_ACTIVE_SQL + "\nFOR UPDATE OF current_registry\n"

ACTIVATE_VERSION_SQL = """
INSERT INTO competency_data.registry_current (
    singleton,
    version_id,
    updated_at
)
VALUES (1, %s, NOW())
ON CONFLICT (singleton)
DO UPDATE SET
    version_id = EXCLUDED.version_id,
    updated_at = NOW()
"""

SET_TRANSACTION_READ_ONLY_SQL = "SET TRANSACTION READ ONLY"


@dataclass(frozen=True)
class PreparedRegistry:
    source_filename: str
    source_markdown: str
    source_sha256: str
    schema_version: str
    document: dict[str, Any]
    item_count: int
    source_format: str = "markdown-v1"
    source_items: tuple[Mapping[str, object], ...] | None = None


@dataclass(frozen=True)
class ActiveRegistry:
    version_id: int
    document: dict[str, Any]
    source_markdown: str
    source_format: str
    source_items: tuple[Mapping[str, object], ...] | None = None


@dataclass(frozen=True)
class UploadResult:
    version_id: int | None
    source_sha256: str
    item_count: int
    created: bool
    activated: bool
    dry_run: bool
    registry_diff: RegistryDiff | None = None


def get_database_url() -> str:
    """비밀값을 출력하지 않고 PostgreSQL 연결 문자열을 검증한다."""

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

    if urlsplit(database_url).scheme not in {"postgres", "postgresql"}:
        raise RuntimeError("DATABASE_URL은 PostgreSQL 연결 문자열이어야 합니다.")

    return database_url


def _v2_source_items(
    source_markdown: str,
) -> tuple[Mapping[str, object], ...]:
    source = parse_source_v2_text(source_markdown)
    return tuple(item.model_dump(mode="json") for item in source.items)


def prepare_registry(
    source_path: Path,
    *,
    source_format: str = "markdown-v1",
) -> PreparedRegistry:
    """원본을 공통 runtime 계약으로 검증하고 업로드 값을 준비한다."""

    if source_format not in SOURCE_FORMATS:
        raise RegistryError(f"지원하지 않는 source format입니다: {source_format}")

    resolved_source = source_path.resolve()
    if not resolved_source.is_file():
        raise RegistryError(f"원본 Markdown 파일이 없습니다: {resolved_source}")

    source_bytes = resolved_source.read_bytes()
    try:
        source_markdown = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegistryError("원본 Markdown은 UTF-8이어야 합니다.") from exc

    registry = build_registry(
        resolved_source,
        source_format=source_format,
    )
    items = registry.get("items")
    if not isinstance(items, list) or not items:
        raise RegistryError("생성된 레지스트리의 items는 비어 있지 않은 배열이어야 합니다.")

    validated_registry = validate_registry_document(registry)
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source = validated_registry.get("source")
    registry_sha256 = source.get("sha256") if isinstance(source, Mapping) else None
    if registry_sha256 != actual_sha256:
        raise RegistryError(
            "원본 Markdown 해시와 생성된 레지스트리 해시가 일치하지 않습니다."
        )

    schema_version = validated_registry["schema_version"]
    source_items = (
        _v2_source_items(source_markdown)
        if source_format == "markdown-v2"
        else None
    )

    return PreparedRegistry(
        source_filename=resolved_source.name,
        source_markdown=source_markdown,
        source_sha256=actual_sha256,
        schema_version=schema_version,
        document=validated_registry,
        item_count=len(validated_registry["items"]),
        source_format=source_format,
        source_items=source_items,
    )


def _row_id(row: object) -> int:
    if isinstance(row, Mapping):
        return int(row["id"])
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        return int(row[0])
    raise RuntimeError("버전 ID를 읽지 못했습니다.")


def _existing_version_values(
    row: object,
) -> tuple[int, str, str, str, str, object, int]:
    if isinstance(row, Mapping):
        return (
            int(row["id"]),
            str(row["source_filename"]),
            str(row["source_markdown"]),
            str(row["source_sha256"]),
            str(row["schema_version"]),
            row["registry_json"],
            int(row["item_count"]),
        )
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        if len(row) < 7:
            raise RuntimeError("기존 버전 행의 필드가 부족합니다.")
        return (
            int(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            row[5],
            int(row[6]),
        )
    raise RuntimeError("기존 버전 행을 읽지 못했습니다.")


def _find_version_id(cursor: Any, prepared: PreparedRegistry) -> int | None:
    cursor.execute(FIND_VERSION_SQL, (prepared.source_sha256,))
    row = cursor.fetchone()
    if row is None:
        return None

    (
        version_id,
        source_filename,
        source_markdown,
        source_sha256,
        schema_version,
        raw_document,
        item_count,
    ) = _existing_version_values(row)
    stored_document = validate_registry_document(raw_document)

    stored_source = stored_document["source"]
    actual_source_sha256 = hashlib.sha256(
        source_markdown.encode("utf-8")
    ).hexdigest()
    if (
        source_sha256 != prepared.source_sha256
        or actual_source_sha256 != source_sha256
        or stored_source["sha256"] != source_sha256
        or stored_source["file"] != source_filename
        or stored_document["schema_version"] != schema_version
        or len(stored_document["items"]) != item_count
    ):
        raise RegistryError("동일 source hash의 기존 DB 버전 metadata가 손상되었습니다.")

    # The version key is the source bytes hash, not the local basename.  A
    # byte-identical private source copied to another directory may therefore
    # reuse the same immutable DB version even though compiler metadata
    # ``source.file`` reflects the caller's local filename.
    stored_comparable = deepcopy(stored_document)
    prepared_comparable = deepcopy(prepared.document)
    stored_comparable["source"]["file"] = "<source-filename>"
    prepared_comparable["source"]["file"] = "<source-filename>"
    if stored_comparable != prepared_comparable:
        raise RegistryError(
            "동일 source hash의 기존 버전이 현재 compiler 결과와 다릅니다."
        )
    return version_id


def _active_row_values(row: object) -> tuple[int, object, str]:
    if isinstance(row, Mapping):
        return (
            int(row["version_id"]),
            row["registry_json"],
            str(row["source_markdown"]),
        )
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        if len(row) < 3:
            raise RuntimeError("활성 레지스트리 행의 필드가 부족합니다.")
        return int(row[0]), row[1], str(row[2])
    raise RuntimeError("활성 레지스트리 행을 읽지 못했습니다.")


def _active_registry_from_row(row: object) -> ActiveRegistry:
    version_id, raw_document, source_markdown = _active_row_values(row)
    document = validate_registry_document(raw_document)
    source = document["source"]
    source_sha256 = source["sha256"]
    actual_sha256 = hashlib.sha256(source_markdown.encode("utf-8")).hexdigest()
    if source_sha256 != actual_sha256:
        raise RegistryError("활성 레지스트리의 원본 해시가 저장 문서와 다릅니다.")

    source_format = source.get("format", "markdown-v1")
    if source_format not in SOURCE_FORMATS:
        raise RegistryError("활성 레지스트리의 source format을 지원하지 않습니다.")
    source_items = (
        _v2_source_items(source_markdown)
        if source_format == "markdown-v2"
        else None
    )
    return ActiveRegistry(
        version_id=version_id,
        document=document,
        source_markdown=source_markdown,
        source_format=source_format,
        source_items=source_items,
    )


def _read_active_registry(
    cursor: Any,
    *,
    for_update: bool = False,
) -> ActiveRegistry | None:
    cursor.execute(READ_ACTIVE_FOR_UPDATE_SQL if for_update else READ_ACTIVE_SQL)
    row = cursor.fetchone()
    return None if row is None else _active_registry_from_row(row)


def _registry_diff(
    active: ActiveRegistry | None,
    prepared: PreparedRegistry,
) -> RegistryDiff:
    current_document: Mapping[str, object] = (
        active.document if active is not None else {"items": []}
    )
    return diff_registries(
        current_document,
        prepared.document,
        current_source_items=None if active is None else active.source_items,
        candidate_source_items=prepared.source_items,
        current_version_id=None if active is None else active.version_id,
    )


def _verify_operation_flags(
    *,
    activate: bool,
    dry_run: bool,
    expected_current_version: int | None | object,
    allow_breaking: bool,
) -> None:
    if expected_current_version is not _EXPECTED_CURRENT_UNSET and (
        expected_current_version is not None
        and (
            isinstance(expected_current_version, bool)
            or not isinstance(expected_current_version, int)
            or expected_current_version <= 0
        )
    ):
        raise RegistryError(
            "expected current version은 양의 정수 또는 none이어야 합니다."
        )
    if activate and dry_run:
        raise RegistryError("--activate와 --dry-run은 동시에 사용할 수 없습니다.")
    if activate and expected_current_version is _EXPECTED_CURRENT_UNSET:
        raise RegistryError("활성화에는 expected current version이 필요합니다.")
    if not activate and expected_current_version is not _EXPECTED_CURRENT_UNSET:
        raise RegistryError("expected current version은 활성화할 때만 지정할 수 있습니다.")
    if allow_breaking and not activate:
        raise RegistryError("breaking 변경 허용은 활성화할 때만 지정할 수 있습니다.")


def _preview_active_diff(
    prepared: PreparedRegistry,
    *,
    database_url: str | None = None,
) -> RegistryDiff:
    """현재 활성 버전을 읽기 전용 트랜잭션으로 비교한다."""

    connection = connect(database_url or get_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(SET_TRANSACTION_READ_ONLY_SQL)
            active = _read_active_registry(cursor)
            registry_diff = _registry_diff(active, prepared)
        connection.rollback()
        return registry_diff
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def upload_prepared_registry(
    prepared: PreparedRegistry,
    *,
    activate: bool = False,
    diff_active: bool = False,
    expected_current_version: int | None | object = _EXPECTED_CURRENT_UNSET,
    allow_breaking: bool = False,
    database_url: str | None = None,
) -> UploadResult:
    """후보를 멱등 업로드하고, 보호 조건을 통과한 경우에만 활성화한다."""

    _verify_operation_flags(
        activate=activate,
        dry_run=False,
        expected_current_version=expected_current_version,
        allow_breaking=allow_breaking,
    )
    connection = connect(database_url or get_database_url())
    created = False
    registry_diff: RegistryDiff | None = None

    try:
        with connection.cursor() as cursor:
            if activate:
                cursor.execute(LOCK_CURRENT_SQL)
                active = _read_active_registry(cursor, for_update=True)
                actual_current_version = (
                    None if active is None else active.version_id
                )
                if actual_current_version != expected_current_version:
                    raise RegistryError(
                        "활성 레지스트리 버전이 기대값과 일치하지 않습니다."
                    )
                registry_diff = _registry_diff(active, prepared)
                if registry_diff.is_breaking and not allow_breaking:
                    raise RegistryError(
                        "breaking 변경은 --allow-breaking 없이 활성화할 수 없습니다."
                    )
            elif diff_active:
                active = _read_active_registry(cursor)
                registry_diff = _registry_diff(active, prepared)

            version_id = _find_version_id(cursor, prepared)
            if version_id is None:
                cursor.execute(
                    INSERT_VERSION_SQL,
                    (
                        prepared.source_filename,
                        prepared.source_markdown,
                        prepared.source_sha256,
                        prepared.schema_version,
                        Jsonb(prepared.document),
                        prepared.item_count,
                    ),
                )
                inserted_row = cursor.fetchone()
                if inserted_row is None:
                    version_id = _find_version_id(cursor, prepared)
                    if version_id is None:
                        raise RuntimeError("업로드된 레지스트리 버전을 찾지 못했습니다.")
                else:
                    version_id = _row_id(inserted_row)
                    created = True

            if activate:
                cursor.execute(ACTIVATE_VERSION_SQL, (version_id,))

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return UploadResult(
        version_id=version_id,
        source_sha256=prepared.source_sha256,
        item_count=prepared.item_count,
        created=created,
        activated=activate,
        dry_run=False,
        registry_diff=registry_diff,
    )


def upload_competency_registry(
    source_path: Path,
    *,
    source_format: str = "markdown-v1",
    activate: bool = False,
    dry_run: bool = False,
    diff_active: bool = False,
    expected_current_version: int | None | object = _EXPECTED_CURRENT_UNSET,
    allow_breaking: bool = False,
    database_url: str | None = None,
) -> UploadResult:
    """원본을 준비한 뒤 dry-run, 비교, 업로드 또는 활성화한다."""

    _verify_operation_flags(
        activate=activate,
        dry_run=dry_run,
        expected_current_version=expected_current_version,
        allow_breaking=allow_breaking,
    )
    prepared = prepare_registry(source_path, source_format=source_format)
    if dry_run:
        registry_diff = (
            _preview_active_diff(prepared, database_url=database_url)
            if diff_active
            else None
        )
        return UploadResult(
            version_id=None,
            source_sha256=prepared.source_sha256,
            item_count=prepared.item_count,
            created=False,
            activated=False,
            dry_run=True,
            registry_diff=registry_diff,
        )

    return upload_prepared_registry(
        prepared,
        activate=activate,
        diff_active=diff_active,
        expected_current_version=expected_current_version,
        allow_breaking=allow_breaking,
        database_url=database_url,
    )


def _parse_expected_current_version(value: str) -> int | None:
    if value.casefold() == "none":
        return None
    try:
        version_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected current version은 양의 정수 또는 'none'이어야 합니다."
        ) from exc
    if version_id <= 0:
        raise argparse.ArgumentTypeError(
            "expected current version은 양의 정수 또는 'none'이어야 합니다."
        )
    return version_id


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="역량 정의 Markdown을 검증해 PostgreSQL에 업로드합니다."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="업로드할 원본 Markdown 경로",
    )
    parser.add_argument(
        "--source-format",
        choices=SOURCE_FORMATS,
        default="markdown-v1",
        help="원본 문서 형식(기본값: markdown-v1)",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="업로드한 버전을 현재 활성 버전으로 지정합니다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="후보를 파싱·검증하되 DB에는 저장하지 않습니다.",
    )
    parser.add_argument(
        "--diff-active",
        action="store_true",
        help="현재 활성 버전과 후보의 안전한 변경 요약을 출력합니다.",
    )
    parser.add_argument(
        "--expected-current-version",
        type=_parse_expected_current_version,
        default=_EXPECTED_CURRENT_UNSET,
        metavar="N|none",
        help="활성화 직전 기대하는 현재 버전 ID 또는 none",
    )
    parser.add_argument(
        "--allow-breaking",
        action="store_true",
        help="breaking 변경의 활성화를 명시적으로 허용합니다.",
    )
    args = parser.parse_args(argv)

    if args.activate and args.dry_run:
        parser.error("--activate와 --dry-run은 동시에 사용할 수 없습니다.")
    if args.activate and args.expected_current_version is _EXPECTED_CURRENT_UNSET:
        parser.error("--activate에는 --expected-current-version이 필요합니다.")
    if not args.activate and args.expected_current_version is not _EXPECTED_CURRENT_UNSET:
        parser.error("--expected-current-version은 --activate와 함께 사용해야 합니다.")
    if args.allow_breaking and not args.activate:
        parser.error("--allow-breaking은 --activate와 함께 사용해야 합니다.")
    return args


def print_result(result: UploadResult) -> None:
    version_id = "dry-run" if result.version_id is None else str(result.version_id)
    print(f"version_id: {version_id}")
    print(f"source_sha256: {result.source_sha256}")
    print(f"item_count: {result.item_count}")
    if result.registry_diff is not None:
        print(result.registry_diff.to_summary_text())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = upload_competency_registry(
        args.source,
        source_format=args.source_format,
        activate=args.activate,
        dry_run=args.dry_run,
        diff_active=args.diff_active,
        expected_current_version=args.expected_current_version,
        allow_breaking=args.allow_breaking,
    )
    print_result(result)
    return 0


def _cli() -> int:
    """원문·정의·DB 연결 정보가 오류 출력에 섞이지 않게 한다."""

    try:
        return main()
    except Exception:
        print("역량 레지스트리 업로드에 실패했습니다.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
