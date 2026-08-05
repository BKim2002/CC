"""검증된 역량 레지스트리 버전을 PostgreSQL에 업로드한다."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from psycopg import connect
from psycopg.types.json import Jsonb

if __package__:
    from scripts.build_registry import (
        EXPECTED_COUNTS,
        RegistryError,
        build_registry,
        validate_items,
    )
else:  # pragma: no cover - 직접 스크립트 실행 경로
    from build_registry import (  # type: ignore[no-redef]
        EXPECTED_COUNTS,
        RegistryError,
        build_registry,
        validate_items,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


FIND_VERSION_SQL = """
SELECT id
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


@dataclass(frozen=True)
class PreparedRegistry:
    source_filename: str
    source_markdown: str
    source_sha256: str
    schema_version: str
    document: dict[str, Any]
    item_count: int


@dataclass(frozen=True)
class UploadResult:
    version_id: int | None
    source_sha256: str
    item_count: int
    created: bool
    activated: bool
    dry_run: bool


def get_database_url() -> str:
    """비밀값을 출력하지 않고 PostgreSQL 연결 문자열을 검증한다."""

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

    if urlsplit(database_url).scheme not in {"postgres", "postgresql"}:
        raise RuntimeError("DATABASE_URL은 PostgreSQL 연결 문자열이어야 합니다.")

    return database_url


def prepare_registry(source_path: Path) -> PreparedRegistry:
    """원본을 변환·검증하고 업로드할 불변 값을 준비한다."""

    resolved_source = source_path.resolve()
    if not resolved_source.is_file():
        raise RegistryError(f"원본 Markdown 파일이 없습니다: {resolved_source}")

    registry = build_registry(resolved_source)

    # 생산 업로드는 build_registry의 전체 구조 검증(62개)을 다시 통과해야 한다.
    items = registry.get("items")
    if not isinstance(items, list):
        raise RegistryError("생성된 레지스트리의 items가 배열이 아닙니다.")
    counts = validate_items(items)
    expected_total = EXPECTED_COUNTS["total"]
    if counts["total"] != expected_total or len(items) != expected_total:
        raise RegistryError(
            f"생산 레지스트리는 {expected_total}개 항목이어야 합니다."
        )

    markdown_bytes = resolved_source.read_bytes()
    try:
        markdown_text = markdown_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegistryError("원본 Markdown은 UTF-8이어야 합니다.") from exc

    actual_sha256 = hashlib.sha256(markdown_bytes).hexdigest()
    source = registry.get("source")
    registry_sha256 = source.get("sha256") if isinstance(source, dict) else None
    if registry_sha256 != actual_sha256:
        raise RegistryError(
            "원본 Markdown 해시와 생성된 레지스트리 해시가 일치하지 않습니다."
        )

    schema_version = registry.get("schema_version")
    if schema_version != "1.0":
        raise RegistryError("지원하지 않는 레지스트리 schema_version입니다.")

    return PreparedRegistry(
        source_filename=resolved_source.name,
        source_markdown=markdown_text,
        source_sha256=actual_sha256,
        schema_version=schema_version,
        document=registry,
        item_count=len(items),
    )


def _row_id(row: object) -> int:
    if isinstance(row, Mapping):
        return int(row["id"])
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        return int(row[0])
    raise RuntimeError("버전 ID를 읽지 못했습니다.")


def _find_version_id(cursor: Any, source_sha256: str) -> int | None:
    cursor.execute(FIND_VERSION_SQL, (source_sha256,))
    row = cursor.fetchone()
    return None if row is None else _row_id(row)


def upload_prepared_registry(
    prepared: PreparedRegistry,
    *,
    activate: bool = False,
    database_url: str | None = None,
) -> UploadResult:
    """준비된 버전을 멱등 업로드하고 선택적으로 같은 트랜잭션에서 활성화한다."""

    connection = connect(database_url or get_database_url())
    created = False

    try:
        with connection.cursor() as cursor:
            version_id = _find_version_id(cursor, prepared.source_sha256)

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
                    # 동시에 같은 해시를 올린 트랜잭션이 먼저 완료된 경우다.
                    version_id = _find_version_id(
                        cursor,
                        prepared.source_sha256,
                    )
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
    )


def upload_competency_registry(
    source_path: Path,
    *,
    activate: bool = False,
    dry_run: bool = False,
    database_url: str | None = None,
) -> UploadResult:
    """원본을 준비한 뒤 dry-run하거나 PostgreSQL에 업로드한다."""

    prepared = prepare_registry(source_path)
    if dry_run:
        return UploadResult(
            version_id=None,
            source_sha256=prepared.source_sha256,
            item_count=prepared.item_count,
            created=False,
            activated=False,
            dry_run=True,
        )

    return upload_prepared_registry(
        prepared,
        activate=activate,
        database_url=database_url,
    )


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
        "--activate",
        action="store_true",
        help="업로드한 버전을 현재 활성 버전으로 지정합니다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB에 연결하지 않고 파싱·검증·해시 계산만 수행합니다.",
    )
    return parser.parse_args(argv)


def print_result(result: UploadResult) -> None:
    version_id = "dry-run" if result.version_id is None else str(result.version_id)
    print(f"version_id: {version_id}")
    print(f"source_sha256: {result.source_sha256}")
    print(f"item_count: {result.item_count}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = upload_competency_registry(
        args.source,
        activate=args.activate,
        dry_run=args.dry_run,
    )
    print_result(result)
    return 0


def _cli() -> int:
    """원문·JSON·DB 연결 정보가 오류 출력에 섞이지 않게 한다."""

    try:
        return main()
    except Exception:
        print("역량 레지스트리 업로드에 실패했습니다.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
