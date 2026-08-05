"""PostgreSQL 역량 레지스트리 스키마를 한 번 준비한다."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from psycopg import connect


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


DDL_STATEMENTS = (
    "CREATE SCHEMA IF NOT EXISTS competency_data",
    """
    CREATE TABLE IF NOT EXISTS competency_data.registry_versions (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        source_filename TEXT NOT NULL,
        source_markdown TEXT NOT NULL,
        source_sha256 TEXT NOT NULL UNIQUE,
        schema_version TEXT NOT NULL,
        registry_json JSONB NOT NULL,
        item_count INTEGER NOT NULL CHECK (item_count > 0),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CHECK (length(source_sha256) = 64),
        CHECK (jsonb_typeof(registry_json) = 'object')
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS competency_data.registry_current (
        singleton SMALLINT PRIMARY KEY DEFAULT 1 CHECK (singleton = 1),
        version_id BIGINT NOT NULL
            REFERENCES competency_data.registry_versions(id),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS registry_versions_created_at_idx
        ON competency_data.registry_versions(created_at DESC)
    """,
)


def get_database_url() -> str:
    """비밀값을 출력하지 않고 PostgreSQL 연결 문자열을 검증한다."""

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

    if urlsplit(database_url).scheme not in {"postgres", "postgresql"}:
        raise RuntimeError("DATABASE_URL은 PostgreSQL 연결 문자열이어야 합니다.")

    return database_url


def setup_competency_registry_database(
    database_url: str | None = None,
) -> None:
    """레지스트리 DDL 전체를 하나의 트랜잭션으로 적용한다."""

    connection = connect(database_url or get_database_url())
    try:
        with connection.cursor() as cursor:
            for statement in DDL_STATEMENTS:
                cursor.execute(statement)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> int:
    setup_competency_registry_database()
    print("PostgreSQL 역량 레지스트리 스키마 준비를 완료했습니다.")
    return 0


def _cli() -> int:
    """DB 예외에 포함될 수 있는 연결 정보를 터미널에 노출하지 않는다."""

    try:
        return main()
    except Exception:
        print(
            "PostgreSQL 역량 레지스트리 스키마 준비에 실패했습니다.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
