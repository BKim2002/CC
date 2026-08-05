"""LangGraph PostgreSQL 체크포인트 테이블을 한 번 준비한다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_database_url() -> str:
    """비밀값을 출력하지 않고 PostgreSQL 연결 문자열을 검증한다."""

    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL 환경 변수가 설정되지 않았습니다."
        )

    return database_url


def setup_checkpoint_database() -> None:
    """LangGraph가 필요한 테이블과 마이그레이션을 적용한다."""

    with PostgresSaver.from_conn_string(get_database_url()) as saver:
        saver.setup()


def main() -> int:
    setup_checkpoint_database()
    print("PostgreSQL 체크포인트 테이블 준비를 완료했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
