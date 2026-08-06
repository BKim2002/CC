"""역량 레지스트리 source-format dispatcher와 파일 기반 CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__:
    from scripts.registry_source_v1 import (
        EXPECTED_COUNTS,
        RegistryError,
        build_registry_v1,
        validate_items,
    )
else:  # pragma: no cover - 직접 스크립트 실행 경로
    from registry_source_v1 import (  # type: ignore[no-redef]
        EXPECTED_COUNTS,
        RegistryError,
        build_registry_v1,
        validate_items,
    )


SOURCE_FORMATS = ("markdown-v1", "markdown-v2")


def _build_registry_v2(source_path: Path) -> dict[str, Any]:
    """V2 source를 공통 compiler가 소비하는 runtime document로 변환한다."""

    if __package__:
        from scripts.registry_compiler import compile_registry
        from scripts.registry_source_v2 import parse_source_v2
    else:  # pragma: no cover - 직접 스크립트 실행 경로
        from registry_compiler import compile_registry
        from registry_source_v2 import parse_source_v2

    source_bytes = source_path.read_bytes()
    source = parse_source_v2(source_path)
    return compile_registry(
        source,
        source_filename=source_path.name,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_format="markdown-v2",
    )


def build_registry(
    source_path: Path,
    source_format: str = "markdown-v1",
) -> dict[str, Any]:
    """명시된 source format의 parser/compiler로 레지스트리를 생성한다.

    포맷은 절대 내용으로 추측하지 않는다. 기본값은 기존 Python 호출의
    하위 호환성을 위해 ``markdown-v1``으로 유지한다.
    """

    if source_format == "markdown-v1":
        return build_registry_v1(source_path)
    if source_format == "markdown-v2":
        return _build_registry_v2(source_path)
    raise RegistryError(
        f"지원하지 않는 source format입니다: {source_format}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="역량 정의 Markdown을 검증된 JSON 레지스트리로 변환합니다."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="원본 Markdown 경로",
    )
    parser.add_argument(
        "--source-format",
        choices=SOURCE_FORMATS,
        default="markdown-v1",
        help="원본 문서 형식(기본값: markdown-v1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="생성한 JSON을 저장할 선택적 경로",
    )
    parser.add_argument(
        "--check",
        type=Path,
        metavar="JSON_PATH",
        help="현재 변환 결과와 비교할 기존 JSON 경로",
    )
    return parser.parse_args(argv)


def _format_counts(registry: Mapping[str, Any]) -> str:
    """특정 source policy의 count key를 가정하지 않고 요약한다."""

    validation = registry.get("validation")
    counts = (
        validation.get("counts")
        if isinstance(validation, Mapping)
        else None
    )
    if isinstance(counts, Mapping) and counts:
        return ", ".join(f"{key} {value}" for key, value in counts.items())

    items = registry.get("items")
    return f"total {len(items)}" if isinstance(items, list) else "확인 불가"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = args.source.resolve()

    if not source_path.is_file():
        raise RegistryError(f"원본 Markdown 파일이 없습니다: {source_path}")
    if args.output is not None and args.check is not None:
        raise RegistryError("--output과 --check는 동시에 사용할 수 없습니다.")

    registry = build_registry(source_path, source_format=args.source_format)

    if args.check is not None:
        check_path = args.check.resolve()
        if not check_path.is_file():
            raise RegistryError(f"검수할 JSON 파일이 없습니다: {check_path}")
        existing = json.loads(check_path.read_text(encoding="utf-8"))
        if existing != registry:
            raise RegistryError(
                "기존 JSON이 현재 Markdown에서 생성되는 결과와 일치하지 않습니다."
            )
        action = f"검수 완료: {check_path}"
    elif args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        action = f"변환 완료: {output_path}"
    else:
        action = "검증 완료"

    source = registry.get("source")
    source_sha256 = (
        source.get("sha256") if isinstance(source, Mapping) else None
    )
    print(action)
    print(f"원본 SHA-256: {source_sha256 or '확인 불가'}")
    print(f"항목 수: {_format_counts(registry)}")
    return 0


def _cli() -> int:
    """원본 YAML/Markdown 값이 traceback에 포함되지 않게 한다."""

    try:
        return main()
    except Exception:
        print("역량 레지스트리 변환 또는 검증에 실패했습니다.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
