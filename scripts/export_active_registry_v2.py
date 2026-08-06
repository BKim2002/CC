"""활성 PostgreSQL 레지스트리를 구조화된 Markdown V2로 내보낸다.

이 스크립트는 활성 스냅샷을 읽기만 하며 DB 버전을 생성하거나 활성 포인터를
변경하지 않는다. 파일을 쓰기 전, 생성한 V2 문서를 다시 파싱·컴파일하여 현재
runtime 레지스트리와 핵심 의미 필드가 같은지도 검증한다.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # 직접 스크립트 실행 경로
    sys.path.insert(0, str(PROJECT_ROOT))

from competency_registry import load_active_registry  # noqa: E402
from scripts.registry_compiler import compile_registry  # noqa: E402
from scripts.registry_source_v2 import (  # noqa: E402
    SourceInstrument,
    SourceItem,
    SourceRegistry,
    parse_source_v2_text,
    render_source_v2,
)


load_dotenv(PROJECT_ROOT / ".env")

# V2 intentionally regenerates these bookkeeping fields.  Every user-facing
# hierarchy and lookup field is compared by ``semantic_projection`` below.
INTENTIONAL_ROUND_TRIP_NORMALIZATIONS = (
    "source metadata/rules/validation",
    "definition_status",
    "source_section",
    "runtime source extras (order/status/replacement_id)",
)


class RegistryExportError(ValueError):
    """내보내기 또는 round-trip 검증이 안전하게 끝나지 못했을 때 발생한다."""


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    version_id: int
    canonical_count: int
    lookup_count: int
    source_sha256: str


def get_database_url() -> str:
    """비밀값을 출력하지 않고 PostgreSQL 연결 문자열만 검증한다."""

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")
    if urlsplit(database_url).scheme not in {"postgres", "postgresql"}:
        raise RuntimeError("DATABASE_URL은 PostgreSQL 연결 문자열이어야 합니다.")
    return database_url


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryExportError(f"{label}가 object가 아닙니다.")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise RegistryExportError(f"{label}가 array가 아닙니다.")
    return value


def source_from_runtime_document(document: Mapping[str, Any]) -> SourceRegistry:
    """runtime 1.0 문서를 stable-ID V2 source model로 변환한다.

    형제 ``order``는 runtime의 ``children_ids`` 순서를 우선 보존한다. root의
    순서는 같은 instrument 안에서 runtime ``items``에 나타난 순서를 사용한다.
    """

    raw_items = _sequence(document.get("items"), "runtime items")
    items = [
        _mapping(item, f"runtime items[{index}]")
        for index, item in enumerate(raw_items)
    ]
    if not items:
        raise RegistryExportError("빈 runtime 레지스트리는 내보낼 수 없습니다.")

    by_id: dict[str, Mapping[str, Any]] = {}
    instrument_order: list[str] = []
    instrument_labels: dict[str, str] = {}
    roots_by_instrument: dict[str, list[str]] = {}

    for index, item in enumerate(items):
        item_id = item.get("id")
        instrument = item.get("instrument")
        instrument_label = item.get("instrument_label")
        if not isinstance(item_id, str) or not item_id.strip():
            raise RegistryExportError(f"runtime items[{index}].id가 올바르지 않습니다.")
        if item_id in by_id:
            raise RegistryExportError(f"중복 runtime item ID가 있습니다: {item_id}")
        if not isinstance(instrument, str) or not instrument.strip():
            raise RegistryExportError(f"{item_id}의 instrument가 올바르지 않습니다.")
        if not isinstance(instrument_label, str) or not instrument_label.strip():
            raise RegistryExportError(
                f"{item_id}의 instrument_label이 올바르지 않습니다."
            )

        previous_label = instrument_labels.get(instrument)
        if previous_label is not None and previous_label != instrument_label:
            raise RegistryExportError(
                f"instrument label이 항목마다 다릅니다: {instrument}"
            )
        if previous_label is None:
            instrument_labels[instrument] = instrument_label
            instrument_order.append(instrument)
            roots_by_instrument[instrument] = []

        by_id[item_id] = item
        if item.get("parent_id") is None:
            roots_by_instrument[instrument].append(item_id)

    order_by_id: dict[str, int] = {}
    for root_ids in roots_by_instrument.values():
        for order, item_id in enumerate(root_ids):
            order_by_id[item_id] = order

    for parent_id, parent in by_id.items():
        child_ids = _sequence(
            parent.get("children_ids"), f"runtime item {parent_id}.children_ids"
        )
        seen_children: set[str] = set()
        for order, child_id in enumerate(child_ids):
            if not isinstance(child_id, str) or not child_id.strip():
                raise RegistryExportError(
                    f"runtime item {parent_id}.children_ids 값이 올바르지 않습니다."
                )
            if child_id in seen_children:
                raise RegistryExportError(
                    f"runtime item {parent_id}에 중복 child ID가 있습니다: {child_id}"
                )
            seen_children.add(child_id)
            child = by_id.get(child_id)
            if child is None or child.get("parent_id") != parent_id:
                raise RegistryExportError(
                    f"runtime parent/child 관계가 일치하지 않습니다: {parent_id} -> {child_id}"
                )
            if child_id in order_by_id:
                raise RegistryExportError(
                    f"runtime item이 둘 이상의 sibling 목록에 있습니다: {child_id}"
                )
            order_by_id[child_id] = order

    instruments: list[SourceInstrument] = []
    for instrument in instrument_order:
        root_ids = roots_by_instrument[instrument]
        if not root_ids:
            raise RegistryExportError(
                f"runtime instrument에 root item이 없습니다: {instrument}"
            )

        common_prefix: tuple[str, ...] | None = None
        for root_id in root_ids:
            root = by_id[root_id]
            path = tuple(
                _sequence(root.get("path"), f"runtime item {root_id}.path")
            )
            name = root.get("name")
            if (
                not path
                or not isinstance(name, str)
                or path[-1] != name
                or any(
                    not isinstance(member, str) or not member.strip()
                    for member in path
                )
            ):
                raise RegistryExportError(
                    f"runtime root path가 정식 이름으로 끝나지 않습니다: {root_id}"
                )
            prefix = path[:-1]
            expected_parent_name = prefix[-1] if prefix else None
            if root.get("parent_name") != expected_parent_name:
                raise RegistryExportError(
                    f"runtime root path와 parent_name이 다릅니다: {root_id}"
                )
            if common_prefix is None:
                common_prefix = prefix
            elif common_prefix != prefix:
                raise RegistryExportError(
                    "같은 instrument의 root path prefix가 서로 다릅니다: "
                    f"{instrument}"
                )

        instruments.append(
            SourceInstrument(
                id=instrument,
                label=instrument_labels[instrument],
                root_path_prefix=list(common_prefix or ()),
            )
        )

    source_items: list[SourceItem] = []
    for item_id, item in by_id.items():
        if item_id not in order_by_id:
            raise RegistryExportError(
                f"runtime item이 parent children_ids에 없습니다: {item_id}"
            )
        aliases = list(
            _sequence(item.get("aliases"), f"runtime item {item_id}.aliases")
        )
        notes = list(_sequence(item.get("notes"), f"runtime item {item_id}.notes"))
        source_items.append(
            SourceItem(
                id=item_id,
                name=item.get("name"),
                instrument=item.get("instrument"),
                node_type=item.get("level"),
                parent_id=item.get("parent_id"),
                order=order_by_id[item_id],
                definition=item.get("definition"),
                aliases=aliases,
                analysis_included=item.get("analysis_included"),
                status="active",
                replacement_id=None,
                notes=notes,
            )
        )

    return SourceRegistry(
        source_schema_version="2.0",
        registry_schema_version="1.0",
        instruments=instruments,
        items=source_items,
    )


def semantic_projection(document: Mapping[str, Any]) -> dict[str, object]:
    """round-trip에 필요한 runtime 의미 필드만 정규화한다."""

    items = _sequence(document.get("items"), "runtime items")
    projected: dict[str, dict[str, object]] = {}
    roots_by_instrument: dict[str, list[str]] = {}
    lookup_count = 0
    for index, raw_item in enumerate(items):
        item = _mapping(raw_item, f"runtime items[{index}]")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise RegistryExportError(f"runtime items[{index}].id가 올바르지 않습니다.")
        if item_id in projected:
            raise RegistryExportError(f"중복 runtime item ID가 있습니다: {item_id}")
        aliases = tuple(
            _sequence(item.get("aliases"), f"runtime item {item_id}.aliases")
        )
        path = tuple(_sequence(item.get("path"), f"runtime item {item_id}.path"))
        children = tuple(
            _sequence(item.get("children"), f"runtime item {item_id}.children")
        )
        children_ids = tuple(
            _sequence(item.get("children_ids"), f"runtime item {item_id}.children_ids")
        )
        notes = tuple(
            _sequence(item.get("notes"), f"runtime item {item_id}.notes")
        )
        projected[item_id] = {
            "name": item.get("name"),
            "aliases": aliases,
            "definition": item.get("definition"),
            "instrument": item.get("instrument"),
            "instrument_label": item.get("instrument_label"),
            "node_type": item.get("level"),
            "path": path,
            "parent_name": item.get("parent_name"),
            "parent_id": item.get("parent_id"),
            "children": children,
            "children_ids": children_ids,
            "analysis_included": item.get("analysis_included"),
            "notes": notes,
        }
        instrument = item.get("instrument")
        if item.get("parent_id") is None:
            if not isinstance(instrument, str):
                raise RegistryExportError(
                    f"runtime item {item_id}.instrument가 올바르지 않습니다."
                )
            roots_by_instrument.setdefault(instrument, []).append(item_id)
        lookup_count += 1 + len(aliases)

    return {
        "canonical_count": len(projected),
        "lookup_count": lookup_count,
        "roots_by_instrument": {
            instrument: tuple(root_ids)
            for instrument, root_ids in roots_by_instrument.items()
        },
        "items": projected,
    }


def compile_rendered_source(markdown_text: str, *, source_filename: str) -> dict[str, Any]:
    """메모리의 V2 Markdown을 다시 파싱·컴파일한다."""

    source_bytes = markdown_text.encode("utf-8")
    source = parse_source_v2_text(markdown_text)
    return compile_registry(
        source,
        source_filename=source_filename,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_format="markdown-v2",
    )


def verify_round_trip(
    current_document: Mapping[str, Any],
    candidate_document: Mapping[str, Any],
) -> dict[str, object]:
    """핵심 의미 필드가 다르면 파일을 쓰기 전에 중단한다."""

    current = semantic_projection(current_document)
    candidate = semantic_projection(candidate_document)
    if current != candidate:
        raise RegistryExportError(
            "V2 export round-trip 결과가 활성 runtime 레지스트리와 다릅니다."
        )
    return current


def _resolve_safe_output(output_path: Path, *, overwrite: bool) -> Path:
    output = Path(output_path).resolve()
    if output == PROJECT_ROOT or output.is_relative_to(PROJECT_ROOT):
        raise RegistryExportError(
            "V2 원본 출력 경로는 프로젝트 저장소 밖이어야 합니다."
        )
    if output.exists() and not overwrite:
        raise RegistryExportError(
            "출력 파일이 이미 있습니다. 덮어쓰려면 --overwrite를 명시하세요."
        )
    if not output.parent.is_dir():
        raise RegistryExportError("출력 파일의 상위 디렉터리가 없습니다.")
    return output


def _write_verified_output(
    output: Path,
    markdown_text: str,
    *,
    overwrite: bool,
) -> None:
    """검증이 끝난 source를 경쟁·부분 덮어쓰기에 안전하게 저장한다."""

    if not overwrite:
        try:
            with output.open("x", encoding="utf-8", newline="") as stream:
                stream.write(markdown_text)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise RegistryExportError(
                "검증 중 출력 파일이 생성되어 내보내기를 중단했습니다."
            ) from exc
        return

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(markdown_text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def export_active_registry_v2(
    output_path: Path,
    *,
    overwrite: bool = False,
    database_url: str | None = None,
) -> ExportResult:
    """활성 snapshot을 읽고 검증된 V2 Markdown 파일 하나를 쓴다."""

    output = _resolve_safe_output(output_path, overwrite=overwrite)
    snapshot = load_active_registry(database_url or get_database_url())
    source = source_from_runtime_document(snapshot.document)
    markdown_text = render_source_v2(
        source,
        preamble=(
            "활성 runtime 레지스트리의 stable ID와 단일 부모 위계를 보존해 "
            "내보낸 structured Markdown V2 문서입니다."
        ),
    )
    candidate = compile_rendered_source(
        markdown_text,
        source_filename=output.name,
    )
    projection = verify_round_trip(snapshot.document, candidate)

    _write_verified_output(output, markdown_text, overwrite=overwrite)

    return ExportResult(
        output_path=output,
        version_id=snapshot.version_id,
        canonical_count=int(projection["canonical_count"]),
        lookup_count=int(projection["lookup_count"]),
        source_sha256=hashlib.sha256(markdown_text.encode("utf-8")).hexdigest(),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="활성 DB 레지스트리를 structured Markdown V2로 내보냅니다."
    )
    parser.add_argument("--output", type=Path, required=True, help="저장소 밖 출력 경로")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 출력 파일 덮어쓰기를 명시적으로 허용합니다.",
    )
    return parser.parse_args(argv)


def print_result(result: ExportResult) -> None:
    print("V2 레지스트리 내보내기와 round-trip 검증을 완료했습니다.")
    print(f"version_id: {result.version_id}")
    print(f"canonical_count: {result.canonical_count}")
    print(f"lookup_count: {result.lookup_count}")
    print(f"source_sha256: {result.source_sha256}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_active_registry_v2(args.output, overwrite=args.overwrite)
    print_result(result)
    return 0


def _cli() -> int:
    """DB URL과 source 전문이 예외 출력에 섞이지 않게 한다."""

    try:
        return main()
    except Exception:
        print("활성 역량 레지스트리 V2 내보내기에 실패했습니다.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
