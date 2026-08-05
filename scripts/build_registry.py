from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WRITTEN_INSTRUMENT = "written_competency"
VIDEO_INSTRUMENT = "video_interview"

OVERALL_NAME = "역검 종합점수"
WRITTEN_LEVEL_1 = ("성과 예측", "관계 예측", "적응 예측")

EXPECTED_COUNTS = {
    "written_overall": 1,
    "written_L1": 3,
    "written_L2": 10,
    "written_L3": 30,
    "written_L4": 9,
    "video_factor": 3,
    "video_item": 6,
    "total": 62,
}

EXPECTED_LEVEL_1_CHILDREN = {
    "성과 예측": 4,
    "관계 예측": 3,
    "적응 예측": 3,
}

STRATEGY_LEVEL_3 = {"기억력", "인지력", "분석력"}


class RegistryError(ValueError):
    """Raised when the Markdown source or generated registry is inconsistent."""


def clean_markdown(value: str) -> str:
    """Remove simple inline Markdown while preserving the source wording."""
    value = value.strip()
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [clean_markdown(cell) for cell in stripped.split("|")]


def is_separator_row(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells
    )


def extract_tables(markdown_text: str) -> list[dict[str, Any]]:
    """Extract Markdown tables and retain their nearest level-2 section."""
    lines = markdown_text.splitlines()
    tables: list[dict[str, Any]] = []
    current_h2 = ""
    index = 0

    while index < len(lines):
        line = lines[index]
        heading_match = re.match(r"^##\s+(.+?)\s*$", line)
        if heading_match:
            current_h2 = clean_markdown(heading_match.group(1))

        if (
            line.lstrip().startswith("|")
            and index + 1 < len(lines)
            and is_separator_row(lines[index + 1])
        ):
            header = split_table_row(line)
            rows: list[list[str]] = []
            row_index = index + 2

            while row_index < len(lines) and lines[row_index].lstrip().startswith("|"):
                row = split_table_row(lines[row_index])
                if len(row) < len(header):
                    row.extend([""] * (len(header) - len(row)))
                rows.append(row[: len(header)])
                row_index += 1

            tables.append(
                {
                    "section": current_h2,
                    "header": header,
                    "rows": rows,
                }
            )
            index = row_index
            continue

        index += 1

    return tables


def make_id(instrument: str, level: str, name: str) -> str:
    prefix = "written" if instrument == WRITTEN_INSTRUMENT else "video"
    return f"{prefix}:{level.lower()}:{name}"


def make_item(
    *,
    name: str,
    instrument: str,
    level: str,
    path: list[str],
    definition: str | None,
    analysis_included: bool,
    source_section: str,
    aliases: list[str] | None = None,
    definition_status: str = "explicit",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    parent_name = path[-2] if len(path) > 1 else None
    return {
        "id": make_id(instrument, level, name),
        "name": name,
        "aliases": aliases or [],
        "instrument": instrument,
        "instrument_label": (
            "필기 역량검사"
            if instrument == WRITTEN_INSTRUMENT
            else "영상면접"
        ),
        "level": level,
        "path": path,
        "parent_name": parent_name,
        "parent_id": None,
        "children": [],
        "children_ids": [],
        "definition": definition,
        "definition_status": definition_status,
        "analysis_included": analysis_included,
        "notes": notes or [],
        "source_section": source_section,
    }


def parse_summary_definition(tables: list[dict[str, Any]]) -> str:
    for table in tables:
        if table["header"][:2] != ["레벨", "명칭"]:
            continue
        for row in table["rows"]:
            if row[0] == "종합점수" and row[1] == OVERALL_NAME:
                if len(row) >= 4 and row[3]:
                    return row[3]
    raise RegistryError("위계 요약표에서 '역검 종합점수' 설명을 찾지 못했습니다.")


def add_unique_item(
    items: list[dict[str, Any]],
    seen_ids: set[str],
    item: dict[str, Any],
) -> None:
    if item["id"] in seen_ids:
        raise RegistryError(f"중복 항목 ID가 생성되었습니다: {item['id']}")
    seen_ids.add(item["id"])
    items.append(item)


def parse_standard_table(
    table: dict[str, Any],
    items: list[dict[str, Any]],
    seen_ids: set[str],
) -> None:
    level_1 = table["section"]
    if level_1 not in WRITTEN_LEVEL_1:
        raise RegistryError(
            f"일반 역량표의 Level 1 영역을 판별할 수 없습니다: {level_1}"
        )

    current_level_2 = ""
    current_level_2_definition = ""

    for row in table["rows"]:
        level_2_name, level_2_definition, level_3_name, level_3_definition = row

        if level_2_name:
            current_level_2 = level_2_name
            current_level_2_definition = level_2_definition
            add_unique_item(
                items,
                seen_ids,
                make_item(
                    name=current_level_2,
                    instrument=WRITTEN_INSTRUMENT,
                    level="L2",
                    path=[OVERALL_NAME, level_1, current_level_2],
                    definition=current_level_2_definition,
                    analysis_included=False,
                    source_section=level_1,
                ),
            )

        if not current_level_2:
            raise RegistryError(
                f"{level_1} 표에서 부모 중위요인 없이 하위요인이 나타났습니다."
            )
        if not level_3_name or not level_3_definition:
            raise RegistryError(
                f"{level_1} > {current_level_2} 표에 이름 또는 정의가 비어 있습니다."
            )

        add_unique_item(
            items,
            seen_ids,
            make_item(
                name=level_3_name,
                instrument=WRITTEN_INSTRUMENT,
                level="L3",
                path=[OVERALL_NAME, level_1, current_level_2, level_3_name],
                definition=level_3_definition,
                analysis_included=True,
                source_section=level_1,
            ),
        )


def parse_strategy_table(
    table: dict[str, Any],
    items: list[dict[str, Any]],
    seen_ids: set[str],
) -> None:
    level_1 = "성과 예측"
    current_level_2 = ""
    current_level_3 = ""

    for row in table["rows"]:
        (
            level_2_name,
            level_2_definition,
            level_3_name,
            level_3_definition,
            level_4_name,
            level_4_definition,
        ) = row

        if level_2_name:
            current_level_2 = level_2_name
            add_unique_item(
                items,
                seen_ids,
                make_item(
                    name=current_level_2,
                    instrument=WRITTEN_INSTRUMENT,
                    level="L2",
                    path=[OVERALL_NAME, level_1, current_level_2],
                    definition=level_2_definition,
                    analysis_included=False,
                    source_section="성과 예측 > 전략성",
                ),
            )

        if level_3_name:
            current_level_3 = level_3_name
            add_unique_item(
                items,
                seen_ids,
                make_item(
                    name=current_level_3,
                    instrument=WRITTEN_INSTRUMENT,
                    level="L3",
                    path=[
                        OVERALL_NAME,
                        level_1,
                        current_level_2,
                        current_level_3,
                    ],
                    definition=level_3_definition,
                    analysis_included=True,
                    source_section="성과 예측 > 전략성",
                    notes=[
                        "전략성의 Level 3 집계 항목으로 분석 변수에 직접 사용한다."
                    ],
                ),
            )

        if not current_level_2 or not current_level_3:
            raise RegistryError(
                "전략성 표에서 부모 Level 2 또는 Level 3 없이 Level 4가 나타났습니다."
            )
        if not level_4_name or not level_4_definition:
            raise RegistryError("전략성 표의 Level 4 이름 또는 정의가 비어 있습니다.")

        add_unique_item(
            items,
            seen_ids,
            make_item(
                name=level_4_name,
                instrument=WRITTEN_INSTRUMENT,
                level="L4",
                path=[
                    OVERALL_NAME,
                    level_1,
                    current_level_2,
                    current_level_3,
                    level_4_name,
                ],
                definition=level_4_definition,
                analysis_included=False,
                source_section="성과 예측 > 전략성",
                notes=[
                    "Level 4 최하위요인으로, 분석 변수에는 직접 사용하지 않는다.",
                    "상위 Level 3 집계 항목을 분석 단위로 사용한다.",
                ],
            ),
        )


def parse_video_table(
    table: dict[str, Any],
    items: list[dict[str, Any]],
    seen_ids: set[str],
) -> None:
    current_factor = ""

    for row in table["rows"]:
        (
            factor_name,
            factor_definition,
            item_name,
            legacy_column_name,
            item_definition,
        ) = row

        if factor_name:
            current_factor = factor_name
            add_unique_item(
                items,
                seen_ids,
                make_item(
                    name=current_factor,
                    instrument=VIDEO_INSTRUMENT,
                    level="factor",
                    path=["영상면접", current_factor],
                    definition=factor_definition,
                    analysis_included=False,
                    source_section="영상 면접 역량",
                    notes=[
                        "필기 역량검사와 별개의 영상면접 평가요인이다.",
                        "competency score_cols에 포함하지 않는다.",
                    ],
                ),
            )

        if not current_factor:
            raise RegistryError("영상면접 표에서 평가요인 없이 세부항목이 나타났습니다.")
        if not item_name or not item_definition:
            raise RegistryError(
                f"영상면접 > {current_factor} 표의 세부항목 이름 또는 정의가 비어 있습니다."
            )

        aliases = (
            [legacy_column_name]
            if legacy_column_name and legacy_column_name != item_name
            else []
        )
        add_unique_item(
            items,
            seen_ids,
            make_item(
                name=item_name,
                aliases=aliases,
                instrument=VIDEO_INSTRUMENT,
                level="item",
                path=["영상면접", current_factor, item_name],
                definition=item_definition,
                analysis_included=False,
                source_section="영상 면접 역량",
                notes=[
                    "필기 역량검사와 별개의 영상면접 세부항목이다.",
                    "competency score_cols에 포함하지 않는다.",
                ],
            ),
        )


def connect_hierarchy(items: list[dict[str, Any]]) -> None:
    by_instrument_and_name: dict[tuple[str, str], dict[str, Any]] = {}
    by_id = {item["id"]: item for item in items}

    for item in items:
        key = (item["instrument"], item["name"])
        if key in by_instrument_and_name:
            raise RegistryError(
                "같은 검사 안에서 중복된 정식 명칭이 발견되었습니다: "
                f"{item['instrument']} / {item['name']}"
            )
        by_instrument_and_name[key] = item

    for item in items:
        parent_name = item["parent_name"]
        if not parent_name:
            continue

        # "영상면접"은 경로 표시용 루트이며 별도 항목으로 저장하지 않는다.
        if item["instrument"] == VIDEO_INSTRUMENT and parent_name == "영상면접":
            continue

        parent = by_instrument_and_name.get((item["instrument"], parent_name))
        if parent is None:
            raise RegistryError(
                f"부모 항목을 찾지 못했습니다: {item['name']} -> {parent_name}"
            )
        item["parent_id"] = parent["id"]
        parent["children"].append(item["name"])
        parent["children_ids"].append(item["id"])

    for item in items:
        for child_id in item["children_ids"]:
            if child_id not in by_id:
                raise RegistryError(
                    f"존재하지 않는 자식 ID가 연결되었습니다: {child_id}"
                )


def build_items(markdown_text: str) -> list[dict[str, Any]]:
    tables = extract_tables(markdown_text)
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    overall_definition = parse_summary_definition(tables)
    add_unique_item(
        items,
        seen_ids,
        make_item(
            name=OVERALL_NAME,
            instrument=WRITTEN_INSTRUMENT,
            level="overall",
            path=[OVERALL_NAME],
            definition=overall_definition,
            definition_status="summary",
            analysis_included=False,
            source_section="위계 구조 (5단계)",
            notes=[
                "원본 문서는 종합점수를 '전체 통합 점수'로 요약한다.",
                "분석 변수는 Level 3 하위요인 30개를 사용한다.",
            ],
        ),
    )

    for level_1 in WRITTEN_LEVEL_1:
        add_unique_item(
            items,
            seen_ids,
            make_item(
                name=level_1,
                instrument=WRITTEN_INSTRUMENT,
                level="L1",
                path=[OVERALL_NAME, level_1],
                definition=None,
                definition_status="not_provided",
                analysis_included=False,
                source_section="위계 구조 (5단계)",
                notes=[
                    "원본 문서에는 이 Level 1 상위요인의 독립적인 정의 문장이 없다.",
                    "소속 Level 2 중위요인을 통해 구성 범위를 설명해야 한다.",
                ],
            ),
        )

    parsed_standard_tables = 0
    parsed_strategy_tables = 0
    parsed_video_tables = 0

    for table in tables:
        header = table["header"]

        if (
            len(header) == 4
            and header[0] == "중위요인"
            and header[2].startswith("하위요인")
        ):
            parse_standard_table(table, items, seen_ids)
            parsed_standard_tables += 1
        elif (
            len(header) == 6
            and header[0] == "중위요인"
            and header[4].startswith("최하위요인")
        ):
            parse_strategy_table(table, items, seen_ids)
            parsed_strategy_tables += 1
        elif (
            len(header) == 5
            and header[0] == "평가요인"
            and header[2].startswith("세부항목")
        ):
            parse_video_table(table, items, seen_ids)
            parsed_video_tables += 1

    if parsed_standard_tables != 3:
        raise RegistryError(
            f"일반 역량표는 3개여야 하지만 {parsed_standard_tables}개를 읽었습니다."
        )
    if parsed_strategy_tables != 1:
        raise RegistryError(
            f"전략성 표는 1개여야 하지만 {parsed_strategy_tables}개를 읽었습니다."
        )
    if parsed_video_tables != 1:
        raise RegistryError(
            f"영상면접 표는 1개여야 하지만 {parsed_video_tables}개를 읽었습니다."
        )

    connect_hierarchy(items)
    return items


def classify_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()

    for item in items:
        if item["instrument"] == WRITTEN_INSTRUMENT:
            counts[f"written_{item['level']}"] += 1
        elif item["instrument"] == VIDEO_INSTRUMENT:
            counts[f"video_{item['level']}"] += 1

    counts["total"] = len(items)
    return {key: counts[key] for key in EXPECTED_COUNTS}


def validate_items(items: list[dict[str, Any]]) -> dict[str, int]:
    errors: list[str] = []
    counts = classify_counts(items)

    for key, expected in EXPECTED_COUNTS.items():
        actual = counts[key]
        if actual != expected:
            errors.append(f"{key}: 기대 {expected}개, 실제 {actual}개")

    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        errors.append("중복된 항목 ID가 있습니다.")

    # 런타임 lookup은 검사 종류와 무관하게 하나의 이름 공간을 사용한다.
    # 업로드 전에 같은 규칙으로 검증해, 저장에는 성공했지만 다음 startup에서
    # 거부되는 레지스트리를 만들지 않는다.
    lookup_owners: dict[str, str] = {}
    for item in items:
        name = item["name"]
        previous_owner = lookup_owners.get(name)
        if previous_owner is not None:
            errors.append(
                f"중복된 정식 명칭이 있습니다: {name} "
                f"({previous_owner}, {item['id']})"
            )
        else:
            lookup_owners[name] = item["id"]

    for item in items:
        if not item["path"] or item["path"][-1] != item["name"]:
            errors.append(f"{item['name']}: path의 마지막 값이 정식 명칭과 다릅니다.")

        if item["definition_status"] == "explicit" and not item["definition"]:
            errors.append(f"{item['name']}: 명시적 정의가 비어 있습니다.")

        if item["level"] == "L1":
            if item["definition"] is not None:
                errors.append(
                    f"{item['name']}: 원본에 없는 Level 1 정의가 생성되었습니다."
                )
            if item["definition_status"] != "not_provided":
                errors.append(
                    f"{item['name']}: Level 1 정의 상태가 not_provided가 아닙니다."
                )

        should_be_analysis_variable = (
            item["instrument"] == WRITTEN_INSTRUMENT
            and item["level"] == "L3"
        )
        if item["analysis_included"] != should_be_analysis_variable:
            errors.append(
                f"{item['name']}: analysis_included 값이 분석 규칙과 다릅니다."
            )

        if item["parent_id"] is None:
            allowed_root = (
                item["level"] == "overall"
                or (
                    item["instrument"] == VIDEO_INSTRUMENT
                    and item["level"] == "factor"
                )
            )
            if not allowed_root:
                errors.append(f"{item['name']}: 부모 연결이 누락되었습니다.")

    item_by_name = {
        (item["instrument"], item["name"]): item for item in items
    }

    for level_1_name, expected_children in EXPECTED_LEVEL_1_CHILDREN.items():
        item = item_by_name.get((WRITTEN_INSTRUMENT, level_1_name))
        if item is None:
            errors.append(f"Level 1 항목이 없습니다: {level_1_name}")
        elif len(item["children"]) != expected_children:
            errors.append(
                f"{level_1_name}: Level 2 자식은 {expected_children}개여야 하지만 "
                f"{len(item['children'])}개입니다."
            )

    for item in items:
        if item["instrument"] == WRITTEN_INSTRUMENT and item["level"] == "L2":
            if len(item["children"]) != 3:
                errors.append(
                    f"{item['name']}: Level 3 자식은 3개여야 하지만 "
                    f"{len(item['children'])}개입니다."
                )
        if item["instrument"] == VIDEO_INSTRUMENT and item["level"] == "factor":
            if len(item["children"]) != 2:
                errors.append(
                    f"{item['name']}: 영상면접 세부항목은 2개여야 하지만 "
                    f"{len(item['children'])}개입니다."
                )

    for level_3_name in STRATEGY_LEVEL_3:
        item = item_by_name.get((WRITTEN_INSTRUMENT, level_3_name))
        if item is None:
            errors.append(f"전략성 Level 3 항목이 없습니다: {level_3_name}")
        elif len(item["children"]) != 3:
            errors.append(
                f"{level_3_name}: Level 4 자식은 3개여야 하지만 "
                f"{len(item['children'])}개입니다."
            )

    for item in items:
        for alias in item["aliases"]:
            previous_owner = lookup_owners.get(alias)
            if previous_owner is not None:
                errors.append(
                    f"조회 이름 '{alias}'가 정식 명칭 또는 다른 별칭과 "
                    f"충돌합니다: {previous_owner}, {item['id']}"
                )
            else:
                lookup_owners[alias] = item["id"]

    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise RegistryError(f"레지스트리 검증에 실패했습니다:\n{formatted}")

    return counts


def build_registry(source_path: Path) -> dict[str, Any]:
    markdown_bytes = source_path.read_bytes()
    markdown_text = markdown_bytes.decode("utf-8")
    items = build_items(markdown_text)
    counts = validate_items(items)

    return {
        "schema_version": "1.0",
        "source": {
            "file": source_path.name,
            "sha256": hashlib.sha256(markdown_bytes).hexdigest(),
            "encoding": "utf-8",
        },
        "rules": {
            "written_analysis_unit": "Level 3 하위요인 30개",
            "level_4": "정의와 데이터 컬럼은 보존하되 분석 변수에서는 제외",
            "video_interview": (
                "필기 역량검사와 별도의 도구이며 competency score_cols에서 제외"
            ),
        },
        "validation": {
            "status": "passed",
            "counts": counts,
        },
        "items": items,
    }


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = args.source.resolve()

    if not source_path.is_file():
        raise RegistryError(f"원본 Markdown 파일이 없습니다: {source_path}")
    if args.output is not None and args.check is not None:
        raise RegistryError("--output과 --check는 동시에 사용할 수 없습니다.")

    registry = build_registry(source_path)

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

    counts = registry["validation"]["counts"]
    print(action)
    print(f"원본 SHA-256: {registry['source']['sha256']}")
    print(
        "항목 수: "
        f"종합 {counts['written_overall']}, "
        f"L1 {counts['written_L1']}, "
        f"L2 {counts['written_L2']}, "
        f"L3 {counts['written_L3']}, "
        f"L4 {counts['written_L4']}, "
        f"영상면접 평가요인 {counts['video_factor']}, "
        f"영상면접 세부항목 {counts['video_item']}, "
        f"총 {counts['total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
