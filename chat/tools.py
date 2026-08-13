"""모델이 추론으로 메우면 안 되는 것만 도구로 노출한다.

레지스트리 전문이 이미 컨텍스트에 있으므로 조회용 도구는 두지 않는다.
1단계에서 정의·목록·비교는 도구 없이 전부 통과했다.  남는 것은 두 종류다.

* **정확한 개수** -- 62개를 눈으로 세면 틀릴 수 있다.
* **빠짐없는 트리 순회** -- 후손 전체는 깊이가 있으면 놓친다.

v1은 질문 유형별로 실행 경로를 두었다.  여기서는 자료구조가 강제하는 것만
둔다.  항목이 62개에서 620개가 되어도 이 파일은 자라지 않는다.
REBUILD_PLAN 3.1 참고.

도구 출력에 **티어 이름을 함께 싣는 것이 이 단계의 핵심**이다.  1단계에서
`g06`은 개수 3은 맞혔지만 자식을 `중위요인`이라 불렀다 -- 전략성이 중위요인
이므로 그 자식은 하위요인인데, 모델이 부모의 티어 이름을 물려주었다.  8회 중
7회가 그랬다.  개수만 돌려주면 이 오류는 그대로 남는다.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping

from chat.registry import RegistrySnapshot

# 필기 역량검사의 level 값에 붙는 한국어 위계 용어.  level 값 자체에서
# 유도되지 않는 도메인 어휘라 여기에 한 번만 적는다 -- chat.prompt 의
# 용어표도 이 사전에서 만들어져, 모델이 읽는 이름과 도구가 돌려주는
# 이름이 갈라질 수 없다.
WRITTEN_TIER_TERMS: Mapping[str, str] = {
    "overall": "종합점수",
    "L1": "상위요인",
    "L2": "중위요인",
    "L3": "하위요인",
    "L4": "최하위요인",
}

VIDEO_TIER_TERMS: Mapping[str, str] = {
    "factor": "평가요인",
    "item": "세부항목",
}

VIDEO_INSTRUMENT_ID = "video_interview"

Relation = Literal["parent", "children", "ancestors", "descendants", "siblings"]


class UnknownCompetencyError(LookupError):
    """등록되지 않은 이름으로 도구를 호출했을 때."""


def tier_term(item: Mapping[str, Any]) -> str:
    """검사별 어휘를 섞지 않고 항목의 위계 이름을 돌려준다."""

    level = str(item.get("level", ""))
    if item.get("instrument") == VIDEO_INSTRUMENT_ID:
        return VIDEO_TIER_TERMS.get(level, level)
    return WRITTEN_TIER_TERMS.get(level, level)


def _items(snapshot: RegistrySnapshot) -> list[Mapping[str, Any]]:
    return list(snapshot.document.get("items", ()))


def _active_children(item: Mapping[str, Any], snapshot: RegistrySnapshot) -> list[str]:
    """등록되지 않은 자식 ID를 걸러낸다 (v1 `_active_children` 이식)."""

    return [
        item_id
        for item_id in item.get("children_ids", ())
        if item_id in snapshot.id_lookup
    ]


def _descendant_ids(
    target_id: str,
    snapshot: RegistrySnapshot,
    max_depth: int | None = None,
) -> list[str]:
    """v1 `competency_query._descendant_ids` 이식. 순환을 seen으로 막는다."""

    found: list[str] = []
    seen = {target_id}

    def visit(item_id: str, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        item = snapshot.id_lookup.get(item_id)
        if item is None:
            return
        for child_id in _active_children(item, snapshot):
            if child_id in seen:
                continue
            seen.add(child_id)
            found.append(child_id)
            visit(child_id, depth + 1)

    visit(target_id, 1)
    return found


def _ancestor_ids(target_id: str, snapshot: RegistrySnapshot) -> list[str]:
    """v1 `competency_query._ancestor_ids` 이식. 뿌리부터 가까운 순으로."""

    nearest_first: list[str] = []
    seen = {target_id}
    current = snapshot.id_lookup.get(target_id)
    while current is not None:
        parent_id = current.get("parent_id")
        if (
            not isinstance(parent_id, str)
            or parent_id not in snapshot.id_lookup
            or parent_id in seen
        ):
            break
        seen.add(parent_id)
        nearest_first.append(parent_id)
        current = snapshot.id_lookup[parent_id]
    return list(reversed(nearest_first))


def _sibling_ids(target: Mapping[str, Any], snapshot: RegistrySnapshot) -> list[str]:
    """같은 검사 안에서 부모가 같은 항목 (v1 `SIBLINGS` 분기 이식).

    검사 조건이 있어야 뿌리가 여럿인 이 레지스트리에서 필기 뿌리와 영상
    뿌리가 서로 형제가 되지 않는다.
    """

    raw_parent_id = target.get("parent_id")
    parent_id = raw_parent_id if raw_parent_id in snapshot.id_lookup else None
    return [
        item["id"]
        for item in _items(snapshot)
        if item["instrument"] == target["instrument"]
        and (
            item.get("parent_id")
            if item.get("parent_id") in snapshot.id_lookup
            else None
        )
        == parent_id
        and item["id"] != target["id"]
    ]


def _described(item_ids: Iterable[str], snapshot: RegistrySnapshot) -> list[dict[str, str]]:
    """이름만 돌려주지 않는다. 위계 이름을 함께 실어 추론 여지를 없앤다."""

    described: list[dict[str, str]] = []
    for item_id in item_ids:
        item = snapshot.id_lookup[item_id]
        described.append(
            {
                "name": item["name"],
                "level": item["level"],
                "tier": tier_term(item),
                "instrument": item["instrument_label"],
            }
        )
    return described


def _resolve(name: str, snapshot: RegistrySnapshot) -> Mapping[str, Any]:
    item = snapshot.lookup.get(name.strip())
    if item is None:
        raise UnknownCompetencyError(
            f"'{name}'은(는) 레지스트리에 등록된 이름이나 별칭이 아닙니다."
        )
    return item


def count_competencies(
    snapshot: RegistrySnapshot,
    *,
    level: str | None = None,
    instrument: str | None = None,
    analysis_included: bool | None = None,
) -> dict[str, Any]:
    """조건에 맞는 등록 항목 수를 센다.

    조건을 그대로 돌려주는 것은 4단계 judge를 위해서다.  숫자만 있으면
    "이 30은 무엇을 센 것인가"를 나중에 대조할 수 없다.
    """

    matched = [
        item
        for item in _items(snapshot)
        if (level is None or item["level"] == level)
        and (instrument is None or item["instrument_label"] == instrument)
        and (analysis_included is None or item["analysis_included"] is analysis_included)
    ]
    criteria = {
        "level": level,
        "instrument": instrument,
        "analysis_included": analysis_included,
    }
    return {
        "count": len(matched),
        "criteria": {key: value for key, value in criteria.items() if value is not None},
        "tier": WRITTEN_TIER_TERMS.get(level or "", None) if level else None,
        "names": [item["name"] for item in matched],
    }


def get_relations(
    snapshot: RegistrySnapshot,
    *,
    name: str,
    relation: Relation,
) -> dict[str, Any]:
    """한 항목의 구조적 관계를 돌려준다.

    ``children``과 ``descendants``를 **함께** 돌려주는 것이 `g06`의 교훈이다.
    "아래에 몇 개"라는 질문에는 정답이 둘이고, 하나만 주면 모델이 어느 쪽을
    센 것인지 사용자가 알 수 없다.
    """

    target = _resolve(name, snapshot)
    target_id = target["id"]

    if relation == "parent":
        parent_id = target.get("parent_id")
        related = [parent_id] if parent_id in snapshot.id_lookup else []
    elif relation == "ancestors":
        related = _ancestor_ids(target_id, snapshot)
    elif relation == "children":
        related = _active_children(target, snapshot)
    elif relation == "descendants":
        related = _descendant_ids(target_id, snapshot)
    elif relation == "siblings":
        related = _sibling_ids(target, snapshot)
    else:  # pragma: no cover - Literal이 막지만 런타임 호출도 있다
        raise ValueError(f"지원하지 않는 relation: {relation!r}")

    result: dict[str, Any] = {
        "target": {
            "name": target["name"],
            "level": target["level"],
            "tier": tier_term(target),
        },
        "relation": relation,
        "count": len(related),
        "items": _described(related, snapshot),
    }

    # "아래에 몇 개"는 직속과 후손 중 어느 쪽인지 질문만으로 확정되지 않는다.
    # 한쪽만 돌려주면 모델이 나머지를 모른 채 단정한다.
    if relation in {"children", "descendants"}:
        children = _active_children(target, snapshot)
        descendants = _descendant_ids(target_id, snapshot)
        if len(children) != len(descendants):
            result["also"] = {
                "direct_children": len(children),
                "all_descendants": len(descendants),
                "note": "직속 자식 수와 전체 후손 수가 다르다. 어느 쪽을 말하는지 밝혀야 한다.",
            }

    return result
