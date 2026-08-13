"""chat.prompt — 레지스트리를 프롬프트 블록으로 만드는 계층.

v1에는 대응하는 테스트가 없다.  등가물이었던 name_catalog/semantic_catalog는
스냅샷이 만들어 주는 문자열이라 별도 검증 대상이 아니었다.

여기서 지키려는 계약은 둘이다.

1. **전문이 실린다.**  v2가 질의 유형 없이 답할 수 있는 근거가 컨텍스트에
   모든 필드가 있다는 것이다.  한 필드라도 빠지면 그 필드를 묻는 질문이
   조용히 답을 못 하게 된다 (REBUILD_PLAN 2.2).
2. **레지스트리 블록이 항상 맨 앞이다.**  프롬프트 캐시 접두를 안정시킨다.
   1단계 실측 적중률이 96~98%였고, 순서가 뒤집히면 그것이 사라진다.
"""

from __future__ import annotations

from chat.prompt import SYSTEM_PROMPT, build_system_message, render_registry
from chat.registry import build_registry_snapshot

SOURCE_HASH = "b" * 64


def _item(
    item_id: str,
    name: str,
    *,
    aliases: list[str] | None = None,
    parent_name: str | None = None,
    parent_id: str | None = None,
    children: list[str] | None = None,
    definition: str | None = None,
    definition_status: str = "explicit",
    analysis_included: bool = False,
    notes: list[str] | None = None,
) -> dict:
    return {
        "id": item_id,
        "name": name,
        "aliases": aliases or [],
        "instrument": "synthetic",
        "instrument_label": "합성 검사",
        "level": "L2" if children else "L3",
        "path": [name],
        "parent_name": parent_name,
        "parent_id": parent_id,
        "children": children or [],
        "children_ids": [f"synthetic:{child}" for child in (children or [])],
        "definition": definition,
        "definition_status": definition_status,
        "analysis_included": analysis_included,
        "notes": notes or [],
        "source_section": "테스트",
    }


def _snapshot():
    document = {
        "schema_version": "1.0",
        "source": {"file": "synthetic.md", "sha256": SOURCE_HASH, "encoding": "utf-8"},
        "rules": {"synthetic": "합성 레지스트리 규칙"},
        "validation": {"status": "passed", "counts": {"total": 2}},
        "items": [
            _item(
                "synthetic:parent",
                "부모 역량",
                aliases=["부모 별칭"],
                children=["자식 역량"],
                definition="부모 역량의 정의",
                analysis_included=True,
                notes=["부모 비고"],
            ),
            _item(
                "synthetic:child",
                "자식 역량",
                parent_name="부모 역량",
                parent_id="synthetic:parent",
                definition=None,
                definition_status="not_provided",
            ),
        ],
    }
    return build_registry_snapshot(
        {
            "id": 11,
            "source_filename": "synthetic.md",
            "source_sha256": SOURCE_HASH,
            "schema_version": "1.0",
            "registry_json": document,
            "item_count": 2,
        }
    )


def test_render_includes_every_field_the_model_may_be_asked_about() -> None:
    rendered = render_registry(_snapshot())

    assert "[부모 역량] (별칭: 부모 별칭)" in rendered
    assert "id=synthetic:parent" in rendered
    assert "level=L2" in rendered
    assert "검사=합성 검사" in rendered
    assert "부모=부모 역량" in rendered
    assert "자식=자식 역량" in rendered
    assert "분석포함=예" in rendered
    assert "분석포함=아니오" in rendered
    assert "정의: 부모 역량의 정의" in rendered
    assert "비고: 부모 비고" in rendered


def test_render_marks_a_missing_definition_instead_of_dropping_it() -> None:
    """정의가 없는 항목을 조용히 비우면 모델이 '정의가 있는데 못 봤다'와

    '정의가 등록되지 않았다'를 구분하지 못한다.  후자는 사용자에게 알려야
    하는 사실이다.
    """

    rendered = render_registry(_snapshot())

    assert "정의: (없음 — definition_status=not_provided)" in rendered


def test_render_reports_the_item_count_it_actually_rendered() -> None:
    rendered = render_registry(_snapshot())

    assert "항목 수: 2" in rendered
    assert rendered.count("id=synthetic:") == 2


def test_registry_block_is_delimited_and_leads_the_system_message() -> None:
    message = build_system_message(_snapshot())

    assert message.startswith("<registry>")
    assert message.index("</registry>") < message.index(SYSTEM_PROMPT.strip()[:20])


def test_tier_glossary_is_optional_and_never_precedes_the_registry() -> None:
    """블록 표지로 확인한다.

    용어 자체("하위요인")로 확인하면 안 된다 -- 3단계에서 시스템 프롬프트가
    도구 사용 지침의 예시로 같은 단어를 쓰기 시작했다.
    """

    with_glossary = build_system_message(_snapshot(), tier_glossary=True)
    without_glossary = build_system_message(_snapshot(), tier_glossary=False)

    assert "<tier_vocabulary>" in with_glossary
    assert "L3      = 하위요인" in with_glossary
    assert "<tier_vocabulary>" not in without_glossary
    assert with_glossary.startswith("<registry>")
    assert without_glossary.startswith("<registry>")


def test_tier_glossary_is_generated_from_the_same_dict_the_tools_use() -> None:
    """모델이 읽는 위계 이름과 도구가 돌려주는 이름이 갈라지면

    도구로 티어 오류를 고칠 수 없게 된다 (3단계의 g06 대응).
    """

    from chat.tools import VIDEO_TIER_TERMS, WRITTEN_TIER_TERMS

    glossary = build_system_message(_snapshot(), tier_glossary=True)

    for level, term in {**WRITTEN_TIER_TERMS, **VIDEO_TIER_TERMS}.items():
        assert f"{level:<8}= {term}" in glossary


def test_system_prompt_states_the_grounding_promise_and_the_prescription_limit() -> None:
    """3.4의 근거 약속과 2.5단계에서 b06을 해소한 처방 금지 줄.

    둘 다 프롬프트에만 존재하는 계약이라 코드가 강제하지 않는다.  지워지면
    조용히 사라지므로 여기서 붙잡아 둔다.
    """

    assert "근거해서만 답한다" in SYSTEM_PROMPT
    assert "방법·절차·해석 기준·권고는 제시하지 않는다" in SYSTEM_PROMPT
