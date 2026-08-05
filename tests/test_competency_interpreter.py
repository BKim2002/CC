"""역량명 추출과 설정 기본값에 대한 핵심 회귀 테스트."""

from __future__ import annotations

import pytest

from competency_interpreter import (
    extract_registered_names,
    selected_model_name,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("수인지력의 정의를 알려줘.", ["수인지력"]),
        ("표현능력_의사표현", ["의사표현"]),
        ("대인매력_매력도", ["매력도"]),
        ("인지력과 수인지력", ["인지력", "수인지력"]),
        ("수인지력과 인지력", ["수인지력", "인지력"]),
    ],
)
def test_longer_registered_name_wins_only_when_spans_overlap(
    query: str,
    expected: list[str],
) -> None:
    assert extract_registered_names(query) == expected


def test_blank_model_setting_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    assert selected_model_name() == "gpt-5.4-mini"
