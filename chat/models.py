"""역할별 모델 선택.

v1 `llm_gateway.ModelRole`(ENTRY/ANSWER)을 이식하고 ``JUDGE``를 더했다.
judge가 하는 일은 추출이라 큰 모델이 필요 없고, 이것이 10초 예산을 지키는
가장 큰 레버다 -- 3단계 실측에서 답변만으로 이미 최대 10.09초가 나왔다.

호출당 타임아웃도 역할별로 나눈다.  v1의 기본값은 30초였는데(`llm_gateway.py`
`create_chat_model`), 턴 총예산이 10초인 상황에서 개별 호출 상한이 30초면
상한이 없는 것과 같다.

Responses API를 쓰는 이유는 3단계에 기록되어 있다 -- gpt-5.6-luna는
chat/completions에서 함수 도구를 받지 않는다.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-5.6-luna"

# judge 기본값을 답변 모델과 같이 둔다.  더 작은 모델이 10초 예산의 주요
# 레버지만, 이 계정에서 무엇이 쓸 수 있는지 확인하지 않은 ID를 기본값으로
# 박지 않는다.  OPENAI_JUDGE_MODEL 로 갈아끼우는 것이 7단계 작업이다.
DEFAULT_JUDGE_MODEL = DEFAULT_MODEL

LEGACY_MODEL_ENV = "OPENAI_MODEL"


class ModelRole(str, Enum):
    ENTRY = "entry"
    ANSWER = "answer"
    JUDGE = "judge"
    APPEAL = "appeal"


# 대화 맥락 상한을 역할별로 나눈다.  단위는 턴이 아니라 **메시지**다 --
# 주고받기로는 절반이다(v1 `RECENT_MESSAGE_LIMIT = 12`와 같은 단위).
#
# 항소심만 방향이 반대다.  맥락이 길수록 "앞에서 역량을 얘기했으니 관련
# 있다"며 거절을 뒤집는 쪽으로 기운다.  나머지는 길수록 이해가 좋아진다.
# 값을 하나로 묶어 두면 나중에 항소심만 줄이는 것이 분기 추가가 된다
# (REBUILD_PLAN 3.7).
MESSAGE_LIMIT: Mapping[ModelRole, int] = {
    ModelRole.ENTRY: 12,
    ModelRole.ANSWER: 12,
    ModelRole.JUDGE: 0,  # judge는 대화를 보지 않는다 (3.8)
    ModelRole.APPEAL: 6,
}


_ROLE_ENV: Mapping[ModelRole, str] = {
    ModelRole.ENTRY: "OPENAI_ENTRY_MODEL",
    ModelRole.ANSWER: "OPENAI_ANSWER_MODEL",
    ModelRole.JUDGE: "OPENAI_JUDGE_MODEL",
    ModelRole.APPEAL: "OPENAI_APPEAL_MODEL",
}

# 합이 턴 예산 아래여야 한다.  judge는 짧은 구조화 출력이라 상한이 낮아도
# 된다.  넘으면 부분 답변을 내보내는 대신 실패로 처리한다.
_ROLE_TIMEOUT: Mapping[ModelRole, float] = {
    ModelRole.ENTRY: 8.0,
    ModelRole.ANSWER: 20.0,
    ModelRole.JUDGE: 8.0,
    ModelRole.APPEAL: 8.0,
}


def _nonblank(name: str, environ: Mapping[str, str]) -> str | None:
    value = environ.get(name, "").strip()
    return value or None


def selected_model_name(
    role: ModelRole,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """역할별 → legacy → 기본값 순으로 모델을 고른다."""

    source = os.environ if environ is None else environ
    checking = role in {ModelRole.JUDGE, ModelRole.APPEAL}
    default = DEFAULT_JUDGE_MODEL if checking else DEFAULT_MODEL
    return (
        _nonblank(_ROLE_ENV[role], source)
        or _nonblank(LEGACY_MODEL_ENV, source)
        or default
    )


def create_model(
    role: ModelRole,
    *,
    model_name: str | None = None,
    timeout: float | None = None,
) -> ChatOpenAI:
    """SDK가 조용히 턴 예산을 넘기지 못하게 재시도를 끄고 상한을 건다."""

    selected = (model_name or "").strip() or selected_model_name(role)
    return ChatOpenAI(
        model=selected,
        max_retries=0,
        timeout=timeout if timeout is not None else _ROLE_TIMEOUT[role],
        use_responses_api=True,
    )
