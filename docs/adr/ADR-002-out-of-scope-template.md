# ADR-002: 범위 밖 응답을 템플릿으로 고정

**Status:** Accepted
**Date:** 2026-08-11
**Deciders:** Seokyoung Kim
**관련:** [ADR-001](ADR-001-registry-writer-grounding-contract.md), [PR #5](https://github.com/BKim2002/CC/pull/5)

## Context

ADR-001은 registry 경로에 "결정적 Python 출력은 요청 경로에서 재검증하지 않는다"는 원칙을 세웠다. `render_grounded_fallback`은 검증된 context에서만 만들어지므로 재검증이 단위 테스트 assertion을 사용자 턴으로 옮기는 일이 되기 때문이다.

Scope 경로는 이 원칙에서 빠져 있었다. `write_scope_answer`는 Scope Writer가 두 번 실패하면 `scope_fallback_draft()`로 결정적 문자열을 만든 뒤 **그것을 다시 `validate_scope_draft`로 검증**했고, 그 검증이 실패하면 `FIXED_FAILURE_MESSAGE`로 끝났다.

동시에 out-of-scope 응답의 성격을 다시 보면, 생성형이 기여하는 바가 거의 없다.

- 출력은 세 문장 고정 구조다: 주제 반영 → 범위 설명 → 역량 질문 유도.
- 세 문장 모두 `_controlled_scope_vocabulary`의 허용 어휘 안에서만 쓸 수 있다.
- 모델이 실제로 결정하는 유일한 값은 `ScopeTopic.summary`(≤160자)이고, 그마저 `sanitize_topic_summary`가 짧은 명사구로 줄이거나 10개 category 중 6개에서는 고정 라벨로 대체한다.

즉 자유 문장 3개를 생성한 뒤 정규식 다발로 검증하는 구조인데, 통과할 수 있는 문장 공간은 이미 템플릿에 가깝다.

## Decision

**`mode == "out_of_scope"`에서 Scope Writer 호출을 제거하고 템플릿을 정상 경로로 삼는다.**

### D1. 템플릿 경로

`scope_template_answer()`가 `out_of_scope_draft()`의 세 필드를 이어 붙여 최종 문자열을 만든다. `write_scope_answer`는 out_of_scope일 때 이 문자열을 바로 공개하고 `response_mode="scope_template"`으로 정상 종료한다. LLM 호출·재시도·요청 경로 검증이 모두 사라진다.

### D1a. 인정 문장은 한계를 먼저 밝힌다

첫 문장은 질문을 되풀이하는 `{주제}에 관해 궁금하신 점을 이해했습니다` 대신 한계를 바로 밝히는 `{주제}에 관한 질문은 제가 알려드릴 수 없습니다`를 쓴다. 되풀이는 사용자가 이미 아는 내용을 한 문장 더 읽게 만든다.

이에 맞춰 두 번째 문장도 조정했다. 첫 문장이 거절을 말하므로 기존 boundary(`…직접 답하거나 판단하지 않습니다`)는 같은 말을 두 번 하게 된다. 이제 boundary는 거절이 아니라 **이유**를 말한다: `이 챗봇은 등록된 역량 정보만 다루는 범위로 한정되어 있습니다`.

세 문장의 역할이 거절 → 이유 → 유도로 갈라져 중복이 없다. 영어도 같은 구조로 맞췄다.

`_acknowledgement_shape_is_safe`에 이 형태를 추가하고 `관한`·`questions`를 허용 어휘에 넣었다. 형태가 늘어도 `_all_direct_answers_are_absent`와 `definition_claim_is_absent`는 그대로 적용되므로, 한계를 밝히면서 답까지 흘리는 문장은 여전히 거부된다.

### D2. redirect 4종 회전

고정 문장 하나면 반복 사용자에게 경직되어 보이므로 registry redirect를 한국어·영어 각 4종 두고 턴마다 회전한다.

**인덱스는 `(len(messages) - 1) // 2 % 4`다.** `messages`는 턴당 정확히 2개(Human+AI) 늘어나므로 `len % 4`는 `{1, 3}` 두 값만 만들어 4종 중 2종이 영원히 사용되지 않는다. 실측으로 확인했고 회귀 테스트로 고정했다.

`random.choice()`는 쓰지 않는다. 한 번의 노드 실행 안에서는 공개 delta와 checkpoint의 `AIMessage`가 같은 변수에서 나오므로 무작위여도 일치하지만, 노드 재실행이나 checkpoint 재개에서 다른 문자열이 나오면 `run_competency_stream`의 일치 불변식이 깨진다.

### D3. meta는 현행 유지

`greeting`, `thanks`, `farewell`, `bot_identity`, `capability_help`는 계속 Scope Writer가 생성한다. 유출 위험이 낮고 자연스러움이 사용자에게 체감되는 구간이며, 실패 시 `scope_fallback` 복구도 그대로 둔다.

### D4. 혼합 의도 꼬리말도 현행 유지

`[지원 범위]` 꼬리말은 Registry Writer가 작성하므로 `validate_registry_scope_note`의 검증이 계속 필요하다.

### D5. 검증기는 삭제하지 않고 테스트 oracle로 이동

`validate_scope_draft`의 out_of_scope 분기는 **남긴다.** 요청 경로에서 호출하지 않을 뿐, 모든 category × 언어 × variant 조합의 템플릿 출력이 이 검증기를 통과함을 테스트가 확인한다. 검증을 없애는 것이 아니라 실행 시점을 요청 경로에서 테스트로 옮기는 것이다.

## Consequences

**쉬워지는 것**

- out-of-scope 턴의 LLM 호출이 1~2회 줄어 Gateway 1회만 남는다.
- 검증 오탐이 `FIXED_FAILURE_MESSAGE`로 이어지던 경로가 사라진다.
- Scope 경로가 ADR-001의 원칙에 정렬된다.
- 적대적 topic summary가 출력에 영향을 줄 수 없음을 문자열 동일성으로 증명할 수 있다.

**어려워지는 것**

- 범위 밖 안내의 표현이 4종으로 고정된다. 늘리려면 각 문장이 `_controlled_scope_vocabulary`를 통과해야 한다.
- `scope_first_failure`·`scope_retry_success`·`scope_fallback` 메트릭이 이제 meta 턴만 설명한다.

**남는 dead branch**

`_scope_writer_for(model_name, mode)`의 `mode == "out_of_scope"` → `OutOfScopeResponseDraft` 분기는 요청 경로에서 도달하지 않는다. 함수가 mode→schema 매핑이고 테스트가 2-인자 시그니처에 의존하므로 남겨 두었다.

## Action Items

1. [x] `scope_template_answer` / `out_of_scope_draft` / redirect 4종 + 회전
2. [x] `write_scope_answer`의 out_of_scope 분기 분리, `response_mode="scope_template"`
3. [x] `_redirect_variant` — 턴 인덱스 기반
4. [x] meta fallback의 증명 가능한 중복 재시도 제거 (meta draft는 `category`/`summary`를 쓰지 않아 2차 시도가 같은 문자열을 재검증)
5. [x] 모든 category × 언어 × variant의 템플릿 계약 회귀 테스트, 적대적 요약 무해화 테스트
6. [x] 기존 out_of_scope writer 테스트를 meta 경로로 이전 (취소·거부·재시도·fallback·문맥)
7. [x] `WEB_CHAT_README.md` 갱신
