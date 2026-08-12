# ADR-001: Registry Writer의 grounding 계약과 실패 복구

**Status:** Accepted
**Date:** 2026-08-11
**Deciders:** Seokyoung Kim
**관련:** [PR #5](https://github.com/BKim2002/CC/pull/5), [PRD#1](../prd/ONE_TIME_LLM_GATEWAY_DUAL_WRITER_PRD.md) (§9 fallback 금지 규칙을 supersede), [PRD#2](../prd/ONE_TIME_REGISTRY_FIRST_INPUT_FLEXIBILITY_PRD.md)

## Context

### 현재 동작

`_registry_answer_is_valid`([competency_interpreter.py:1354](../../competency_interpreter.py))는 Registry Writer의 출력에 두 겹의 검사를 적용한다.

1. **완전 일치 검사** — `_registry_reference_framing_is_valid`([competency_interpreter.py:1190](../../competency_interpreter.py))가 `answer.count(reference) != 1`을 요구한다. `reference`는 `render_grounded_fallback(context)`가 만든 결정적 문자열이다. 인사도 미지원 혼합 주제도 없는 일반 `result` 모드에서는 prefix·suffix가 모두 비어야 하므로 **`answer == reference` 완전 일치**가 강제된다.
2. **사실 보존 검증** — `validate_grounded_answer`([competency_query.py:3047](../../competency_query.py))가 정의 원문 연결, 위계 원문, 항목 순서, 미허용 이름·수치 차단, 그룹/관계 사실 연결을 검사한다.

검사 1을 통과하지 못하면 2회 재시도 후 사용자는 `FIXED_FAILURE_MESSAGE`("답변을 만드는 중 문제가 발생했습니다")를 받는다. 즉 **Python이 이미 만들어 둔 정상 텍스트가 존재하는데도 장애 메시지가 나간다.**

### 두 PRD의 충돌

| 출처 | 규칙 |
|---|---|
| PRD#1 §9 Registry Writer 계약 | "render_grounded_fallback은 Writer prompt의 기준 답변이나 검증 보조로 사용할 수 있지만, **정상 사용자 응답을 Python이 대신 작성하는 공개 fallback으로 사용하지 않는다.** Writer가 최종적으로 실패하면 공통 고정 장애 안내문을 사용한다." |
| PRD#2 §13 Scope Writer 복구와 fallback | "**정상 primary 응답은 항상 생성형이며 fallback만 결정적이다.**" |
| PRD#2 구현 범위 | Scope Writer는 "두 번 실패한 경우에도 fixed failure가 아닌 scope fallback으로 끝난다." |

PRD#2가 Scope 경로에 도입한 "primary는 생성형, fallback은 결정적, fallback도 정상 종료"는 PRD#1의 절대 금지를 실질적으로 대체한 정책이다. 그러나 이 정책이 Scope 경로에만 적용되고 Registry 경로에는 적용되지 않아 **동일 시스템 안에 두 개의 실패 정책이 공존**한다.

### 결정적 렌더러의 품질

`_render_grounded_fallback_unbounded`([competency_query.py:2844](../../competency_query.py))의 출력은 기계적 나열이 아니라 완성된 한국어 산문이다.

```
현재 조건에 맞는 등록 항목은 총 3개입니다.

1. 책임감
책임감의 등록된 정의는 다음과 같습니다: …
책임감의 등록된 직접 하위 항목: …
```

공개 품질에 문제가 없다. 따라서 "Python 텍스트는 사용자에게 보여줄 수 없다"는 PRD#1의 전제는 사실이 아니다.

### 사실 보존 검증기의 실제 완성도

`validate_grounded_answer`는 이미 다음을 강제한다. 재서술을 허용해도 사실이 훼손되지 않는다.

- `exact_definitions`의 정의 원문이 해당 항목 블록 안에 그대로 존재
- `hierarchy_text` 원문 그대로 존재
- 항목 라벨이 등록 순서대로, 리스트 마커에 anchored
- `allowed_names` 밖의 등록명 추가 차단
- **`allowed_numbers` 밖의 모든 수치 차단** (리스트 서수 제외)
- 그룹 label↔개수, 관계 대상↔이름↔개수↔관계어 연결, parent/children/path/analysis 연결

**남는 구멍:** 등록명도 수치도 포함하지 않는 해석성 산문(예: "이 역량은 협업과 밀접한 관련이 있습니다")은 통과한다. 완전 일치 검사가 지금까지 이 구멍을 대신 막고 있었다.

### 파생 문제

- **재시도 피드백 부재** — `write_registry_answer`([competency_interpreter.py:1473](../../competency_interpreter.py))는 `model_input`을 루프 밖에서 한 번 만들고 재시도 시 동일 프롬프트를 재전송한다. Scope Writer는 `retry_issue`를 매 시도 주입한다([competency_interpreter.py:1681](../../competency_interpreter.py)).
- **guidance 전부 아니면 전무** — `_guidance_is_safe`([competency_interpreter.py:1222](../../competency_interpreter.py))가 `(`·`)`·`:` 전면 금지, 모든 문장에 제안 토큰 요구, 등록명 전면 금지 등 누적 규칙을 적용한다. 하나라도 걸리면 검증된 `[등록 정보]` 섹션까지 버려지고 장애 메시지가 나간다.
- **예산 고갈** — `MAX_LLM_API_CALLS_PER_TURN = 3`. Gateway가 1회 재시도하면(2슬롯) semantic selector(1슬롯) 후 잔여 0이 되어 `write_registry_answer`의 `while attempts < 2 and budget.remaining_calls > 0`이 한 번도 돌지 않고 곧장 장애로 간다.

## Decision

**Registry 경로를 PRD#2의 정책("primary는 생성형, fallback은 결정적, fallback도 정상 종료")에 정렬하고, 완전 일치 대신 사실 보존을 grounding 계약으로 삼는다.**

네 부분으로 구성한다.

### D1. 완전 일치 검사를 구조 검사로 교체

`_registry_reference_framing_is_valid`의 `answer.count(reference) == 1` 요구를 제거한다. 대신 다음 구조만 검사한다.

- `acknowledge_greeting=false`이면 인사 문장 부재, `true`이면 인사가 선두 한 문장이고 `_registry_greeting_is_safe`를 통과
- `scope_topic=None`이면 `[지원 범위]` 섹션 부재, 아니면 말미 한 섹션이고 `validate_registry_scope_note`를 통과
- 그 사이 본문은 `validate_grounded_answer`가 사실 권위를 갖는다

`validate_grounded_answer`는 변경하지 않는다. 이미 필요한 보증을 전부 제공한다.

### D2. 해석성 산문 차단기 신설

D1이 여는 유일한 구멍을 막는 검사를 추가한다. **검증된 원문 span(등록 정의, `hierarchy_text`)을 제거한 나머지 전체**에 적용한다.

- 정의성 단정 금지 — `scope_response.definition_claim_is_absent` 재사용
- 개인화·전문조언·채용 판단 토큰 금지 — `_PERSONALIZATION_RISK_TOKENS`로 추출해 `_guidance_is_safe`와 공유
- 등록명을 문장 주어로 쓰는 형태 금지 — `context.allowed_names` + `은|는` (결정적 렌더러는 의/에는/이(가)만 사용하므로 비용 없음)

**구현 시 정정 두 가지.**

1. `manifest_claims_are_safe`는 **제외한다.** 금지 토픽이 `"약"`, `"외부"`, `"개인"`, `"문서"` 같은 맨 substring이라 `"요약하면 총 3개입니다"` 같은 정상 registry 산문이 거부된다. Scope 산문 전용으로 튜닝된 검사이며 장르가 다르다.
2. "anchored 사실 줄 식별" 대신 **검증된 span 제거 후 나머지 전체 검사**를 택했다. 줄 단위 분류는 `책임성은 리더십의 핵심입니다`처럼 등록명으로 시작하는 조작 문장이 사실 줄로 오분류되어 검사를 우회한다. span 제거 방식은 이 우회가 구조적으로 불가능하고 구현도 더 단순하다.

검증: 8개 intent의 결정적 reference가 전부 통과하고 6개 조작 문장이 전부 거부됨을 회귀 테스트로 고정했다.

### D3. Registry fallback 도입 (Scope와 대칭)

Writer가 2회 실패하면 `FIXED_FAILURE_MESSAGE` 대신 `render_grounded_fallback(context)`를 **정상 종료**로 공개한다.

- 새 `response_mode="registry_fallback"` 추가 (`Literal["llm", "scope_fallback", "registry_fallback", "failure"]`)
- `_record_runtime_metric("registry_fallback", mode)`로 운영 관측
- `FIXED_FAILURE_MESSAGE`는 **실제 시스템 실패에만** 남긴다: Gateway 자체 실패, grounding context 구성 실패, checkpoint 실패, 안전 route 부재

fallback 텍스트는 이미 `validate_grounded_answer`를 정의상 통과하므로 추가 검증 비용이 없다.

### D4. guidance 부분 성공

`[일반 활용 제안]` 섹션이 `_guidance_is_safe`에 걸리면 **섹션만 버리고** 검증된 `[등록 정보]` 섹션을 정상 공개한다. `_guidance_is_safe`의 규칙 자체는 완화하지 않는다 — 안전망 강도를 유지한 채 실패 반경만 줄인다.

### D5. 예산 재배분

D3이 도입되면 Writer 실패가 더 이상 치명적이지 않으므로 예산 상향 없이 해결된다. `MAX_LLM_API_CALLS_PER_TURN = 3`을 유지하고, 슬롯 부족으로 Writer가 실행되지 못한 경우 곧바로 D3의 fallback으로 정상 종료한다. semantic 경로의 구조적 장애가 사라진다.

## Options Considered

### Option A: 결정적 fallback만 추가 (D3 단독)

| Dimension | Assessment |
|---|---|
| 복잡도 | 낮음 |
| 작업량 | 1~2시간 |
| 위험 | 매우 낮음 |
| PRD 정합성 | PRD#2 정책의 단순 확장 |

**Pros:** 장애 메시지 노출이 즉시 사라진다. 정책 변경이 아니라 누락 보완이다.
**Cons:** Writer는 여전히 완전 일치만 통과하므로 LLM 호출이 사실상 복사기로 남는다. 비용·지연 그대로. 실패율이 그대로라 fallback이 상시 경로가 되어 "primary는 생성형"이 이름뿐이 된다.

### Option B: 완전 일치 모드는 LLM 생략

| Dimension | Assessment |
|---|---|
| 복잡도 | 낮음 |
| 작업량 | 2~3시간 |
| 위험 | 낮음 (기능적), 높음 (정책적) |
| PRD 정합성 | PRD#1 명시 규칙을 정면으로 뒤집음 |

**Pros:** 가장 흔한 경로에서 1회 호출과 지연이 사라지고 실패 모드 자체가 소멸한다.
**Cons:** 모든 레지스트리 답변이 byte 단위로 동일한 결정적 텍스트가 된다. 문맥·어조 적응이 영구히 불가능해진다. PRD#1을 명시적으로 폐기해야 한다.

### Option C: 사실 보존 검증 + fallback ✅ **채택**

| Dimension | Assessment |
|---|---|
| 복잡도 | 중간 |
| 작업량 | 6~10시간 (당초 추정 대비 하향 — 검증기가 이미 존재) |
| 위험 | 중간 — D2가 유일한 신규 안전 로직 |
| PRD 정합성 | PRD#2 정책과 완전 일치, PRD#1의 fallback 금지만 폐기 |

**Pros:** Writer가 처음으로 실질적 가치를 갖는다(문맥 적응, 자연스러운 연결, 인사·혼합 주제 통합). 실패 시에도 사용자는 항상 정상 답변을 받는다. 사실 보증 강도는 `validate_grounded_answer`가 그대로 유지한다.
**Cons:** D2를 설계·검증해야 한다. 재서술을 허용하는 만큼 회귀 테스트 표면이 넓어진다. Writer 출력이 결정적이지 않으므로 기존 테스트 double(`_reference_answer`)을 재작성해야 한다.

## Trade-off Analysis

**핵심 판단:** 완전 일치 검사는 "사실 훼손 방지"라는 목적에 대해 **과도하게 강한 수단**이다. 목적에 정확히 대응하는 수단(`validate_grounded_answer`)이 이미 존재하고 이미 실행 중이며, 완전 일치는 그 위에 얹힌 중복 방어다. 이 중복이 막아 주던 유일한 잔여 위험이 해석성 산문(D2)이고, 그것을 막는 데 필요한 부품도 이미 세 곳(`_guidance_is_safe`, `_definition_claim_is_absent`, `manifest_claims_are_safe`)에 구현되어 있다.

**Option A vs C:** A는 증상(장애 메시지)만 없앤다. 원인(LLM이 복사기)은 남고, fallback이 상시 경로가 되면서 "생성형 primary"라는 설계 의도가 사문화된다. C는 원인을 제거한다.

**Option B vs C:** B가 비용·지연·실패율에서 우월하지만 대가로 시스템의 생성 능력을 영구 포기한다. 인사 반영과 미지원 혼합 주제 처리는 이미 구현된 기능인데, B는 그 경로만 LLM을 남기는 이원 구조를 만든다. 일관성 손실이 절감분을 상쇄한다.

**D2의 잔여 위험:** anchored 사실 줄 식별이 부정확하면 정상 문장을 사실 줄로 오인해 검사를 건너뛸 수 있다. 보수적으로 설계한다 — **사실 줄로 확신되지 않는 모든 줄을 산문으로 간주**해 검사를 적용한다. 오탐(정상 문장 거부)은 D3의 fallback이 흡수하므로 사용자 피해가 없고, 미탐(해석성 산문 통과)만이 실제 위험이다.

## Consequences

**쉬워지는 것**

- Registry와 Scope 경로가 동일한 실패 정책을 갖는다. 신규 route 추가 시 따를 기준이 하나가 된다.
- `FIXED_FAILURE_MESSAGE`가 실제 시스템 장애의 신호가 된다. 현재는 Writer 서식 실패와 DB 장애가 구분되지 않는다.
- 예산 상향 없이 semantic 경로의 구조적 장애가 해소된다.
- Writer prompt를 개선할 여지가 생긴다. 완전 일치 하에서는 프롬프트 튜닝이 무의미했다.

**어려워지는 것**

- Writer 출력이 비결정적이 되어 `tests/test_competency_interpreter.py`의 `_reference_answer` 기반 double(52개 테스트 중 다수)이 "재서술된 유효 출력"을 만들어야 한다.
- D2의 오탐/미탐 경계를 지속적으로 관찰해야 한다.
- fallback 비율이 새 SLO 지표가 된다.

**재검토 시점**

- `registry_fallback` 비율이 전체 registry 턴의 20%를 넘으면 D2가 과도하거나 프롬프트가 부적절한 것이므로 재검토한다.
- D2 미탐이 실제로 관측되면 Option B(완전 일치 모드 LLM 생략)로 후퇴하는 것을 재검토한다.

## Action Items

1. [ ] `docs/adr/` 도입에 맞춰 루트의 `ONE_TIME_*_PRD.md` 2건을 `docs/prd/`로 이동하고 PRD#1 `:316-318`을 본 ADR로 supersede 표기 — *Phase 2 (R9)*
2. [x] **D1** `_registry_reference_framing_is_valid` → `_registry_structure_is_valid`로 교체. 인사는 첫 문장, scope note는 마커 이후로 위치만 검사하고 본문은 `validate_grounded_answer`가 소유
3. [x] **D2** `_registry_prose_is_safe` 신설 (검증 span 제거 후 3개 검사). 위 "구현 시 정정" 참조
4. [x] **D3** `response_mode="registry_fallback"` 추가, `validate_registry_answer_node`가 단일 공개 지점으로 복구 담당, `route_after_registry_validation`을 `route_after_scope_writer`와 동일 형태로 정렬
5. [x] **D4** `_guidance_partial_answer` 추가, `response_mode="guidance_partial"`. 살린 본문은 일반 `result` 계약으로 재검증
6. [x] **D5** 예산 소진 시 fallback 직행을 graph-level 테스트로 고정 (Gateway 재시도 + semantic selector = 3슬롯)
7. [x] **D6** `_registry_writer_input(..., retry_issue=...)`를 매 시도 재구성
8. [x] **D7** 기존 `_reference_answer` double 20개는 유지하고 재서술 계열 테스트를 병행 추가 (회귀 안전망 보존). 사실 변조 4종·해석성 산문 6종 거부를 회귀로 고정
9. [x] `WEB_CHAT_README.md`의 실패 정책·메트릭·테스트 목록 서술 갱신

**부수 수정:** `_is_greeting_text`를 분리하며 영어 인사 토큰을 word-anchored로 교정했다. 기존 `"hi"` 맨 substring은 `"hierarchy"`에도 반응했다.
