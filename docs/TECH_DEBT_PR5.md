# PR #5 기술 부채 우선순위와 정리 계획

**작성일:** 2026-08-11
**대상:** [PR #5](https://github.com/BKim2002/CC/pull/5) (draft) · `agent/future-proof-competency-registry`
**설계 근거:** [ADR-001](adr/ADR-001-registry-writer-grounding-contract.md)

우선순위 = (Impact + Risk) × (6 − Effort). 각 축 1~5.

**진행 상황 (2026-08-12):** Phase 1·2 완료 (`c6b0a49`, `2d2fc0d`, PR #5로 머지). 이후 범위 밖 응답을 템플릿으로 전환했고 ([ADR-002](adr/ADR-002-out-of-scope-template.md)), 미해결 mention 3단 판정과 Phase 3를 PR #6에 담았다. **Phase 1~3 전부 완료.** 테스트 740 → 920.

---

## 1. 항목 목록과 점수

### 아키텍처 부채 — ADR-001 실행 항목

| ID | 항목 | Impact | Risk | Effort | 점수 | 근거 |
|---|---|:-:|:-:|:-:|:-:|---|
| **D3** | `registry_fallback` 도입 | 4 | 5 | 2 | **36** | Writer 서식 실패가 곧 사용자 장애 메시지. Python이 만든 정상 텍스트가 있는데도 버려진다. fallback 텍스트는 정의상 `validate_grounded_answer`를 통과하므로 추가 검증 비용 없음 |
| **D2** | 해석성 산문 차단기 | 3 | 5 | 2 | **32** | D1의 전제조건이자 유일한 신규 안전 로직. 미탐 시 사실 왜곡이 사용자에게 도달 — 시스템의 핵심 보증 위반. 줄 앵커링 정규식이 이미 존재해 Effort 하향 |
| **D4** | guidance 부분 성공 | 2 | 3 | 1 | **25** | 검증된 `[등록 정보]`까지 함께 버려지는 전부-아니면-전무 구조. 규칙 완화 없이 실패 반경만 축소 |
| **D5** | 예산 시나리오 테스트 | 2 | 3 | 1 | **25** | Gateway 재시도 + semantic = 구조적 장애. D3이 실질 해소하므로 남는 작업은 회귀 테스트 |
| **D1** | 완전 일치 → 구조 검사 | 4 | 3 | 3 | **21** | LLM 호출이 복사기로 낭비되고 프롬프트 개선이 무의미. 검사 교체 자체는 단순하나 D2 없이 단독 적용 불가 |
| **D6** | Writer 재시도 피드백 | 2 | 2 | 1 | **20** | 동일 프롬프트 재전송으로 슬롯만 소모. Scope Writer의 `retry_issue` 패턴 복사 |
| **D7** | 테스트 double 20개 재작성 | 2 | 4 | 3 | **18** | D1의 실제 비용. 부실하면 재서술 허용의 회귀를 잡지 못한다 |

### 코드 부채 — 리뷰 지적사항

| ID | 항목 | Impact | Risk | Effort | 점수 | 근거 |
|---|---|:-:|:-:|:-:|:-:|---|
| **R6** | `_public_stream_event` fallthrough | 1 | 2 | 1 | **15** | `_SSE_EVENT_NAMES`에 항목 추가 시 조용히 `error`로 매핑. 현재는 무해 |
| **R7** | `RegistrySnapshot.__post_init__` KeyError | 1 | 2 | 1 | **15** | 직접 생성 경로에서 `item["id"]` 노출. 프로덕션 경로는 검증됨 |
| **R8** | `test_chat_js` Node hard fail | 1 | 2 | 1 | **15** | CI 이미지에 Node 없으면 skip이 아닌 red |
| **R10** | PR 본문 stale | 2 | 1 | 1 | **15** | "General Writer", "450 passed" — 리뷰어가 읽는 문서. 실제는 Scope Writer, 740 passed |
| **R5a** | 폴링 지수 백오프 | 1 | 2 | 1 | **15** | 10ms 고정 → 10~50ms 백오프. 저비용 완화 |
| **R1** | `normalize_registry_query` 분해 | 4 | 3 | 4 | **14** | 550줄 단일 함수, early return 20개. 신규 제약 추가 시 삽입 위치 판단이 어려워 실제 개발 속도를 떨어뜨린다 |
| **R2** | `instrument_labels=instrument_ids` | 2 | 1 | 2 | **12** | 필드명과 실제 값 불일치. 동작은 정상(이중 lookup) |
| **R3** | `target_names` 중복 계산 | 1 | 1 | 1 | **10** | 1766의 값이 1772~1778에서만 쓰이고 2161에서 재계산 |
| **R4** | `astream` → `ainvoke` | 1 | 1 | 1 | **10** | **D1 이후 재판단 필요** — 재서술이 허용되면 실제 증분 delta를 낼 여지가 생겨 `astream` 유지가 옳을 수 있다 |
| **R9** | `ONE_TIME_*_PRD.md` 루트 누적 | 1 | 1 | 1 | **10** | 1,626줄. `docs/` 도입으로 이동 경로 확보됨 |
| **R5b** | 폴링 근본 해결 | 1 | 2 | 4 | **6** | **권장하지 않음** — 아래 참조 |

---

## 2. 리팩터링 가능 여부 탐색 결과

### ✅ 저항 없음

**D2 (해석성 산문 차단기)** — 당초 "신규 검증기 설계"로 추정했으나 부품이 전부 존재한다.

- 줄 분류: `_anchored_line_contains_facts`의 앵커 정규식 `^\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?{label}`을 그대로 재사용
- 등록명 주어 서술 금지: `_guidance_is_safe`의 `(?:성|력|능력)(?:은|는)\s+` 패턴
- 정의성 단정 금지: `scope_response._definition_claim_is_absent`
- 개인화·전문조언 토큰: `_guidance_is_safe`의 `guidance_risk_tokens`
- manifest 밖 기능 주장: `scope_response.manifest_claims_are_safe`

신규 코드는 줄 분류기 ~10줄과 조립부뿐이다. **보수적 설계 원칙: 사실 줄로 확신되지 않는 모든 줄을 산문으로 간주해 검사를 적용한다.** 오탐은 D3의 fallback이 흡수하므로 사용자 피해가 없고 미탐만이 실제 위험이다.

**D3 (registry_fallback)** — `write_scope_answer`([competency_interpreter.py:1715](../competency_interpreter.py))에 동일 패턴이 완성되어 있다. `response_mode` Literal 확장, 실패 분기 전환, 메트릭 추가로 끝난다.

**R6·R7·R3·R8·R9** — 전부 국소 변경. 의존관계 없음.

### ⚠️ 조건부

**D1 (완전 일치 제거)** — 코드 변경은 작지만 **테스트 20개가 결합**되어 있다. `_reference_answer` double이 프롬프트에서 reference를 추출해 그대로 반환하는 구조([tests/test_competency_interpreter.py:254](../tests/test_competency_interpreter.py))라, 재서술을 허용하면 이 double이 "통과하는 유일한 출력"을 만들어 내지 못한다.

권장 접근: `_reference_answer`를 유지하되(여전히 유효한 출력이어야 함) **별도의 `_rephrased_answer` double을 추가**해 두 계열을 병행 검증한다. 기존 20개를 폐기하지 않으므로 회귀 안전망이 유지된다.

**R1 (normalizer 분해)** — 가능하지만 자명하지 않다. 함수 전체가 `rule_ids`, `target_ids`, `accepted_constraints`, `relation`, `related_tier`, `instrument_ids`, `hierarchy_tiers`, `scope`, `filter_flags`를 순차 축적하는 파이프라인이고 각 단계가 `RegistryNormalizationResult`로 조기 이탈한다.

실현 가능한 형태: 가변 컨텍스트 dataclass + 각 단계가 `RegistryNormalizationResult | None`을 반환하는 스테이지 함수열.

```
_NormalizationContext(raw_query, draft, snapshot, previous, rule_ids, ...)
stages = (_stage_targets, _stage_relation, _stage_instrument,
          _stage_tier, _stage_scope, _stage_filters, _stage_intent)
for stage in stages:
    if (early := stage(ctx)) is not None:
        return early
return _build_plan(ctx)
```

**단, 이 리팩터링은 순서 의존성을 코드 구조에 고정시킨다.** 현재 순서가 우연이 아니라 의도인지(예: instrument 모호성이 tier 모호성보다 먼저 보고되어야 하는지) 확인하지 않은 채 분해하면 회귀 위험이 있다. 104개 테스트가 안전망이지만 순서 자체를 검증하는 테스트가 있는지 먼저 확인해야 한다. **머지 후 별도 PR 권장.**

### ❌ 권장하지 않음

**R5b (폴링 근본 해결)** — 리뷰에서 과대평가한 항목이다. `_thread_execution`(sync, [:2105](../competency_interpreter.py))과 `_async_thread_execution`(async, [:2139](../competency_interpreter.py))이 **동일한 `threading.Lock`을 공유**한다. `/api/chat`과 `/api/chat/stream`이 같은 `thread_id`에 대해 상호 직렬화되어야 하므로 `asyncio.Lock` 교체는 정합성을 깬다.

`await asyncio.to_thread(entry.lock.acquire)`는 [:2047](../competency_interpreter.py)의 주석이 명시한 취소 안전성을 잃는다 — 취소된 코루틴이 워커 스레드에 blocking acquire를 남긴다.

올바른 해결은 sync/async 겸용 프리미티브 도입이며 현재 부하에서 정당화되지 않는다. **R5a(지수 백오프)만 채택하고 근본 해결은 보류한다.**

---

## 3. 단계별 실행 계획

### Phase 1 — 머지 전, 사용자 대면 동작 (필수)

ADR-001의 정책 비대칭을 `main`에 들이지 않는 것이 목적. PR이 아직 draft이므로 같은 PR에서 처리한다.

| 순서 | 항목 | 이유 |
|---|---|---|
| 1 | **D3** registry_fallback | 독립적. 먼저 넣으면 이후 모든 변경의 실패 안전망이 된다 |
| 2 | **D2** 해석성 산문 차단기 | D1의 전제조건 |
| 3 | **D1** 완전 일치 → 구조 검사 | D2 완료 후에만 |
| 4 | **D7** 테스트 double 병행 계열 추가 | D1과 동일 커밋 |
| 5 | **D4** guidance 부분 성공 | 독립적 |
| 6 | **D6** 재시도 피드백 | 독립적 |
| 7 | **D5** 예산 시나리오 테스트 | D3 이후 동작 확인 |

**순서가 중요합니다.** D1을 D2·D3보다 먼저 넣으면 검증 완화 상태에서 안전망도 fallback도 없는 구간이 생깁니다.

### Phase 2 — 머지 전, 저비용 동봉 ✅ 완료

전부 Effort 1이고 의존관계가 없으므로 같은 PR에 묶는다.

- [x] **R6** fallthrough를 `if event == "error"`로 좁히고 미처리 event는 `None` 반환
- [x] **R7** `_derive_id_lookup` 분리, ID 누락 시 `RegistryValidationError`
- [x] **R3** 중복 `target_names` 제거
- [x] **R8** Node 부재 시 `pytest.skip`
- [x] **R9** `ONE_TIME_*_PRD.md` → `docs/prd/` 이동, PRD#1 §9에 supersede 블록 추가
- [x] **R5a** 폴링 지수 백오프 (10ms → 100ms 상한), sync/async 락 공유 이유를 주석으로 고정
- [ ] **R10** PR 본문 갱신 — push 이후

### Phase 3 — 완료 (PR #6에 동봉)

- [x] **R1** normalizer 분해 — 599 → 408줄. 계획의 전면 stage 분해 대신 입출력 면이 좁은 세 블록(filter·instrument·relation)만 순수 함수로 추출했다. 결합도를 먼저 측정한 결과 `rule_ids`가 574줄에 걸쳐 36회, `target_ids`가 520줄에 걸쳐 17회 읽혀, context 객체가 20개 필드에 이른다. 복잡도를 줄이는 게 아니라 옮기는 셈이라 채택하지 않았다. 호출 순서가 그대로라 분기 우선순위는 구성상 보존되며, 테스트 파일 수정 0줄이 그 증거다.
- [x] **R2** `instrument_labels` → `instrument_refs`. 이 필드는 label만 담은 적이 없다 — 정규화기가 id를 넣고 `validate_parsed_query`가 양쪽을 해석한다. 호출부 5곳, 전부 내부.
- [x] **R4** `astream` **유지**. 코드 변경 없이 근거만 기록했다.

**R4 판단 근거.** 증분 delta 공개는 D1과 무관하게 불가능하다. 모든 grounding 검사가 전문을 읽는다 — 필수 이름과 그 순서, 총계, truncation 안내, 인사와 `[지원 범위]`의 위치. 부분 접두사는 검증할 수 없고, 검증 전 공개는 이 노드가 버퍼링하는 이유 자체를 무너뜨린다.

그렇다면 `ainvoke`로 단순화할 수 있는가? 없다. `astream`이 주는 것은 증분 길이 가드다. 폭주 생성을 끝까지 버퍼링한 뒤 거부하는 대신 조기에 포기한다. 바꾸면 이걸 잃고 얻는 게 없다.

---

## 4. 예상 작업량

| Phase | 항목 수 | 예상 |
|---|:-:|---|
| Phase 1 | 7 | 8~12시간 (D7이 절반) |
| Phase 2 | 7 | 2~3시간 |
| Phase 3 | 3 | 6~10시간 |

---

## 5. 관측 지표

D3 도입 후 다음을 `runtime_metric_snapshot()`으로 추적한다.

- `registry_fallback.*` 비율 — 전체 registry 턴의 **20% 초과 시** D2 과도 또는 프롬프트 부적절로 판단해 재검토 (ADR-001 재검토 조건)
- `scope_fallback.*` 대비 비율 — 두 경로의 실패율이 크게 다르면 한쪽 검증이 불균형
- `fixed_failure.all` — D3 이후에는 **실제 시스템 장애만** 남아야 한다. 이 값이 유의미하게 남으면 실패 분기 전환이 불완전한 것
