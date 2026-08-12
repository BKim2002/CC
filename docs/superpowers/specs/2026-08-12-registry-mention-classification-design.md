# 미해결 mention 3단 판정 설계

**작성일:** 2026-08-12
**상태:** 승인됨
**관련:** [ADR-001](../../adr/ADR-001-registry-writer-grounding-contract.md), [ADR-002](../../adr/ADR-002-out-of-scope-template.md), PR #5 (main 병합 완료)

## 문제

프로덕션에서 `역량 종류 좀 알려줘`, `역량 목록 좀 알려줘`, `전체 역량 목록 좀 알려줘`가 모두 미등록 안내로 거부된다.

> '…'은(는) 현재 등록된 정식 이름이나 별칭으로 확인되지 않습니다. 정확한 역량명 또는 관찰한 행동 특징을 알려 주세요.

## 진단

의도 분류는 정상이다. **대상 추출이 정답을 덮어쓴다.**

같은 질문을 Gateway 초안의 `target_mentions`만 바꿔 정규화기에 통과시킨 결과:

| 입력 | `target_mentions` | 결과 |
|---|---|---|
| 역량 목록 좀 알려줘 | `["역량 목록"]` | unregistered_target |
| 역량 목록 좀 알려줘 | `[]` | plan · catalog_query |

세 겹이 겹쳤다.

1. **Fast path가 좁다.** `competency_query.py:520`은 `전체\s*역량\s*목록` + 제한된 존댓말 접미의 `fullmatch`를 요구한다. `좀`이 끼거나 `전체`가 없거나 `종류`면 실패한다.
2. **Gateway는 규칙대로 행동했다.** 프롬프트가 "사용자가 실제로 쓴 target의 짧은 표현을 보존하라"고 지시한다. 일반 범위 명사가 target이 아니라는 말은 어디에도 없다.
3. **stopword가 정확히-일치다.** `competency_query.py:1106`에 `목록`과 `역량`이 각각 있지만 합성어 `역량 목록`은 없다. 커버리지가 조합적으로 늘어난다.

## 측정

24개 입력을 두 가지 Gateway 행동(명사구 복사 / 복사 안 함)으로 각각 실행했다.

| 판정 | 수 |
|---|:-:|
| SAFE | 12 |
| FRAGILE (복사할 때만 깨짐) | **10** |
| BROKEN | 2 |
| LEAK (미등록명 누출) | 0 |

FRAGILE 10건은 catalog 6, hierarchy 2, aggregate 1, filter 1로 **4개 계열**에 걸쳐 있다. 전부 동일 서명이다: 복사만 없으면 올바른 plan이 나온다.

BROKEN 2건은 이번 사안이 아니다.

- `필기 역량만 보여줘` — 합성 레지스트리에 검사가 `합성 검사` 하나뿐이라 발생한 **테스트 픽스처 문제**. 프로덕션에는 필기 역량검사가 있다.
- `책임성의 상위 항목 알려줘` — `ambiguous_relation` 되물음. 거부가 아니라 선택지를 주는 의도된 동작.

두 가지 추가 사실이 설계를 결정했다.

- **등록된 이름은 초안 없이도 원문에서 찾는다.** `책임성 목록 보여줘`는 초안 mention이 없어도 `target_ids=['responsibility']`가 나온다(`_raw_registered_targets`). 초안 mention은 등록명에 대해 아무것도 기여하지 않는다.
- **진짜 미등록 판정도 원문에서 나온다.** `혁신성이 뭐야?`는 초안 mention 없이도 UNREGISTERED다(`_unknown_mentions_from_raw`의 `뭐야` 패턴).

## 결정

### 미해결 mention 3단 판정

```
초안 mention이 레지스트리 항목으로 해석되지 않음
   ├─ 근접 매칭 후보 있음 → CLARIFICATION  "책임성을 찾으시나요?" (최대 3개)
   └─ 후보 없음          → DROP, 원문 해석으로 진행
                           (초안 mention은 단독으로 미등록 판정을 만들지 않는다)

원문 유래 unknown (_unknown_mentions_from_raw)
   → semantic → near-match → UNREGISTERED 순서
```

기대 동작:

| 입력 | 현재 | 결정 후 |
|---|---|---|
| FRAGILE 10건 | UNREGISTERED | 올바른 plan |
| `책임쎵 알려줘` | UNREGISTERED | CLARIFICATION · 후보 `책임성` |
| `혁신성이 뭐야?` | UNREGISTERED | UNREGISTERED (원문 유래) |
| `그릿의 정의 알려줘` | UNREGISTERED | UNREGISTERED (원문 유래) |
| `책임성 목록 보여줘` | 정상 | 정상 (원문 스캔) |

### 채택하지 않은 대안

**어휘 목록 정교화** — `_is_registry_field_stopword`를 토큰 단위 판정으로 바꾸는 안. 조합 폭발을 선형으로 줄이지만 `종류`·`전체`·`구조`·`트리`·`역량들`·`검사별`을 계속 추가해야 하는 목록이 남는다. **이번 버그를 만든 것과 같은 종류의 물건**이라 채택하지 않는다. 근접 매칭이 "이름 같은가"를 판정하므로 이 목록 없이도 문제가 해결된다.

**Gateway 프롬프트만 수정** — 모델 순응은 보장되지 않는다. 그것을 전제할 수 없어서 정규화기가 존재한다. 단독 수단으로는 불가하며, 위생 조치로만 병행한다.

### 이번 범위에서 제외

`혁신성 목록 보여줘`는 근접 후보가 없어 DROP되어 **조용히 전체 목록**이 나온다.

이는 새 규칙의 퇴행이 아니다. 초안 mention이 없는 실행에서 이미 `catalog_query`가 나오는 기존 동작이며, Gateway의 과잉 mention이 우연히 가려주고 있었다. 닫으려면 위에서 기각한 어휘 목록이 필요하고, 프로덕션에서 관측된 적이 없다. **실제 관측 시 데이터를 갖고 재검토한다.**

## 근접 매칭 임계값

`difflib`(표준 라이브러리)로 오타 6종 / 비오타 6종을 측정했다.

| cutoff | 12개 중 정답 |
|---|:-:|
| 0.5 | 12 |
| **0.6** | **12** |
| 0.7 | 9 |
| 0.8 | 8 |

두 집단이 넓게 갈린다.

| 오타 (발화해야 함) | 유사도 | 비오타 (침묵해야 함) | 유사도 |
|---|:-:|---|:-:|
| 책임쎵 → 책임성 | 0.67 | 혁신성 | 0.33 |
| 책잉성 → 책임성 | 0.67 | 회복탄력성 | 0.25 |
| 인지렵 → 인지력 | 0.67 | 위계 구조 | 0.22 |
| 의사표혅 → 의사표현 | 0.75 | 역량 목록 | 0.00 |
| 책임 → 책임성 | 0.80 | 그릿 | 0.00 |

**0.6**을 채택한다. 최고 오탐(0.33)과 최저 정탐(0.67) 사이 중앙이라 여유가 크다.

**측정 한계:** 합성 레지스트리는 이름이 6개, 프로덕션은 약 52개다. 이름이 늘면 충돌 확률이 오른다. 아래 속성 테스트로 방어하며, **릴리스 전 프로덕션 레지스트리 스냅샷으로 한 번 실행해야 한다.**

## 변경 지점

### `competency_query.py`

1. **`NormalizationIssueCode`에 `NEAR_MATCH_TARGET = "near_match_target"` 추가.**
   `UNKNOWN_TARGET`은 `UnregisteredTargetResult`의 validator가 독점해 재사용할 수 없다. `AMBIGUOUS_TARGET` 재사용도 가능하나, 별도 코드를 두면 `normalization_issue` 메트릭에서 "사용자 오타"와 "이름 중의성"이 분리되어 운영 신호로 쓸 수 있다.

2. **상수 추가** (기존 `MAX_*` 옆)

   ```python
   NEAR_MATCH_CUTOFF = 0.6
   MAX_NEAR_MATCH_CANDIDATES = 3   # manifest의 max_semantic_candidates와 일치
   MIN_NEAR_MATCH_LENGTH = 2       # 1글자는 무엇에나 걸린다
   ```

3. **`_near_registered_names(mention, snapshot) -> list[str]`**
   `MIN_NEAR_MATCH_LENGTH` 미만이면 빈 목록. `difflib.get_close_matches`를 `snapshot.lookup`(정식명 + 별칭) 키에 적용하고, 매치된 라벨을 정식명으로 환원해 item id 기준 중복 제거 후 `MAX_NEAR_MATCH_CANDIDATES`개로 자른다.

4. **`_result_with_near_match(mention, candidates, rule_ids)`**
   `NormalizationOutcome.CLARIFICATION` 결과를 만들고 후보를 `NormalizationOption(label=정식명, description="현재 레지스트리의 정식 이름")`으로 싣는다. 인터프리터의 `_clarification_context`가 옵션 라벨을 `validate_registry_names`로 재검증하는데, 후보가 실제 등록명이므로 통과한다.

5. **`normalize_registry_query` 분기 재구성**

   `unknown_mentions` 단일 목록을 둘로 나눈다.

   - `raw_unknown_mentions` ← `_unknown_mentions_from_raw` (`competency_query.py:1716`)
   - `draft_unknown_mentions` ← 초안 loop (`competency_query.py:1749-1759`)

   **두 출처의 우선순위:** `raw_unknown_mentions`가 비어 있지 않으면 그쪽을 먼저 처리한다. 모듈이 선언한 권위 비대칭("the active snapshot and explicit raw-query spans come first")을 따른다. 초안 근접 매칭이 원문 유래 미등록 판정을 가로채지 않는다.

   초안 loop 직후: `raw_unknown_mentions`가 비어 있을 때만, `draft_unknown_mentions`를 초안 순서대로 훑어 근접 후보가 있는 첫 항목에서 `CLARIFICATION`을 반환한다. 하나도 없으면 전부 버리고 진행한다. 미등록 판정으로 승격하지 않는다.

   기존 unknown 분기(`competency_query.py:2032`)는 `raw_unknown_mentions`만 본다. 순서는 **semantic → near-match → unregistered**. 행동 묘사 신호가 철자 유사도보다 강하므로 semantic이 앞선다.

   초안 unknown을 일찍 버리면 `_intent_from_raw`의 `has_unknown_targets`가 `False`가 된다. 이것이 측정표의 "복사 안 함" 열을 재현하는 지점이다.

   새 rule id: `draft_mention_dropped`, `near_match_suggestion`

### `competency_interpreter.py`

6. **`_gateway_prompt`에 한 줄 추가**

   > `target_mentions`에는 등록 역량의 이름처럼 보이는 표현만 넣으세요. 목록·종류·전체·위계·구조·개수처럼 범위나 형식을 가리키는 말은 target이 아니라 constraint입니다.

   안전망이 있으므로 발동 빈도를 낮추는 위생 조치다. 단독으로 신뢰하지 않는다.

## 오류 처리

근접 매칭은 순수 문자열 연산이며 `difflib`는 표준 라이브러리라 새 의존성이 없다. 새 실패 모드도 없다. 예외가 발생하면 `normalize_registry_query_node`의 기존 `try/except`가 `fixed_failure_message`로 보낸다.

`CLARIFICATION`은 이미 존재하는 출력 경로이므로 Writer·grounding 검증·SSE 계약에 변경이 없다.

## 테스트

| 종류 | 내용 |
|---|---|
| 회귀 | 측정한 FRAGILE 10건 — 초안이 명사구를 복사해도 올바른 plan |
| 보존 | `혁신성이 뭐야?`, `그릿의 정의 알려줘` → UNREGISTERED |
| 보존 | `책임성 목록 보여줘` → 원문 스캔이 이름을 찾아 정상 처리 |
| 신규 | `책임쎵 알려줘` → CLARIFICATION, 후보에 `책임성` |
| 신규 | 후보 최대 3개, `MIN_NEAR_MATCH_LENGTH` 미만 mention은 매칭 없음 |
| 신규 | semantic 신호가 있으면 near-match보다 우선 |
| 신규 | 원문 유래 unknown이 있으면 초안 근접 매칭이 가로채지 않음 |
| 신규 | `near_match_target` 메트릭 기록 |
| 속성 | 로드된 레지스트리의 서로 다른 정식명 두 개가 `NEAR_MATCH_CUTOFF` 안에 들어오면 실패 |

속성 테스트는 합성 레지스트리(6개)에서는 자명하게 통과한다. 프로덕션 스냅샷을 픽스처로 넣으면 충돌을 잡는다. **릴리스 전 프로덕션 레지스트리로 실행할 것.**

## 성공 기준

1. 측정한 FRAGILE 10건이 Gateway 행동과 무관하게 올바른 plan을 낸다.
2. 미등록명 2건(`혁신성`, `그릿`)이 계속 UNREGISTERED로 남는다.
3. 오타 1건(`책임쎵`)이 근접 후보 안내를 받는다.
4. 프로덕션 레지스트리에서 정식명 간 근접 충돌이 없다.
5. 기존 테스트 874건이 통과한다.
