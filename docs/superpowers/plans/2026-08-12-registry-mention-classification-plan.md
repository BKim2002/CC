# 구현 계획 — 미해결 mention 3단 판정

**설계:** [2026-08-12-registry-mention-classification-design.md](../specs/2026-08-12-registry-mention-classification-design.md)
**기준선:** 874 passed
**대상 파일:** `competency_query.py`, `competency_interpreter.py`, `tests/test_competency_query.py`, `tests/test_competency_interpreter.py`, `WEB_CHAT_README.md`

각 단계는 끝난 시점에 스위트가 통과해야 한다. 순서가 중요하다 — 2단계(순수 리팩터)가 4단계(동작 변경)의 안전망이다.

---

## 1단계 — 근접 매칭 기본기 (동작 변경 없음)

`competency_query.py`

- `import difflib` 추가
- 상수 추가 (기존 `MAX_*` 블록 옆, `competency_query.py:29-36`)
  ```python
  NEAR_MATCH_CUTOFF = 0.6
  MAX_NEAR_MATCH_CANDIDATES = 3
  MIN_NEAR_MATCH_LENGTH = 2
  ```
- `NormalizationIssueCode`에 `NEAR_MATCH_TARGET = "near_match_target"` 추가
- `_near_registered_names(mention: str, snapshot: RegistrySnapshot) -> list[str]`
  - `mention`이 `MIN_NEAR_MATCH_LENGTH` 미만이면 `[]`
  - `difflib.get_close_matches(mention, list(snapshot.lookup), n=MAX_NEAR_MATCH_CANDIDATES * 2, cutoff=NEAR_MATCH_CUTOFF)`
  - 매치된 라벨을 `snapshot.lookup[label]["name"]`으로 환원, item id 기준 중복 제거
  - `MAX_NEAR_MATCH_CANDIDATES`개로 절단
  - `n`을 2배로 뽑는 이유: 별칭과 정식명이 같은 항목을 가리키면 중복 제거 후 개수가 줄어든다
- `_result_with_near_match(mention, candidates, rule_ids) -> RegistryNormalizationResult`
  - `_result_with_issue(NormalizationIssueCode.NEAR_MATCH_TARGET, question, options=..., rule_ids=...)` 형태
  - question: `f"'{mention}'과(와) 가장 가까운 등록 역량을 찾았습니다. 어느 것을 뜻하는지 선택해 주세요."`
  - options: `NormalizationOption(label=name, description="현재 레지스트리의 정식 이름")`

이 단계는 아직 아무 데서도 호출되지 않는다.

**검증**
- `_near_registered_names` 단위 테스트: 오타 6종 발화 / 비오타 6종 침묵 (설계 문서의 측정표)
- 1글자 mention은 `[]`
- 후보 최대 3개
- 별칭 매치가 정식명으로 환원되고 중복 제거됨 (`표현능력_의사표현` → `의사표현`)
- **속성 테스트:** 로드된 레지스트리의 서로 다른 정식명 두 개가 `NEAR_MATCH_CUTOFF` 안에 들어오면 실패
- 스위트: 874 + 신규, 기존 테스트 변경 없음

---

## 2단계 — unknown 출처 분리 (순수 리팩터, 동작 변경 없음)

`competency_query.py` · `normalize_registry_query`

- `unknown_mentions` 단일 목록을 둘로 분리
  - `raw_unknown_mentions` ← `_unknown_mentions_from_raw` (`competency_query.py:1716`)
  - `draft_unknown_mentions` ← 초안 loop의 `else` 분기 (`competency_query.py:1749-1759`)
- 두 목록 각각 `_unique` + `_resolve_target_mention` 재확인 필터 적용 (현재 `competency_query.py:1761-1765`가 하는 일)
- 기존 사용처는 **합집합으로 그대로 유지**
  ```python
  unknown_mentions = _unique([*raw_unknown_mentions, *draft_unknown_mentions])
  ```
  `_intent_from_raw`의 `has_unknown_targets`와 `competency_query.py:2032` 분기 모두 이 합집합을 계속 사용한다.

**검증**
- 스위트 통과, **기존 테스트 파일 수정 0줄**. 이것이 순수 리팩터임의 증명이다.
- 통과하지 않으면 진행하지 말고 원인을 먼저 찾는다.

---

## 3단계 — 원문 유래 unknown에 근접 매칭 적용

`competency_query.py:2032` 분기를 `raw_unknown_mentions`만 보도록 좁히고 순서를 정한다.

```
if raw_unknown_mentions:
    if grounded_semantic_query is not None:
        return semantic request          # 기존
    near = _near_registered_names(raw_unknown_mentions[0], snapshot)
    if near:
        return _result_with_near_match(raw_unknown_mentions[0], near, (*rule_ids, "near_match_suggestion"))
    return _result_with_unregistered(raw_unknown_mentions, ...)   # 기존
```

이 시점에서 `unknown_mentions` 합집합은 `_intent_from_raw`에만 남는다.

**검증**
- 신규: `책임쎵이 뭐야?` → CLARIFICATION, 후보에 `책임성`
- 보존: `혁신성이 뭐야?` → UNREGISTERED
- 보존: `그릿의 정의 알려줘` → UNREGISTERED
- 신규: semantic 신호가 있으면 near-match보다 우선
- 스위트 통과

---

## 4단계 — 초안 유래 unknown 처리 변경 (실제 수정)

두 가지를 동시에 바꾼다. 분리하면 중간 상태가 일관되지 않는다.

**4a. `_intent_from_raw` 입력 교정** (`competency_query.py:2006`)

```python
has_unknown_targets=bool(raw_unknown_mentions),   # 합집합이 아니라 원문 유래만
```

`has_unknown_targets`는 `ITEM_LOOKUP` fallback 하나에만 영향한다(`competency_query.py:1542`). 버려질 초안 unknown이 이 값을 켜면 catalog 질문이 item lookup으로 오분류된다. **이 줄을 빠뜨리면 4단계가 동작하지 않는다.**

**4b. 초안 unknown 분기 추가** — 3단계의 `if raw_unknown_mentions:` 블록 바로 뒤

```
if draft_unknown_mentions:
    for mention in draft_unknown_mentions:
        near = _near_registered_names(mention, snapshot)
        if near:
            return _result_with_near_match(mention, near, (*rule_ids, "near_match_suggestion"))
    rule_ids.append("draft_mention_dropped")
    # 아무것도 반환하지 않고 계속 진행 — 미등록 판정으로 승격하지 않는다
```

`raw_unknown_mentions`가 비어 있을 때만 이 블록에 도달한다(3단계 블록이 먼저 반환하므로). 설계의 권위 비대칭 규칙이 제어 흐름으로 자연히 표현된다.

이제 `unknown_mentions` 합집합 변수는 쓰이지 않으므로 제거한다.

**검증**
- 회귀: 측정한 FRAGILE 10건이 초안 명사구 복사 상태에서 올바른 plan을 낸다
  - catalog 6: `역량 목록 좀 알려줘`, `역량 종류 좀 알려줘`, `전체 역량 목록 좀 알려줘`, `등록된 역량 다 보여줘`, `어떤 역량들이 있어?`, `역량 리스트 뽑아줘`
  - hierarchy 2: `위계 구조 알려줘`, `역량 트리 보여줘`
  - aggregate 1: `검사별 역량 수 알려줘`
  - filter 1: `분석에 포함되는 역량만`
- 신규: `책임쎵 알려줘` → CLARIFICATION, 후보에 `책임성`
- 신규: 원문 유래 unknown이 있으면 초안 근접 매칭이 가로채지 않음
- 보존: `책임성 목록 보여줘` → 원문 스캔이 이름을 찾아 정상 처리
- 보존: 미등록 2건 계속 UNREGISTERED
- 스위트 통과

---

## 5단계 — Gateway 프롬프트 위생

`competency_interpreter.py` · `_gateway_prompt`

registry_query 초안 지침 단락에 한 줄 추가:

> `target_mentions`에는 등록 역량의 이름처럼 보이는 표현만 넣으세요. 목록·종류·전체·위계·구조·개수처럼 범위나 형식을 가리키는 말은 target이 아니라 constraint입니다.

**검증**
- 프롬프트에 해당 지침이 포함되는지 확인하는 테스트 (기존 프롬프트 테스트 옆)
- 스위트 통과

---

## 6단계 — 문서

- `WEB_CHAT_README.md`: 미해결 mention 3단 판정과 오타 근접 안내를 동작 설명에 반영
- 메트릭 문단에 `near_match_target` 추가
  - **코드 변경 없음.** `competency_interpreter.py:617`이 모든 issue의 `code.value`를 기록하고, `_SAFE_METRIC_COMPONENT`가 `near_match_target`을 허용한다. 문서만 갱신한다.

**검증**
- 스위트 통과
- `compileall`

---

## 릴리스 전 별도 확인

설계 성공 기준 4번은 이 저장소에서 검증할 수 없다. 합성 레지스트리는 이름 6개, 프로덕션은 약 52개다.

**프로덕션 레지스트리 스냅샷으로 1단계의 속성 테스트를 한 번 실행할 것.** 정식명 간 근접 충돌이 나오면 `NEAR_MATCH_CUTOFF`를 올리거나 충돌 쌍을 예외 처리해야 한다. 이 확인 전에는 cutoff 0.6이 프로덕션에서 안전하다고 단정할 수 없다.

---

## 되돌리기

각 단계가 독립 커밋이다. 4단계에서 문제가 생기면 3단계까지 되돌려도 시스템은 일관된다(근접 매칭이 원문 경로에만 적용된 상태). 1~2단계는 동작 변경이 없어 언제든 안전하다.
