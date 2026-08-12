# PRD: 모든 입력을 해석하는 LLM Gateway와 이중 LLM Writer

> 이 문서는 메인 Codex 세션이 별도의 대화 맥락 없이 구현을 이어가기 위한
> 단일 인계 문서다. 구현자는 이 문서를 처음부터 끝까지 읽고, 현재 작업 트리와
> 테스트를 확인한 뒤 작업한다.

- 상태: 사용자 승인 완료
- 작성일: 2026-08-07
- 대상 저장소: C:\Users\ksy0823\.vscode\LangGraph
- 구현 상태: 이 문서 작성 시점에는 소스 구현을 시작하지 않음
- 변경 성격: 기존 역량 질의 기능을 보존하면서 대화 입구와 출구를 재설계

## 1. 문서 사용 규칙

1. 이 문서를 이번 변경의 요구사항 단일 원본으로 사용한다.
2. 작업 전 git status와 현재 diff를 확인하고 기존의 관련 없는 변경을 보존한다.
3. 현재 코드가 이 문서의 사실 설명과 달라졌다면 최신 코드를 먼저 확인한다.
   요구사항을 바꿀 정도의 충돌만 사용자에게 질문한다.
4. 구현과 테스트가 끝나기 전에 완료했다고 보고하지 않는다.
5. 기존 파이프라인과 새 파이프라인을 장기적으로 함께 유지하지 않는다.

완료 기준: 아래 인수 조건과 전체 회귀 테스트가 모두 통과하고, 제거 대상으로
지정된 기존 분기와 사용되지 않는 코드가 남아 있지 않아야 한다.

## 2. 문제 정의

현재 competency_interpreter.py의 정상 진입점은 interpret_query다. 이 노드는
Python으로 정확한 등록명 추출과 detect_deterministic_query를 먼저 수행하고,
그 결과가 없을 때만 llm_interpret_query를 호출한다. 따라서 자연어 해석 책임이
Python 규칙과 LLM에 나뉘어 있고, 새로운 질문 유형을 추가할 때 여러 분기와
호환 경로를 함께 이해해야 한다.

출구도 혼합되어 있다. 레지스트리 조회 결과 일부는 write_streamed_answer의
LLM이 문장화하지만, 후보 제시, 도움말, 재질문, 알 수 없는 역량, 범위 밖 안내는
Python 함수가 최종 AIMessage를 직접 작성한다.

사용자가 원하는 결과는 다음과 같다.

- 모든 요청을 입구 LLM이 먼저 이해한다.
- 일반적인 인사와 제한된 간단한 질문에도 자연스럽게 답한다.
- 챗봇의 주 목적인 역량 질문으로 무리 없이 유도한다.
- 모든 정상 사용자 응답은 목적별 LLM Writer가 작성한다.
- 역량 이름, 정의, 위계, 관계, 개수는 계속 레지스트리와 결정적 Python
  실행기만 진실의 원천으로 사용한다.
- 일반적인 정상 경로의 LLM 호출 수는 2회, 의미 검색을 포함한 경로까지
  최대 3회를 넘지 않는다.

## 3. 설계 원칙

### 3.1 LLM과 Python의 책임

LLM은 다음을 담당한다.

- 사용자 의도와 대화 유형 해석
- 역량 질의를 기존 RegistryQueryPlan 계열로 구조화
- 일반 대화 응답 작성
- 검증된 레지스트리 결과를 자연어로 작성

Python은 다음을 담당한다.

- LLM 구조화 출력의 스키마 검증
- LLM이 반환한 enum을 LangGraph 노드에 연결하는 얇은 dispatch
- 레지스트리 이름과 stable ID 검증
- 질의 계획 검증 및 결정적 조회와 계산
- 역량 답변과 후보명의 사실 검증
- 턴당 LLM 호출 예산 관리
- 최종 실패 시 고정 장애 안내문 반환

Python은 정상 경로에서 정규식, 키워드, 정확한 역량명 포함 여부로 사용자
의도를 분류하지 않는다. LangGraph conditional edge 함수는 필요하지만,
검증된 LLM route를 노드 이름에 매핑하는 역할만 수행한다.

### 3.2 레지스트리 우선

LLM은 역량의 이름, 정의, 위계, 관계, 개수를 만들어내지 않는다.
competency_query.py와 competency_registry.py의 검증 및 조회 계층은 계속
권위 있는 원천이다. LLM이 생성한 역량 답변은 사용자에게 공개되기 전에
검증되어야 한다.

### 3.3 두 개의 전문 Writer

- Registry Writer: 역량 조회 결과, 후보, 역량 관련 재질문과 미등록 안내
- General Writer: 인사, 가벼운 대화, 챗봇 소개와 사용법, 간단한 일반 개념,
  범위 밖 안내와 일반적인 재질문

두 Writer 모두 같은 ANSWER_MODEL 설정을 사용할 수 있지만 prompt, 입력
컨텍스트, 검증과 스트리밍 정책은 분리한다.

## 4. 범위

### 4.1 이번 구현에 포함

- 단일 LLM Gateway 노드와 엄격한 구조화 출력
- 기존 RegistryQueryPlan 계열과의 연결
- Registry Writer와 General Writer
- capability manifest
- 최근 대화 문맥을 사용하는 후속 질문
- 혼합 입력과 모호한 역량 후보 처리
- 턴당 최대 3회의 실제 LLM API 요청 예산
- 목적별 스트리밍과 checkpoint 정책
- 관련 기존 분기와 사용되지 않는 코드 제거
- 자동 테스트와 WEB_CHAT_README.md 갱신

### 4.2 이번 구현에서 제외

- 로그인과 사용자 계정
- 사용자별 대화 소유권
- 여러 과거 대화의 목록, 선택, 검색과 삭제 UI
- 대화 보존 기간 및 자동 삭제 정책
- 웹 검색, 날씨, 뉴스 등 최신 정보 도구
- 의료, 법률, 금융 전문 상담
- 개인 역량 점수 추정, 진단, 채용 판단과 직무 추천
- 역량 레지스트리 스키마 또는 DB 마이그레이션
- UI 시각 디자인 개편
- 구 파이프라인과 신 파이프라인을 선택하는 장기 feature flag

## 5. 현재 시스템에서 보존할 사실

구현 전에 다음 파일과 심볼을 다시 확인한다.

- competency_interpreter.py
  - CompetencyState와 stable-ID 기반 후속 문맥
  - validate_query_plan_node
  - execute_registry_query_node
  - find_competencies
  - find_semantic_candidates
  - build_grounded_answer_context를 사용하는 응답 검증
  - 같은 thread 직렬화와 서로 다른 thread 병렬 실행
  - PostgresSaver 기반 checkpoint runtime
- competency_query.py
  - ParsedRegistryQuery
  - RegistryQueryPlan
  - validate_parsed_query
  - execute_registry_query
  - build_grounded_answer_context
  - render_grounded_fallback
  - validate_grounded_answer
- web_api.py
  - 기존 비스트리밍 API
  - start, status, delta, replace, done, error SSE 계약
  - thread 생성 및 메시지 복원 endpoint
- static/chat.js
  - localStorage의 현재 thread_id
  - 현재 대화 복원
  - 새 대화 생성
  - delta, replace, done 처리

PostgreSQL checkpoint와 현재 브라우저의 마지막 thread_id 복원은 유지한다.
thread_id는 대화 식별자일 뿐 인증 수단이 아니라는 현재 보안 경계도 유지한다.
로그인과 여러 세션을 탐색하는 기능만 향후로 보류한다.

## 6. 목표 그래프

정상 흐름은 다음과 같다.

    START
      -> llm_gateway
      -> validate_gateway_decision
      -> route_after_gateway

    registry_query
      -> validate_query_plan
      -> existing deterministic registry execution
      -> write_registry_answer
      -> validate_registry_answer
      -> END

    semantic_search
      -> find_semantic_candidates
      -> either execute a confident single match
         or prepare up to three validated candidates
      -> write_registry_answer
      -> validate_registry_answer
      -> END

    general_conversation / capability_help / unsupported / general_clarification
      -> write_general_answer
      -> END

    unrecoverable LLM or validation failure
      -> fixed_failure_message
      -> END

입구에서 자연어를 다시 판별하는 Python fast path는 존재하지 않는다.

## 7. LLM Gateway 계약

### 7.1 출력 형식

새 Gateway는 Pydantic의 strict JSON schema를 사용하는 discriminated union으로
구현한다. 구체적인 클래스명은 코드 맥락에 맞출 수 있지만 최소한 다음 route를
표현해야 한다.

- registry_query
  - 기존 ParsedRegistryQuery 또는 동등한 구조화 질의 포함
- general_conversation
  - greeting, small_talk, bot_identity, simple_concept 중 하나
- capability_help
- unsupported
  - current_information, sensitive_advice, unsafe_or_other_unsupported 구분 가능
- needs_clarification

Gateway는 사용자에게 보여줄 자연어 답변을 생성하지 않는다. route와 조회에
필요한 구조만 반환한다.

### 7.2 한 번의 호출에서 수행할 일

Gateway 한 번이 다음 두 수준을 함께 결정한다.

1. 역량 질의인지 일반 대화인지 결정
2. 역량 질의라면 기존 intent와 조회 조건을 작성

역량 질의를 위해 별도의 llm_interpret_query를 직렬로 다시 호출하지 않는다.
현재 llm_interpret_query의 역할은 Gateway에 흡수한다.

### 7.3 Gateway 입력

Gateway에는 다음 정보만 필요한 범위로 제공한다.

- 현재 사용자 메시지
- 같은 thread의 최근 10~12개 user/assistant 메시지
- 현재 레지스트리에서 재검증한 이전 결과 stable ID와 표시 이름
- 이전 query plan의 안전한 요약
- 허용된 검사, node type, 정식 역량명과 별칭 catalog
- capability manifest와 route 정책

전체 checkpoint 메시지는 저장하되 매 호출마다 무제한 전달하지 않는다.
이전 대화의 사용자 주장을 레지스트리 사실로 취급하지 않는다.

### 7.4 혼합 입력

- 인사와 역량 질문이 섞이면 registry_query를 우선한다.
- Registry Writer가 인사를 자연스럽게 짧게 반영할 수 있다.
- 지원되는 일반 질문과 역량 질문이 함께 있으면 역량 답변을 중심으로 처리한다.
- 서로 무관한 여러 실질적 요청이 섞이고 일부가 범위 밖이면 지원되는 부분을
  먼저 답하고 나머지 범위를 짧게 안내한다.

## 8. 역량 질의 동작

다음 기존 intent와 기능은 회귀 없이 유지한다.

- item_lookup
- semantic_search
- catalog_query
- hierarchy_query
- relation_query
- aggregate_query
- comparison_query

help와 out_of_scope가 현재 ParsedRegistryQuery에 남아 있더라도 새 정상 흐름에서는
Gateway의 capability_help와 unsupported route를 사용한다. QueryIntent에서
기존 값을 삭제할지는 저장소 전체 호출자를 확인한 뒤 결정한다.

### 8.1 모호한 역량

- Gateway의 이름 선택은 최종 권위가 아니다.
- 모든 이름은 active registry에서 검증한다.
- 의미 검색 결과가 높은 확신의 단일 후보라면 해당 후보를 자동 선택해 조회한다.
- 비슷한 후보가 여러 개면 정식 이름과 레지스트리 정의를 최대 3개까지 제시하고
  번호 또는 정확한 이름으로 선택하도록 요청한다.
- 확신할 후보가 없으면 역량이나 행동 특징을 더 구체적으로 묻는다.
- 후보 목록 밖 이름을 Writer가 추가하면 검증 실패다.

SemanticSelection에 단일 후보 자동 확정을 표현할 명시적 confidence 또는
auto_select 필드를 추가한다. 단순히 후보가 한 개라는 이유만으로 고확신으로
간주하지 않는다.

### 8.2 역량 활용 질문

다음과 같은 비개인화 질문은 지원한다.

- 협업 역량을 높이기 위한 일반적인 방법
- 책임감이 업무 행동에서 나타나는 일반적인 모습
- 특정 역량을 연습할 때 참고할 일반적인 활동

답변은 레지스트리 사실과 생성형 제안을 명확히 구분한다.

- 레지스트리 기준: 정의와 위계 등 검증 가능한 내용
- 일반적인 활용 제안: 레지스트리 원문이 아닌 비개인화된 참고 제안

개인 점수, 성향 진단, 채용 적합성, 직무 추천 또는 사용자의 원인을 추론하지
않는다. 기존 validate_grounded_answer의 사실 검증을 전체적으로 느슨하게
만들어서는 안 된다. 활용 제안이 필요하면 별도 answer mode 또는 별도 필드로
사실 부분과 구분한다.

## 9. Registry Writer 계약

Registry Writer는 다음 응답을 모두 담당한다.

- 성공한 레지스트리 조회 결과
- 의미 검색 후보 제시
- 역량 관련 clarification
- 등록되지 않은 역량 안내
- 레지스트리 사실을 바탕으로 한 비개인화 활용 설명

Writer 입력에는 검증된 query plan, query result, grounding context, 현재 질문과
필요한 최근 대화만 포함한다. stable ID, 내부 prompt, DB 정보는 출력하지 않는다.

다음 규칙은 유지하거나 강화한다.

- 정식 이름, 개수, 순서, 위계와 관계를 바꾸거나 추가하지 않는다.
- exact definition은 생략하거나 의역하지 않고 원문 그대로 포함한다.
- 긴 목록과 tree의 순서를 보존한다.
- 후보는 제공된 최대 3개만 사용한다.
- 개인 평가나 점수 해석을 추가하지 않는다.

### 9.1 검증 후 공개

Registry Writer의 생성 토큰은 사용자에게 바로 보내지 않는다.

1. 전체 답변을 메모리에서 완성한다.
2. 최대 길이와 레지스트리 grounding을 검증한다.
3. 검증 성공 후에만 delta 또는 replace로 최종 내용을 공개한다.
4. checkpoint에는 검증된 AIMessage 하나만 기록한다.

> **SUPERSEDED — [ADR-001](../adr/ADR-001-registry-writer-grounding-contract.md) (2026-08-11)**
> 아래 단락의 fallback 금지 규칙은 더 이상 유효하지 않다. Registry Writer가
> 검증을 통과하지 못하면 같은 grounding context로 렌더링한 답변을
> `registry_fallback`으로 공개하며, 고정 장애 안내는 grounding context 구성
> 실패처럼 결정적 복구조차 불가능한 경우에만 쓴다. 이는 이 문서의 후속 PRD가
> Scope 경로에 도입한 "정상 primary는 생성형, fallback만 결정적" 정책을 Registry
> 경로에도 적용한 것이다.

기존 render_grounded_fallback은 Writer prompt의 기준 답변이나 검증 보조로
사용할 수 있지만, 정상 사용자 응답을 Python이 대신 작성하는 공개 fallback으로
사용하지 않는다. Writer가 최종적으로 실패하면 공통 고정 장애 안내문을 사용한다.

## 10. General Writer 계약

General Writer가 지원하는 범위는 다음과 같다.

- 인사와 가벼운 대화
- 챗봇의 정체성과 사용법
- 시사성이 없는 짧은 일반 개념 설명
- 지원 범위 밖 질문에 대한 경계 안내
- 역량과 무관한 모호한 입력에 대한 자연스러운 재질문

응답 정책은 다음과 같다.

- 사용자의 질문에 먼저 1~3문장으로 유용하게 답한다.
- 자연스러운 한 문장으로 역량 질문을 제안한다.
- 최근 assistant 응답에서 이미 같은 유도를 했다면 반복하지 않는다.
- 초기 구현에서는 별도 Python counter를 추가하지 않고 최근 대화를 보고
  Writer가 판단한다. 실제 반복 문제가 테스트나 운영에서 확인될 때만 상태
  필드를 추가한다.
- 사용자 언어를 따르며 현재 UI에서는 한국어를 기본으로 한다.
- 최신 정보가 필요한 질문에는 확인할 수 없음을 투명하게 말한다.
- 의료, 법률, 금융 조언과 위험한 행동 요청에는 실질적인 답을 만들지 않는다.

General Writer는 실시간 delta를 보낼 수 있다. 다만 모델 호출이 끝나기 전의
부분 답변은 checkpoint하지 않는다. 취소나 최종 실패 시 부분 답변을 제거하거나
replace로 공통 장애 안내문과 동기화한다.

## 11. Capability manifest

챗봇이 자신의 기능을 임의로 추론하지 않도록 하나의 capability manifest를
단일 원본으로 관리한다. 구현 위치는 기존 모듈 구조를 보고 정하되 같은 내용을
prompt와 도움말 함수에 중복하지 않는다.

manifest에는 최소한 다음이 포함된다.

- 지원
  - 역량 정의와 등록 정보
  - 목록, 위계, 부모와 자식, 조상과 후손, 형제 관계
  - 검사 및 node type별 필터와 집계
  - 최대 3개 역량 비교
  - 행동 설명을 통한 관련 역량 후보 찾기
  - 비개인화된 일반 활용 제안
  - 인사, 챗봇 소개와 간단한 일반 개념
- 제한
  - 레지스트리에 없는 정의 생성
  - 사용자 개인 평가와 점수 추정
  - 채용 또는 직무 적합성 판단
  - 최신 정보 검색
  - 의료, 법률, 금융 전문 조언

General Writer의 챗봇 소개와 사용 예시는 반드시 이 manifest를 근거로 한다.

## 12. 모델 설정과 호출 예산

역할별 모델 설정을 분리한다.

- OPENAI_ENTRY_MODEL
- OPENAI_ANSWER_MODEL

두 설정의 초기 기본값은 gpt-5.6-luna다. 기존 배포 호환을 위해 필요하면
OPENAI_MODEL을 중간 fallback으로 읽을 수 있다.

예시 우선순위는 다음과 같다.

    role-specific environment variable
      -> existing OPENAI_MODEL
      -> gpt-5.6-luna

General Writer와 Registry Writer는 ANSWER_MODEL을 공유한다. 의미 검색 selector는
ENTRY_MODEL을 사용한다.

### 12.1 실제 요청 상한

한 사용자 턴에서 실제 OpenAI API 요청을 최대 3회로 제한한다.

- 일반 대화: Gateway + General Writer = 2회
- 직접 해석 가능한 역량 질의: Gateway + Registry Writer = 2회
- 의미 검색: Gateway + semantic selector + Registry Writer = 3회
- 구조 또는 출력 복구: 남아 있는 예산 안에서만 1회 재시도

semantic selector까지 사용해 이미 3회를 소비했다면 Writer 검증 실패를 네 번째
호출로 복구하지 않고 공통 장애 안내문으로 종료한다.

현재 ChatOpenAI의 max_retries=1처럼 SDK 내부 재시도가 실제 요청 상한을 넘길 수
있다. 실제 API 요청 기준 3회라는 요구를 지키도록 내부 retry와 애플리케이션
retry를 하나의 예산으로 조정한다. 가장 단순한 구현은 SDK 자동 retry를 끄고
턴 단위 예산 안에서 명시적으로 복구하는 방식이다.

## 13. 실패 정책

LLM API 오류, timeout, 잘못된 구조화 출력, Registry Writer 검증 실패는 남은
호출 예산 안에서 한 번만 복구할 수 있다. 최종 실패 시 Python이 다음 문장만
사용자에게 반환한다.

> 답변을 만드는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.

이 문구는 하나의 상수로 관리한다. 내부 예외, prompt, stable ID, DB URL 또는
레지스트리 원문을 API나 SSE error에 노출하지 않는다.

범위 밖 질문은 실패가 아니다. 정상적으로 General Writer를 거쳐 답한다.

## 14. 대화 상태와 checkpoint

- 전체 MessagesState는 현재 PostgreSQL checkpointer에 계속 저장한다.
- Gateway와 Writer에는 최근 10~12개 메시지만 전달한다.
- 후속 역량 문맥은 이름 문자열보다 현재 snapshot에서 재검증한 stable ID를
  우선한다.
- last_query_plan, last_result_ids 계열의 안전한 문맥은 유지한다.
- 새 턴을 시작할 때 transient route, query result, candidate와 retry 상태를
  초기화한다.
- 같은 thread_id의 요청 직렬화와 다른 thread의 병렬 실행을 유지한다.
- 새 채팅은 새로운 UUID thread_id를 만들며 이전 thread를 삭제하지 않는다.
- 브라우저의 현재 thread_id localStorage 복원과 메시지 endpoint를 유지한다.

이번 변경에서는 사용자와 thread의 소유 관계를 만들지 않는다. 공개 배포 시
thread_id가 인증 수단이 아니라는 기존 제한도 그대로다.

## 15. SSE와 웹 API 계약

기존 공개 event 이름을 유지한다.

- start
- status
- delta
- replace
- done
- error

General Writer는 생성 중 delta를 전송할 수 있다. Registry Writer는 검증 전에
delta를 보내지 않는다. 검증 후 전체 답변을 하나의 delta로 보내거나 검증된
문자열을 안전하게 나눈 delta로 보낼 수 있다.

어떤 경로에서도 다음 조건을 만족해야 한다.

- done.answer와 checkpoint의 마지막 assistant message가 같다.
- 비스트리밍 API의 answer도 같은 최종 문자열이다.
- 부분 출력은 checkpoint에 남지 않는다.
- 취소된 요청은 부분 assistant message를 남기지 않는다.
- replace 이후 브라우저 말풍선과 done.answer가 다시 일치한다.
- 후보명은 공개 전에 active registry 기준으로 필터링한다.

API endpoint와 frontend payload 모양은 꼭 필요한 경우가 아니면 바꾸지 않는다.

## 16. 기존 코드에서 목표 코드로의 이동

| 현재 요소 | 목표 | 조치 |
|---|---|---|
| interpret_query의 Python 우선 분류 | llm_gateway | 정상 START 경로에서 제거 |
| detect_deterministic_query 호출 | LLM Gateway 구조화 판단 | interpreter의 정상 경로 import와 호출 제거 |
| extract_exact_registered_names fast path | Gateway + registry validation | 분류 용도로 제거 |
| llm_interpret_query | llm_gateway | 역할 통합 후 기존 노드 제거 |
| ParsedNaturalLanguageQuery 호환 모델 | 새 strict Gateway union | 외부 사용 여부 확인 후 제거 |
| _legacy_parser_update | 새 Gateway 계약 | 호환 요구가 없으면 제거 |
| route_after_interpret와 route_after_llm | route_after_gateway | 하나의 얇은 dispatch로 통합 |
| find_semantic_candidates | 의미 검색의 선택적 3번째 경로 | confidence 계약을 추가해 유지 가능 |
| write_streamed_answer | write_registry_answer | 검증 전 공개를 중단하고 retry 정책 적용 |
| produce_answer | Registry Writer | 직접 최종 AIMessage 작성 경로 제거 |
| present_candidates | Registry Writer candidate mode | 직접 최종 AIMessage 작성 경로 제거 |
| answer_help | General Writer + manifest | 직접 최종 AIMessage 작성 경로 제거 |
| clarify_query | route에 따라 두 Writer | 직접 최종 AIMessage 작성 경로 제거 |
| handle_unknown | Registry Writer | 직접 최종 AIMessage 작성 경로 제거 |
| handle_out_of_scope | General Writer | 직접 최종 AIMessage 작성 경로 제거 |
| PostgresSaver runtime | 기존 동작 | 유지 |
| web_api의 thread/history endpoint | 기존 동작 | 유지 |
| chat.js의 localStorage 복원 | 기존 동작 | 유지 |

detect_deterministic_query 자체가 competency_query.py의 다른 공개 기능이나 테스트에
필요하면 그 모듈에서 즉시 삭제하지 않는다. 이번 요구는 interpreter의 정상
입구에서 Python 자연어 분류를 제거하는 것이다.

## 17. 구현 순서와 단계별 완료 기준

### 단계 0. 현재 상태 고정

작업:

- git status와 diff 확인
- 관련 테스트를 먼저 실행
- 현재 graph, API와 SSE 계약 확인
- 기존의 관련 없는 수정 보존

완료 기준:

- 변경 전 실패와 통과 상태가 기록되어 있고, 작업 대상과 기존 사용자 변경이
  구분되어 있다.

### 단계 1. Gateway schema와 모델 설정

작업:

- strict discriminated union 정의
- capability manifest 단일 원본 정의
- ENTRY_MODEL과 ANSWER_MODEL 설정 분리
- 최근 메시지와 stable-ID 문맥을 만드는 helper 정의
- Gateway prompt와 구조화 호출 작성

완료 기준:

- 인사, 도움말, 범위 밖 질문과 모든 기존 registry intent가 구조화 결과로
  구분되고, Gateway가 자연어 답변을 직접 반환하지 않는다.

### 단계 2. START와 registry routing 교체

작업:

- START를 llm_gateway에 연결
- route_after_gateway를 얇은 enum dispatch로 구현
- 기존 validate_query_plan과 결정적 실행기 연결
- 의미 검색 단일 확정과 다중 후보 흐름 구현

완료 기준:

- 정확한 등록명 질의를 포함한 모든 테스트 입력에서 Gateway mock 호출이
  정확히 한 번 발생하고, Python detector가 정상 route 결정에 사용되지 않는다.

### 단계 3. 두 Writer 구현

작업:

- write_registry_answer 구현
- write_general_answer 구현
- capability help와 unsupported prompt 구현
- Registry Writer의 facts와 guidance mode 구분
- 공통 고정 장애 문구와 호출 예산 적용

완료 기준:

- 모든 정상 terminal route가 두 Writer 중 하나로 끝나고, 고정 장애 경로 외의
  Python 함수가 최종 사용자 문장을 직접 작성하지 않는다.

### 단계 4. 검증과 스트리밍 정합성

작업:

- Registry Writer 전체 출력 buffering
- 공개 전 grounded validation
- General Writer의 실시간 delta와 취소 처리
- done, checkpoint, 비스트리밍 answer 동기화
- 실제 LLM 요청 최대 3회 계측

완료 기준:

- 검증 전 registry token이 공개되지 않고, 모든 성공 및 실패 경로에서 화면,
  done.answer와 checkpoint가 일치한다.

### 단계 5. 기존 코드 제거

작업:

- 새 graph에서 도달하지 않는 기존 노드 제거
- 호환 전용 parser와 helper의 실제 외부 사용 여부 확인
- 사용되지 않는 import, type, state field와 테스트 제거 또는 교체
- 장기 feature flag를 남기지 않음

완료 기준:

- builder에 구 terminal 노드가 등록되어 있지 않고, dead code 검사와 전체
  테스트가 통과한다.

### 단계 6. 문서와 전체 검증

작업:

- WEB_CHAT_README.md의 graph, 모델 env, 스트리밍과 지원 범위 갱신
- 관련 unit, graph, API, frontend 테스트 실행
- 전체 pytest 실행
- git diff --check 실행

완료 기준:

- 아래 인수 테스트가 모두 통과하고 문서가 실제 동작과 일치한다.

## 18. 테스트 전략

가능하면 가장 높은 기존 seam을 사용한다.

- graph 테스트: tests/test_competency_interpreter.py에서 builder를
  InMemorySaver로 compile하고 Gateway와 Writer 모델만 stub
- query engine 테스트: 기존 tests/test_competency_query.py 유지
- API와 SSE 테스트: tests/test_web_api.py
- frontend event와 복원 테스트: tests/test_chat_js.py
- PostgreSQL runtime 설정과 checkpoint 초기화 테스트 유지

모델 helper마다 과도한 mock seam을 추가하지 않는다. 전체 graph 한 턴에서
route, 호출 횟수, 최종 message와 checkpoint를 함께 검증하는 테스트를 우선한다.

## 19. 필수 인수 시나리오

| ID | 입력 또는 상황 | 기대 결과 |
|---|---|---|
| A01 | 정확한 역량명과 정의 요청 | Gateway를 먼저 호출하고 검증된 원문 정의 반환 |
| A02 | 목록, 위계, 관계, 집계, 비교 질문 | 기존 결정적 query 기능과 결과 유지 |
| A03 | 안녕하세요 | 짧은 인사 후 자연스러운 역량 질문 제안 |
| A04 | 시사성이 없는 간단한 개념 질문 | 먼저 답한 뒤 최근에 반복하지 않았다면 역량 질문 제안 |
| A05 | 무엇을 할 수 있어? | capability manifest에 있는 기능과 제한만 안내 |
| A06 | 오늘 날씨 또는 최신 뉴스 | 현재 확인 범위를 안내하고 역량 질문으로 연결 |
| A07 | 안녕, 성실성 정의를 알려줘 | registry route를 선택하고 인사를 자연스럽게 반영 |
| A08 | 협업 역량을 높이려면? | 레지스트리 기준과 일반 활용 제안을 분리 |
| A09 | 사용자 개인 역량 점수를 추정해줘 | 개인 평가를 하지 않고 가능한 역량 정보로 전환 |
| A10 | 주도적으로 일하는 역량은? | 단일 고확신 후보는 조회, 다중 후보는 최대 3개 제시 |
| A11 | 그 역량의 하위요인은? | 같은 thread의 stable-ID 문맥으로 후속 질문 해결 |
| A12 | 같은 질문을 다른 thread에서 입력 | 이전 thread 문맥을 사용하지 않음 |
| A13 | Gateway schema 오류 | 예산 안에서 한 번 복구 후 최종 실패 시 고정 안내문 |
| A14 | Registry Writer가 정의를 변경 | 공개하지 않고 복구하거나 고정 안내문 반환 |
| A15 | General Writer stream 도중 취소 | 부분 AIMessage가 checkpoint에 남지 않음 |
| A16 | Registry Writer 생성 중 | 검증 전 token이 SSE delta로 공개되지 않음 |
| A17 | 의미 검색 경로 | 실제 LLM API 요청이 3회를 넘지 않음 |
| A18 | 일반 및 직접 registry 경로 | 정상적으로 2회의 LLM 호출 사용 |
| A19 | 새로고침 | localStorage의 현재 thread_id 대화 복원 유지 |
| A20 | 새 대화 버튼 | 새 thread_id를 만들고 이전 대화 문맥을 사용하지 않음 |
| A21 | 등록되지 않은 후보명 생성 | 공개 후보에서 제거되거나 답변 검증 실패 |
| A22 | 전체 기존 테스트 | registry V2, API, frontend와 runtime 회귀 없음 |

## 20. 예상 변경 파일

반드시 확인하거나 수정할 가능성이 높은 파일:

- competency_interpreter.py
- tests/test_competency_interpreter.py
- WEB_CHAT_README.md

스트리밍 동작에 필요한 경우에만 수정:

- web_api.py
- tests/test_web_api.py
- static/chat.js
- tests/test_chat_js.py

정상적으로는 변경하지 않을 파일:

- competency_registry.py와 registry DB schema
- registry compiler와 migration scripts
- static UI 디자인 파일

새 외부 dependency는 예상하지 않는다. .env, .env.local 또는 실제 secret 값을
문서나 테스트 출력에 기록하지 않는다.

## 21. 완료 정의

다음을 모두 만족해야 한다.

- START 다음 정상 노드는 llm_gateway다.
- 정확한 역량명을 포함한 모든 입력이 Gateway를 거친다.
- Python 자연어 fast path가 정상 graph에서 제거됐다.
- 기존 registry intent가 모두 유지된다.
- 정상 terminal 응답은 Registry Writer 또는 General Writer가 작성한다.
- 고정 장애 안내 외의 Python 최종 응답 템플릿이 제거됐다.
- Registry Writer는 검증 전에 출력을 공개하지 않는다.
- General Writer의 부분 stream은 checkpoint되지 않는다.
- 같은 thread 후속 질문과 다른 thread 격리가 유지된다.
- PostgreSQL checkpoint와 현재 대화 복원이 유지된다.
- 실제 LLM API 요청이 턴당 최대 3회다.
- 기존과 새 인수 테스트가 모두 통과한다.
- WEB_CHAT_README.md가 새 graph와 설정을 설명한다.
- git diff --check가 통과한다.
- 관련 없는 사용자 변경을 덮어쓰지 않았다.

## 22. 향후 작업

다음은 이번 PRD가 끝난 뒤 별도 요구사항으로 다룬다.

- 로그인과 사용자 계정
- user_id와 thread_id의 소유권 연결
- 여러 대화 목록, 제목, 복원과 삭제
- 보존 기간과 개인정보 삭제
- 여러 서버 인스턴스의 분산 잠금
- 웹 검색 및 최신 정보 도구
- 입구와 출구 모델의 비용 기반 분리
- 장기 대화 요약

## 23. 메인 세션에 전달할 시작 지시

메인 세션에는 다음과 같이 요청하면 된다.

> C:\Users\ksy0823\.vscode\LangGraph\ONE_TIME_LLM_GATEWAY_DUAL_WRITER_PRD.md를
> 처음부터 끝까지 읽고, 현재 작업 트리와 관련 테스트를 확인한 뒤 문서에
> 명시된 범위만 구현해 주세요. 기존의 관련 없는 변경을 보존하고, 구현 후
> 필수 인수 시나리오와 전체 테스트를 검증해 주세요.
