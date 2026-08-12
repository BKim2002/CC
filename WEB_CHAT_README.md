# 역량 해석 웹 챗봇 실행 안내

이 프로젝트는 LangGraph 역량 해석기를 FastAPI와 HTML·CSS·JavaScript 기반의 웹 채팅 화면에 연결한다. 로컬 Uvicorn 실행과 Vercel 배포 진입점을 모두 제공한다.

## 구성

- `app.py`: Vercel이 자동 탐지하는 FastAPI 진입점
- `web_api.py`: FastAPI 서버, 요청 검증, 정적 파일 제공, LangGraph 연결
- `static/index.html`: 채팅 화면 구조
- `static/style.css`: 반응형 채팅 UI
- `static/chat.js`: 대화 생성·SSE 스트리밍·복원과 후보 버튼 처리
- `competency_query.py`: Gateway 초안을 active registry와 stable ID로 정규화하는 Query Normalizer, 목록·위계·관계·집계·비교 엔진과 grounding 검증
- `llm_gateway.py`: 세 가지 strict Gateway route, Scope 응답 계약, 단일 capability manifest, 역할별 모델 설정과 턴당 호출 예산
- `scope_response.py`: 범위 밖 응답 템플릿과 redirect 회전, 메타 대화의 필드별 안전성 검증, 결정적 fallback, 공개 문자열 정규화
- `competency_interpreter.py`: Gateway → Query Normalizer/Scope Writer로 분기하는 Registry-first LangGraph, 활성 레지스트리 스냅샷, PostgreSQL 체크포인터
- `competency_registry.py`: PostgreSQL 활성 레지스트리의 검증·인덱스 생성
- `scripts/setup_checkpoint_database.py`: LangGraph 체크포인트 테이블 준비
- `scripts/setup_competency_registry_database.py`: 버전형 역량 레지스트리 스키마 준비
- `scripts/build_registry.py`: V1/V2 원본 형식을 명시적으로 선택해 runtime JSON 1.0으로 변환
- `scripts/registry_source_v1.py`: 기존 표 형식 Markdown과 62개 규칙을 보존한 legacy adapter
- `scripts/registry_source_v2.py`: fenced YAML 기반 structured Markdown V2 parser
- `scripts/registry_compiler.py`: stable ID와 `parent_id`를 runtime 위계로 컴파일
- `scripts/registry_diff.py`: 활성 버전과 후보를 stable ID 기준으로 비교
- `scripts/upload_competency_registry.py`: 외부 원본을 검증·비교·업로드·활성화
- `scripts/export_active_registry_v2.py`: 활성 DB 버전을 읽기 전용으로 V2 원본에 내보내기
- `tests/`: 해석 로직, API·상태 격리·복원, 레지스트리와 DB 도구 테스트
- `requirements.txt` / `requirements-dev.txt`: 운영 / 테스트 의존성

## 처음 한 번 설치

현재 가상환경에 프로젝트 의존성을 설치한다.

```powershell
cd C:\Users\ksy0823\.vscode\LangGraph
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

테스트까지 실행하려면 테스트 의존성을 함께 설치한다.

```powershell
python -m pip install -r requirements-dev.txt
```

모든 질문을 LLM Gateway와 두 Writer로 처리하려면 프로젝트의 `.env`에 서버용 OpenAI 키를 설정한다. 키를 HTML이나 JavaScript에 넣지 않는다.

```text
OPENAI_API_KEY=발급받은_API_KEY
OPENAI_ENTRY_MODEL=gpt-5.6-luna
OPENAI_ANSWER_MODEL=gpt-5.6-luna
DATABASE_URL=postgresql://사용자명:비밀번호@호스트:5432/데이터베이스명?sslmode=require
```

모든 입력은 먼저 Entry 모델의 LLM Gateway를 거친다. 레지스트리 질의의 설명은 Registry Writer가, 인사·소개·범위 밖 질문의 경계 안내는 Scope Writer가 작성하므로 OpenAI 키가 필요하다. 각 역할별 설정이 비어 있으면 기존 `OPENAI_MODEL`, 그것도 비어 있으면 `gpt-5.6-luna`를 사용한다. SDK 자동 재시도는 끄고 애플리케이션이 실제 API 요청을 턴당 최대 3회로 제한한다.

## PostgreSQL 최초 준비와 레지스트리 적재

새 데이터베이스에서는 LangGraph 체크포인트와 애플리케이션 레지스트리를 각각 준비한다. 두 setup 스크립트는 책임이 다르며, 기존 구조가 있는 데이터베이스에서 다시 실행해도 안전하다.

```powershell
python .\scripts\setup_checkpoint_database.py
python .\scripts\setup_competency_registry_database.py
```

역량 원본은 저장소에 두지 않는다. source format은 다음 두 가지이며 파일 내용으로 자동 추측하지 않는다.

- `markdown-v1`: 기존 표 형식 원본. 현재 62개 구성과 Level별 규칙을 그대로 검사하는 legacy 형식이다.
- `markdown-v2`: stable ID와 명시적 `parent_id`를 사용하는 structured Markdown. 항목 수와 위계 깊이를 고정하지 않는다.

### Structured Markdown V2 작성 규칙

V2 문서에는 아래 형식의 `competency-registry-yaml` fenced block이 정확히 하나 있어야 한다. 바깥쪽 제목·설명·표는 자유롭게 편집할 수 있으며 parser는 데이터로 해석하지 않는다.

````markdown
# 역량 정의서

```competency-registry-yaml
source_schema_version: "2.0"
registry_schema_version: "1.0"
instruments:
  - id: written
    label: 필기 역량검사
    root_path_prefix: []
items:
  - id: competency_0017
    name: 성실 수행
    instrument: written
    node_type: analysis_factor
    parent_id: competency_0004
    order: 10
    definition: 맡은 일을 꼼꼼하고 꾸준하게 수행하는 특성
    aliases:
      - 성실성
    analysis_included: true
    status: active
    replacement_id: null
    notes: []
```
````

핵심 운영 규칙:

- `id`는 이름과 무관한 영구 식별자다. 개명·위계 이동 때 바꾸지 않으며 새 ID도 이름에서 자동 생성하지 않는다.
- 개명할 때 이전 정식 이름을 `aliases`에 남기면 기존 정확 일치 질의를 계속 지원할 수 있다.
- `parent_id: null`은 root다. 그 외 항목은 같은 instrument의 active 부모 ID를 지정한다.
- 실제 item ID가 없는 가상 instrument 조상을 path에 표시해야 할 때만 instrument의 `root_path_prefix`를 사용한다.
- `order`는 같은 부모 아래 표시 순서다. active 형제끼리는 중복할 수 없고, active root끼리는 같은 instrument 안에서 비교한다. retired 항목의 order는 runtime 정렬에 참여하지 않으므로 대체 active 항목과 같은 값을 유지할 수 있다.
- `path`, 부모 이름, 자식 이름과 ID, 깊이, 정의 상태, instrument label은 compiler가 계산하므로 원본에 중복 작성하지 않는다.
- 단일 부모 tree와 instrument별 복수 root를 지원한다. 다중 부모 DAG는 지원하지 않는다.
- 퇴역 항목은 `status: retired`로 남기고 가능한 경우 active `replacement_id`를 지정한다. 퇴역 항목은 runtime snapshot에서 제외된다.
- duplicate YAML key, 알 수 없는 필드, 이름·별칭 충돌, orphan·self-parent·cycle·instrument 간 parent 연결은 거부된다.

### 기존 활성 버전을 V2 원본으로 한 번 내보내기

기존 stable ID를 보존해 V2로 전환하려면 활성 DB를 저장소 밖의 접근 통제된 경로로 내보낸다.

```powershell
python .\scripts\export_active_registry_v2.py `
  --output "C:\secure-source\competency-definitions-v2.md"
```

내보내기는 DB를 읽기만 하며 새 버전을 만들거나 활성 포인터를 바꾸지 않는다. 생성한 문서를 메모리에서 즉시 재파싱·재컴파일해 ID, 이름, 별칭, 정의, 사용자에게 보이는 전체 path, 부모·자식 관계와 root/child 순서, 분석 플래그가 현재 runtime과 같은지 확인한 뒤에만 파일을 쓴다. 실제 item이 아닌 가상 instrument 조상이 기존 path에 있으면 instrument의 `root_path_prefix`로 보존한다. 기존 출력 파일은 기본적으로 거부하며, 정말 교체할 때만 `--overwrite`를 명시한다. 출력 경로는 프로젝트 저장소 밖이어야 한다.

V2 compiler가 새 형식에 맞춰 다시 만드는 bookkeeping 값은 `source` metadata·rules·validation, `definition_status`, `source_section`, runtime source extra(`order/status/replacement_id`)다. 특히 legacy Markdown 표 위치를 나타내던 `source_section`은 V2에서 instrument label로 정규화된다. 이 값들은 사용자 lookup·정의·위계 비교에서는 제외되지만, 그 밖의 runtime 의미 필드는 round-trip에서 같아야 한다.

### Build, diff, upload, activate

먼저 외부 V2 원본을 DB 연결 없이 build할 수 있다.

```powershell
python .\scripts\build_registry.py `
  --source "C:\secure-source\competency-definitions-v2.md" `
  --source-format markdown-v2 `
  --output "C:\temp\candidate-registry.json"
```

업로드 전에는 dry-run과 활성 버전 diff를 함께 실행한다. 이 명령은 활성 DB를 읽기 전용 트랜잭션으로 조회하지만 버전 행이나 포인터를 변경하지 않는다.

```powershell
python .\scripts\upload_competency_registry.py `
  --source "C:\secure-source\competency-definitions-v2.md" `
  --source-format markdown-v2 `
  --dry-run `
  --diff-active
```

검토할 후보를 비활성 버전으로만 업로드하려면 `--dry-run`과 `--activate`를 모두 빼고 실행한다. `--diff-active`를 함께 쓰면 업로드와 안전한 요약을 한 번에 확인할 수 있다.

```powershell
python .\scripts\upload_competency_registry.py `
  --source "C:\secure-source\competency-definitions-v2.md" `
  --source-format markdown-v2 `
  --diff-active
```

활성화할 때는 diff를 확인했던 현재 버전 ID를 반드시 명시한다. 그 사이 다른 버전이 활성화됐으면 트랜잭션이 rollback된다.

```powershell
python .\scripts\upload_competency_registry.py `
  --source "C:\secure-source\competency-definitions-v2.md" `
  --source-format markdown-v2 `
  --activate `
  --expected-current-version 1
```

아직 활성 버전이 없는 새 DB의 최초 활성화는 기대값을 명시적으로 `none`으로 둔다.

```powershell
python .\scripts\upload_competency_registry.py `
  --source "C:\secure-source\competency-definitions-v2.md" `
  --source-format markdown-v2 `
  --activate `
  --expected-current-version none
```

active 항목 제거·퇴역, 부모 또는 instrument 이동, root/가상 root path prefix 변경, node type 또는 분석 플래그 변경, 기존 lookup 이름 제거, 이전 이름을 alias로 남기지 않은 개명은 기본적으로 breaking이다. 이미 retired인 항목에서 replacement를 제거하는 변경도 breaking이다. breaking 후보는 활성화 전에 거부되며, 내용을 별도로 검토하고 승인한 경우에만 `--allow-breaking`을 추가한다. 정의 문구와 lookup을 보존하는 별칭 추가도 diff에는 항상 나타난다.

업로드는 원본 bytes의 SHA-256을 버전 키로 사용하므로 같은 내용의 파일은 로컬 basename이 달라도 같은 버전이다. 같은 원본을 다시 실행하면 중복 행을 만들지 않고 기존 버전을 사용하되, `source.file`을 제외한 DB runtime JSON이 현재 compiler 결과와 다르면 활성화하지 않는다. 활성화 트랜잭션은 현재 포인터를 잠근 뒤 기대 버전 확인과 breaking diff를 다시 수행하고, 그 후에만 후보 INSERT와 포인터 전환을 처리한다. 출력에는 정의 전문이나 DB URL을 포함하지 않는다.

서버 시작 코드는 테이블을 생성하지 않는다. 시작할 때 `registry_current`가 가리키는 JSONB를 한 번 읽어 검증한 뒤 메모리 스냅샷으로 유지한다. 활성 버전이 없거나 손상되었으면 로컬 파일로 폴백하지 않고 시작에 실패한다. 따라서 새 `DATABASE_URL`로 바꾼 뒤에는 위 두 setup과 upload/activate를 서버보다 먼저 실행해야 한다.

## 개발 서버 실행

```powershell
cd C:\Users\ksy0823\.vscode\LangGraph
.\.venv\Scripts\Activate.ps1
python -m uvicorn web_api:app --reload
```

브라우저에서 다음 주소를 연다.

```text
http://127.0.0.1:8000
```

`--reload`는 파일 변경을 감지해 개발 서버를 재시작하는 개발용 옵션이다.

## 로컬 일반 실행

자동 재시작이 필요하지 않으면 다음처럼 실행한다.

```powershell
python -m uvicorn web_api:app --host 127.0.0.1 --port 8000 --workers 1
```

대화 상태는 PostgreSQL 체크포인터에 저장된다. 역량 정의는 서버 프로세스가 시작될 때 활성 DB 버전에서 한 번 로드된다. 런타임 초기화와 대화 실행 잠금은 분리되어 같은 `thread_id`의 요청은 겹치지 않고, 서로 다른 thread는 병렬로 실행할 수 있다. 이 제어는 한 프로세스 안에서만 유효하므로 로컬에서는 `--workers 1`로 실행한다. 여러 프로세스나 서버 인스턴스로 확장할 때는 분산 잠금과 사용자 인증을 별도로 설계해야 한다.

## 지원 질문과 용어

챗봇은 활성 레지스트리에 등록된 사실만 사용해 다음 질문을 처리한다.

- 하나 이상의 정확한 이름·별칭 조회와 의미상 가까운 역량 후보 검색
- 전체·검사별·조건별 목록: `전체 역량 목록`, `영상면접 세부항목 목록`, `정의가 없는 역량`
- 전체·검사별·부분 위계: `전체 위계 구조`, `자기관리 아래의 전체 구조`
- 구조 관계: 직접 부모·자식, 모든 조상·후손, 형제 역량
- 전체 수와 검사·node type·공식 단계·깊이별 집계
- 최대 3개 역량의 정의·위계 등 레지스트리 필드 비교
- 직전 결과에 대한 필터: `그중 영상면접만`, `그중 분석에 포함되는 것만`
- 행동 설명을 통한 최대 3개의 등록 후보 탐색과 고확신 단일 후보 자동 조회
- 등록 사실과 분리된 비개인화 일반 활용 제안: `협력 역량을 높이려면?`
- 인사·감사·작별, 챗봇 소개와 `어떤 질문을 할 수 있어?` 같은 사용법 안내

개인 역량 점수·진단, 채용 판단, 직무 적합성·추천, 최신 날짜·시간·날씨·뉴스, 일반 지식 설명, 의료·법률·금융 전문 조언, 외부 작업 실행은 지원하지 않는다. 미지원 질문도 오류로 취급하지 않고 Scope Writer가 해당 주제를 짧게 확인한 뒤 범위를 설명하고 지원되는 역량 질문으로 연결한다. Scope Writer 출력이 두 번 검증에 실패하면 질문 원문을 그대로 되풀이하지 않는 topic-aware Python 안내로 정상 종료한다.

필기 역량검사의 `상위요인`, `중위요인`, `하위요인`, `최하위요인`은 현재 공식 단계인 `L1`, `L2`, `L3`, `L4`를 뜻한다. 현재 활성 버전의 회귀 기준은 각각 3개, 10개, 30개, 9개이며, `역검 종합점수`는 이 네 단계 위의 종합점수이므로 상위요인 3개에 포함하지 않는다. 이 숫자는 답변에 하드코딩하지 않고 매번 활성 스냅샷을 필터링해 계산한다.

특정 대상을 붙인 `자기긍정의 상위요인`과 `자기관리의 하위요인`은 각각 직접 부모와 직접 자식을 뜻한다. `모든 상위요인`과 `아래의 모든 역량`은 모든 조상과 후손을 뜻하며, `자기긍정이 속한 중위요인`처럼 `속한`과 공식 단계를 함께 쓰면 해당 단계의 자기 자신 또는 가장 가까운 조상을 찾는다. 구조적인 `루트/말단`은 공식 `상위/최하위요인`과 서로 다른 개념이다.

영상면접에는 필기검사의 4단계 용어를 적용하지 않는다. `factor`는 `요인`, `item`은 `세부항목`으로 표시한다. path에 `영상면접` 같은 가상 상위 라벨이 있더라도 등록 item이 아니면 부모 역량으로 만들지 않는다.

목록·위계는 레지스트리의 원래 순서와 `children_ids` 순서를 보존한다. 한 답변은 기본적으로 최대 100개 항목을 렌더링하며, 이를 넘으면 전체 개수와 앞부분을 보여 주고 범위를 좁히도록 안내한다. 정의·경로·자식 목록처럼 긴 상세 사실은 writer 호출 전에 별도의 문자 예산으로 더 일찍 제한하고, 최종 답변은 20,000자를 넘지 않게 검증한다. 비교는 모바일 가독성과 grounding 범위를 위해 최대 3개로 제한한다.

정확한 이름 입력도 예외 없이 `START → llm_gateway → validate_gateway_decision`을 거친다. Gateway는 `registry_query`, `meta_conversation`, `out_of_scope` 중 하나와 작은 초안만 만들며 사용자 답변이나 최종 query plan을 쓰지 않는다. Query Normalizer가 사용자 원문과 초안을 대조하고, 이름 검증·목록 필터·관계 탐색·개수 계산·비교 사실 구성은 Python이 active registry와 stable ID로 수행한다. 역량처럼 보이는 미등록 이름은 일반 지식으로 설명하지 않고 미등록 대상으로 안내한다.

레지스트리에서 해석되지 않은 표현은 사용자가 얼마나 강하게 단언했는지로 갈라 처리한다. 사용자 원문에서 직접 뽑혔거나 조사로 대상임을 표시한 표현은 **declared**로 보고, 근접한 등록명이 있으면 후보를 제시하고 없으면 미등록 대상으로 안내한다. 다른 대상이 없어 남은 초안 표현은 **incidental**로 보고, 근접한 등록명이 있으면 후보를 제시하고 없으면 버린 뒤 원문 해석을 계속한다. `역량 목록 좀 알려줘`처럼 Gateway가 범위 명사를 target으로 복사해도 목록 질문이 미등록 안내로 바뀌지 않는 이유다.

근접 후보는 정식명과 별칭 전체를 대상으로 찾고, 정식명 기준으로 최대 3개까지 제시한다. `공감셩` 같은 한 음절 오타는 `공감성`을 되찾는다. 형제 이름이 서로 매우 유사한 레지스트리에서도 안전한데, 정확히 입력된 라벨은 근접 매칭에 도달하기 전에 해석되기 때문이다.

레지스트리 결과·후보·역량 관련 재질문은 Registry Writer가, 인사·소개·도움말·미지원 범위는 Scope Writer가 담당한다. 활용 제안은 `[등록 정보]`와 `[일반 활용 제안]`을 분리하고 등록 정보 섹션에 기존 grounding 검증을 그대로 적용한다. 정의는 레지스트리 원문을 유지한다.

범위 밖 질문은 LLM을 호출하지 않고 `scope_template`으로 답한다. 출력 중 모델에서 온 텍스트는 `sanitize_topic_summary`가 짧은 명사구 또는 고정 category 라벨로 줄인 주제 요약 하나뿐이며, 나머지 문장은 전부 고정 템플릿이다. 안내 문장이 매번 같아 보이지 않도록 registry redirect는 4종을 두고 턴 번호로 회전한다. 결정적 출력을 요청 경로에서 다시 검증하지 않는 대신, 모든 category·언어·variant 조합이 Scope Writer와 동일한 계약을 만족함을 `tests/test_scope_response.py`가 고정한다.

인사·감사·작별·챗봇 소개·기능 안내의 `meta_conversation`은 계속 Scope Writer가 생성한다. 두 Writer의 정상 응답 실패는 모두 결정적 복구로 끝난다. Registry Writer가 검증을 통과하지 못하면 같은 grounding context로 렌더링한 답변을 `registry_fallback`으로, Scope Writer가 실패하면 `scope_fallback`으로 공개한다. 공통 고정 장애 안내는 Gateway 자체 실패, grounding context 구성 실패, checkpoint 실패처럼 결정적 복구조차 불가능한 실제 시스템 실패에만 쓴다.

## 채팅 API와 스트리밍

기존 `POST /api/chat`은 다음 최종 응답을 계속 반환한다.

```json
{
  "thread_id": "UUID",
  "answer": "최종 답변",
  "candidates": []
}
```

`POST /api/chat/stream`은 같은 요청 본문을 받고 UTF-8 `text/event-stream`을 반환한다. 공개 event는 `start`, `status`, `delta`, `replace`, `done`, `error`다. 브라우저는 `fetch()`의 `ReadableStream`으로 frame을 읽어 한 assistant 말풍선에 `delta`를 이어 붙이고, 최종 오류에서 `replace`를 받으면 말풍선 전체를 확정 문자열로 바꾼다. 마지막 `done`에는 기존 API와 같은 `thread_id`, 최종 `answer`, active registry에서 재검증한 최대 3개의 `candidates`가 들어간다.

Registry Writer와 Scope Writer는 모두 전체 출력을 서버에서 버퍼링하고 검증한다. Registry Writer는 이름·수치·순서·정의 grounding을, Scope Writer는 route·category·capability manifest·직접 답변 부재를 확인한 뒤 checkpoint에 동일한 AIMessage가 저장된 경우에만 확정 delta를 공개한다. 거부된 초안과 예외 전문은 상태나 event에 넣지 않는다. 취소된 요청은 부분 assistant message를 남기지 않으며, `done.answer`, 비스트리밍 응답, history의 마지막 assistant message는 byte-for-byte 같은 문자열이다. 중간 상태와 오류 event에는 stable ID, 내부 plan, prompt, DB 연결 정보나 예외 전문을 싣지 않는다.

## Vercel 배포

Vercel은 루트의 `app.py`를 FastAPI 진입점으로 사용하고, `.python-version`과 `requirements.txt`로 런타임을 준비한다. 배포 전 Vercel 환경 변수에 `DATABASE_URL`, `OPENAI_API_KEY`, `OPENAI_ENTRY_MODEL`, `OPENAI_ANSWER_MODEL`을 설정한다. 기존 `OPENAI_MODEL`은 두 역할별 설정이 없을 때만 호환 fallback으로 읽는다.

배포 전에 운영 DB에 레지스트리 버전을 먼저 활성화해야 한다. 새 버전을 활성화한 뒤에는 각 인스턴스가 새 스냅샷을 읽도록 재배포하거나 재시작한다.

현재 애플리케이션에는 로그인과 요청 횟수 제한이 없다. Vercel Deployment Protection 같은 접근 보호를 켜거나 애플리케이션 인증과 요청 제한을 구현하기 전에는 공개 URL로 운영하지 않는다. 여러 인스턴스에서 같은 `thread_id`를 동시에 처리하는 사용 방식도 지원하지 않는다.

## 레지스트리 업데이트와 롤백

새 정의를 배포할 때는 현재 활성 버전 ID를 먼저 기록하고, 외부 원본을 `--dry-run --diff-active`로 검토한 뒤 비활성 업로드와 활성화를 구분한다. 활성화 뒤에는 SQL 또는 관리 도구로 버전의 해시와 항목 수를 확인하고 애플리케이션을 재배포하거나 재시작한다.

```sql
SELECT version_id
FROM competency_data.registry_current
WHERE singleton = 1;
```

이전 원본을 보관하고 있다면 같은 hash가 멱등 조회되므로, 원본 형식을 명시해 다시 활성화할 수 있다. 되돌림 자체가 breaking으로 분류되면 검토 후 `--allow-breaking`도 필요하다.

```powershell
python .\scripts\upload_competency_registry.py `
  --source "C:\secure-source\previous-competencies-v2.md" `
  --source-format markdown-v2 `
  --activate `
  --expected-current-version 현재_ID `
  --allow-breaking
```

원본을 사용할 수 없어 DB 관리 작업으로 직접 되돌릴 때는 기록해 둔 ID와 현재 포인터를 트랜잭션 안에서 확인한 뒤 포인터를 바꾸고 모든 애플리케이션 인스턴스를 다시 시작한다.

아래 관리용 SQL의 두 숫자는 실행 전에 실제 기대 현재 버전과 되돌릴 버전으로 바꾼다. 잠금 뒤 기대값이 다르면 예외를 발생시켜 포인터를 변경하지 않는다.

```sql
BEGIN;
LOCK TABLE competency_data.registry_current IN SHARE ROW EXCLUSIVE MODE;

DO $$
DECLARE
    expected_current_version BIGINT := 2;
    rollback_version BIGINT := 1;
    actual_current_version BIGINT;
BEGIN
    SELECT version_id
    INTO STRICT actual_current_version
    FROM competency_data.registry_current
    WHERE singleton = 1
    FOR UPDATE;

    IF actual_current_version <> expected_current_version THEN
        RAISE EXCEPTION 'current registry version changed';
    END IF;

    UPDATE competency_data.registry_current
    SET version_id = rollback_version,
        updated_at = NOW()
    WHERE singleton = 1;
END $$;

COMMIT;
```

`registry_versions` 행의 Markdown이나 JSON은 수정하지 않는다. 내용이 달라지면 항상 새 버전으로 업로드한다. 저장소에서 원본 파일을 삭제하는 일반 커밋은 최신 트리에서만 파일을 없애며, Git 과거 이력·기존 클론·기존 배포에는 남을 수 있다. 과거 이력 삭제와 기존 배포 폐기는 별도 승인과 절차가 필요한 파괴적 작업이다.

## 테스트

일반 테스트는 운영 PostgreSQL 대신 가짜 DB 연결과 임시 SQLite 체크포인터를 사용하며 OpenAI API 호출을 비활성화한다. 레지스트리 테스트도 저장소의 운영 JSON 대신 작은 합성 스냅샷을 사용한다.

```powershell
python -m pytest -q
```

검증 범위:

- 메인 HTML, CSS, JavaScript 제공
- 서버 상태와 UUID 대화 생성
- 공백·길이·잘못된 UUID·추가 필드 검증
- 정확한 역량 질문과 체크포인트 기록
- 정확한 이름을 포함한 모든 입력의 Gateway 우선 처리와 직접 질의·Scope 안내 2회/의미 검색 3회 호출 상한
- Gateway 초안과 사용자 원문을 active registry에 대조하는 Query Normalizer의 plan·clarification·semantic candidate·미등록 대상 분기
- 복사된 범위 명사를 미등록 대상으로 승격하지 않는 declared/incidental 구분과, 오타에 대한 근접 등록명 제안(정식명 62개 전체의 한 음절 오타 복원 포함)
- 전체·검사별·공식 단계별 목록과 동적 집계
- 임의 깊이·복수 root·가상 상위 라벨의 전체/부분 위계
- 부모·조상·자식·후손·형제 관계와 registry-backed 비교
- 공식 단계와 target-relative 관계의 구분, 영상면접 용어 분리
- stable ID 후속 결과와 stale ID 안전 제외
- 동일 대화의 후속 질문
- 서로 다른 대화 간 상태 격리
- 의미 검색의 고확신 단일 자동 조회, 저확신·다중 후보 저장과 후보 선택
- 런타임 재초기화 후 대화 복원
- 기존 `ask_competency()` 호출 방식 호환성
- 겹치는 역량명에서 가장 긴 정확한 이름 선택
- 체크포인트와 레지스트리 PostgreSQL setup 스크립트의 환경 변수·트랜잭션 검증
- 레지스트리 JSON 손상, 이름·별칭 충돌, 항목 수·해시 불일치 거부
- V1 62개 legacy 회귀와 V2 61/63개 동적 registry build
- V2 duplicate key·unknown field·collision·orphan·cycle·가변 깊이 검증
- stable ID 기반 add·retire·rename·move·instrument/root-prefix diff와 breaking 분류
- 업로드 dry-run, active read-only diff, 동일 해시 멱등성, JSONB 바인딩
- expected-current-version과 breaking guard, 원자적 활성화와 rollback
- 활성 runtime → V2 source → runtime 의미 동등성 round-trip
- 런타임 레지스트리의 프로세스당 1회 로드와 재초기화
- 공개 정적 파일의 비밀정보 비노출
- 내부 오류 응답의 경로·예외 정보 비노출
- Registry/Scope Writer의 검증 전 delta 차단, Registry 검증 실패 후 `registry_fallback` 복구와 grounding context 부재 시의 고정 장애, 범위 밖 템플릿의 무호출·redirect 회전·적대적 요약 무해화, meta Scope Writer의 재시도·fallback·취소, SSE event 순서·한국어 chunk 경계·최종 history 일치
- Node.js에서 실행하는 브라우저 SSE parser·한 말풍선 동기화·후보·중복 전송·새 대화 실패 복원·안전한 `textContent` 렌더링
- 같은 thread 직렬화와 서로 다른 thread 병렬 실행

## 상태와 보안 한계

- 현재 `thread_id`는 브라우저 `localStorage`의 `competency_chat_thread_id` 키에 저장된다.
- 실제 대화 상태는 `DATABASE_URL`이 가리키는 PostgreSQL 데이터베이스에 저장된다.
- 역량 Markdown 원문과 생성 JSON도 같은 데이터베이스의 `competency_data.registry_versions`에 버전별로 저장된다.
- 현재 프로세스는 시작 시점의 활성 레지스트리만 사용하므로 포인터 변경 뒤 재시작이 필요하다.
- 새 대화 버튼은 새 UUID를 만들며 기존 PostgreSQL 기록을 삭제하지 않는다.
- `thread_id`는 대화를 구분하는 값일 뿐 인증 수단이 아니다.
- 로그인과 요청 제한이 없는 상태로 인터넷에 공개하지 않는다.
- PostgreSQL에는 사용자의 질문과 챗봇 답변이 저장된다.
- 같은 프로세스에서는 같은 대화의 동시 요청을 직렬화한다. 여러 프로세스·인스턴스에 걸친 동시 실행 제어는 지원하지 않는다.
- 브라우저에는 API 키, DB 경로, 시스템 프롬프트, 전체 LangGraph 내부 상태를 전달하지 않는다.
- 프로세스별 안전 집계는 `runtime_metric_snapshot()`과 `runtime_metric` 로그로 확인할 수 있다. Gateway route, normalizer rule/issue(`near_match_target`, `draft_mention_dropped` 포함), Registry fallback(응답 모드별), Scope 템플릿(category별), meta Scope Writer의 첫 실패·재시도·fallback, fixed failure와 경로별 호출 수만 기록하며 원문 질문·정의·prompt·URL·예외 전문은 기록하지 않는다.

## 수동 확인 항목

1. 사용자 말풍선이 오른쪽, 챗봇 말풍선이 왼쪽에 나타나는지 확인한다.
2. 답변의 줄바꿈이 유지되는지 확인한다.
3. 답변 생성 중 입력창과 전송 버튼이 비활성화되는지 확인한다.
4. 오타 질문 뒤 후보 버튼을 눌러 같은 대화에서 답변이 이어지는지 확인한다.
5. 페이지 새로고침 뒤 메시지와 후보 버튼이 복원되는지 확인한다.
6. 새 대화 버튼을 누르면 화면과 대화 맥락이 초기화되는지 확인한다.
7. 모바일 폭에서 메시지와 입력 영역이 화면 밖으로 벗어나지 않는지 확인한다.
