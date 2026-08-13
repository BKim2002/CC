# 역량 해석 웹 챗봇 실행 안내

이 프로젝트는 LangGraph 역량 해석기를 FastAPI와 HTML·CSS·JavaScript 기반의 웹 채팅 화면에 연결한다. 로컬 Uvicorn 실행과 Vercel 배포 진입점을 모두 제공한다.

## 구성

런타임은 `chat/` 패키지 하나다. 질의 유형을 미리 정의하지 않고 레지스트리 전문을
컨텍스트에 실어 답한다. 설계 근거와 이력은 [docs/REBUILD_PLAN.md](docs/REBUILD_PLAN.md)에 있다.

- `app.py`: Vercel이 자동 탐지하는 FastAPI 진입점
- `chat/web.py`: FastAPI 서버, 요청 검증, 정적 파일 제공, SSE
- `chat/runtime.py`: 프로세스 수명, 활성 레지스트리 스냅샷, PostgreSQL 체크포인터
- `chat/turn.py`: 한 턴의 흐름 — 작성 → 검수 → 복구
- `chat/answer.py`: 도구 호출 루프
- `chat/tools.py`: 개수와 트리 순회. 모델이 추론으로 메우면 안 되는 것만 노출한다
- `chat/judge.py`: grounding judge. 주장 추출은 LLM이, 참·거짓 판정은 코드가 한다
- `chat/scope.py`: 거절에만 거는 항소심
- `chat/prompt.py`: 레지스트리 전문을 프롬프트로 만드는 계층
- `chat/models.py`: 역할별 모델·타임아웃·대화 맥락 상한
- `chat/registry.py`: PostgreSQL 활성 레지스트리의 검증·인덱스 생성
- `static/index.html`, `static/style.css`, `static/chat.js`: 채팅 화면과 SSE 처리
- `evals/cases.yaml`: 평가 케이스 53건
- `evals/spike.py`: 평가 실행기 (`--run` 없이는 견적만 낸다)
- `scripts/setup_checkpoint_database.py`: LangGraph 체크포인트 테이블 준비
- `scripts/setup_competency_registry_database.py`: 버전형 역량 레지스트리 스키마 준비
- `scripts/build_registry.py`: V1/V2 원본 형식을 명시적으로 선택해 runtime JSON 1.0으로 변환
- `scripts/registry_source_v1.py`: 기존 표 형식 Markdown과 62개 규칙을 보존한 legacy adapter
- `scripts/registry_source_v2.py`: fenced YAML 기반 structured Markdown V2 parser
- `scripts/registry_compiler.py`: stable ID와 `parent_id`를 runtime 위계로 컴파일
- `scripts/registry_diff.py`: 활성 버전과 후보를 stable ID 기준으로 비교
- `scripts/upload_competency_registry.py`: 외부 원본을 검증·비교·업로드·활성화
- `scripts/export_active_registry_v2.py`: 활성 DB 버전을 읽기 전용으로 V2 원본에 내보내기
- `tests/`: 런타임, API, 레지스트리와 DB 도구 테스트
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
python -m uvicorn chat.web:app --reload
```

브라우저에서 다음 주소를 연다.

```text
http://127.0.0.1:8000
```

`--reload`는 파일 변경을 감지해 개발 서버를 재시작하는 개발용 옵션이다.

## 로컬 일반 실행

자동 재시작이 필요하지 않으면 다음처럼 실행한다.

```powershell
python -m uvicorn chat.web:app --host 127.0.0.1 --port 8000 --workers 1
```

대화 상태는 PostgreSQL 체크포인터에 저장된다. 역량 정의는 서버 프로세스가 시작될 때 활성 DB 버전에서 한 번 로드된다. 런타임 초기화와 대화 실행 잠금은 분리되어 같은 `thread_id`의 요청은 겹치지 않고, 서로 다른 thread는 병렬로 실행할 수 있다. 이 제어는 한 프로세스 안에서만 유효하므로 로컬에서는 `--workers 1`로 실행한다. 여러 프로세스나 서버 인스턴스로 확장할 때는 분산 잠금과 사용자 인증을 별도로 설계해야 한다.

## 지원 질문과 용어

챗봇은 활성 레지스트리에 등록된 사실에 근거해서만 답한다. **답할 수 있는 질문의
종류를 미리 정해두지 않는다.** 레지스트리 62개 항목의 모든 필드가 매 턴 컨텍스트에
실리므로, 표현이 낯설다는 이유로 되묻지 않는다.

지원하지 않는 것은 명확하다. 날짜·시각·날씨·뉴스, 일반 개념 설명, 코딩, 개인 역량
점수·진단, 채용 판단, 의료·법률·금융 조언, 외부 작업 실행. 이런 질문은 오류가 아니라
정상적인 범위 안내로 끝나며, 무엇을 도울 수 있는지 함께 알린다.

레지스트리에 없는 방법·절차·해석 기준·권고도 제시하지 않는다. `역량검사 결과를 어떻게
해석해?`처럼 물으면 등록되어 있지 않다고 알리고 등록된 구조만 설명한다.

역량명 형태이지만 등록되지 않은 이름(`정의감`, `혁신성`, `협업`)은 일반 상식으로
정의하지 않는다. 미등록임을 알리고 비슷한 등록 이름이 있으면 후보로 제시한다.

필기 역량검사의 `상위요인`, `중위요인`, `하위요인`, `최하위요인`은 각각 `L1`, `L2`,
`L3`, `L4`를 뜻한다. 현재 활성 버전에서 3개, 10개, 30개, 9개이며 `역검 종합점수`는 그
위의 종합점수다. 이 숫자는 하드코딩하지 않고 매번 활성 스냅샷에서 계산한다.

영상면접에는 이 4단계 용어를 쓰지 않는다. `factor`는 `평가요인`, `item`은 `세부항목`이다.

**개수와 관계는 도구로 확인한다.** 눈으로 세면 틀리기 때문이다. 도구는 각 항목의 위계
이름을 함께 돌려주므로 자식을 부모의 단계 이름으로 부르는 일이 없고, `전략성 아래에 몇
개`처럼 직속과 후손이 갈리는 질문에는 양쪽을 함께 답한다.

**답변은 공개 전에 검수한다.** 답변에 등장한 숫자가 무엇을 주장하는지 판정 모델이
구조화하고, 그 주장이 참인지는 코드가 레지스트리에서 직접 계산해 대조한다. 등록되지
않은 이름을 등록된 역량처럼 다루면 그 자리에서 걸린다. 틀린 값이 있으면 참값을 주입해
한 번 다시 쓰고, 그래도 실패하면 확인된 사실만 나열한다.

**거절은 한 번 더 본다.** 등록 사실을 하나도 말하지 않은 답변에 대해, "이 질문에 답하려면
레지스트리에서 무엇을 찾아야 하는가"를 따로 묻는다. 구체적인 등록 역량을 댈 수 있으면
거절을 뒤집고 다시 쓴다. 역량 질문이 잘못 거절되는 것을 막기 위한 장치이며, 반대 방향은
grounding 검수가 막는다.

## 채팅 API와 스트리밍

`POST /api/chat`은 최종 응답을 반환한다.

```json
{
  "thread_id": "UUID",
  "answer": "최종 답변",
  "candidates": []
}
```

`candidates`는 항상 빈 배열이다. 후보는 답변 본문에서 제시하며, 필드는 프런트엔드
호환을 위해 남겨 두었다.

`POST /api/chat/stream`은 같은 요청 본문을 받고 UTF-8 `text/event-stream`을 반환한다.
공개 event는 `start`, `status`, `done`, `error`다.

**`delta`는 보내지 않는다.** 검수를 통과하지 않은 토큰은 공개하지 않기 때문이다. 대신
`status`로 진행 단계를 알려 대기 중 화면이 비어 있지 않게 한다. `done.answer`와
비스트리밍 응답, history의 마지막 assistant 메시지는 같은 문자열이다.

거부된 초안, prompt, DB 연결 정보, 예외 전문은 상태나 event에 넣지 않는다.

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

- 메인 HTML, CSS, JavaScript 제공과 브라우저 SSE parser
- 서버 상태, UUID 대화 생성, 공백·길이·잘못된 UUID 검증
- v1과 동일한 API 계약, `delta` 미발신, 진행 단계 알림, 스트리밍·비스트리밍 답변 일치
- 내부 오류의 경로·예외·연결 정보 비노출
- 레지스트리 전문 렌더링에 모든 필드가 실리는지, 블록이 항상 프롬프트 앞에 오는지
- 도구의 트리 순회·집계 정확성과 위계 이름 동봉, 직속/후손이 갈릴 때의 안내
- 도구 호출 루프의 결과 반영·기록, 미등록 이름 처리, 폭주 중단
- grounding judge에 판정 필드가 없다는 것, 틀린 수·미등록 이름 적발, 부분 판정 거부
- 재생성이 참값을 들고 가는지, 두 번 실패하면 도구 결과만 렌더링하는지
- 항소심이 거절에만 걸리는지, 지어낸 이름으로는 뒤집히지 않는지, 원문 맥락을 보는지
- 레지스트리 JSON 손상, 이름·별칭 충돌, 항목 수·해시 불일치 거부
- V1 62개 legacy 회귀와 V2 동적 registry build, duplicate·orphan·cycle·가변 깊이 검증
- stable ID 기반 add·retire·rename·move diff와 breaking 분류
- 업로드 dry-run, 멱등성, expected-current-version guard, 원자적 활성화와 rollback
- 활성 runtime → V2 source → runtime 의미 동등성 round-trip
- 체크포인트와 레지스트리 PostgreSQL setup 스크립트의 환경 변수·트랜잭션 검증

실제 모델을 부르는 평가는 `pytest`에 포함하지 않는다. `evals/spike.py`가 담당하며
`--run` 없이는 견적만 낸다.

```powershell
python evals\spike.py --run
python evals\spike.py --run --id g06 --repeat 3
```

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
