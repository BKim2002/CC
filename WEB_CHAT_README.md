# 역량 해석 웹 챗봇 실행 안내

이 프로젝트는 LangGraph 역량 해석기를 FastAPI와 HTML·CSS·JavaScript 기반의 웹 채팅 화면에 연결한다. 로컬 Uvicorn 실행과 Vercel 배포 진입점을 모두 제공한다.

## 구성

- `app.py`: Vercel이 자동 탐지하는 FastAPI 진입점
- `web_api.py`: FastAPI 서버, 요청 검증, 정적 파일 제공, LangGraph 연결
- `static/index.html`: 채팅 화면 구조
- `static/style.css`: 반응형 채팅 UI
- `static/chat.js`: 대화 생성·전송·복원과 후보 버튼 처리
- `competency_interpreter.py`: 역량 해석 그래프, 활성 레지스트리 스냅샷, PostgreSQL 체크포인터
- `competency_registry.py`: PostgreSQL 활성 레지스트리의 검증·인덱스 생성
- `scripts/setup_checkpoint_database.py`: LangGraph 체크포인트 테이블 준비
- `scripts/setup_competency_registry_database.py`: 버전형 역량 레지스트리 스키마 준비
- `scripts/upload_competency_registry.py`: 외부 Markdown 원본을 검증·업로드·활성화
- `scripts/build_registry.py`: 외부 Markdown을 로컬에서 검증하거나 JSON으로 변환
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

자연어 의미 검색에 OpenAI를 사용하려면 프로젝트의 `.env`에 서버용 키를 설정한다. 키를 HTML이나 JavaScript에 넣지 않는다.

```text
OPENAI_API_KEY=발급받은_API_KEY
OPENAI_MODEL=gpt-5.4-mini
DATABASE_URL=postgresql://사용자명:비밀번호@호스트:5432/데이터베이스명?sslmode=require
```

정확한 역량명·별칭 검색은 OpenAI 키 없이도 작동한다. 자연어 질의, 오타 해석, 의미 검색에는 OpenAI 키가 필요하다.

## PostgreSQL 최초 준비와 레지스트리 적재

새 데이터베이스에서는 LangGraph 체크포인트와 애플리케이션 레지스트리를 각각 준비한다. 두 setup 스크립트는 책임이 다르며, 기존 구조가 있는 데이터베이스에서 다시 실행해도 안전하다.

```powershell
python .\scripts\setup_checkpoint_database.py
python .\scripts\setup_competency_registry_database.py
```

역량 원본은 저장소에 두지 않는다. 접근이 통제된 외부 경로의 Markdown을 먼저 dry-run으로 검증한 뒤 같은 파일을 업로드·활성화한다.

```powershell
python .\scripts\upload_competency_registry.py `
  --source "C:\secure-source\competency-definitions.md" `
  --dry-run

python .\scripts\upload_competency_registry.py `
  --source "C:\secure-source\competency-definitions.md" `
  --activate
```

업로드는 원본 SHA-256을 버전 키로 사용한다. 같은 원본을 다시 실행하면 중복 행을 만들지 않고 기존 버전을 사용하며, `--activate` 전환은 버전 INSERT와 같은 트랜잭션에서 처리된다. 출력에는 버전 ID, SHA-256, 항목 수만 나타난다.

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

대화 상태는 PostgreSQL 체크포인터에 저장된다. 역량 정의는 서버 프로세스가 시작될 때 활성 DB 버전에서 한 번 로드된다. 현재 같은 프로세스 안의 동시 실행만 락으로 보호하므로 로컬에서는 `--workers 1`로 실행한다. 여러 프로세스나 서버 인스턴스로 확장할 때는 같은 `thread_id`에 대한 동시 요청 제어와 사용자 인증을 추가로 설계해야 한다.

## Vercel 배포

Vercel은 루트의 `app.py`를 FastAPI 진입점으로 사용하고, `.python-version`과 `requirements.txt`로 런타임을 준비한다. 배포 전 Vercel 환경 변수에 `DATABASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`을 설정한다.

배포 전에 운영 DB에 레지스트리 버전을 먼저 활성화해야 한다. 새 버전을 활성화한 뒤에는 각 인스턴스가 새 스냅샷을 읽도록 재배포하거나 재시작한다.

현재 애플리케이션에는 로그인과 요청 횟수 제한이 없다. Vercel Deployment Protection 같은 접근 보호를 켜거나 애플리케이션 인증과 요청 제한을 구현하기 전에는 공개 URL로 운영하지 않는다. 여러 인스턴스에서 같은 `thread_id`를 동시에 처리하는 사용 방식도 지원하지 않는다.

## 레지스트리 업데이트와 롤백

새 정의를 배포할 때는 현재 활성 버전 ID를 먼저 기록하고, 새 외부 Markdown을 dry-run한 뒤 업로드·활성화한다. SQL 또는 관리 도구로 활성 버전의 해시와 항목 수를 확인한 다음 애플리케이션을 재배포하거나 재시작한다.

```sql
SELECT version_id
FROM competency_data.registry_current
WHERE singleton = 1;
```

데이터만 이전 버전으로 되돌릴 때는 기록해 둔 ID로 현재 포인터를 바꾸고 모든 애플리케이션 인스턴스를 다시 시작한다.

```sql
UPDATE competency_data.registry_current
SET version_id = 이전_VERSION_ID,
    updated_at = NOW()
WHERE singleton = 1;
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
- 동일 대화의 후속 질문
- 서로 다른 대화 간 상태 격리
- 오타 후보 저장과 후보 선택
- 런타임 재초기화 후 대화 복원
- 기존 `ask_competency()` 호출 방식 호환성
- 겹치는 역량명에서 가장 긴 정확한 이름 선택
- 체크포인트와 레지스트리 PostgreSQL setup 스크립트의 환경 변수·트랜잭션 검증
- 레지스트리 JSON 손상, 이름·별칭 충돌, 항목 수·해시 불일치 거부
- 업로드 dry-run, 동일 해시 멱등성, JSONB 바인딩, 원자적 활성화와 rollback
- 런타임 레지스트리의 프로세스당 1회 로드와 재초기화
- 공개 정적 파일의 비밀정보 비노출
- 내부 오류 응답의 경로·예외 정보 비노출

## 상태와 보안 한계

- 현재 `thread_id`는 브라우저 `localStorage`의 `competency_chat_thread_id` 키에 저장된다.
- 실제 대화 상태는 `DATABASE_URL`이 가리키는 PostgreSQL 데이터베이스에 저장된다.
- 역량 Markdown 원문과 생성 JSON도 같은 데이터베이스의 `competency_data.registry_versions`에 버전별로 저장된다.
- 현재 프로세스는 시작 시점의 활성 레지스트리만 사용하므로 포인터 변경 뒤 재시작이 필요하다.
- 새 대화 버튼은 새 UUID를 만들며 기존 PostgreSQL 기록을 삭제하지 않는다.
- `thread_id`는 대화를 구분하는 값일 뿐 인증 수단이 아니다.
- 로그인과 요청 제한이 없는 상태로 인터넷에 공개하지 않는다.
- PostgreSQL에는 사용자의 질문과 챗봇 답변이 저장된다.
- 같은 대화를 여러 탭에서 동시에 호출하는 동작은 이번 버전의 지원 범위가 아니다.
- 브라우저에는 API 키, DB 경로, 시스템 프롬프트, 전체 LangGraph 내부 상태를 전달하지 않는다.

## 수동 확인 항목

1. 사용자 말풍선이 오른쪽, 챗봇 말풍선이 왼쪽에 나타나는지 확인한다.
2. 답변의 줄바꿈이 유지되는지 확인한다.
3. 답변 생성 중 입력창과 전송 버튼이 비활성화되는지 확인한다.
4. 오타 질문 뒤 후보 버튼을 눌러 같은 대화에서 답변이 이어지는지 확인한다.
5. 페이지 새로고침 뒤 메시지와 후보 버튼이 복원되는지 확인한다.
6. 새 대화 버튼을 누르면 화면과 대화 맥락이 초기화되는지 확인한다.
7. 모바일 폭에서 메시지와 입력 영역이 화면 밖으로 벗어나지 않는지 확인한다.
