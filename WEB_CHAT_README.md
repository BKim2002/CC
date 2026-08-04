# 역량 해석 웹 챗봇 실행 안내

이 프로젝트는 기존 LangGraph 역량 해석기를 FastAPI와 HTML·CSS·JavaScript 기반의 로컬 웹 채팅 화면에 연결한다.

## 구성

- `web_api.py`: FastAPI 서버, 요청 검증, 정적 파일 제공, LangGraph 연결
- `static/index.html`: 채팅 화면 구조
- `static/style.css`: 반응형 채팅 UI
- `static/chat.js`: 대화 생성·전송·복원과 후보 버튼 처리
- `competency_interpreter_v1.py`: 역량 해석 그래프와 PostgreSQL 체크포인터
- `tests/test_web_api.py`: API, 상태 격리, 복원 테스트

## 처음 한 번 설치

현재 가상환경에 프로젝트 의존성을 설치한다.

```powershell
cd C:\Users\ksy0823\.vscode\LangGraph
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

테스트까지 실행하려면 테스트 전용 패키지도 설치한다.

```powershell
python -m pip install pytest httpx langgraph-checkpoint-sqlite
```

자연어 의미 검색에 OpenAI를 사용하려면 프로젝트의 `.env`에 서버용 키를 설정한다. 키를 HTML이나 JavaScript에 넣지 않는다.

```text
OPENAI_API_KEY=발급받은_API_KEY
OPENAI_MODEL=gpt-5.4-mini
DATABASE_URL=postgresql://사용자명:비밀번호@호스트:5432/데이터베이스명?sslmode=require
```

정확한 역량명·별칭 검색과 오타 후보 검색은 OpenAI 키 없이도 작동한다.

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

대화 상태는 PostgreSQL 체크포인터에 저장된다. 현재 같은 프로세스 안의 동시 실행만 락으로 보호하므로 로컬에서는 `--workers 1`로 실행한다. 여러 프로세스나 서버 인스턴스로 확장할 때는 같은 `thread_id`에 대한 동시 요청 제어와 사용자 인증을 추가로 설계해야 한다.

## 테스트

테스트는 운영 PostgreSQL 대신 임시 SQLite 테스트 더블을 사용하며 OpenAI API 호출을 비활성화한다.

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
- 공개 정적 파일의 비밀정보 비노출
- 내부 오류 응답의 경로·예외 정보 비노출

## 상태와 보안 한계

- 현재 `thread_id`는 브라우저 `localStorage`의 `competency_chat_thread_id` 키에 저장된다.
- 실제 대화 상태는 `DATABASE_URL`이 가리키는 PostgreSQL 데이터베이스에 저장된다.
- 새 대화 버튼은 새 UUID를 만들며 기존 PostgreSQL 기록을 삭제하지 않는다.
- `thread_id`는 대화를 구분하는 값일 뿐 인증 수단이 아니다.
- 로그인 기능이 없으므로 인터넷에 공개하지 않고 로컬·소규모 용도로만 사용한다.
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
