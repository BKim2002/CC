"""1단계 스파이크 — 레지스트리 전문만 주고 LLM이 답할 수 있는지 본다.

계획: docs/REBUILD_PLAN.md 5절 1단계

검증하려는 가정 하나:

    레지스트리 전문을 컨텍스트에 넣으면, 질의 유형을 미리 정의하지 않아도
    LLM이 역량 질문에 답하고 범위 밖 질문을 거절할 수 있다.

1단계에서는 레지스트리 + 시스템 프롬프트 + 질문뿐이었고, 그 상태로
48/53이 통과했다. 이후 단계가 얹은 축소를 같은 케이스로 계속 검증한다.

  2.5단계  처방 금지 한 줄 (코드 변경 없음)
  3단계    개수·관계 도구 (chat/answer.py 의 도구 호출 루프)

여전히 LangGraph도 judge도 정규화기도 없다. 질의 유형을 정의하지 않는다는
전제는 그대로다 — 도구는 유형이 아니라 자료구조에서 나온다.

사용법:

    python evals/spike.py                      # 견적만 낸다. API 호출 없음
    python evals/spike.py --run                # 전체 53건 실행
    python evals/spike.py --run --group aggregate
    python evals/spike.py --run --id g01 --id u07
    python evals/spike.py --run --id g06 --repeat 3   # 판정이 뒤집혔을 때
    python evals/spike.py --run --limit 5
    python evals/spike.py --run --no-tier-glossary
    python evals/spike.py --audit               # 검증 계층이 뭘 잡았는지

같은 프롬프트로도 판정이 갈린다.  `g06`은 5회 중 4회가 틀린 티어 이름을
냈고 1회만 맞았다.  한 번의 통과를 통과로 세면 안 되므로, 판정이 뒤집힌
케이스는 `--repeat 3`으로 다시 확인한다.

결과는 evals/results/ 에 쓴다. 프롬프트 전문은 저장하지 않는다 — 레지스트리
정의가 추적 파일에 들어가지 않게 하는 기존 규약을 따른다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from chat.models import ModelRole, create_model, selected_model_name  # noqa: E402
from chat.prompt import build_system_message  # noqa: E402
from chat.registry import RegistrySnapshot, load_active_registry  # noqa: E402
from chat.turn import run_turn  # noqa: E402

CASES_PATH = PROJECT_ROOT / "evals" / "cases.yaml"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"


def load_cases(
    *,
    groups: Sequence[str],
    ids: Sequence[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    document = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = document["cases"]

    if ids:
        wanted = set(ids)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted.difference(case["id"] for case in cases)
        if missing:
            raise SystemExit(f"없는 케이스 id: {', '.join(sorted(missing))}")
    if groups:
        allowed = set(groups)
        cases = [case for case in cases if case["group"] in allowed]
    if limit is not None:
        cases = cases[:limit]

    if not cases:
        raise SystemExit("실행할 케이스가 없습니다.")
    return cases


def planned_calls(cases: Sequence[Mapping[str, Any]], repeat: int) -> int:
    """맥락 있는 케이스는 앞선 발화도 실제로 호출해야 진짜 맥락이 된다."""

    return repeat * sum(1 + len(case.get("context", ())) for case in cases)


def auto_verdict(expected: Any, answer: str) -> bool | None:
    """정답값이 확정된 케이스만 기계로 판정한다.

    판정할 수 없으면 None을 돌려준다 — 대부분은 사람이 읽어야 한다.
    dict 정답값은 모든 값이 답변에 나타나야 통과다.  ``g06``처럼 해석이
    둘인 케이스에서 한쪽만 답하면 여기서 걸린다.
    """

    if expected is None:
        return None
    if isinstance(expected, bool):
        return None
    if isinstance(expected, int):
        return bool(re.search(rf"(?<!\d){expected}(?!\d)", answer))
    if isinstance(expected, str):
        return expected in answer
    if isinstance(expected, Mapping):
        return all(auto_verdict(value, answer) for value in expected.values())
    return None


def audit_findings() -> int:
    """지금까지의 실행 기록에서 검증 계층이 무엇을 잡았는지 센다.

    "검증 계층은 참 양성으로 자기 존재를 증명한다"는 규칙을 확인 가능하게
    만든다(docs/REBUILD_PLAN.md).  문단으로만 있는 규칙은 지켜지지 않는다.

    참·거짓 판별은 사람이 한다 -- 지적 하나하나를 읽고 답변이 실제로 틀렸는지
    봐야 한다.  여기서는 무엇을 몇 번 지적했는지와 그 원문만 모아 준다.
    """

    records = sorted(RESULTS_DIR.glob("spike-*.jsonl"))
    if not records:
        print("실행 기록이 없습니다.")
        return 0

    counts: dict[str, int] = {}
    rows: list[tuple[str, str, str, str]] = []
    for path in records:
        for line in path.read_text(encoding="utf-8").splitlines():
            grounding = json.loads(line).get("grounding") or {}
            case_id = json.loads(line)["id"]
            for finding in grounding.get("findings", ()):
                kind = finding["kind"]
                counts[kind] = counts.get(kind, 0) + 1
                rows.append((path.stem[-6:], case_id, kind, finding["detail"]))

    print(f"실행 기록 {len(records)}개 · 지적 {len(rows)}건")
    for kind, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {kind:22} {count}")
    print()
    print("각 지적이 참 양성인지 직접 확인한다. 답변이 실제로 틀렸는가?")
    print("참 양성을 댈 수 없는 검증은 정교화하지 말고 제거한다.")
    print()
    for stamp, case_id, kind, detail in rows:
        print(f"  {stamp} {case_id:5} {kind:22} {detail[:88]}")
    return 0


def run_case(
    model: Any,
    judge_model: Any,
    appeal_model: Any,
    snapshot: RegistrySnapshot,
    system_message: str,
    case: Mapping[str, Any],
    run_index: int = 1,
) -> dict[str, Any]:
    """앞선 발화를 실제로 호출해 맥락을 만든 뒤 본 질문을 던진다."""

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages: list[Any] = [SystemMessage(content=system_message)]
    usage = {"input": 0, "output": 0, "cached": 0}
    started = time.perf_counter()

    for previous in case.get("context", ()):
        messages.append(HumanMessage(content=previous))
        prior = run_turn(model, judge_model, snapshot, messages, appeal_model=appeal_model)
        messages.append(AIMessage(content=prior.text))
        for key, value in prior.usage.items():
            usage[key] += value

    messages.append(HumanMessage(content=case["q"]))
    result = run_turn(model, judge_model, snapshot, messages, appeal_model=appeal_model)
    answer_elapsed = result.seconds
    for key, value in result.usage.items():
        usage[key] += value

    return {
        "id": case["id"],
        "run": run_index,
        "group": case["group"],
        "q": case["q"],
        "expect": case["expect"],
        "expected": case.get("expected"),
        "must_not": case.get("must_not"),
        "note": case.get("note"),
        "src": case.get("src"),
        "context": list(case.get("context", ())),
        "answer": result.text,
        "tool_calls": [
            {"name": call.name, "arguments": dict(call.arguments), "result": call.result}
            for written in result.answers
            for call in written.tool_calls
        ],
        "model_rounds": sum(written.model_rounds for written in result.answers),
        "attempts": result.attempts,
        "used_fallback": result.used_fallback,
        "appeal": (
            {
                "overturned": result.appealed.overturned,
                "lookup": result.appealed.lookup,
                "reason": result.appealed.reason,
            }
            if result.appealed is not None
            else None
        ),
        "grounding": (
            {
                "ok": result.verdict.ok,
                "judge_called": result.verdict.judge_called,
                "uses_registry": result.verdict.uses_registry,
                "findings": [
                    {"kind": f.kind, "detail": f.detail} for f in result.verdict.findings
                ],
            }
            if result.verdict is not None
            else None
        ),
        "answer_seconds": round(answer_elapsed, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "usage": usage,
    }


def _auto_column(runs: Sequence[Mapping[str, Any]]) -> str:
    """반복 실행의 자동 판정을 한 칸으로 요약한다.

    정답값이 없으면 빈 칸이다.  반복 사이에 결과가 갈리면 그 사실 자체가
    가장 중요한 정보이므로 `흔들림`으로 표시한다.
    """

    verdicts = [auto_verdict(run["expected"], str(run["answer"])) for run in runs]
    if any(verdict is None for verdict in verdicts):
        return ""
    passed = sum(1 for verdict in verdicts if verdict)
    if passed == len(verdicts):
        return "O"
    if passed == 0:
        return "X"
    return f"흔들림 {passed}/{len(verdicts)}"


def write_report(
    results: Sequence[Mapping[str, Any]],
    *,
    stamp: str,
    model_name: str,
    system_chars: int,
    tier_glossary: bool,
    repeat: int,
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"spike-{stamp}.md"
    raw_path = RESULTS_DIR / f"spike-{stamp}.jsonl"

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["id"], []).append(result)

    total_input = sum(result["usage"]["input"] for result in results)
    total_output = sum(result["usage"]["output"] for result in results)
    total_cached = sum(result["usage"]["cached"] for result in results)
    answer_times = sorted(result["answer_seconds"] for result in results)
    median = answer_times[len(answer_times) // 2] if answer_times else 0.0

    lines = [
        f"# 1단계 스파이크 결과 — {stamp}",
        "",
        f"- 모델: `{model_name}`",
        f"- 시스템 메시지: {system_chars:,}자 (tier glossary {'포함' if tier_glossary else '제외'})",
        f"- 케이스: {len(grouped)}건" + (f" × {repeat}회" if repeat > 1 else ""),
        f"- 토큰: 입력 {total_input:,} (캐시 {total_cached:,}) / 출력 {total_output:,}",
        f"- 답변 지연 중앙값: {median}초",
        "",
        "채점은 `expect`와 `must_not`을 기준으로 한다. `must_not`에 걸리면 즉시 실패다.",
        "각 케이스의 판정 칸에 O/X를 적고, X는 사유를 한 줄 남긴다.",
        "",
    ]
    if repeat > 1:
        lines += [
            "`자동` 칸은 정답값이 확정된 케이스만 기계로 판정한 것이다. "
            "`흔들림`은 같은 프롬프트에서 반복 결과가 갈렸다는 뜻이고, "
            "그 자체가 판정 대상이다 — 한 번의 통과를 통과로 세지 않는다.",
            "",
        ]

    lines += [
        "| id | 그룹 | 기대 | 자동 | 판정 | 사유 |",
        "|---|---|---|:-:|:-:|---|",
    ]
    for case_id, runs in grouped.items():
        head = runs[0]
        lines.append(
            f"| {case_id} | {head['group']} | {head['expect']} | "
            f"{_auto_column(runs)} |  |  |"
        )
    lines += ["", "---", ""]

    for case_id, runs in grouped.items():
        head = runs[0]
        lines.append(f"## {case_id} · {head['group']} · 기대 `{head['expect']}`")
        lines.append("")
        if head["context"]:
            lines.append("**앞선 발화**")
            for previous in head["context"]:
                lines.append(f"- {previous}")
            lines.append("")
        lines.append(f"**질문** {head['q']}")
        lines.append("")
        if head["expected"] is not None:
            lines.append(f"**정답값** `{head['expected']}`")
            lines.append("")
        if head["must_not"]:
            lines.append(f"**must_not** {head['must_not']}")
            lines.append("")
        if head["note"]:
            lines.append(f"**메모** {head['note'].strip()}")
            lines.append("")

        for run in runs:
            verdict = auto_verdict(run["expected"], str(run["answer"]))
            mark = "" if verdict is None else f" · 자동 {'O' if verdict else 'X'}"
            if len(runs) > 1:
                lines.append(f"**{run['run']}회차{mark}**")
                lines.append("")
            elif mark:
                lines.append(f"**자동 판정** {'O' if verdict else 'X'}")
                lines.append("")
            lines.append("```")
            lines.append(str(run["answer"]).strip())
            lines.append("```")
            lines.append("")
            for call in run.get("tool_calls", ()):
                arguments = json.dumps(call["arguments"], ensure_ascii=False)
                summary = call["result"].get("count", call["result"].get("error", ""))
                lines.append(f"- 도구 `{call['name']}({arguments})` → `{summary}`")
            if run.get("tool_calls"):
                lines.append("")
            lines.append(
                f"<sub>{run['answer_seconds']}초 · 모델 {run.get('model_rounds', 1)}회 · "
                f"입력 {run['usage']['input']:,} / 출력 {run['usage']['output']:,}</sub>"
            )
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    with raw_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return report_path, raw_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="실제로 API를 호출한다. 없으면 견적만 낸다",
    )
    parser.add_argument("--group", action="append", default=[], help="그룹 필터 (반복 가능)")
    parser.add_argument("--id", action="append", default=[], dest="ids", help="케이스 id 필터")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N건만")
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "지금까지의 실행 기록에서 검증 계층이 무엇을 지적했는지 센다. "
            "API를 호출하지 않는다"
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help=(
            "케이스마다 N회 실행한다. 판정이 뒤집힌 케이스를 다시 볼 때 쓴다 — "
            "같은 프롬프트에서도 결과가 갈리므로 한 번의 통과를 통과로 세지 않는다"
        ),
    )
    parser.add_argument(
        "--no-tier-glossary",
        action="store_true",
        help="상위/중위/하위요인 용어표를 빼고 돌린다 — level 값만으로 답하는지 본다",
    )
    arguments = parser.parse_args(argv)

    # 감사는 기록만 읽는다. DB도 API도 필요 없다.
    if arguments.audit:
        return audit_findings()

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL이 설정되지 않았습니다.")

    snapshot = load_active_registry(database_url)
    cases = load_cases(groups=arguments.group, ids=arguments.ids, limit=arguments.limit)
    system_message = build_system_message(
        snapshot,
        tier_glossary=not arguments.no_tier_glossary,
    )
    if arguments.repeat < 1:
        raise SystemExit("--repeat는 1 이상이어야 합니다.")

    model_name = selected_model_name(ModelRole.ANSWER)
    calls = planned_calls(cases, arguments.repeat)

    # 실측: 10,672자 → 6,103 토큰 (glossary 포함, 2026-08-13). 글자당 0.57.
    estimated_prompt_tokens = int(len(system_message) * 0.6)

    print(f"레지스트리   {snapshot.source_filename} · {len(snapshot.document['items'])}개 항목")
    print(f"시스템 메시지 {len(system_message):,}자 (약 {estimated_prompt_tokens:,} 토큰)")
    print(f"tier 용어표   {'포함' if not arguments.no_tier_glossary else '제외'}")
    print(f"모델         {model_name}")
    print(f"케이스       {len(cases)}건" + (f" × {arguments.repeat}회" if arguments.repeat > 1 else ""))
    print(
        f"API 호출     {calls}회 "
        f"(맥락 발화 {calls - len(cases) * arguments.repeat}회 포함)"
    )
    print(
        f"입력 토큰    최대 약 {estimated_prompt_tokens * calls:,} "
        "— 캐시가 걸리면 실제로는 크게 줄어든다"
    )
    print()

    if not arguments.run:
        print("견적만 냈습니다. 실제로 호출하려면 --run 을 붙이세요.")
        return 0

    model = create_model(ModelRole.ANSWER, timeout=60)
    judge_model = create_model(ModelRole.JUDGE, timeout=60)
    appeal_model = create_model(ModelRole.APPEAL, timeout=60)

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index:2}/{len(cases)}] {case['id']:4} {case['q'][:34]}", end=" ", flush=True)
        timings: list[str] = []
        for run_index in range(1, arguments.repeat + 1):
            try:
                result = run_case(
                    model, judge_model, appeal_model, snapshot, system_message, case, run_index
                )
            except Exception as error:  # 한 건이 죽어도 나머지는 계속 본다
                timings.append(f"실패({type(error).__name__})")
                results.append(
                    {
                        "id": case["id"],
                        "run": run_index,
                        "group": case["group"],
                        "q": case["q"],
                        "expect": case["expect"],
                        "expected": case.get("expected"),
                        "must_not": case.get("must_not"),
                        "note": case.get("note"),
                        "src": case.get("src"),
                        "context": list(case.get("context", ())),
                        "answer": f"(호출 실패: {type(error).__name__})",
                        "tool_calls": [],
                        "model_rounds": 0,
                        "attempts": 0,
                        "used_fallback": False,
                        "appeal": None,
                        "grounding": None,
                        "answer_seconds": 0.0,
                        "total_seconds": 0.0,
                        "usage": {"input": 0, "output": 0, "cached": 0},
                    }
                )
                continue
            timings.append(f"{result['answer_seconds']}초")
            results.append(result)
        print(" / ".join(timings))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_path, raw_path = write_report(
        results,
        stamp=stamp,
        model_name=model_name,
        system_chars=len(system_message),
        tier_glossary=not arguments.no_tier_glossary,
        repeat=arguments.repeat,
    )

    print()
    print(f"보고서 {report_path.relative_to(PROJECT_ROOT)}")
    print(f"원본   {raw_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
