"""B-E1 on LongMemEval_M (MEM2-220): the mem arm only, one question at a time, resumable.

M is the same 500 questions as S, each with about 500 sessions (2.7 GB). So the runner never holds the
file: `lme.iter_questions` streams one question (about 5 MB), `retrieve` builds a fresh store for it and
deletes it afterwards, the answerer and judge run through `e1.run_e1` for that one question, and the
result is appended to a jsonl before the next question is read.

Files (all next to `--out`): `<out stem>.jsonl` one line per finished question; `<out stem>.STATUS` the
checkpoint (state running / complete / stopped / crashed, done, last qid, spend, and the `meta` the run
started with); `--out` the summary, recomputed from the jsonl alone (`summarize`). A restart skips the
qids in the jsonl, drops a torn last line, counts what earlier sessions spent against the budget, and
refuses (`ResumeMismatch`) if the answerer, judge, prompts, gate or config fingerprint differ from the
ones the STATUS file recorded: a run is one measurement, not a patchwork.

M uses S's questions, and S's test split was seen once (25 Oct): no tuning on M (`NOTE`).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench import e1
from bench.datasets import lme
from bench.llm import LLM
from bench.lme_eval import Retrieved
from bench.metrics import ir
from bench.metrics.tokens import est_tokens

QUESTIONS = 500
NOTE = (
    "LongMemEval_M uses S's questions, and S's test split was seen once (25 Oct): M is a second "
    "read of the same questions with a harder haystack, not an independent test. No tuning on M."
)
CHECKED = (
    "answerer",
    "judge",
    "prompt_version",
    "prompts_sha256",
    "judge_version",
    "gate",
    "config",
)  # resume must match


class ResumeMismatch(RuntimeError):
    """The STATUS file belongs to a run with other settings."""


def paths(out: Path) -> tuple[Path, Path]:
    """(jsonl, STATUS) next to the summary file."""
    return out.with_suffix(".jsonl"), out.with_suffix(".STATUS")


def read_rows(jsonl: Path) -> list[dict[str, Any]]:
    """The finished questions; a torn last line (a kill mid-write) is cut off the file, not trusted."""
    if not jsonl.exists():
        return []
    rows, good = [], 0
    data = jsonl.read_bytes()
    for raw in data.splitlines(keepends=True):
        try:
            row = json.loads(raw)
        except ValueError:
            break
        if not raw.endswith(b"\n"):
            break
        rows.append(row)
        good += len(raw)
    if good != len(data):
        with open(jsonl, "r+b") as f:
            f.truncate(good)
    return rows


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {"n": len(ordered), "mean": sum(ordered) / len(ordered),
            "p95": ordered[-(-95 * len(ordered) // 100) - 1]}  # fmt: skip


def summarize(jsonl: Path, *, meta: dict[str, Any], planned: int | None,
              stopped: str | None = None) -> dict[str, Any]:  # fmt: skip
    """The run record, from the jsonl alone (offline, deterministic up to the bootstrap seed)."""
    rows = read_rows(jsonl)
    usd = round(sum(r["usd"] for r in rows), 4)
    run = {"items": rows, "conditions": ["mem"], "complete": False, "cost": {"usd": usd}}
    metrics, detail = e1.score_run(run)
    mem = detail["per_condition"]["mem"]
    r5 = [r["recall_any5"] for r in rows if r.get("recall_any5") is not None]
    complete = stopped is None and len(rows) == (planned if planned is not None else len(rows))
    return {**meta, "n": len(rows), "planned": planned, "complete": complete,
            "stopped": stopped, "conditions": ["mem"], "prompt_version": e1.PROMPT_VERSION,
            "prompts_sha256": e1.prompts_sha256(), "judge_version": e1.JUDGE_VERSION,
            "judge_flagged": int(metrics["judge_flagged"]), "accuracy": metrics["accuracy"],
            "accuracy_ci_lo": metrics["accuracy_ci_lo"], "accuracy_ci_hi": metrics["accuracy_ci_hi"],
            "accuracy_macro": metrics["accuracy_macro"],
            "abstention_accuracy": metrics["abstention_accuracy"], "per_type": mem["per_type"],
            "recall_any5": sum(r5) / len(r5) if r5 else None, "recall_any5_n": len(r5),
            "delivered_tokens": _dist([r["delivered_tokens"] for r in rows]),
            "ingest_ms": _dist([r["ingest_ms"] for r in rows]),
            "query_ms": _dist([r["query_ms"] for r in rows]),
            "wall_s": _dist([r["wall_s"] for r in rows]),
            "cost": {"usd": usd, "answer_judge_per_question": usd / len(rows) if rows else 0.0},
            "note": NOTE}  # fmt: skip


def estimate(rec: dict[str, Any], questions: int = QUESTIONS) -> dict[str, Any]:
    """What the whole file would cost and take, projected from the questions run so far."""
    n = rec["n"]
    return {"questions": questions, "basis_n": n,
            "usd": rec["cost"]["usd"] / n * questions if n else 0.0,
            "host_hours": rec["wall_s"]["mean"] * questions / 3600}  # fmt: skip


def _write_status(status: Path, state: str, rows: list[dict[str, Any]], meta: dict[str, Any],
                  started: str, planned: int | None) -> None:  # fmt: skip
    doc = {"state": state, "done": len(rows), "planned": planned,
           "last_qid": rows[-1]["qid"] if rows else None, "usd": round(sum(r["usd"] for r in rows), 4),
           "started": started, "updated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "meta": meta}  # fmt: skip
    tmp = status.with_name(status.name + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, status)


def _check_resume(status: Path, meta: dict[str, Any]) -> str | None:
    if not status.exists():
        return None
    old = json.loads(status.read_text(encoding="utf-8"))
    for key in CHECKED:
        if old["meta"].get(key) != meta.get(key):
            raise ResumeMismatch(f"{status}: {key} differs from the run being resumed "
                                 f"({old['meta'].get(key)!r} != {meta.get(key)!r}); "
                                 "use another --out for a new measurement")  # fmt: skip
    return old.get("started")


def run_m(
    path: Path,
    retrieve: Callable[[dict[str, Any]], Retrieved],
    answerer: LLM,
    judge: LLM,
    *,
    jsonl: Path,
    status: Path,
    meta: dict[str, Any],
    budget_usd: float,
    limit: int | None = None,
    planned: int | None = None,
    questions: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stream `path`, ask and judge every question not yet in `jsonl`, append, checkpoint. `limit`: only
    the first N questions of the file. `meta` carries the answerer, judge, gate and config
    fingerprints the record and a resume are checked against."""
    meta = {**meta, "prompt_version": e1.PROMPT_VERSION, "prompts_sha256": e1.prompts_sha256(),
            "judge_version": e1.JUDGE_VERSION}  # fmt: skip
    started = _check_resume(status, meta) or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = read_rows(jsonl)
    done = {r["qid"] for r in rows}
    planned = limit if limit is not None else planned
    spent = sum(r["usd"] for r in rows)
    stopped: str | None = None
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_status(status, "running", rows, meta, started, planned)
    try:
        with open(jsonl, "a", encoding="utf-8") as out:
            for i, q in enumerate(questions if questions is not None else lme.iter_questions(path)):
                if limit is not None and i >= limit:
                    break
                qid = q["question_id"]
                if qid in done:
                    continue
                got: list[Retrieved] = []

                def fetch(q: dict[str, Any], got: list[Retrieved] = got) -> str:
                    got.append(retrieve(q))
                    return e1.mem_context(got[0])

                t0 = time.perf_counter()
                rec = e1.run_e1([q], fetch, answerer, judge, conditions=("mem",),
                                budget_usd=max(budget_usd - spent, 0.0))  # fmt: skip
                if rec["stopped"] or not rec["items"]:
                    stopped = rec["stopped"] or "no item"
                    break
                r, item = got[0], rec["items"][0]
                gold = set(q.get("answer_session_ids") or ())
                abstain = item["judge_kind"] == e1.ABSTENTION
                row = item | {
                    "qid": qid,
                    "answerer": answerer.model, "judge": judge.model,
                    "prompt_version": e1.PROMPT_VERSION, "prompts_sha256": e1.prompts_sha256(),
                    "judge_version": e1.JUDGE_VERSION,
                    "recall_any5": None if abstain or not gold else
                    ir.recall_any(lme.dedupe_sessions(r.R), gold, 5),
                    "delivered_tokens": est_tokens(e1.mem_context(r)),
                    "delivered_rows": len(r.S), "truncated_rows": r.truncated,
                    "n_sessions": len(q["haystack_session_ids"]),
                    "ingest_ms": round(r.ingest_ms, 1), "query_ms": round(r.latency_ms, 1),
                    "wall_s": round(time.perf_counter() - t0, 2),
                }  # fmt: skip
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                out.flush()
                os.fsync(out.fileno())
                rows.append(row)
                done.add(qid)
                spent += row["usd"]
                _write_status(status, "running", rows, meta, started, planned)
    except BaseException:
        _write_status(status, "crashed", rows, meta, started, planned)
        raise
    result = summarize(jsonl, meta=meta, planned=planned, stopped=stopped)
    _write_status(status, "stopped" if stopped else "complete", rows, meta, started, planned)
    return result
