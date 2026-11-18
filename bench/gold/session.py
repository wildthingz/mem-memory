"""Labelling sittings: draft review, pooled-pair judging, status (benchmarks-v2 §3.1 "Authoring").

All state is append-only JSONL under the work dir (`data/gold/v1/`, git-ignored), so a sitting can
stop at any point (time limit, `q`, a crash) and the next one resumes. The latest event for a key
wins. Nothing here talks to a model or the store: the sittings get the chunk index and, for the
no-answer check, a BM25 callable from the caller.

Files: drafts.jsonl, checks.jsonl, review.jsonl, pool.jsonl, labels.jsonl (each row has a `judge`:
claude | human, and a `source`: draft | pool | kappa | undo), sittings.jsonl, kappa_sample.json.
A label of the human's wins over Claude's for the same pair, whatever the order of the rows.
Time is measured per item (`seconds`), so `status` reports the real pace against the budget the
spec assumes (≈ 25 s per pre-filtered draft, ≈ 8 s per pooled pair).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.gold import pg
from bench.gold.chunks import ChunkIndex
from bench.gold.draft import read_drafts
from bench.gold.kappa import KAPPA_MIN, KappaResult, draw_sample, result_of

Ask = Callable[[str], str]
Out = Callable[[str], None]
BM25 = Callable[[dict[str, Any], int], list[str]]  # (draft, k) -> BM25 top-k gold ids

REVIEW_BUDGET_S, LABEL_BUDGET_S = (
    25,
    8,
)  # per item, §3.1 (review ≈ 1 h for ≈ 140 pre-filtered drafts)
SHOW_CHARS = 700
SHORT = {"single-fact": "sf", "multi-note": "mn", "temporal": "tp", "knowledge-update": "ku",
         "alias-entity": "al", "relational": "rl", "no-answer-near-miss": "nm",
         "no-answer-off-topic": "ot"}  # fmt: skip


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


class Work:
    """The work directory of one PG build."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, name: str) -> Path:
        return self.root / name

    # -- drafts and checks
    def drafts(self) -> list[dict[str, Any]]:
        return read_drafts(self.path("drafts.jsonl"))

    def checks(self) -> dict[str, dict[str, Any]]:
        return {c["draft_id"]: c for c in _read(self.path("checks.jsonl"))}

    # -- review decisions: draft_id -> latest event
    def decisions(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for e in _read(self.path("review.jsonl")):
            out[e["draft_id"]] = e
        return out

    def accepted(self) -> list[pg.Question]:
        """Accepted, not voided, questions in acceptance order, gold from the latest labels."""
        drafts = {d["draft_id"]: d for d in self.drafts()}
        state = self.label_state()
        labels = {k: rel for k, (rel, _) in state.items()}
        out = []
        for did, ev in self.decisions().items():
            if ev["action"] not in ("accept", "edit"):
                continue
            d = drafts[did]
            q = pg.Question(
                qid=ev["qid"], category=d["category"], question=ev.get("question") or d["question"],
                answer=ev.get("answer") if ev.get("answer") is not None else d["answer"],
                filters=d.get("filters", {}), reworded=bool(ev.get("reworded")),
                source=d.get("source", []), draft_id=did, store=d.get("store", "personal"),
                gold={g: r for (qid, g), r in labels.items() if qid == ev["qid"] and r >= 1},
                nonrel=sorted(g for (qid, g), r in labels.items() if qid == ev["qid"] and r == 0),
                judges={g: j for (qid, g), (_, j) in state.items() if qid == ev["qid"]})  # fmt: skip
            out.append(q)
        return out

    def judgments(self, judge: str, source: str | None = None) -> dict[tuple[str, str], int]:
        """The latest grade of one judge per (qid, gold id); an undone label (rel -1) is absent.
        `source` keeps only rows from that source (the latest such row wins). A row with no
        `judge` is the human's (labels written before the judge column existed)."""
        out: dict[tuple[str, str], int] = {}
        for e in _read(self.path("labels.jsonl")):
            if e.get("judge", "human") != judge or (
                source and e.get("source") != source and e.get("source") != "undo"
            ):
                continue
            out[(e["qid"], e["gold_id"])] = int(e["rel"])
        return {k: r for k, r in out.items() if r >= 0}

    def label_state(self) -> dict[tuple[str, str], tuple[int, str]]:
        """(qid, gold id) -> (rel, judge): the human's label wins over Claude's."""
        state = {k: (r, "claude") for k, r in self.judgments("claude").items()}
        state |= {k: (r, "human") for k, r in self.judgments("human").items()}
        return state

    def labels(self) -> dict[tuple[str, str], int]:
        return {k: rel for k, (rel, _) in self.label_state().items()}

    # -- the blind kappa sample
    def kappa_sample(self, *, n: int, seed: int) -> list[tuple[str, str]]:
        """Claude's pooled judgments for accepted questions, a seeded random n of them. The first
        draw is saved, so a resumed sitting and a later `status` see the same sample."""
        path = self.path("kappa_sample.json")
        accepted = {q.qid for q in self.accepted()}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return [(a, b) for a, b in saved["keys"] if a in accepted]
        keys = [k for k in self.judgments("claude", "pool") if k[0] in accepted]
        sample = draw_sample(keys, n, seed)
        path.write_text(
            json.dumps({"n": n, "seed": seed, "keys": sample}, indent=1) + "\n", encoding="utf-8"
        )
        return sample

    def kappa_result(self) -> tuple[KappaResult | None, int]:
        """(result over the sampled pairs the human answered, sampled pairs still unanswered); None if
        no sample was drawn yet."""
        path = self.path("kappa_sample.json")
        if not path.exists():
            return None, 0
        keys = self.kappa_sample(n=0, seed=0)
        claude, human = self.judgments("claude", "pool"), self.judgments("human", "kappa")
        pairs = [(claude[k], human[k]) for k in keys if k in claude and k in human]
        return result_of(pairs), len(keys) - len(pairs)

    def pool(self) -> list[dict[str, Any]]:
        return _read(self.path("pool.jsonl"))

    def sittings(self) -> list[dict[str, Any]]:
        return _read(self.path("sittings.jsonl"))

    def next_qid(self, category: str) -> str:
        used = {e["qid"] for e in _read(self.path("review.jsonl")) if e.get("qid")}
        n = 1
        while f"pg1-{SHORT[category]}-{n:02d}" in used:
            n += 1
        return f"pg1-{SHORT[category]}-{n:02d}"

    def log_event(self, name: str, row: dict[str, Any]) -> None:
        _append(self.path(name), row)


def counts(work: Work) -> Counter[str]:
    return Counter(q.category for q in work.accepted())


def _clip(text: str, n: int = SHOW_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " […]"


def review_order(
    drafts: list[dict[str, Any]], decided: set[str], flagged: set[str]
) -> list[dict[str, Any]]:
    """Undecided, unflagged drafts, interleaved across categories (round-robin in QUOTAS order,
    each category by sha1(draft_id)) so every quota fills at about the same rate."""
    by_cat: dict[str, list[dict[str, Any]]] = {c: [] for c in pg.QUOTAS}
    for d in sorted(drafts, key=lambda d: pg.sha1_key(d["draft_id"])):
        if d["draft_id"] not in decided and d["draft_id"] not in flagged:
            by_cat[d["category"]].append(d)
    out = []
    while any(by_cat.values()):
        for queue in by_cat.values():
            if queue:
                out.append(queue.pop(0))
    return out


@dataclass
class SittingResult:
    kind: str
    items: int
    seconds: float
    stopped: str  # time | quit | done
    remaining: int


def _log_sitting(work: Work, res: SittingResult, started: str) -> None:
    work.log_event("sittings.jsonl", {"kind": res.kind, "items": res.items,
                                      "seconds": round(res.seconds, 1), "stopped": res.stopped,
                                      "started_at": started, "remaining": res.remaining})  # fmt: skip


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def review_sitting(
    work: Work, index: ChunkIndex, bm25: BM25 | None, *, minutes: float | None,
    ask: Ask = input, out: Out = print, clock: Callable[[], float] = time.monotonic,
) -> SittingResult:  # fmt: skip
    """Accept (a), edit (e), reject (r), skip (s) or quit (q) each undecided draft. Stops at
    `minutes`. A no-answer draft shows the BM25 top 20: `a` means none of them answers it, `x`
    means one does (rejected). A category at its quota is not shown any more."""
    decisions = work.decisions()
    flagged = {did for did, c in work.checks().items() if not c["ok"]}
    queue = review_order(work.drafts(), set(decisions), flagged)
    have = counts(work)
    started, t_start, done, stopped = _now_iso(), clock(), 0, "done"
    skipped: set[str] = set()
    for d in queue:
        if have[d["category"]] >= pg.TOTAL[d["category"]]:
            continue
        if minutes is not None and clock() - t_start >= minutes * 60:
            stopped = "time"
            break
        t_item = clock()
        out(f"\n[{d['draft_id']}] {d['category']}   accepted {have[d['category']]}"
            f"/{pg.TOTAL[d['category']]}")  # fmt: skip
        out(f"Q: {d['question']}")
        if d["category"] in pg.NO_ANSWER:
            out("(no-answer: the vault should not answer this)")
            top = bm25(d, 20) if bm25 else []
            for i, g in enumerate(top, 1):
                c = index.by_gold.get(g)
                out(f"  {i:>2}. {g}  {_clip(c.text, 90) if c else ''}")
            if not top:
                out("  (no BM25 results)")
            prompt = (
                "[a] none of these answers it  [x] one does  [e]dit  [r]eject  [s]kip  [q]uit > "
            )
        else:
            out(f"A: {d['answer']}")
            for g, rel in sorted(d["gold"].items(), key=lambda kv: -kv[1]):
                c = index.by_gold.get(g)
                out(f"  rel {rel}  {g}\n      {_clip(c.text) if c else '(not in the vault)'}")
            prompt = "[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > "
        while True:
            k = ask(prompt).strip().lower()[:1]
            if k in ("a", "e", "r", "s", "q") or (k == "x" and d["category"] in pg.NO_ANSWER):
                break
        if k == "q":
            stopped = "quit"
            break
        if k == "s":
            skipped.add(d["draft_id"])
            continue
        secs = round(clock() - t_item, 1)
        if k in ("r", "x"):
            work.log_event("review.jsonl", {"draft_id": d["draft_id"], "action": "reject",
                                            "why": "answered" if k == "x" else "", "seconds": secs})  # fmt: skip
        else:
            question, answer = d["question"], d["answer"]
            if k == "e":
                question = ask("question (blank = keep) > ").strip() or question
                if d["category"] not in pg.NO_ANSWER:
                    answer = ask("short answer (blank = keep) > ").strip() or answer
            qid = work.next_qid(d["category"])
            work.log_event("review.jsonl", {
                "draft_id": d["draft_id"], "action": "edit" if k == "e" else "accept", "qid": qid,
                "question": question, "answer": answer, "reworded": question != d["question"],
                "bm25_confirmed": d["category"] in pg.NO_ANSWER, "seconds": secs})  # fmt: skip
            if d["category"] not in pg.NO_ANSWER:
                for g, rel in d["gold"].items():
                    work.log_event("labels.jsonl", {"qid": qid, "gold_id": g, "rel": rel,
                                                    "source": "draft", "judge": "human"})  # fmt: skip
            have[d["category"]] += 1
        done += 1
    remaining = sum(1 for d in queue if d["draft_id"] not in work.decisions()
                    and have[d["category"]] < pg.TOTAL[d["category"]])  # fmt: skip
    res = SittingResult("review", done, clock() - t_start, stopped, remaining)
    _log_sitting(work, res, started)
    out(status_text(work))
    return res


# ---- pooled pairs -------------------------------------------------------------------------------


def record_label(
    work: Work, q: pg.Question, gold_id: str, rel: int, *, judge: str, source: str, out: Out = print
) -> bool:
    """Append one grade. A no-answer question that gets a 1 or 2 is not no-answer: it is voided
    (rejected), its quota reopens and its remaining pairs are dropped. Returns True if voided."""
    work.log_event("labels.jsonl", {"qid": q.qid, "gold_id": gold_id, "rel": rel,
                                    "source": source, "judge": judge})  # fmt: skip
    if rel and not q.answerable:
        ev = next(e for e in work.decisions().values() if e.get("qid") == q.qid)
        work.log_event("review.jsonl", {"draft_id": ev["draft_id"], "action": "reject",
                                        "why": f"voided: {gold_id} judged rel {rel} by {judge}"})  # fmt: skip
        out(f"  {q.qid} is not a no-answer question: voided, its quota reopens.")
        return True
    return False


def show_pair(q: pg.Question, c: Any, gold_id: str, out: Out) -> None:
    """What the person sees for a pair: never the system that found it, never Claude's grade."""
    out(f"\n[{q.qid}] {q.category}   Q: {q.question}"
        + (f"\n   short answer: {q.answer}" if q.answer else "   (no answer expected)"))  # fmt: skip
    out(f"  {gold_id}\n  {_clip(c.text, 900) if c else '(chunk is gone)'}")


def label_sitting(
    work: Work, index: ChunkIndex, *, minutes: float | None, ask: Ask = input, out: Out = print,
    clock: Callable[[], float] = time.monotonic, relabel_claude: bool = False,
) -> SittingResult:  # fmt: skip
    """Judge pooled (question, chunk) pairs: 0 no, 1 supports, 2 answers, s skip, u undo, q quit.
    Your grade is recorded as `human` and wins over Claude's. `relabel_claude` (the kappa < 0.7
    path) also offers the pairs Claude judged; your blind 50 are already yours and are not asked
    again. A no-answer question that gets a 1 or 2 is voided (see `record_label`)."""
    questions = {q.qid: q for q in work.accepted()}
    state = work.label_state()
    todo = [p for p in work.pool() if p["qid"] in questions
            and ((p["qid"], p["gold_id"]) not in state
                 or (relabel_claude and state[(p["qid"], p["gold_id"])][1] == "claude"))]  # fmt: skip
    started, t_start, done, stopped = _now_iso(), clock(), 0, "done"
    voided: set[str] = set()
    history: list[tuple[str, str]] = []
    skipped: set[tuple[str, str]] = set()
    i = 0
    while i < len(todo):
        p = todo[i]
        key = (p["qid"], p["gold_id"])
        if p["qid"] in voided or key in skipped:
            i += 1
            continue
        if minutes is not None and clock() - t_start >= minutes * 60:
            stopped = "time"
            break
        q = questions[p["qid"]]
        show_pair(q, index.by_gold.get(p["gold_id"]), p["gold_id"], out)
        k = ask("[0] no  [1] supports  [2] answers  [s]kip  [u]ndo  [q]uit > ").strip().lower()[:1]
        if k == "q":
            stopped = "quit"
            break
        if k == "u":
            if history:
                last = history.pop()
                work.log_event("labels.jsonl", {"qid": last[0], "gold_id": last[1], "rel": -1,
                                                "source": "undo", "judge": "human"})  # fmt: skip
                i = next(j for j, x in enumerate(todo) if (x["qid"], x["gold_id"]) == last)
                skipped.discard(last)
            continue
        if k == "s":
            skipped.add(key)
            i += 1
            continue
        if k not in ("0", "1", "2"):
            continue
        if record_label(work, q, p["gold_id"], int(k), judge="human", source="pool", out=out):
            voided.add(q.qid)
        history.append(key)
        done += 1
        i += 1
    live = {x.qid for x in work.accepted()}
    state = work.label_state()
    remaining = sum(1 for p in work.pool() if p["qid"] in live
                    and ((p["qid"], p["gold_id"]) not in state
                         or (relabel_claude and state[(p["qid"], p["gold_id"])][1] == "claude")))  # fmt: skip
    res = SittingResult("label", done, clock() - t_start, stopped, remaining)
    _log_sitting(work, res, started)
    out(status_text(work))
    return res


def kappa_sitting(
    work: Work, index: ChunkIndex, *, n: int, seed: int, ask: Ask = input, out: Out = print,
    clock: Callable[[], float] = time.monotonic,
) -> SittingResult:  # fmt: skip
    """The blind check: grade the seeded sample of Claude's pooled judgments 0 / 1 / 2 (or q to
    stop and resume later). Claude's grade is not shown, not even as a hint. Your grades are
    recorded as `human` (source `kappa`) and win over Claude's for those pairs."""
    questions = {q.qid: q for q in work.accepted()}
    sample = work.kappa_sample(n=n, seed=seed)
    answered = work.judgments("human", "kappa")
    todo = [k for k in sample if k not in answered and k[0] in questions]
    started, t_start, done, stopped = _now_iso(), clock(), 0, "done"
    voided: set[str] = set()
    for qid, gid in todo:
        if qid in voided:
            continue
        q = questions[qid]
        show_pair(q, index.by_gold.get(gid), gid, out)
        k = ""
        while k not in ("0", "1", "2", "q"):
            k = ask(f"({done + 1}/{len(todo)}) [0] no  [1] supports  [2] answers  [q]uit > "
                    ).strip().lower()[:1]  # fmt: skip
        if k == "q":
            stopped = "quit"
            break
        if record_label(work, q, gid, int(k), judge="human", source="kappa", out=out):
            voided.add(qid)
        done += 1
    left = sum(1 for k in sample if k not in work.judgments("human", "kappa"))
    res = SittingResult("kappa", done, clock() - t_start, stopped, left)
    _log_sitting(work, res, started)
    return res


# ---- status -------------------------------------------------------------------------------------


def status_text(work: Work) -> str:
    acc = counts(work)
    lines = ["PG-v1 status", f"  {'category':<22}{'accepted':>9}{'quota':>7}"]
    for cat in pg.QUOTAS:
        lines.append(f"  {cat:<22}{acc[cat]:>9}{pg.TOTAL[cat]:>7}")
    qs = work.accepted()
    reworded = sum(q.reworded for q in qs)
    lines.append(
        f"  accepted {len(qs)}/{pg.N_QUESTIONS}; reworded {reworded} ({reworded / len(qs):.0%})"
        if qs
        else f"  accepted 0/{pg.N_QUESTIONS}"
    )
    drafts, dec = work.drafts(), work.decisions()
    flagged = {did for did, c in work.checks().items() if not c["ok"]}
    shortfall = {c: pg.TOTAL[c] - acc[c] for c in pg.QUOTAS if acc[c] < pg.TOTAL[c]}
    left = Counter(d["category"] for d in drafts if d["draft_id"] not in dec
                   and d["draft_id"] not in flagged)  # fmt: skip
    need_more = {c: n - left[c] for c, n in shortfall.items() if left[c] < n}
    lines.append(f"  drafts {len(drafts)}; decided {len(dec)}; flagged by check {len(flagged)}")
    if need_more:
        lines.append("  NOT ENOUGH DRAFTS LEFT for: "
                     + ", ".join(f"{c} (short by {n})" for c, n in need_more.items())
                     + "  -> run `bench gold draft --more`")  # fmt: skip
    pool = work.pool()
    lab = work.labels()
    pool_done = sum((p["qid"], p["gold_id"]) in lab for p in pool)
    lines.append(f"  pooled pairs {pool_done}/{len(pool)} judged")
    judged = Counter(j for _, j in work.label_state().values())
    lines.append(f"  labels by judge: claude {judged['claude']}, human {judged['human']}")
    res, unanswered = work.kappa_result()
    if res is None:
        lines.append("  blind kappa: not started (`bench gold kappa-sample`)")
    elif unanswered:
        lines.append(f"  blind kappa: {unanswered} sampled pair(s) still to grade")
    else:
        lines.append("  " + res.text().splitlines()[0]
                     + (f": Claude's labels stand (>= {KAPPA_MIN})" if res.stands
                        else f": < {KAPPA_MIN}, you label the rest (`bench gold label`)"))  # fmt: skip
    for kind, budget in (
        ("review", REVIEW_BUDGET_S),
        ("label", LABEL_BUDGET_S),
        ("kappa", LABEL_BUDGET_S),
    ):
        sit = [s for s in work.sittings() if s["kind"] == kind]
        items, secs = sum(s["items"] for s in sit), sum(s["seconds"] for s in sit)
        if items:
            pace = secs / items
            lines.append(f"  {kind}: {items} items in {secs / 60:.0f} min, {pace:.0f} s each"
                         f" (spec assumes {budget} s)")  # fmt: skip
    return "\n".join(lines)
