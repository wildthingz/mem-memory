"""PG-v2 (benchmarks-v2 §3, B-E4 "Also in P6"): 60 real `recall` queries sampled from `events`,
labelled by them, the fresh holdout for B-R3. Recorded in P6, a gate only from the next version.

Pipeline (all file based, each step deterministic):
  sample   `sample_events`: 60 live recall events. Reviewed ones first, then the rest, each group in
           sha1(event ref) order; one event per (store, query text).
  sheet    `make_sheet`: per question the candidate rows to judge = the rows the event returned, the
           row named by a should-have-found review, and (optionally) the top-`depth` of ours, BM25
           and dense (pooling, so a better system is not scored against the old system's output).
           Writes `sheet.jsonl` (questions) and `labels.tsv` (one line per candidate, `rel` blank).
  freeze   `freeze`: reads the filled `labels.tsv` (rel 0 not relevant, 1 supports, 2 answers; a
           question with no rel-2 row is a no-answer question), writes `questions.yaml`,
           `qrels.tsv`, `SNAPSHOT` (sha256 of both files + the sample provenance). Refuses a blank
           label, a half-labelled question, or a frozen directory.
  load     `load`: re-checks the hashes, returns `pg.Question`s (store = the store the query was
           asked of, filters = the scope filters of the event), so `bench.pg_eval` scores them
           exactly like PG-v1.

Ids are display ids as the events log them: `m<id>` memories, and for imported chunks the gold id
`path::chunk_key` (mapped from the event's `c<chunk id>` at sheet time, which is stable across
re-imports where the chunk id is not).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from bench import p6_use
from bench.gold import pg

N_QUESTIONS = 60
POOL_DEPTH = 3
VERSION = "v2"
ANSWERABLE = "daily"
NO_ANSWER = "no-answer-near-miss"  # the pg.NO_ANSWER category PG-v2 uses for "nothing stored"
FILTER_KEYS = pg.FILTER_KEYS
SNIPPET = 220
Pool = Callable[[pg.Question], Mapping[str, Sequence[str]]]  # question -> {system: ranked ids}


def qid_of(ref: str) -> str:
    return "pv2-" + hashlib.sha1(ref.encode()).hexdigest()[:10]


def sample_events(
    events: Sequence[p6_use.Event], reviews: Mapping[str, p6_use.Review], n: int = N_QUESTIONS
) -> list[p6_use.Event]:  # fmt: skip
    recalls = [e for e in events if e.tool == "recall" and str(e.args.get("query", "")).strip()]
    key = lambda e: (e.ref not in reviews, p6_use.sha1_key(e.ref))
    out, seen = [], set()
    for e in sorted(recalls, key=key):
        k = (e.store, " ".join(str(e.args["query"]).split()).lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
        if len(out) == n:
            break
    return out


def normalise_id(conn: sqlite3.Connection, shown: str) -> str:
    """`c<chunk id>` of an imported note -> `path::chunk_key`; anything else unchanged."""
    if shown.startswith("c") and shown[1:].isdigit():
        row = conn.execute(
            "SELECT n.source_ref, c.chunk_key FROM chunks c JOIN nodes n ON n.id = c.node_id"
            " WHERE c.id = ? AND n.origin = 'import' AND n.source_ref IS NOT NULL",
            (int(shown[1:]),)).fetchone()  # fmt: skip
        if row and row[1]:
            return f"{row[0]}::{row[1]}"
    return shown


def describe(conn: sqlite3.Connection, ident: str) -> str:
    """One line to judge an id by."""
    if ident.startswith("m") and ident[1:].isdigit():
        row = conn.execute("SELECT type, date, name, body FROM nodes WHERE id = ?",
                           (int(ident[1:]),)).fetchone()  # fmt: skip
        if row:
            return f"{row[0]} · {row[1]} · {row[2]} — {' '.join((row[3] or '').split())}"[:SNIPPET]
    elif "::" in ident:
        path, _, key = ident.partition("::")
        row = conn.execute(
            "SELECT c.text FROM chunks c JOIN nodes n ON n.id = c.node_id"
            " WHERE n.source_ref = ? AND c.chunk_key = ? ORDER BY c.id LIMIT 1",
            (path, key)).fetchone()  # fmt: skip
        if row:
            return f"{ident} — {' '.join(row[0].split())}"[:SNIPPET]
    return f"{ident} — (no longer in the store)"


def _filters(args: Mapping[str, Any]) -> dict[str, Any]:
    f: dict[str, Any] = {}
    for k in FILTER_KEYS:
        v = args.get(k)
        if v not in (None, [], ""):
            f[k] = v
    return f


def make_sheet(
    sample: Sequence[p6_use.Event], reviews: Mapping[str, p6_use.Review],
    conns: Mapping[str, sqlite3.Connection], *, pool: Pool | None = None, depth: int = POOL_DEPTH,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:  # fmt: skip
    """(questions, label lines). Candidates keep their first source in the order
    returned < found < pool, so the printed order is the order they saw them in."""
    qs, lines = [], []
    for e in sorted(sample, key=lambda e: qid_of(e.ref)):
        conn = conns[e.store]
        rv = reviews.get(e.ref)
        cands: dict[str, str] = {}
        for i in e.ids:
            cands.setdefault(normalise_id(conn, i), "returned")
        if rv and rv.mark == "should-have-found" and rv.found and rv.found != p6_use.NOT_STORED:
            cands.setdefault(normalise_id(conn, rv.found), "found")
        q = pg.Question(qid=qid_of(e.ref), category=ANSWERABLE, question=str(e.args["query"]),
                        filters=_filters(e.args), store=e.store)  # fmt: skip
        if pool is not None:
            for system, ranked in sorted(pool(q).items()):
                for i in list(ranked)[:depth]:
                    cands.setdefault(
                        normalise_id(conn, i) if "::" not in i else i, f"pool:{system}"
                    )
        qs.append({"qid": q.qid, "ref": e.ref, "at": e.at, "host": e.host, "store": e.store,
                   "query": q.question, "filters": q.filters,
                   "review": {"mark": rv.mark, "found": rv.found} if rv else None,
                   "event_ids": [normalise_id(conn, i) for i in e.ids]})  # fmt: skip
        lines += [{"qid": q.qid, "id": i, "rel": "", "source": src, "text": describe(conn, i)}
                  for i, src in cands.items()]  # fmt: skip
    return qs, lines


LABEL_COLS = ["qid", "id", "rel", "source", "text"]


def write_sheet(
    out_dir: Path, qs: Sequence[dict[str, Any]], lines: Sequence[dict[str, str]]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sheet.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n" for q in qs),
        encoding="utf-8")  # fmt: skip
    with (out_dir / "labels.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, LABEL_COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(lines)


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out.setdefault(row["qid"], {})[row["id"]] = row["rel"].strip()
    return out


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def questions_from_labels(
    qs: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[str, str]]
) -> tuple[list[pg.Question], list[str]]:  # fmt: skip
    """(questions, errors). Errors list every blank or invalid label and every question without
    any line; freezing needs none."""
    errs, out = [], []
    for q in sorted(qs, key=lambda q: q["qid"]):
        rows = labels.get(q["qid"])
        if not rows:
            errs.append(f"{q['qid']}: no candidate lines")
            continue
        bad = [i for i, r in rows.items() if r not in ("0", "1", "2")]
        errs += [f"{q['qid']} {i}: rel {rows[i]!r} (0, 1 or 2)" for i in bad]
        gold = {i: int(r) for i, r in rows.items() if r in ("1", "2")}
        nonrel = sorted(i for i, r in rows.items() if r == "0")
        if gold and 2 not in gold.values():
            errs.append(f"{q['qid']}: supports (1) but no answers (2): mark the best one 2, or "
                        "all 0 for no-answer")  # fmt: skip
        out.append(pg.Question(
            qid=q["qid"], category=ANSWERABLE if gold else NO_ANSWER, question=q["query"],
            gold=gold, nonrel=nonrel, filters=dict(q.get("filters") or {}), store=q["store"],
            source=[q["ref"]], judges=dict.fromkeys([*gold, *nonrel], "human")))  # fmt: skip
    return out, errs


def freeze(
    sheet_dir: Path, out_dir: Path, *, built_at: str, window: Mapping[str, Any]
) -> dict[str, str]:
    if (out_dir / "SNAPSHOT").exists():
        raise pg.Frozen(f"{out_dir} is frozen: write a new version (v2.1), never overwrite")
    qs = [json.loads(ln) for ln in (sheet_dir / "sheet.jsonl").read_text(encoding="utf-8").splitlines()
          if ln.strip()]  # fmt: skip
    questions, errs = questions_from_labels(qs, read_labels(sheet_dir / "labels.tsv"))
    if errs:
        raise ValueError("cannot freeze PG-v2:\n  " + "\n  ".join(errs))
    meta = {q["qid"]: q for q in qs}
    qyaml = yaml.safe_dump(
        {"version": VERSION, "questions": [
            {**q.to_dict(), "event": meta[q.qid]["ref"], "asked_at": meta[q.qid]["at"],
             "host": meta[q.qid]["host"]} for q in questions]},
        sort_keys=True, allow_unicode=True, width=1000).encode("utf-8")  # fmt: skip
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(["qid", "gold_id", "rel", "judge"])
    for q in questions:
        for i in sorted(q.gold):
            w.writerow([q.qid, i, q.gold[i], "human"])
        for i in sorted(q.nonrel):
            w.writerow([q.qid, i, 0, "human"])
    qrels = buf.getvalue().encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.yaml").write_bytes(qyaml)
    (out_dir / "qrels.tsv").write_bytes(qrels)
    hashes = {"questions.yaml": sha256(qyaml), "qrels.tsv": sha256(qrels)}
    n_ans = sum(q.answerable for q in questions)
    (out_dir / "SNAPSHOT").write_text(
        "".join(
            f"{k}: {v}\n"
            for k, v in [
                ("version", VERSION),
                ("built_at", built_at),
                ("window", json.dumps(dict(window))),
                ("questions_yaml_sha256", hashes["questions.yaml"]),
                ("qrels_tsv_sha256", hashes["qrels.tsv"]),
                ("n_questions", len(questions)),
                ("n_answerable", n_ans),
                ("n_no_answer", len(questions) - n_ans),
            ]
        ),
        encoding="utf-8",
    )
    return hashes


def load(gold_dir: Path) -> list[pg.Question]:
    snap = pg.read_snapshot(gold_dir / "SNAPSHOT")
    for fname, key in (
        ("questions.yaml", "questions_yaml_sha256"),
        ("qrels.tsv", "qrels_tsv_sha256"),
    ):
        got = sha256((gold_dir / fname).read_bytes())
        if got != snap[key]:
            raise ValueError(f"{gold_dir / fname}: sha256 {got[:12]} != SNAPSHOT {snap[key][:12]} "
                             "(frozen files are never edited: write v2.1)")  # fmt: skip
    raw = yaml.safe_load((gold_dir / "questions.yaml").read_text(encoding="utf-8"))
    rels: dict[str, dict[str, int]] = {}
    with (gold_dir / "qrels.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rels.setdefault(row["qid"], {})[row["gold_id"]] = int(row["rel"])
    out = []
    for d in raw["questions"]:
        r = rels.get(d["qid"], {})
        out.append(pg.Question(
            qid=d["qid"], category=d["category"], question=d["question"],
            gold={g: v for g, v in r.items() if v >= 1}, nonrel=sorted(g for g, v in r.items() if v == 0),
            filters=d.get("filters") or {}, store=d.get("store", "personal"),
            source=[d["event"]] if d.get("event") else [],
            judges=dict.fromkeys(r, "human")))  # fmt: skip
    if len(out) != int(snap["n_questions"]):
        raise ValueError(f"{gold_dir}: {len(out)} questions, SNAPSHOT says {snap['n_questions']}")
    return out
