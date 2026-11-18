"""PG-v1 data model, quotas, split, validation and freeze (benchmarks-v2 §3.1).

Definitions (decision record 0010):
- **Quota** per category, tune / test, exact by construction: the category total is 3 x tune, and
  within a category the questions are sorted by `sha1(qid)` and the first third is tune.
- **Gold** per question: `{path::chunk_key: rel}`, rel 2 = answers, 1 = supports. A no-answer
  question has an empty gold set. Every answerable question has at least one rel-2 chunk.
- **Lexical overlap** of a question = |T(question) ∩ T(chunk)| / |T(question)|, the best value over
  its rel-2 chunks; T() = lower-cased `\\w+` tokens of 2+ characters, minus a fixed stop list, no
  stemming. The control: the median over answerable questions is ≤ 0.5 (§3.1).
- **Reworded**: the person edited the drafted question text. At least 50% of the questions are.
- Frozen files: `questions.yaml`, `qrels.tsv` (qid, gold id, rel, chunk content hash at label
  time, judge = claude | human; rel 0 = judged not relevant), `split.tsv`, and `SNAPSHOT` (tarball sha256 + the three file hashes). Any edit is a new
  version directory (v1.1), never an overwrite.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bench.gold.chunks import ChunkIndex, split_gold_id
from bench.gold.kappa import KAPPA_MIN

VERSION = "v1"
NOTE_SEED = 20261022
# PG-v1 holds no work-store content (spec change 2026-10-24, approved by the owner): notes come from
# these two folders only, and only those routed to the personal store.
PG_FOLDERS = ("personal", "projects")
N_NOTES = 40  # single-note units, proportional to folder (largest remainder, `folder_quotas`)
N_PAIRS = 10
JUDGES = ("claude", "human")
# category -> (tune, test). Order is the reporting order.
QUOTAS: dict[str, tuple[int, int]] = {
    "single-fact": (10, 20),
    "multi-note": (5, 10),
    "temporal": (5, 10),
    "knowledge-update": (4, 8),
    "alias-entity": (4, 8),
    "relational": (2, 4),
    "no-answer-near-miss": (7, 14),
    "no-answer-off-topic": (3, 6),
}
NO_ANSWER = frozenset({"no-answer-near-miss", "no-answer-off-topic"})
PAIR_ONLY = frozenset({"multi-note", "relational"})  # need two linked notes
TOTAL = {c: t + s for c, (t, s) in QUOTAS.items()}
N_QUESTIONS = sum(TOTAL.values())  # 120
FILTER_KEYS = ("about", "type", "since", "until")
MAX_MEDIAN_OVERLAP = 0.5
MIN_REWORDED = 0.5

STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "after",
        "all",
        "also",
        "any",
        "because",
        "before",
        "being",
        "between",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "over",
        "same",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "only",
        "own",
        "out",
        "through",
        "under",
        "until",
        "while",
    ]
)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) >= 2 and t not in STOP}


def overlap(question: str, chunk_text: str) -> float:
    q = content_tokens(question)
    return len(q & content_tokens(chunk_text)) / len(q) if q else 0.0


def folder_quotas(counts: dict[str, int], total: int = N_NOTES) -> dict[str, int]:
    """`total` notes split in proportion to `counts` (eligible notes per folder), largest-remainder
    rounding so it sums to `total`; ties go to the larger folder, then the name."""
    n = sum(counts.values())
    if n < total:
        raise ValueError(f"{n} eligible notes in {sorted(counts)}, need {total}")
    exact = {f: total * c / n for f, c in counts.items()}
    out = {f: int(x) for f, x in exact.items()}
    order = sorted(counts, key=lambda f: (-(exact[f] - out[f]), -counts[f], f))
    for f in order[: total - sum(out.values())]:
        out[f] += 1
    return out


def sha1_key(qid: str) -> str:
    return hashlib.sha1(qid.encode()).hexdigest()


@dataclass
class Question:
    qid: str
    category: str
    question: str
    answer: str = ""
    gold: dict[str, int] = field(default_factory=dict)  # gold id -> rel (1 supports, 2 answers)
    nonrel: list[str] = field(default_factory=list)  # judged rel 0 (qrels.tsv only: pooling bias)
    filters: dict[str, Any] = field(default_factory=dict)  # about / type / since / until
    reworded: bool = False
    source: list[str] = field(default_factory=list)  # vault paths the question was drafted from
    draft_id: str | None = None
    store: str = "personal"  # the store the question is asked of (routing of its source notes)
    judges: dict[str, str] = field(default_factory=dict)  # gold / non-relevant id -> claude | human

    @property
    def answerable(self) -> bool:
        return self.category not in NO_ANSWER

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"qid": self.qid, "category": self.category, "question": self.question,
                             "answer": self.answer, "filters": dict(self.filters),
                             "reworded": self.reworded, "source": list(self.source), "store": self.store}  # fmt: skip
        if self.draft_id:
            d["draft_id"] = self.draft_id
        return d


def split_of(questions: Iterable[Question]) -> dict[str, str]:
    """qid -> 'tune' | 'test'. Within a category, sha1(qid) order, first third is tune. A category
    that is not at its exact total still gets a deterministic split (`ceil(n / 3)`), but
    `validate` rejects it, so a frozen set always has the exact quotas."""
    by_cat: dict[str, list[str]] = {}
    for q in questions:
        by_cat.setdefault(q.category, []).append(q.qid)
    out: dict[str, str] = {}
    for qids in by_cat.values():
        ordered = sorted(qids, key=sha1_key)
        n_tune = -(-len(ordered) // 3)
        for i, qid in enumerate(ordered):
            out[qid] = "tune" if i < n_tune else "test"
    return out


def quota_counts(questions: Iterable[Question]) -> dict[str, int]:
    counts = dict.fromkeys(QUOTAS, 0)
    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1
    return counts


def lexical_overlaps(questions: Iterable[Question], index: ChunkIndex) -> dict[str, float]:
    """qid -> best overlap with a rel-2 gold chunk (answerable questions with resolvable gold)."""
    out = {}
    for q in questions:
        texts = [index.by_gold[g].text for g, r in q.gold.items() if r == 2 and g in index.by_gold]
        if q.answerable and texts:
            out[q.qid] = max(overlap(q.question, t) for t in texts)
    return out


def validate(questions: list[Question], index: ChunkIndex) -> list[str]:
    """Every reason the set cannot be frozen; empty = freezable."""
    errs: list[str] = []
    qids = [q.qid for q in questions]
    if len(set(qids)) != len(qids):
        errs.append("duplicate qids")
    for cat, n in TOTAL.items():
        have = sum(q.category == cat for q in questions)
        if have != n:
            errs.append(f"category {cat}: {have} questions, the quota needs exactly {n}")
    for q in questions:
        if q.category not in QUOTAS:
            errs.append(f"{q.qid}: unknown category {q.category!r}")
            continue
        if not q.question.strip():
            errs.append(f"{q.qid}: empty question")
        bad = set(q.filters) - set(FILTER_KEYS)
        if bad:
            errs.append(f"{q.qid}: unknown filters {sorted(bad)}")
        if not q.answerable:
            if q.gold:
                errs.append(f"{q.qid}: a no-answer question has gold {sorted(q.gold)}")
            continue
        if not any(r == 2 for r in q.gold.values()):
            errs.append(f"{q.qid}: no rel-2 (answers) gold chunk")
        if q.category in PAIR_ONLY and len({split_gold_id(g)[0] for g in q.gold}) < 2:
            errs.append(f"{q.qid}: {q.category} needs gold chunks in at least two notes")
        for g, r in q.gold.items():
            if r not in (1, 2):
                errs.append(f"{q.qid}: {g} has rel {r} (only 1 and 2 are stored)")
            if g not in index.by_gold:
                errs.append(f"{q.qid}: gold id {g} is not a chunk of the imported vault")
            elif index.by_gold[g].store != q.store:
                errs.append(f"{q.qid}: gold {g} is in the {index.by_gold[g].store} store, the "
                            f"question is asked of {q.store}")  # fmt: skip
        if not q.answer.strip():
            errs.append(f"{q.qid}: no short answer")
    if questions:
        share = sum(q.reworded for q in questions) / len(questions)
        if share < MIN_REWORDED:
            errs.append(f"reworded {share:.0%} of questions, at least {MIN_REWORDED:.0%} needed")
    ov = lexical_overlaps(questions, index)
    if ov and (med := statistics.median(ov.values())) > MAX_MEDIAN_OVERLAP:
        errs.append(f"median lexical overlap {med:.2f} > {MAX_MEDIAN_OVERLAP}")
    return errs


# ---- files --------------------------------------------------------------------------------------


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def questions_yaml(questions: Iterable[Question]) -> bytes:
    rows = [q.to_dict() for q in sorted(questions, key=lambda q: q.qid)]
    return yaml.safe_dump({"version": VERSION, "questions": rows}, sort_keys=True,
                          allow_unicode=True, width=1000).encode("utf-8")  # fmt: skip


def qrels_tsv(questions: Iterable[Question], index: ChunkIndex) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(["qid", "gold_id", "rel", "content_hash", "judge"])
    for q in sorted(questions, key=lambda q: q.qid):
        for g in sorted(q.gold):
            w.writerow([q.qid, g, q.gold[g], index.by_gold[g].content_hash, q.judges.get(g, "")])
        for g in sorted(set(q.nonrel) - set(q.gold)):  # judged not relevant: explicit rel 0
            w.writerow([q.qid, g, 0, index.by_gold[g].content_hash if g in index.by_gold else "",
                        q.judges.get(g, "")])  # fmt: skip
    return buf.getvalue().encode("utf-8")


def split_tsv(questions: list[Question]) -> bytes:
    sp = split_of(questions)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(["qid", "category", "split"])
    for q in sorted(questions, key=lambda q: q.qid):
        w.writerow([q.qid, q.category, sp[q.qid]])
    return buf.getvalue().encode("utf-8")


class Frozen(RuntimeError):
    """The version directory is already frozen: an edit is a new version (v1.1)."""


def judge_errors(questions: Iterable[Question], kappa: float | None) -> list[str]:
    """The judge column (§3.1): every labelled pair has a judge, and Claude's labels stand only
    with a recorded kappa >= KAPPA_MIN (otherwise every label is the human's)."""
    errs: list[str] = []
    claude = 0
    for q in questions:
        for g in sorted(set(q.gold) | set(q.nonrel)):
            j = q.judges.get(g, "")
            if j not in JUDGES:
                errs.append(f"{q.qid} {g}: judge {j!r} (must be one of {', '.join(JUDGES)})")
            claude += j == "claude"
    if claude and (kappa is None or math.isnan(kappa) or kappa < KAPPA_MIN):
        got = "none recorded" if kappa is None or math.isnan(kappa) else f"{kappa:.3f}"
        errs.append(f"{claude} label(s) are Claude's but the blind kappa is {got} "
                    f"(< {KAPPA_MIN}): the human labels the rest (`bench gold label`)")  # fmt: skip
    return errs


def work_routed(
    questions: Iterable[Question], index: ChunkIndex, routes: Mapping[str, str]
) -> list[str]:
    """PG-v1 holds no work-store content: every question source, gold and pooled (rel 0) path must
    be routed to the personal store (`vault.routes` of config.yaml over the importer's defaults)
    and sit in a personal store of the import."""
    from mem.importer import route_of

    bad: list[str] = []
    for q in questions:
        paths = {*q.source, *(split_gold_id(g)[0] for g in {*q.gold, *q.nonrel})}
        for p in sorted(paths):
            if route_of(p, routes) == "work" or index.store_of_path.get(p) == "work":
                bad.append(f"{q.qid}: {p} is in a folder routed to work")
    return bad


def freeze(
    questions: list[Question], index: ChunkIndex, out_dir: Path, *, snapshot_sha256: str,
    snapshot_name: str, snapshot_date: str, version: str = VERSION,
    routes: Mapping[str, str] | None = None, kappa: float | None = None, kappa_n: int = 0,
) -> dict[str, str]:  # fmt: skip
    """Write the three hashed files and SNAPSHOT into `out_dir`. Refuses unless `validate` is
    clean, every label has a judge, no path is work-routed and the directory is not frozen yet.
    Returns the file hashes."""
    if (out_dir / "SNAPSHOT").exists():
        raise Frozen(f"{out_dir} is frozen: write a new version (v1.1), never overwrite")
    errs = validate(questions, index) + judge_errors(questions, kappa)
    errs += work_routed(questions, index, routes or {})
    if errs:
        raise ValueError("cannot freeze PG-v1:\n  " + "\n  ".join(errs))
    files = {"questions.yaml": questions_yaml(questions),
             "qrels.tsv": qrels_tsv(questions, index), "split.tsv": split_tsv(questions)}  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
        hashes[name] = sha256_bytes(data)
    (out_dir / "SNAPSHOT").write_text(
        "".join(f"{k}: {v}\n" for k, v in [
            ("version", version), ("snapshot_file", snapshot_name),
            ("snapshot_sha256", snapshot_sha256), ("snapshot_date", snapshot_date),
            ("questions_yaml_sha256", hashes["questions.yaml"]),
            ("qrels_tsv_sha256", hashes["qrels.tsv"]), ("split_tsv_sha256", hashes["split.tsv"]),
            ("n_questions", len(questions)),
            ("judge_kappa", "none" if kappa is None else f"{kappa:.4f}"),
            ("judge_kappa_n", kappa_n), ("judge_kappa_weights", "linear")]),
        encoding="utf-8")  # fmt: skip
    return hashes


@dataclass
class Gold:
    """A frozen PG version, loaded."""

    version: str
    snapshot: dict[str, str]
    questions: list[Question]
    split: dict[str, str]
    qrels: dict[str, dict[str, tuple[int, str]]]  # qid -> gold id -> (rel, content hash)
    judges: dict[tuple[str, str], str] = field(default_factory=dict)  # (qid, id) -> claude | human

    def of_split(self, split: str) -> list[Question]:
        return [q for q in self.questions if self.split[q.qid] == split]


def read_snapshot(path: Path) -> dict[str, str]:
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        k, _, v = ln.partition(": ")
        out[k] = v
    return out


def load(gold_dir: Path) -> Gold:
    """Load a frozen version and re-check its hashes and its split rule (a hand edit fails)."""
    snap = read_snapshot(gold_dir / "SNAPSHOT")
    for fname, key in (("questions.yaml", "questions_yaml_sha256"), ("qrels.tsv", "qrels_tsv_sha256"),
                       ("split.tsv", "split_tsv_sha256")):  # fmt: skip
        got = sha256_bytes((gold_dir / fname).read_bytes())
        if got != snap[key]:
            raise ValueError(f"{gold_dir / fname}: sha256 {got[:12]} != SNAPSHOT {snap[key][:12]} "
                             "(frozen files are never edited: write v1.1)")  # fmt: skip
    raw = yaml.safe_load((gold_dir / "questions.yaml").read_text(encoding="utf-8"))
    qrels: dict[str, dict[str, tuple[int, str]]] = {}
    judges: dict[tuple[str, str], str] = {}
    with (gold_dir / "qrels.tsv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "judge" not in (reader.fieldnames or []):
            raise ValueError(f"{gold_dir / 'qrels.tsv'} has no judge column (claude | human)")
        for row in reader:
            judges[(row["qid"], row["gold_id"])] = row["judge"]
            qrels.setdefault(row["qid"], {})[row["gold_id"]] = (
                int(row["rel"]),
                row["content_hash"],
            )
    questions = []
    for d in raw["questions"]:
        questions.append(Question(
            qid=d["qid"], category=d["category"], question=d["question"], answer=d.get("answer", ""),
            gold={g: rel for g, (rel, _) in qrels.get(d["qid"], {}).items() if rel >= 1},
            nonrel=sorted(g for g, (rel, _) in qrels.get(d["qid"], {}).items() if rel == 0),
            filters=d.get("filters") or {}, reworded=bool(d.get("reworded")),
            source=d.get("source", []), draft_id=d.get("draft_id"),
            store=d.get("store", "personal"),
            judges={g: judges[(d["qid"], g)] for g in qrels.get(d["qid"], {})}))  # fmt: skip
    with (gold_dir / "split.tsv").open(encoding="utf-8", newline="") as f:
        split = {r["qid"]: r["split"] for r in csv.DictReader(f, delimiter="\t")}
    if split != split_of(questions):
        raise ValueError("split.tsv does not follow the sha1(qid) rule")
    have = quota_counts(questions)
    off = [f"{c}: {have.get(c, 0)} != {n}" for c, n in TOTAL.items() if have.get(c, 0) != n]
    off += [f"unknown category {c!r}" for c in have if c not in TOTAL]
    if off:
        raise ValueError(f"{gold_dir}: the question quotas do not hold ({'; '.join(off)})")
    kappa = _kappa_of(snap)
    if errs := judge_errors(questions, kappa):
        raise ValueError(f"{gold_dir}: " + "; ".join(errs[:5]))
    return Gold(snap.get("version", VERSION), snap, questions, split, qrels, judges)


def _kappa_of(snap: dict[str, str]) -> float | None:
    try:
        return float(snap.get("judge_kappa", ""))
    except ValueError:
        return None


def stale_gold(gold: Gold, index: ChunkIndex) -> list[str]:
    """The stale-gold check run on each import (§3.1): a gold id with no chunk (`missing`) or whose
    chunk text changed since labelling (`changed`). Any entry fails a B-R3 run (§2 B-R3 step 3)."""
    out = []
    for qid, golds in sorted(gold.qrels.items()):
        for gid, (rel, h) in sorted(golds.items()):
            if rel == 0:  # a judged non-relevant chunk may change or go without staling the gold
                continue
            c = index.by_gold.get(gid)
            if c is None:
                out.append(f"{qid} {gid}: missing")
            elif c.content_hash != h:
                out.append(f"{qid} {gid}: changed")
    return out
