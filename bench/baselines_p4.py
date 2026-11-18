"""P4 descriptive baselines over the vault snapshot (benchmarks-v2 B-R3): qmd and `rg -il`.

Both return files, so a run is `{qid: [vault-relative paths, best first]}` (top 10) and is scored at
path level (`bench.pg_eval`, DR-4). Both are subprocess calls to tools on the person's machine; the
command runner is a parameter so T0 tests supply a scripted one (nothing is faked in a real run:
the run file records the tool version output and the exact commands).

qmd (https://github.com/tobi/qmd, flags per its README, 2026-10-24):

    qmd --index pgv1 collection add <snapshot dir> --name vault --mask "**/*.md"
    qmd --index pgv1 embed
    qmd --index pgv1 query --json -n 10 "<question>"       # hybrid + rerank, the product default

`file` in the JSON is a `qmd://vault/<path>` URI; the path is taken after the collection name. A
named index keeps the person's own qmd index untouched. qmd sees the same snapshot files the
importer reads, minus nothing: it indexes `**/*.md` (the importer's skip list is not applied to it).

rg floor: for each question the content tokens (`bench.gold.pg.content_tokens`) are searched
one at a time with `rg -ilw -e <token> <dir>`; a file's score is the number of distinct tokens it
contains, ties by path; the top 10 files are the ranking. Scope filters of the question are not
applied to either baseline (they only exist in our `recall`).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bench.gold import pg

Runner = Callable[[list[str]], str]  # argv -> stdout; raises on a non-zero exit (rg: 1 = no match)
INDEX = "pgv1"
COLLECTION = "vault"
TOP = 10


def run_cmd(argv: list[str]) -> str:
    p = subprocess.run(argv, capture_output=True, text=True, check=False)
    if p.returncode not in (0, 1):  # rg exits 1 when nothing matches
        raise RuntimeError(f"{' '.join(argv)[:120]} exited {p.returncode}: {p.stderr[:300]}")
    return p.stdout


def qmd_path(uri: str) -> str:
    """`qmd://vault/work/a.md` -> `work/a.md`; a bare path is returned as is."""
    if uri.startswith("qmd://"):
        return uri.removeprefix("qmd://").partition("/")[2]
    return uri


def qmd_setup(vault: Path, run: Runner = run_cmd) -> dict[str, Any]:
    """Index the snapshot in qmd's own named index. Returns what the run record keeps."""
    version = run(["qmd", "--version"]).strip()
    run(["qmd", "--index", INDEX, "collection", "add", str(vault), "--name", COLLECTION,
         "--mask", "**/*.md"])  # fmt: skip
    run(["qmd", "--index", INDEX, "embed"])
    return {"tool": "qmd", "version": version, "index": INDEX, "collection": COLLECTION}


def qmd_rank(question: str, run: Runner = run_cmd) -> list[str]:
    out = run(["qmd", "--index", INDEX, "query", "--json", "-n", str(TOP), question])
    rows = json.loads(out) if out.strip() else []
    paths: dict[str, None] = {}
    for r in rows:
        paths.setdefault(qmd_path(r["file"]), None)
    return list(paths)[:TOP]


def qmd_run(questions: Sequence[pg.Question], vault: Path, run: Runner = run_cmd) -> dict[str, Any]:
    meta = qmd_setup(vault, run)
    return {**meta, "runs": {q.qid: qmd_rank(q.question, run) for q in questions}}


def rg_rank(question: str, vault: Path, run: Runner = run_cmd) -> list[str]:
    score: dict[str, int] = {}
    for tok in sorted(pg.content_tokens(question)):
        for line in run(["rg", "-ilw", "--glob", "*.md", "-e", tok, str(vault)]).splitlines():
            if line.strip():
                rel = Path(line).resolve().relative_to(Path(vault).resolve()).as_posix()
                score[rel] = score.get(rel, 0) + 1
    return [p for p, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][:TOP]


def rg_run(questions: Sequence[pg.Question], vault: Path, run: Runner = run_cmd) -> dict[str, Any]:
    version = run(["rg", "--version"]).splitlines()[0]
    return {"tool": "rg", "version": version,
            "runs": {q.qid: rg_rank(q.question, vault, run) for q in questions}}  # fmt: skip
