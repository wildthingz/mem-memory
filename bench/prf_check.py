"""`bench prf-check` (MEM2-216): Option C, falsify first. No rerank, no gate, no API.

For each recorded LME tune answerable, rebuild its haystack store on the models of the running
`mem serve` (embedding only; the reranker is never called), take the first pass's top-10 chunks from
the recorded pool, derive expansion terms (`mem.prf`), add an FTS and a vec leg for the expanded
query to the fusion, widen the pool 30 -> 40 and count the gold chunks that are in the new pool and
were not in the recorded one. The verdict line is the spec's: C is DROPPED if fewer than 5% of the
tune answerables gain a gold chunk. The same widening without the expansion is the control: it shows
how much of the gain is just ten more slots.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from mem import prf
from mem.serve import DEFAULT_PORT

MIN_SHARE = 0.05  # C is dropped if fewer than this share of the tune answerables gain a gold chunk
POOL_BEFORE, POOL_AFTER = 30, 40
EXAMPLES = 8


def ingest(store: Any, question: dict[str, Any], haystack: Any) -> dict[int, str]:
    """Chunk id -> `<sid>#<round>` for the question's haystack, written with the production path."""
    from bench.adapters.lme_ours import ingest_haystack

    return ingest_haystack(store, question["question_id"], haystack)


def probe(
    store: Any, refs: Mapping[int, str], query: str, as_of: str, *, before: Sequence[str],
    gold: Collection[str], tagger: Callable[[str], prf.Tagged], top: int = 10,
    min_chunks: int = 2, cap: int = 10, size: int = POOL_AFTER,
) -> dict[str, Any]:  # fmt: skip
    """One query: the recorded pool `before` (refs, pool order), the expansion terms from its first
    `top` chunks, the widened pools with and without the expanded legs, and the gold gained."""
    from mem import read

    ids = {ref: cid for cid, ref in refs.items()}
    texts = [store.conn.execute("SELECT text FROM chunks WHERE id = ?", (ids[r],)).fetchone()[0]
             for r in before[:top]]  # fmt: skip
    terms = prf.expansion_terms(query, texts, tagger, top=top, min_chunks=min_chunks, cap=cap)
    first, _ = read.candidate_pool(store, query, t=as_of, size=len(before))
    plain, _ = read.candidate_pool(store, query, t=as_of, size=size)
    after = (expanded_pool(store, query, prf.expand_query(query, terms), t=as_of, size=size)
             if terms else plain)  # fmt: skip
    seen = set(before)
    to_refs = lambda pool: [refs[c] for c, _ in pool]
    return {
        "terms": terms,
        "after": to_refs(after),
        "gained": sorted((set(to_refs(after)) & set(gold)) - seen),
        "gained_plain": sorted((set(to_refs(plain)) & set(gold)) - seen),
        "first_pass_match": set(to_refs(first)) == seen,
    }


def expanded_pool(store: Any, query: str, expand: str, *, t: str | None,
                  size: int) -> list[tuple[int, float]]:  # fmt: skip
    """`read.candidate_pool` with an FTS and a vec leg for the expanded query added to the fusion,
    in the serving leg order (query FTS, query vec, expanded FTS, expanded vec, session). Option C
    lives here, not in serving (MEM2-249 F13): C stays on its own branch until its verdict."""
    from mem import read

    conn, k = store.conn, store.retrieval["leg_k"]
    read.build_allowed(conn, t)
    legs = [read.fts_leg(conn, query, k),
            read.vec_leg(conn, store.vec_table, store.query_vector(query), k),
            read.fts_leg(conn, expand, k),
            read.vec_leg(conn, store.vec_table, store.query_vector(expand), k)]  # fmt: skip
    if store.session_leg:
        legs.append(read.session_leg(conn, store.vec_table, store.query_vector(query), k,
                                     store.retrieval["per_node_cap"]))  # fmt: skip
    return read.pool_from_legs(store, legs, size)


def _share(n: int, d: int) -> float:
    return n / d if d else 0.0


def summarise(rows: Sequence[dict[str, Any]], *, threshold: float = MIN_SHARE,
              examples: int = EXAMPLES) -> dict[str, Any]:  # fmt: skip
    """Counts by type. `dropped`: the share of rows with a gained gold chunk is below `threshold`."""
    by_type: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: r["category"]):
        t = by_type.setdefault(r["category"], {"n": 0, "gained": 0})
        t["n"] += 1
        t["gained"] += bool(r["gained"])
    for t in by_type.values():
        t["share"] = _share(t["gained"], t["n"])
    n, gained = len(rows), sum(bool(r["gained"]) for r in rows)
    share = _share(gained, n)
    plain = sum(bool(r.get("gained_plain")) for r in rows)
    beyond = sum(bool(set(r["gained"]) - set(r.get("gained_plain", ()))) for r in rows)
    ex = [{k: r[k] for k in ("qid", "category", "question", "terms", "gained")}
          for r in sorted(rows, key=lambda r: r["qid"]) if r["gained"]][:examples]  # fmt: skip
    return {
        "n": n,
        "gained": gained,
        "share": share,
        "threshold": threshold,
        "dropped": share < threshold and not math.isclose(share, threshold),
        "control_gained": plain,
        "beyond_control": beyond,
        "by_type": by_type,
        "examples": ex,
    }


def verdict(s: Mapping[str, Any]) -> str:
    head = "C is DROPPED" if s["dropped"] else "C is NOT dropped"
    cmp = "<" if s["dropped"] else ">="
    return (
        f"{head}: {s['gained']}/{s['n']} tune answerables gain a gold chunk "
        f"({s['share']:.1%} {cmp} {s['threshold']:.0%} threshold)"
    )


def report(s: Mapping[str, Any]) -> str:
    lines = [
        f"tune answerables: {s['n']}; gain a gold chunk in the 40-pool: {s['gained']} ({s['share']:.1%})"
    ]
    lines += [
        f"  {t:<28} {v['gained']}/{v['n']} ({v['share']:.1%})" for t, v in s["by_type"].items()
    ]
    lines.append(f"control (pool 40, no expansion): {s['control_gained']}/{s['n']} gain a gold chunk; "
                 f"{s['beyond_control']} gain one only through the expansion")  # fmt: skip
    for e in s["examples"]:
        lines.append(f"  e.g. {e['qid']} [{e['category']}] {e['question']!r}: +{e['gained']} "
                     f"via {e['terms']}")  # fmt: skip
    lines.append(verdict(s))
    return "\n".join(lines)


def run(labels: Path, lme_file: Path, records: Path, cfg: Any, port: int,
        tagger: Callable[[str], prf.Tagged], *, top: int = 10, min_chunks: int = 2,
        cap: int = 10, size: int = POOL_AFTER) -> dict[str, Any]:  # fmt: skip
    from bench.adapters.lme_ours import OursLmeBackend
    from bench.datasets import calibration, lme
    from mem import calibrate, models_http

    fp_all, pools = calibrate.read_records(records)
    if fp_all != cfg.fp_all():
        raise ValueError(f"{records} was scored with fp_all {fp_all[:8]}, the config is "
                         f"{cfg.fp_all()[:8]}: pass the config the pools were scored with")  # fmt: skip
    recorded = {p.qid: [i.key for i in p.items] for p in pools}
    rows = [r for r in calibration.read_tune_labels(labels) if r["kind"] == "answerable"]
    questions = {q["question_id"]: q for q in lme.load_questions(lme_file)}
    backend = OursLmeBackend(cfg=cfg, loaded=models_http.connect(port, cfg))
    out: list[dict[str, Any]] = []
    try:
        for row in rows:
            q = questions[row["qid"]]
            before = recorded.get(f"{row['qid']}:answerable")
            if before is None:
                raise ValueError(f"{records}: no pool for {row['qid']}:answerable")
            as_of = lme.question_as_of(q["question_date"])
            db, store = backend._open(f"{row['qid']}-prf", as_of)
            try:
                refs = ingest(store, q, calibration.label_haystack(q, row))
                p = probe(store, refs, q["question"], as_of, before=before, gold=set(row["rel"]),
                          tagger=tagger, top=top, min_chunks=min_chunks, cap=cap, size=size)  # fmt: skip
            finally:
                store.close()
                db.unlink(missing_ok=True)
            out.append({"qid": row["qid"], "category": row["category"], "question": q["question"],
                        "terms": p["terms"], "gained": p["gained"],
                        "gained_plain": p["gained_plain"], "first_pass_match": p["first_pass_match"]})  # fmt: skip
    finally:
        backend.close()
    s = summarise(out)
    s["first_pass_mismatches"] = sum(not r["first_pass_match"] for r in out)
    s["verdict"] = verdict(s)
    s["rows"] = out
    return s


def register(app: typer.Typer) -> None:
    @app.command("prf-check")
    def prf_check_cmd(
        labels: Annotated[Path, typer.Option(help="Tune labels from `bench export-tune`.")],
        lme_file: Annotated[Path, typer.Option("--lme", help="longmemeval_s_cleaned.json.")],
        records: Annotated[Path, typer.Option(help="Tune pools from `mem calibrate --records-out`.")],
        out: Annotated[Path, typer.Option(help="Result JSON (summary + one row per query).")] = Path("data/prf-check.json"),
        port: Annotated[int, typer.Option(help="The mem serve whose embedder is used.")] = DEFAULT_PORT,
        config: Annotated[Path | None, typer.Option(help="The config the pools were scored with.")] = None,
        embedder: Annotated[str, typer.Option(help="st = pinned models; hash = test stubs.")] = "st",
    ) -> None:  # fmt: skip
        """Option C, falsify first: do the first pass's top-10 terms pull gold into the 40-pool?
        Embeds through the running `mem serve --http --models-only`; no rerank, no gate, no API."""
        from mem import cli as mem_cli
        from mem import models_http

        cfg = mem_cli._config(config, embedder)
        try:
            tagger = prf.spacy_tagger()
            s = run(labels, lme_file, records, cfg, port, tagger)
        except prf.NlpUnavailable as e:
            raise typer.BadParameter(str(e)) from e
        except models_http.ServeUnavailable as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from e
        except (OSError, ValueError, KeyError) as e:
            typer.echo(f"prf-check: {e}", err=True)
            raise typer.Exit(1) from e
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(report(s))
        if s["first_pass_mismatches"]:
            typer.echo(f"WARNING: the recomputed first pass differs from the recorded pool for "
                       f"{s['first_pass_mismatches']} queries (counts are against the recorded pool)")  # fmt: skip
        typer.echo(f"wrote {out}")
