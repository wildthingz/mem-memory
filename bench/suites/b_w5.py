"""B-W5: agent-replay write quality (benchmarks-v2 B-W5; T2, gated at P3).

Two halves. The recorded run (`bench w5-run`, needs the Anthropic key and money) saves a run file;
THIS module scores a run file offline and deterministically (every metric recomputes from the file),
writes the 50-memory audit sheet and applies the filled sheet. So the paid part never has to repeat
for a scoring or alignment-list fix.

Metric definitions (numerator / denominator; unit = dialogue for the CIs):
- capture      gold facts with >= 1 aligned memory recorded at or after their dialogue's start /
               gold facts (whatever happened to the memory later).
- noise_rate   memories aligned to no gold fact, or whose text carries an `ignore` marker /
               all memories the writer created (any status).
- atomicity    memories aligned to <= 1 gold fact / memories (reported).
- about_fill   memories that mention a gold entity by name or alias AND have that entity on a live
               `about` edge / memories that mention a gold entity (the writer filled `about` with the
               subject it was talking about).
- conflict_action  surfaced true conflicts (an update fact whose new memory's `possible_conflicts`
               contains a memory aligned to its predecessor fact) acted on with `supersedes`
               (a `link` either direction or inline `supersedes`) / surfaced true conflicts.
- wrong_invalidation  gold-current facts that have an aligned memory but no CURRENT aligned memory
               (status active, valid_to NULL) / gold-current facts with an aligned memory.
- slot_accuracy (reported) slots where exactly the gold-current value has a current aligned memory.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.metrics.stats import bootstrap_ratio
from bench.suites.base import BackendUnavailable, Context, SuiteResult
from bench.synth import data as D
from bench.synth import dialogue as dlg
from bench.synth.align import aligns, find, mentions_marker, tokens
from bench.synth.generator import generate
from bench.synth.run import OPT as _SYN_OPT  # noqa: F401

OPT = "w5"
AUDIT_SEED = 7
AUDIT_ALIGNED, AUDIT_UNALIGNED = 30, 20
SHEET_COLS = (
    "memory",
    "kind",
    "aligned_to_or_dialogue_facts",
    "about",
    "text",
    "human_check",
    "note",
)


@dataclass
class W5Setup:
    run_path: Path
    audit_path: Path | None
    split: str  # tune (dev seed) | test (held-out seed 101)
    seed: int = 101
    size: str = "M"


def dialogues_for(seed: int, size: str = "M") -> tuple[Any, list[dlg.Dialogue]]:
    syn = generate(seed, size)
    return syn, dlg.build_dialogues(syn)


def _entity_keys(snapshot: dict[str, Any], syn: Any) -> dict[int, str]:
    """server entity id -> gold entity key, by canonical name or taught alias."""
    by_name = {(" ".join(tokens(e["name"])), e["type"]): e["key"] for e in syn.expected["entities"]}
    by_alias = {" ".join(tokens(e["alias"])): e["key"] for e in syn.expected["entities"]
                if e["alias"]}  # fmt: skip
    out: dict[int, str] = {}
    for ent in snapshot["entities"]:
        hit = by_name.get((" ".join(tokens(ent["name"])), ent["type"]))
        if hit:
            out[ent["id"]] = hit
    for a in snapshot["aliases"]:
        key = by_alias.get(" ".join(tokens(a["alias"])))
        if key:
            out.setdefault(a["node"], key)
    return out


def _dialogue_of(starts: list[str], recorded_at: str) -> int:
    idx = 0
    for i, s in enumerate(starts):
        if s <= recorded_at:
            idx = i
    return idx


def score_run(run: dict[str, Any], seed: int = 101, size: str = "M") -> tuple[dict[str, float],
                                                                               dict[str, Any]]:  # fmt: skip
    """Metrics and details from a run file. Deterministic, no network."""
    syn, dialogues = dialogues_for(seed, size)
    snap = run["snapshot"]
    ekeys = _entity_keys(snap, syn)
    facts = dlg.gold_facts(dialogues)
    tl = dlg.timelines(dialogues)
    cur_gold = dlg.gold_current(dialogues)
    markers = [m for d in dialogues for m in d.ignores]
    starts = [d.start for d in dialogues]
    mems = snap["memories"]

    aligned: dict[int, list[tuple[int, int]]] = {}
    for m in mems:
        about = {ekeys[e] for e in m["about"] if e in ekeys}
        aligned[m["id"]] = [(d, i) for d, i, f in facts if aligns(m["text"], about, f,
                                                                  tl[(f.subject, f.relation)])]  # fmt: skip
    fact_mems: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for m in mems:
        for key in aligned[m["id"]]:
            fact_mems.setdefault(key, []).append(m)

    n_d = len(dialogues)
    cap_num, cap_den = [0.0] * n_d, [0.0] * n_d
    for d, i, _f in facts:
        cap_den[d] += 1
        ok = any(m["recorded_at"] >= starts[d] for m in fact_mems.get((d, i), []))
        cap_num[d] += ok
    noise_num, noise_den = [0.0] * n_d, [0.0] * n_d
    atom = 0
    for m in mems:
        d = _dialogue_of(starts, m["recorded_at"])
        noise_den[d] += 1
        is_noise = not aligned[m["id"]] or any(mentions_marker(m["text"], k) for k in markers)
        noise_num[d] += is_noise
        atom += len(aligned[m["id"]]) <= 1
    strata = ["d"] * n_d
    cap, cap_lo, cap_hi = bootstrap_ratio(cap_num, cap_den, strata)
    noise, n_lo, n_hi = bootstrap_ratio(noise_num, noise_den, strata)

    # about-fill
    ent_tokens = {e["key"]: [tokens(e["name"])] + ([tokens(e["alias"])] if e["alias"] else [])
                  for e in syn.expected["entities"]}  # fmt: skip
    mention_n = fill_n = 0
    for m in mems:
        toks = tokens(m["text"])
        about = {ekeys[e] for e in m["about"] if e in ekeys}
        named = [k for k, forms in ent_tokens.items() if any(find(toks, f) for f in forms)]
        if named:
            mention_n += 1
            fill_n += any(k in about for k in named)

    # conflict action
    calls_by_d = {r["dialogue"]: r["calls"] for r in run["dialogues"]}
    surfaced = acted = 0
    for dl in dialogues:
        for i, f in enumerate(dl.facts):
            if not f["conflict_with"]:
                continue
            d0, i0 = f["conflict_with"]
            old_ids = {f"m{m['id']}" for m in fact_mems.get((d0, i0), [])}
            new_ids = {f"m{m['id']}" for m in fact_mems.get((dl.index, i), [])}
            done = False
            hit = None
            for c in calls_by_d.get(dl.index, []):
                if c["tool"] == "remember" and c["structured"].get("id") in new_ids:
                    cp = {
                        x["id"] if isinstance(x, dict) else str(x).split()[0]
                        for x in c["structured"].get("possible_conflicts") or []
                    }
                    if cp & old_ids:
                        hit = (c["structured"]["id"], sorted(cp & old_ids))
                        if c["input"].get("supersedes") in old_ids:
                            done = True
            if hit:
                surfaced += 1
                for c in calls_by_d.get(dl.index, []):
                    if (
                        c["tool"] == "link"
                        and c["input"].get("rel") == "supersedes"
                        and not c["error"]
                    ):
                        pair = {c["input"].get("src"), c["input"].get("dst")}
                        if hit[0] in pair and pair & set(hit[1]) - {hit[0]}:
                            done = True
                acted += done

    # wrong invalidation and slot accuracy
    def current(m: dict[str, Any]) -> bool:
        return m["status"] == "active" and m["valid_to"] is None

    wi_den = wi_bad = 0
    for key in sorted(cur_gold):
        ms = fact_mems.get(key, [])
        if ms:
            wi_den += 1
            wi_bad += not any(current(m) for m in ms)
    slots: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for d, i, f in facts:
        slots.setdefault((f.subject, f.relation), []).append((d, i))
    slot_ok = 0
    for keys in slots.values():
        want = {k for k in keys if k in cur_gold}
        got = {k for k in keys if any(current(m) for m in fact_mems.get(k, []))}
        slot_ok += want == got

    metrics = {
        "capture": cap, "capture_ci_lo": cap_lo, "capture_ci_hi": cap_hi,
        "noise_rate": noise, "noise_ci_lo": n_lo, "noise_ci_hi": n_hi,
        "atomicity": atom / len(mems) if mems else math.nan,
        "about_fill": fill_n / mention_n if mention_n else math.nan,
        "conflict_action": acted / surfaced if surfaced else math.nan,
        "conflicts_surfaced": float(surfaced),
        "wrong_invalidation": wi_bad / wi_den if wi_den else math.nan,
        "slot_accuracy": slot_ok / len(slots),
        "memories": float(len(mems)), "gold_facts": float(len(facts)), "dialogues": float(n_d),
        "run_complete": float(bool(run.get("complete"))),
        "cost_usd": float(run["cost"]["usd"]), "model_calls": float(run["cost"]["calls"]),
    }  # fmt: skip
    details = {
        "aligned": {str(k): v for k, v in aligned.items()},
        "unaligned_memories": [m["id"] for m in mems if not aligned[m["id"]]],
        "uncaptured": [
            [d, i]
            for d, i, _ in facts
            if not any(m["recorded_at"] >= starts[d] for m in fact_mems.get((d, i), []))
        ],
    }
    return metrics, details


# ---- audit ------------------------------------------------------------------------------------


def write_audit_sheet(
    run: dict[str, Any], path: Path, seed: int = 101, size: str = "M"
) -> dict[str, int]:
    """30 aligned + 20 unaligned memories sampled with `random.Random(7)` (ids sorted first)."""
    _, details = score_run(run, seed, size)
    _, dialogues = dialogues_for(seed, size)
    mems = {m["id"]: m for m in run["snapshot"]["memories"]}
    aligned_ids = sorted(int(k) for k, v in details["aligned"].items() if v)
    unaligned_ids = sorted(details["unaligned_memories"])
    rng = random.Random(AUDIT_SEED)
    pick_a = sorted(rng.sample(aligned_ids, min(AUDIT_ALIGNED, len(aligned_ids))))
    pick_u = sorted(rng.sample(unaligned_ids, min(AUDIT_UNALIGNED, len(unaligned_ids))))
    ents = {e["id"]: e["name"] for e in run["snapshot"]["entities"]}
    starts = [d.start for d in dialogues]

    def fact_str(d: int, i: int) -> str:
        f = dialogues[d].facts[i]
        return f"{f['subject']}|{f['relation']}={f['value']}"

    rows = []
    for mid in pick_a:
        m = mems[mid]
        al = "; ".join(fact_str(d, i) for d, i in details["aligned"][str(mid)])
        rows.append((f"m{mid}", "aligned", al, ", ".join(ents.get(e, f"e{e}") for e in m["about"]),
                     m["text"], "", ""))  # fmt: skip
    for mid in pick_u:
        m = mems[mid]
        d = _dialogue_of(starts, m["recorded_at"])
        al = "; ".join(fact_str(d, i) for i in range(len(dialogues[d].facts)))
        rows.append((f"m{mid}", "unaligned", al, ", ".join(ents.get(e, f"e{e}") for e in m["about"]),
                     m["text"], "", ""))  # fmt: skip
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(SHEET_COLS)
        w.writerows(rows)
    return {"aligned": len(pick_a), "unaligned": len(pick_u)}


def apply_audit(path: Path) -> dict[str, float]:
    """`human_check`: for an `aligned` row, `y` = the memory states the listed gold fact, `n` = the
    alignment is wrong; for an `unaligned` row, `y` = it DOES state a listed fact (a miss), `n` = it
    does not. Every row must be filled."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    bad = [r["memory"] for r in rows if r["human_check"].strip().lower() not in ("y", "n")]
    if bad:
        raise ValueError(f"{path}: fill human_check (y/n) on every row; empty on {bad[:5]}")
    al = [r for r in rows if r["kind"] == "aligned"]
    un = [r for r in rows if r["kind"] == "unaligned"]
    wrong = sum(r["human_check"].strip().lower() == "n" for r in al)
    miss = sum(r["human_check"].strip().lower() == "y" for r in un)
    return {"alignment_audit_wrong": float(wrong), "alignment_audit_aligned_n": float(len(al)),
            "alignment_audit_miss_rate": miss / len(un) if un else math.nan,
            "alignment_audit_unaligned_n": float(len(un))}  # fmt: skip


def setup_of(ctx: Context) -> W5Setup:
    s = ctx.options.get(OPT)
    if s is None:
        raise BackendUnavailable(
            "B-W5 needs a recorded run: `bench w5-run ...` (T2, Anthropic key), then "
            "`bench gate P3 --w5-run <file> --w5-audit <filled sheet>` (deliverables/runbooks/p3-gate.md)"
        )
    return s


def run(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    res = SuiteResult("B-W5", tier="t2", mode="agent_replay")
    run_rec = json.loads(Path(s.run_path).read_text(encoding="utf-8"))
    seed, size = run_rec.get("seed", s.seed), run_rec.get("size", s.size)
    metrics, details = score_run(run_rec, seed, size)
    res.metrics |= metrics
    audit = {"alignment_audit_wrong": math.nan}
    if s.audit_path is not None and Path(s.audit_path).exists():
        audit = apply_audit(Path(s.audit_path))
    else:
        res.notes.append("no filled audit sheet: alignment_audit_wrong is NaN (the gate fails)")
    res.metrics |= audit
    res.items = [{"metric": k, "value": v} for k, v in res.metrics.items()]
    res.extra |= {"dataset": {"name": "SYN dialogues", "version": D.SYN_VERSION, "seed": seed,
                              "size": size, "n": int(metrics["dialogues"])},
                  "cost": run_rec["cost"], "model": run_rec["model"],
                  "skill_sha256": run_rec["skill_sha256"], "run_file": str(s.run_path),
                  "unaligned_memories": details["unaligned_memories"][:50]}  # fmt: skip
    return res
