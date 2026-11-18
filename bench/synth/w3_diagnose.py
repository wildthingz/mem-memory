"""Why does B-W3 miss, and does a miss leak into B-R5? (MEM2-156)

Three offline analyses over dev data only (no model, no network):

- `classify_records`: per `update` / `dup_paraphrase` op of the `syn-score-dev` conflict records, the
  cause of a miss under a fitted scorer (`syn-fit-theta --out`): `not_picked` (the counterpart is not
  among the picked candidates), `below_theta` (scored under theta_conf), `cut_delta` (within theta
  but more than `delta` under the list's best), `cut_top_k` (kept by both but beyond `top_k`), or
  `surfaced`. Counts per subtype; per seed the in-sample surfacing on update ops (the number the
  gate's B-W3 per-seed surfacing should equal if a miss cost nothing later); per miss the
  counterpart's score against theta and the list's best, its rank, and the per-feature contribution
  gap `w * (x_best - x_counterpart)`, so a paste shows why the score was low.
- `report` joins the in-sample per-seed surfacing to the gate's (`gate_gap`: positive = what the
  misses cost later ops through the store state, the compounding the owner described) and counts the
  slots with two or more missed ops (`multi_miss_slots`, the compounding signature).
- `chain_report`: for every single-valued slot whose final chain differs from gold, whether the
  slot holds a missed link (a downstream effect of surfacing) or every update was surfaced and
  linked (`surfaced_all`: a chain bug of its own).
- `join_leaks`: leaked B-R5 current-value queries against the slots that hold a missed update. The
  hypothesis "leakage comes from missed conflicts" holds when every leaked query's slot has a miss.

Everything is generic: it reads the records' own fields and the generator's gold, nothing else.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from bench.synth import score
from bench.synth.replay import OpLog
from mem import confmodel

CAUSES = ("surfaced", "not_picked", "below_theta", "cut_delta", "cut_top_k")


def classify_op(rec: dict[str, Any], model: confmodel.ConfModel, theta: float) -> str:
    cands = rec["candidates"]
    idx = [i for i, c in enumerate(cands) if c["counterpart"]]
    if not idx:
        return "not_picked"
    scores = [model.score(c["features"]) for c in cands]
    keys = [c["id"] for c in cands]
    if idx[0] in model.cut(scores, theta, keys):
        return "surfaced"
    s = scores[idx[0]]
    if s < theta:
        return "below_theta"
    if s < max(scores) - model.delta:
        return "cut_delta"
    return "cut_top_k"


def miss_row(
    rec: dict[str, Any], model: confmodel.ConfModel, theta: float, cause: str
) -> dict[str, Any]:
    """One missed op: the counterpart's score, rank and features against the list's best candidate,
    and per feature `w * (x_best - x_counterpart)` (what opened the gap; the scorer is linear)."""
    cands = rec["candidates"]
    scores = [model.score(c["features"]) for c in cands]
    order = sorted(range(len(cands)), key=lambda i: (-scores[i], cands[i]["id"]))
    best = cands[order[0]]["features"] if cands else None
    cp = next((i for i, c in enumerate(cands) if c["counterpart"]), None)
    cpf = cands[cp]["features"] if cp is not None else None
    gap = {f: w * (best[f] - cpf[f]) for f, w in zip(model.features, model.weights, strict=True)} \
        if best is not None and cpf is not None else None  # fmt: skip
    return {"seed": rec["seed"], "op": rec["op"], "kind": rec["kind"],
            "subtype": rec["subtype"] or rec["kind"], "cause": cause,
            "score": scores[cp] if cp is not None else None, "theta": theta,
            "best": max(scores) if scores else None, "rank": order.index(cp) if cp is not None else None,
            "n": len(cands), "counterpart": cpf, "best_candidate": best, "gap_by_feature": gap}  # fmt: skip


def classify_records(records: Sequence[dict[str, Any]], fit: dict[str, Any]) -> dict[str, Any]:
    """`fit`: the `syn-fit-theta --out` JSON (theta_conf, conf_model). Misses are listed per op
    (`misses`, with `miss_row`); `per_seed` is the in-sample surfacing on update ops only."""
    model = confmodel.ConfModel.from_dict(fit["conf_model"])
    theta = float(fit["theta_conf"])
    by_sub: dict[str, Counter] = defaultdict(Counter)
    seeds: dict[str, list[bool]] = defaultdict(list)
    ops, misses = [], []
    for r in records:
        cause = classify_op(r, model, theta)
        sub = r["subtype"] or r["kind"]
        by_sub[sub][cause] += 1
        ops.append({"seed": r["seed"], "op": r["op"], "subtype": sub, "cause": cause})
        if r["kind"] == "update":
            seeds[str(r["seed"])].append(cause == "surfaced")
        else:
            seeds.setdefault(str(r["seed"]), [])
        if cause != "surfaced":
            misses.append(miss_row(r, model, theta, cause))
    total = Counter()
    for c in by_sub.values():
        total.update(c)
    per_seed = {s: {"updates": len(h), "surfaced": sum(h),
                    "surfacing": sum(h) / len(h) if h else None}
                for s, h in sorted(seeds.items(), key=lambda kv: int(kv[0]))}  # fmt: skip
    return {"fit": {"theta_conf": theta, "delta": model.delta, "top_k": model.top_k},
            "by_subtype": {s: dict(c) for s, c in sorted(by_sub.items())},
            "total": dict(total), "per_seed": per_seed, "ops": ops, "misses": misses}  # fmt: skip


def chain_report(
    syn: Any, logs: list[OpLog], mem: dict[int, score.MemRow], mids: dict[int, int]
) -> list[dict[str, Any]]:
    """One row per single-valued slot with an update whose final chain is not gold."""
    by_op = {lg.op["i"]: lg for lg in logs}
    out = []
    for key, s in syn.expected["slots"].items():
        if s["multi"] or not s["had_update"]:
            continue
        gold = syn.expected["memories"]
        want = [(mids.get(op), gold[str(op)]["valid_from"], gold[str(op)]["valid_to"])
                for op in s["chain"]]  # fmt: skip
        ids = {mids[op] for op in s["members"] if op in mids}
        act = sorted((mem[i] for i in ids if i in mem and mem[i].status == "active"),
                     key=lambda m: (m.valid_from, m.recorded_at, m.id))  # fmt: skip
        got = [(m.id, m.valid_from, m.valid_to) for m in act]
        if got == want:
            continue
        links = [by_op[op] for op in s["members"]
                 if op in by_op and by_op[op].op["kind"] in ("update", "dup_paraphrase")]  # fmt: skip
        missed = [lg.op["i"] for lg in links if lg.status == "saved" and not lg.surfaced]
        failed = [lg.op["i"] for lg in links if lg.acted and lg.action_ok is False]
        unsaved = [lg.op["i"] for lg in links if lg.status != "saved"]
        out.append({"slot": key, "want": want, "got": got, "missed_ops": missed,
                    "link_failed_ops": failed, "unsaved_ops": unsaved,
                    "subtypes": [lg.op["gold"].get("subtype") for lg in links],
                    "cause": "missed_link" if missed else "link_failed" if failed
                    else "unsaved_update" if unsaved else "surfaced_all"})  # fmt: skip
    return out


def missed_slots(logs: list[OpLog]) -> dict[str, list[int]]:
    """slot -> ops of saved updates and paraphrase duplicates whose counterpart was not surfaced.
    A missed paraphrase leaves a duplicate active, which B-R5 counts as a leak when recalled."""
    out: dict[str, list[int]] = defaultdict(list)
    for lg in logs:
        if lg.op["kind"] in ("update", "dup_paraphrase") and lg.status == "saved" \
                and not lg.surfaced:  # fmt: skip
            out[lg.op["gold"]["slot"]].append(lg.op["i"])
    return dict(out)


def join_leaks(
    leaked: Sequence[dict[str, Any]],
    missed: dict[str, list[int]],
    slot_of_op: dict[int, str],
) -> dict[str, Any]:
    """`leaked`: B-R5 probe details with `leaked` true and `slot_ops`. A leaked query is explained
    when one of its slot's ops is a missed update. Returns counts and the unexplained queries."""
    explained, unexplained = [], []
    for q in leaked:
        slots = {slot_of_op[o] for o in q["slot_ops"] if o in slot_of_op}
        hit = sorted(o for s in slots for o in missed.get(s, []))
        (explained if hit else unexplained).append({"id": q["id"], "missed_ops": hit})
    n = len(leaked)
    return {"leaked": n, "explained_by_miss": len(explained), "unexplained": unexplained,
            "hypothesis_holds": n > 0 and not unexplained}  # fmt: skip


def report(
    records: Sequence[dict[str, Any]] | None, fit: dict[str, Any] | None,
    w3_extra: dict[str, Any] | None, r5_extra: dict[str, Any] | None,
    slot_of_op: dict[int, dict[int, str]] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Everything available is analysed; a missing input leaves its section out (never invented).
    `slot_of_op`: seed -> op -> slot key (from `generate(seed, size)`), for the leak join."""
    out: dict[str, Any] = {}
    if records and fit:
        out["causes"] = classify_records(records, fit)
    if w3_extra:
        diag = w3_extra.get("diagnosis") or {}
        out["chain"] = {seed: Counter(r["cause"] for r in d["chain"]) for seed, d in diag.items()}
        out["chain_rows"] = {seed: d["chain"] for seed, d in diag.items()}
        out["missed_by_subtype"] = {
            seed: dict(Counter(d["subtype_misses"].values())) for seed, d in diag.items()
        }
        out["multi_miss_slots"] = {
            seed: sum(len(r.get("missed_ops") or []) >= 2 for r in d["chain"])
            for seed, d in diag.items()
        }
        per = w3_extra.get("per_seed") or {}
        if "causes" in out and per:
            out["gate_gap"] = {}
            for seed, g in per.items():
                mine = out["causes"]["per_seed"].get(seed)
                if mine and mine["surfacing"] is not None and "surfacing_recall" in g:
                    out["gate_gap"][seed] = {"in_sample": mine["surfacing"],
                                             "gate": g["surfacing_recall"],
                                             "gap": mine["surfacing"] - g["surfacing_recall"]}  # fmt: skip
    if w3_extra and r5_extra and slot_of_op:
        joins = {}
        for seed, qs in (r5_extra.get("leaked_queries") or {}).items():
            missed = (w3_extra.get("diagnosis") or {}).get(seed, {}).get("missed_slots", {})
            joins[seed] = join_leaks(qs, missed, slot_of_op[int(seed)])
        out["leak_join"] = joins
    return out


def format_report(rep: dict[str, Any]) -> str:
    lines = []
    if c := rep.get("causes"):
        lines.append("B-W3 miss causes per subtype (in-sample final scorer):")
        lines.append(f"  {'subtype':<14}" + "".join(f"{k:>12}" for k in CAUSES))
        for sub, cnt in [*c["by_subtype"].items(), ("ALL", c["total"])]:
            lines.append(f"  {sub:<14}" + "".join(f"{cnt.get(k, 0):>12}" for k in CAUSES))
        f = c["fit"]
        if c["misses"]:
            lines.append(f"misses (counterpart vs the list's best; theta {f['theta_conf']:g}, "
                         f"delta {f['delta']:g}, top_k {f['top_k']}):")  # fmt: skip
            lines.append(f"  {'seed':>4} {'op':>5} {'subtype':<14}{'cause':<12}{'score':>8} "
                         f"{'best':>8} {'rank':>5}  gap by feature (best - counterpart, top 3)")  # fmt: skip
        for m in c["misses"]:
            top = sorted((m["gap_by_feature"] or {}).items(), key=lambda kv: -abs(kv[1]))[:3]
            feats = "  ".join(f"{k} {v:+.2f}" for k, v in top)
            sc = f"{m['score']:8.2f}" if m["score"] is not None else f"{'-':>8}"
            rk = f"{m['rank']:5d}" if m["rank"] is not None else f"{'-':>5}"
            best = f"{m['best']:8.2f}" if m["best"] is not None else f"{'-':>8}"
            lines.append(f"  {m['seed']:>4} {m['op']:>5} {m['subtype']:<14}{m['cause']:<12}{sc} "
                         f"{best} {rk}  {feats}")  # fmt: skip
    for seed, g in (rep.get("gate_gap") or {}).items():
        ps = rep["causes"]["per_seed"][seed]
        lines.append(f"seed {seed}: in-sample {g['in_sample']:.3f} ({ps['surfaced']}/{ps['updates']} "
                     f"updates), gate {g['gate']:.3f}, gap {g['gap']:+.3f}")  # fmt: skip
    mm = rep.get("multi_miss_slots") or {}
    if any(mm.values()):
        lines.append("slots with 2+ misses (compounding): "
                     + ", ".join(f"seed {s}: {n}" for s, n in mm.items()))  # fmt: skip
    for seed, cnt in (rep.get("chain") or {}).items():
        lines.append(f"chain failures seed {seed}: {dict(cnt) or 'none'}"
                     + ("   <- surfaced_all is a chain bug of its own" if cnt.get("surfaced_all")
                        else ""))  # fmt: skip
    for seed, j in (rep.get("leak_join") or {}).items():
        lines.append(
            f"leaks seed {seed}: {j['leaked']} leaked, {j['explained_by_miss']} in a slot "
            f"with a missed update, {len(j['unexplained'])} unexplained -> "
            + ("hypothesis holds" if j["hypothesis_holds"] else "hypothesis NOT shown")
        )
    return "\n".join(lines) or "nothing to analyse"
