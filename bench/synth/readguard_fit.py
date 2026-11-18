"""The θ_read fit of the read-time conflict guard (ADR 0018, MEM2-159; architecture-v2 §3.3 step 6c).

Data: the SYN DEV seeds replayed under the gate being completed (the server's own conflict cut, so
every missed `supersedes` leaves its stale member open, as in B-R5), then every read probe
(`queries.jsonl`) through the production `recall` with the guard OFF. For each probe the record
holds the delivered set (gold validity of every row at the probe's t) and every same-subject pair
of it scored by the gate's conflict model exactly as the guard scores it (`mem.readguard`).

Labels come from the SYN gold only: a pair's `earlier` row is **stale** when the gold says it is
not valid at t (a missed update: hiding it is right) and **co_valid** when it is (two relations of
one subject, two values of a multi-valued relation, a B-W2 paraphrase and its original: hiding it
is wrong). A paraphrase duplicate takes its original's validity.

The rule: θ_read(train) = the smallest cut that hides **no** co-valid training pair (just above the
best-scoring co-valid pair; the midpoint to the next stale score when there is one), plus a safety
`margin` ∈ MARGINS, never below θ_conf (the read cut is strict). The margin is the one
hyper-parameter: chosen by inner leave-one-seed-out CV on the training seeds (worst inner fold:
fewest wrongly hidden rows, then lowest leakage after, then the larger margin), so the estimate is
NESTED: for each outer dev seed the margin and θ_read see the other seeds only, and the outer seed
is scored once. `feasible` = every outer fold hides zero co-valid rows AND reaches leakage 0. The
final θ_read: the same selection on all dev seeds (flat, optimistic, labelled so), then the rule on
all records. Nothing is relaxed: an infeasible fit is reported with its numbers and no gate is made.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

MARGINS = (0.0, 0.25, 0.5, 1.0)  # score units above the best co-valid pair
EPS = 1e-6


# ---- records --------------------------------------------------------------------------------------


def _valid_at(mem: dict[str, Any], t: str) -> bool:
    if mem["status"] != "active":
        return False
    return mem["valid_from"] <= t and (mem["valid_to"] is None or t < mem["valid_to"])


def probe_records(run: Any, client: Any) -> list[dict[str, Any]]:
    """One record per SYN read probe of `run`, read through `client` (an `OursClient` over the
    replayed store with the gate in force and the guard OFF): the delivered set with gold validity,
    and every same-subject pair scored as the guard would score it."""
    from mem import readguard

    store = client._server.store
    if store.conflict_guard:
        raise ValueError("probe_records needs the guard off (P3Backend(conflict_guard=False))")
    syn, inv = run.syn, {f"m{m}": op for op, m in run.mids.items()}
    mems, ops = syn.expected["memories"], syn.ops
    slot_of = {int(k): v["slot"] for k, v in mems.items()}
    multi = {k: v["multi"] for k, v in syn.expected["slots"].items()}

    def validity(op: int | None, t: str) -> bool | None:
        if op is None:
            return None
        m = mems[str(op)]
        if m["status"] == "duplicate":  # a paraphrase takes its original's validity
            orig = ops[op]["gold"].get("counterpart")
            return validity(orig, t) if orig is not None else None
        return _valid_at(m, t)

    out = []
    for q in syn.queries:
        args = {"query": q["query"]} | ({"as_of": q["as_of"]} if q.get("as_of") else {})
        r = client.call("recall", args, now=q["now"])
        t = q.get("as_of") or q["now"]
        slot_ops = set(q.get("slot_ops", []))
        rows = []
        for row in r.structured["rows"]:
            op = inv.get(row["id"])
            rows.append({"id": row["id"], "op": op, "slot": op in slot_ops,
                         "valid": validity(op, t),
                         "gold": op is not None and op in set(q["gold"]),
                         "dup": op is not None and mems[str(op)]["status"] == "duplicate"})  # fmt: skip
        node_ids = [int(x["id"][1:]) for x in rows if x["id"].startswith("m")]
        pairs = []
        for p in readguard.score_pairs(store, node_ids):
            e, l = inv.get(f"m{p.earlier}"), inv.get(f"m{p.later}")
            v = validity(e, t)
            label = "other" if v is None else ("co_valid" if v else "stale")
            if e is None or l is None:
                sub = "other"
            elif slot_of[e] != slot_of[l]:
                sub = "cross_slot"
            elif mems[str(e)]["status"] == "duplicate" or mems[str(l)]["status"] == "duplicate":
                sub = "paraphrase"
            elif multi.get(slot_of[e]):
                sub = "multi"
            else:
                sub = "same_slot"
            pairs.append({"later": f"m{p.later}", "earlier": f"m{p.earlier}",
                          "score": round(p.score, 6), "label": label, "sub": sub})  # fmt: skip
        out.append({"seed": syn.seed, "qid": f"syn{syn.seed}:{q['id']}", "type": q["type"],
                    "t": t, "rows": rows, "pairs": pairs})  # fmt: skip
    return out


# ---- evaluation at a cut ----------------------------------------------------------------------------


def hidden_at(record: dict[str, Any], theta: float) -> dict[str, str]:
    """earlier id -> later id hidden at `theta` (the guard's own rule: best pair names it)."""
    out: dict[str, str] = {}
    for p in sorted(record["pairs"], key=lambda p: (-p["score"], p["later"], p["earlier"])):
        if p["score"] >= theta and p["earlier"] not in out:
            out[p["earlier"]] = p["later"]
    return out


def evaluate(records: Sequence[dict[str, Any]], theta: float) -> dict[str, Any]:
    """The guard's effect at `theta` on `records`: wrongly hidden co-valid rows, stale rows hidden,
    and B-R5 / B-R2 (ii) before -> after (leakage counts a stale slot row in the set, as the suite
    does; recall@1 the first remaining row being gold; as-of the first remaining slot row)."""
    wrong = stale_hidden = stale_total = 0
    cur_n = asof_n = 0
    leak_b = leak_a = cur_b = cur_a = asof_b = asof_a = 0
    residual: list[dict[str, Any]] = []
    for rec in records:
        hidden = hidden_at(rec, theta)
        rows = rec["rows"]
        for row in rows:
            if row["valid"] is False:
                stale_total += 1
                stale_hidden += row["id"] in hidden
            elif row["valid"] is True and row["id"] in hidden:
                wrong += 1
        left = [r for r in rows if r["id"] not in hidden]
        if rec["type"] == "current":
            cur_n += 1
            lb = any(r["slot"] and r["valid"] is False for r in rows)
            la = any(r["slot"] and r["valid"] is False for r in left)
            leak_b += lb
            leak_a += la
            cur_b += bool(rows) and rows[0]["gold"]
            cur_a += bool(left) and left[0]["gold"]
            if la:
                residual.append({"qid": rec["qid"], "leaking": [
                    r["id"] for r in left if r["slot"] and r["valid"] is False],
                    "cause": _cause(rec, left, theta)})  # fmt: skip
        elif rec["type"] == "as_of":
            asof_n += 1
            sb = [r for r in rows if r["slot"]]
            sa = [r for r in left if r["slot"]]
            asof_b += bool(sb) and sb[0]["gold"]
            asof_a += bool(sa) and sa[0]["gold"]
    rate = lambda n, d: n / d if d else math.nan
    return {
        "wrongly_hidden": wrong, "stale_hidden": stale_hidden, "stale_total": stale_total,
        "leakage_before": rate(leak_b, cur_n), "leakage_after": rate(leak_a, cur_n),
        "current_recall_at_1_before": rate(cur_b, cur_n),
        "current_recall_at_1_after": rate(cur_a, cur_n),
        "as_of_accuracy_before": rate(asof_b, asof_n), "as_of_accuracy_after": rate(asof_a, asof_n),
        "current_n": cur_n, "as_of_n": asof_n, "residual_leaks": residual,
    }  # fmt: skip


def _cause(rec: dict[str, Any], left: list[dict[str, Any]], theta: float) -> str:
    """Why a stale slot row is still delivered at `theta`: `paraphrase` (a missed duplicate: the
    ruling keeps it), `below_theta` (a same-subject later row scored it under the cut),
    `no_partner` (nothing later about the subject is in the set, e.g. the gold current was not
    delivered)."""
    stale = {r["id"] for r in left if r["slot"] and r["valid"] is False}
    if any(r["dup"] for r in left if r["id"] in stale):
        return "paraphrase"
    scored = {p["earlier"] for p in rec["pairs"]}
    if stale & scored:
        return "below_theta"
    return "no_partner"


# ---- the rule and the nested CV -----------------------------------------------------------------------


def choose_theta(
    records: Sequence[dict[str, Any]], theta_conf: float, margin: float = 0.0
) -> tuple[float, dict[str, Any]]:
    """θ_read on `records`: just above the best co-valid pair (midpoint to the next stale score when
    one lies above it), plus `margin`, never below `theta_conf`. Rounded up to 1e-6 so the stored
    cut never hides a training co-valid pair."""
    co = [p["score"] for r in records for p in r["pairs"] if p["label"] == "co_valid"]
    st = [p["score"] for r in records for p in r["pairs"] if p["label"] == "stale"]
    note = ""
    if co:
        lo = max(co)
        above = [s for s in st if s > lo]
        theta = (lo + min(above)) / 2 if above else lo + EPS
        if not above:
            note = "no stale pair scores above the best co-valid pair: the cut hides nothing stale"
    elif st:
        theta = min(st)
        note = "no co-valid pairs on these seeds: the cut sits at the lowest stale score"
    else:
        theta = theta_conf
        note = "no pairs at all: the cut is theta_conf"
    theta = max(theta + margin, theta_conf)
    rounded = math.ceil(theta * 1e6) / 1e6
    if co and rounded <= max(co):
        rounded = max(co) + EPS
    return float(rounded), {"max_co_valid": max(co) if co else None,
                            "min_stale_above": min([s for s in st if s > max(co)], default=None)
                            if co else (min(st) if st else None),
                            "co_valid_pairs": len(co), "stale_pairs": len(st), "note": note}  # fmt: skip


def _fold_key(ev: dict[str, Any], margin: float) -> tuple[Any, ...]:
    return (ev["wrongly_hidden"], ev["leakage_after"], -margin)


def _worst(evs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"wrongly_hidden": max(e["wrongly_hidden"] for e in evs),
            "leakage_after": max(e["leakage_after"] for e in evs),
            "leakage_before": max(e["leakage_before"] for e in evs),
            "current_recall_at_1_after": min(e["current_recall_at_1_after"] for e in evs),
            "current_recall_at_1_before": min(e["current_recall_at_1_before"] for e in evs),
            "as_of_accuracy_after": min(e["as_of_accuracy_after"] for e in evs),
            "as_of_accuracy_before": min(e["as_of_accuracy_before"] for e in evs)}  # fmt: skip


def select_margin(
    records: Sequence[dict[str, Any]], seeds: Sequence[int], theta_conf: float,
    margins: Sequence[float] = MARGINS,
) -> dict[str, Any]:  # fmt: skip
    """Inner leave-one-seed-out CV over `seeds`: for each margin, θ_read on all but one seed and
    the held-out seed's evaluation; the margin with the best worst inner fold wins (fewest wrongly
    hidden rows, then the lowest leakage after, then the larger margin)."""
    by_seed = {s: [r for r in records if r["seed"] == s] for s in seeds}
    rows = []
    for m in margins:
        evs = []
        for s in seeds:
            train = [r for t, rs in by_seed.items() if t != s for r in rs]
            theta, _ = choose_theta(train, theta_conf, m)
            evs.append(evaluate(by_seed[s], theta))
        w = _worst(evs)
        rows.append({"margin": m, "worst": w, "key": _fold_key(w, m)})
    best = min(rows, key=lambda r: r["key"])
    return {"margin": best["margin"], "worst_inner": best["worst"],
            "grid": [{"margin": r["margin"], **r["worst"]} for r in rows]}  # fmt: skip


def fit_read_guard(
    records: Sequence[dict[str, Any]], theta_conf: float, conf_model_fingerprint: str,
    margins: Sequence[float] = MARGINS,
) -> dict[str, Any]:  # fmt: skip
    """The nested fit (module docstring). Needs records of at least three dev seeds (two inner
    folds); refuses held-out seeds."""
    from bench.synth import data as D
    from mem.readguard import ReadGuard

    seeds = sorted({r["seed"] for r in records})
    if bad := [s for s in seeds if s not in D.DEV_SEEDS]:
        raise ValueError(f"seeds {bad} are not dev seeds; only {D.DEV_SEEDS} feed the fit")
    if len(seeds) < 3:
        raise ValueError("nested CV needs records of at least three seeds (two inner folds)")
    if not math.isfinite(theta_conf):
        raise ValueError("theta_conf must be the gate's finite write-time cut")
    by_seed = {s: [r for r in records if r["seed"] == s] for s in seeds}
    outer = []
    for s in seeds:
        others = [t for t in seeds if t != s]
        train = [r for t in others for r in by_seed[t]]
        sel = select_margin(train, others, theta_conf, margins)
        theta, how = choose_theta(train, theta_conf, sel["margin"])
        ev = evaluate(by_seed[s], theta)
        outer.append({"seed": s, "selected_on": others, "margin": sel["margin"],
                      "inner_worst": sel["worst_inner"], "theta_read": theta, "rule": how,
                      **{k: v for k, v in ev.items() if k != "residual_leaks"},
                      "residual_leaks": ev["residual_leaks"][:10]})  # fmt: skip
    worst = _worst(outer)
    feasible = worst["wrongly_hidden"] == 0 and worst["leakage_after"] == 0
    sel = select_margin(records, seeds, theta_conf, margins)
    theta, how = choose_theta(records, theta_conf, sel["margin"])
    ins = evaluate(records, theta)
    guard = ReadGuard.build(theta, conf_model_fingerprint)
    n_pairs = sum(len(r["pairs"]) for r in records)
    by_sub: dict[str, dict[str, int]] = {}
    for r in records:
        for p in r["pairs"]:
            d = by_sub.setdefault(p["sub"], {"co_valid": 0, "stale": 0, "other": 0})
            d[p["label"]] += 1
    return {
        "theta_read": theta, "theta_conf": theta_conf, "read_guard": guard.to_dict(),
        "feasible": feasible, "chosen": {"margin": sel["margin"]}, "rule": how,
        "seeds": seeds, "queries": len(records), "pairs": n_pairs, "pairs_by_sub": by_sub,
        "margins": list(margins),
        "nested": {"outer_folds": outer, "worst": worst,
                   "target": {"wrongly_hidden": 0, "leakage_after": 0.0}},
        "cv": {"chosen": sel, "note": "flat leave-one-seed-out CV that picks the final margin:"
               " optimistic, every seed took part; `nested` is the honest estimate"},
        "in_sample": {k: v for k, v in ins.items() if k != "residual_leaks"},
        "residual_leaks": ins["residual_leaks"],
    }  # fmt: skip


def format_report(fit: dict[str, Any]) -> str:
    n, w = fit["nested"], fit["nested"]["worst"]
    f3 = lambda x: (
        "    nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:7.3f}"
    )
    head = (
        f"read guard nested CV: outer = held-out dev seed {fit['seeds']}, inner = leave-one-seed-out"
        f" on the other {len(fit['seeds']) - 1} (margin grid {fit['margins']}); {fit['queries']}"
        f" probes, {fit['pairs']} same-subject pairs in the delivered sets"
        f" (by kind: {fit['pairs_by_sub']}); theta_conf {fit['theta_conf']:g}"
    )
    cols = (" seed  margin  theta_read  wrong_hidden  stale_hidden/total  leakage b->a   cur@1 b->a"
            "   as_of b->a")  # fmt: skip
    lines = [
        head,
        "outer fold: margin chosen on the other seeds only (theta_read on them too)",
        cols,
    ]
    for f in n["outer_folds"]:
        lines.append(
            f"{f['seed']:>5}  {f['margin']:6.2f}  {f['theta_read']:10.4f}  {f['wrongly_hidden']:12d}"
            f"  {f['stale_hidden']:9d}/{f['stale_total']:<8d} {f3(f['leakage_before'])}->"
            f"{f3(f['leakage_after'])} {f3(f['current_recall_at_1_before'])}->"
            f"{f3(f['current_recall_at_1_after'])} {f3(f['as_of_accuracy_before'])}->"
            f"{f3(f['as_of_accuracy_after'])}"
        )
    lines.append(
        f"worst outer fold: wrongly hidden {w['wrongly_hidden']}, leakage after"
        f" {f3(w['leakage_after']).strip()} (before {f3(w['leakage_before']).strip()}), cur@1"
        f" {f3(w['current_recall_at_1_after']).strip()}, as_of {f3(w['as_of_accuracy_after']).strip()}"
        f" (target: 0 wrongly hidden, leakage 0): " + ("PASS" if fit["feasible"] else "MISS")
    )
    ins = fit["in_sample"]
    lines.append(
        f"final fit (all dev seeds, margin {fit['chosen']['margin']:g} by flat CV: optimistic):"
        f" theta_read {fit['theta_read']:g} ({
            fit['rule']['note'] or 'rule: just above the best co-valid pair'
        }; best co-valid {fit['rule']['max_co_valid']}, next stale"
        f" {fit['rule']['min_stale_above']}); in-sample wrongly hidden {ins['wrongly_hidden']},"
        f" stale hidden {ins['stale_hidden']}/{ins['stale_total']}, leakage"
        f" {f3(ins['leakage_before']).strip()}->{f3(ins['leakage_after']).strip()}; feasible"
        f" {fit['feasible']} (nested)"
    )
    for r in fit["residual_leaks"][:10]:
        lines.append(f"  residual leak {r['qid']}: {r['leaking']} ({r['cause']})")
    return "\n".join(lines)
