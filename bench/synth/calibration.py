"""SYN dev data for `mem calibrate` (architecture-v2 §3.3 steps 1, 2, 5, 6; benchmarks-v2 §4 P3).

Only the dev seeds (1-5) are ever exported, so calibration cannot see held-out data.

- `dev_pools`: replay each dev seed (oracle-client, real store), then score the read probes' rerank
  pools with `mem.calibrate.score_pool` (steps 1-5, gate off): current-value and as-of probes are
  answerable (the memory valid at the probe time is rel 2, every other memory of the slot is rel 0:
  a stale value must not be delivered), missing-slot probes are no-answer queries. They are
  concatenated with the LME tune pools (`merge_records`) before `mem calibrate --records` fits
  Platt, tau_keep, tau_high and g.
- `conflict_records` + `fit_conf_model`: the same records carry every candidate picked before the
  list cut with its generic features (`mem.confmodel.FEATURES`); `fit_conf_model` fits the small
  logistic scorer and chooses (C, label, top_k, delta, theta_conf) by NESTED leave-one-seed-out CV
  (MEM2-154): for each outer held-out dev seed the choice is made on the other four seeds only,
  with a train precision floor of 0.52; the report gives each outer fold, the worst fold, per
  subtype and the paraphrase rows, then the final fit on all five seeds. The list cut it evaluates
  is `mem.confmodel.list_cut`, the one `mem.conflicts.find` applies. `fit_theta_conf` (below) is
  the v1 raw-logit baseline.
- `fit_cut`: the list cut of a `syn-fit-theta --out` fit as a callable over the server's unfiltered
  rows, for `Replay(cut=...)` (`bench syn-score-dev --fit`, MEM2-156): the records then come from a
  replay whose store state is the gate's (every miss leaves its stale memory active), while each
  record still holds the whole pool. `gated_by` on the record names the fit. Caveat, stated: the
  gating fit saw all five seeds, so it shapes the outer fold's store state too; the labels and
  features are untouched, the selection itself stays nested.
- `conflict_records` + `fit_theta_conf`: for every `update` and `dup_paraphrase` op, the candidate
  list the server returned (run with theta_conf at 0 so the list is not pre-filtered) with its
  scores and whether each candidate shares the gold subject and relation. `fit_theta_conf` picks
  theta_conf maximising surfacing recall after the top-5 cap subject to conflict-list precision
  >= 0.50 (architecture-v2 §3.3 step 6).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from bench.synth.run import SeedRun

TOP_K = 5
PRECISION_FLOOR = 0.50


class FitCut:
    """`mem.confmodel.list_cut` of a fit, applied client-side to the unfiltered rows (see `fit_cut`)."""

    def __init__(self, fit: dict[str, Any]) -> None:
        from mem import confmodel

        if not fit.get("conf_model"):
            raise ValueError("fit has no conf_model (re-run bench syn-fit-theta)")
        self.model = confmodel.ConfModel.from_dict(fit["conf_model"])
        self.theta = float(fit["theta_conf"])
        self.gated_by = {"fingerprint": self.model.fingerprint(), "theta_conf": self.theta}

    def __call__(self, rows: list[dict[str, Any]]) -> list[str]:
        if any(r.get("features") is None for r in rows):
            raise ValueError("rows without features: the server must run unfiltered (theta -inf)")
        scores = [self.model.score(r["features"]) for r in rows]
        keys = [int(str(r["id"]).lstrip("m")) for r in rows]  # node ids: the server's tie-break
        return [rows[i]["id"] for i in self.model.cut(scores, self.theta, keys)]


def fit_cut(fit: dict[str, Any]) -> FitCut:
    """The callable `Replay(cut=...)` takes: the fit's scorer, theta_conf, delta and top_k."""
    return FitCut(fit)


def conflict_records(run: SeedRun) -> list[dict[str, Any]]:
    inv = {f"m{m}": op for op, m in run.mids.items()}
    mems = run.syn.expected["memories"]
    gated_by = getattr(run.replay.cut, "gated_by", None)
    out = []
    for lg in run.logs:
        if lg.op["kind"] not in ("update", "dup_paraphrase") or lg.status != "saved":
            continue
        gold = run.replay.mid.get(lg.op["gold"]["counterpart"])
        slot = lg.op["gold"]["slot"]
        cands = []
        for row in lg.conflict_rows:
            op = inv.get(row["id"])
            cands.append({"id": row["id"], "score": float(row["score"]),
                          "same_slot": op is not None and mems[str(op)]["slot"] == slot,
                          "counterpart": row["id"] == gold,
                          "features": row.get("features"),
                          # validity at pick time (MEM2-156 round 4): store-time signals offline
                          "valid_from": row.get("valid_from"), "valid_to": row.get("valid_to")})  # fmt: skip
        out.append({"seed": run.syn.seed, "op": lg.op["i"], "kind": lg.op["kind"],
                    "subtype": lg.op["gold"].get("subtype"), "as_of": lg.op["args"].get("as_of"),
                    "counterpart_returned": any(c["counterpart"] for c in cands),
                    "surfaced": lg.surfaced, "gated_by": gated_by,
                    "candidates": cands})  # fmt: skip
    return out


def _evaluate(records: Sequence[dict[str, Any]], theta: float) -> tuple[float, float, int]:
    """(surfacing recall, list precision, #returned) of the lists cut at `theta`, top 5."""
    hit = tot = ret = 0
    for r in records:
        kept = sorted((c for c in r["candidates"] if c["score"] >= theta),
                      key=lambda c: -c["score"])[:TOP_K]  # fmt: skip
        hit += any(c["counterpart"] for c in kept)
        tot += 1
        ret += len(kept)
    prec_hit = sum(
        c["same_slot"]
        for r in records
        for c in sorted(
            (c for c in r["candidates"] if c["score"] >= theta), key=lambda c: -c["score"]
        )[:TOP_K]
    )
    return (hit / tot if tot else float("nan"), prec_hit / ret if ret else float("nan"), ret)


def fit_theta_conf(
    records: Sequence[dict[str, Any]], floor: float = PRECISION_FLOOR
) -> dict[str, Any]:
    """theta_conf = the cut that maximises surfacing recall with precision >= floor. Ties go to the
    higher precision, then to the higher theta. `feasible` False = no cut reaches the floor: the
    returned theta then maximises precision (and the phase must fix the scorer on dev, A16-style)."""
    scores = sorted({c["score"] for r in records for c in r["candidates"]})
    best: tuple[tuple[float, float, float], float] | None = None
    fallback: tuple[tuple[float, float], float] | None = None
    for theta in [0.0, *scores]:
        rec, prec, ret = _evaluate(records, theta)
        if not ret:
            continue
        if fallback is None or (prec, rec) > fallback[0]:
            fallback = ((prec, rec), theta)
        if prec >= floor and (best is None or (rec, prec, theta) > best[0]):
            best = ((rec, prec, theta), theta)
    chosen = best[1] if best else (fallback[1] if fallback else 0.0)
    rec, prec, ret = _evaluate(records, chosen)
    base_rec, base_prec, base_ret = _evaluate(records, 0.0)
    return {"theta_conf": chosen, "feasible": best is not None, "surfacing_recall": rec,
            "list_precision": prec, "returned": ret, "ops": len(records),
            "at_theta_0": {"surfacing_recall": base_rec, "list_precision": base_prec,
                           "returned": base_ret}}  # fmt: skip


def dev_pools(runs: Iterable[tuple[SeedRun, Any]]) -> list[Any]:
    """`mem.calibrate.QueryPool`s of the dev probes. `runs`: (SeedRun, server) pairs, the server
    opened on the replayed store with its reranker and the clock at `end_now`."""
    from mem import calibrate
    from mem.calibrate import PoolItem, QueryPool

    pools = []
    for run, server in runs:
        syn, store = run.syn, server.store
        inv = {m: op for op, m in run.mids.items()}
        chunk_node = dict(store.conn.execute("SELECT id, node_id FROM chunks"))
        for q in syn.queries:
            items = calibrate.score_pool(store, q["query"], q.get("as_of") or q["now"])
            rel_op = set(q["gold"])
            labelled = []
            for it in items:
                op = inv.get(chunk_node.get(int(it.key[1:])))
                rel = 2 if op in rel_op else 0
                labelled.append(
                    PoolItem(it.key, it.z, rel, it.cos, it.mcos, it.mz, it.pcos, it.pov)
                )
            pools.append(QueryPool(f"syn{syn.seed}:{q['id']}", q["type"] != "missing_slot",
                                   labelled, f"syn:{q['type']}"))  # fmt: skip
    return pools


def merge_records(paths: Sequence[Path], out: Path) -> int:
    """Concatenate scored-record files (`mem.calibrate.write_records`); one fp_all for all."""
    header, lines = None, []
    for p in paths:
        rows = Path(p).read_text(encoding="utf-8").splitlines()
        h = json.loads(rows[0])
        if header is None:
            header = h
        elif h != header:
            raise ValueError(f"{p}: header {h} differs from {header}: score both with one config")
        lines += [ln for ln in rows[1:] if ln.strip()]
    if header is None:
        raise ValueError("no record files")
    out.write_text(json.dumps(header) + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# ---- the conflict scorer v2: logistic model, nested leave-one-seed-out CV ------------------------

TRAIN_FLOOR = 0.52  # the cut is fitted on a floor above the 0.50 target; the held-out fold decides
TARGET_SURFACING = 0.95
TARGET_PRECISION = 0.50
CS = (0.1, 1.0, 10.0)
LABELS = ("counterpart", "same_slot")
TOP_KS = (3, 5)  # <= confmodel.MAX_TOP_K
DELTAS = (2.0, 3.0, 5.0, 10.0)  # score units under the list's best; 10 is close to no relative cut
GRID = [{"C": c, "label": lab, "top_k": k, "delta": dl}
        for lab in LABELS for c in CS for k in TOP_KS for dl in DELTAS]  # fmt: skip
THETA_GRID = 400


def _ops(records: Sequence[dict[str, Any]], features: Sequence[str]) -> dict[str, Any]:
    """Padded arrays: X (ops, cands, F), valid mask, counterpart and same_slot flags, `upd` = the
    update ops. B-W3 surfacing counts update ops only; precision counts every op's list."""
    import numpy as np

    n = max((len(r["candidates"]) for r in records), default=0)
    x = np.zeros((len(records), n, len(features)))
    ok, cp, ss = (np.zeros((len(records), n), dtype=bool) for _ in range(3))
    for i, r in enumerate(records):
        for j, c in enumerate(r["candidates"]):
            x[i, j] = [c["features"][f] for f in features]
            ok[i, j], cp[i, j], ss[i, j] = True, c["counterpart"], c["same_slot"]
    upd = np.array([r["kind"] == "update" for r in records], dtype=bool)
    return {"x": x, "ok": ok, "cp": cp, "ss": ss, "upd": upd}


def _prep(score: Any, d: dict[str, Any]) -> dict[str, Any]:
    """Lists sorted best first (stable: ties keep record position, as `list_cut` with no keys),
    padding at -inf, with the per-list best score. Sorting once makes every cut cheap."""
    import numpy as np

    s = np.where(d["ok"], score, -np.inf)
    order = np.argsort(-s, axis=1, kind="stable")
    return {"s": np.take_along_axis(s, order, 1), "cp": np.take_along_axis(d["cp"], order, 1),
            "ss": np.take_along_axis(d["ss"], order, 1), "upd": d["upd"],
            "top": s.max(axis=1) if s.shape[1] else np.full(len(s), -np.inf)}  # fmt: skip


def _keep(prep: dict[str, Any], theta: Any, delta: float, k: int) -> Any:
    """Kept mask (thetas, ops, k) of `mem.confmodel.list_cut`: score >= theta and score >= top -
    delta, best k. `theta` may be an array: every cut of a theta grid at once."""
    import numpy as np

    s = prep["s"][:, :k]
    base = s >= (prep["top"] - delta)[:, None]  # -inf padding never passes
    th = np.atleast_1d(np.asarray(theta, dtype=float))
    return base[None] & (s[None] >= th[:, None, None])


def _cut(prep: dict[str, Any], theta: float, delta: float, k: int) -> tuple[Any, int, int]:
    """(per-op surfaced flags, #kept same-slot, #kept) of the lists cut at theta, delta, best k."""
    keep = _keep(prep, theta, delta, k)[0]
    hit = (keep & prep["cp"][:, :k]).any(axis=1)
    return hit, int((keep & prep["ss"][:, :k]).sum()), int(keep.sum())


def _pick_theta(prep: dict[str, Any], delta: float, k: int, floor: float) -> tuple[float, bool]:
    """The cut maximising surfacing (update ops) with precision >= floor; ties: higher precision,
    then higher theta. Infeasible: the cut with the best precision, flagged."""
    import numpy as np

    vals = np.unique(prep["s"][np.isfinite(prep["s"])])
    if not len(vals):
        return 0.0, False
    grid = vals if len(vals) <= THETA_GRID else np.unique(
        np.quantile(vals, np.linspace(0, 1, THETA_GRID)))  # fmt: skip
    keep = _keep(prep, grid, delta, k)  # (thetas, ops, k)
    ret = keep.sum(axis=(1, 2))
    live = ret > 0
    if not live.any():
        return 0.0, False
    hit = (keep & prep["cp"][None, :, :k]).any(axis=2)  # (thetas, ops)
    rec = hit[:, prep["upd"]].mean(axis=1) if prep["upd"].any() else np.zeros(len(grid))
    prec = np.where(
        live, (keep & prep["ss"][None, :, :k]).sum(axis=(1, 2)) / np.maximum(ret, 1), -1.0
    )
    good = live & (prec >= floor)
    if good.any():
        i = max(np.flatnonzero(good), key=lambda i: (rec[i], prec[i], grid[i]))
        return float(grid[i]), True
    i = max(np.flatnonzero(live), key=lambda i: (prec[i], rec[i]))
    return float(grid[i]), False


def _fit_lr(d: dict[str, Any], cfg: dict[str, Any]) -> tuple[Any, float]:
    """Raw-feature coefficients (standardisation folded back in) and bias."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x = d["x"][d["ok"]]
    y = d["cp" if cfg["label"] == "counterpart" else "ss"][d["ok"]]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    lr = LogisticRegression(C=cfg["C"], max_iter=2000, solver="lbfgs")
    lr.fit((x - mu) / sd, y)
    w = lr.coef_[0] / sd
    return w, float(lr.intercept_[0] - (w * mu).sum())


def _subset(d: dict[str, Any], idx: Any) -> dict[str, Any]:
    return {k: v[idx] for k, v in d.items()}


def _scored(d: dict[str, Any], w: Any, b: float) -> Any:
    return d["x"] @ w + b


def _score_cut(prep: dict[str, Any], theta: float, delta: float, k: int) -> tuple[Any, float, int]:
    """(per-op surfaced flags, list precision over all ops, #returned)."""
    hit, ss, ret = _cut(prep, theta, delta, k)
    return hit, (ss / ret if ret else float("nan")), ret


def _fold(hit: Any, prec: float, ret: int, ts: dict[str, Any], sub: Sequence[str],
          kinds: Sequence[str], theta: float, ok: bool) -> dict[str, Any]:  # fmt: skip
    upd = ts["upd"]
    # an empty list set has no precision; 0.0 keeps `min` over folds honest (it cannot pass)
    return {"surfacing": float(hit[upd].mean()) if upd.any() else float("nan"),
            "precision": prec if ret else 0.0,
            "returned": ret, "ops": len(hit), "updates": int(upd.sum()), "theta": theta,
            "train_feasible": ok, "by_subtype": _by_subtype([s for s, u in zip(sub, upd, strict=True) if u], hit[upd]),
            "paraphrase": _paraphrase_rows(kinds, hit)}  # fmt: skip


def _simpler(cfg: dict[str, Any]) -> tuple[Any, ...]:
    """Tie-break among equal CV results: shorter lists, wider delta, more regularised, counterpart."""
    return (-cfg["top_k"], cfg["delta"], -(cfg["C"] or 0.0), cfg["label"] == "counterpart")


def _select(
    d: dict[str, Any], seed_of: Any, sub: Sequence[str], kinds: Sequence[str],
    grid: Sequence[dict[str, Any]], floor: float, fit_fn: Any = None,
    select_floor: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # fmt: skip
    """Leave-one-seed-out CV over the seeds present in `d` ONLY: per fold and config, fit on the
    other seeds, cut at the train-optimal theta (precision >= floor on train), score the held-out
    seed. Returns (chosen result, every config's result). Chosen: the best worst-fold surfacing among
    configs whose worst-fold precision >= `select_floor` (default: `floor`, the same margin theta
    is fitted with; none: the best worst-fold surfacing), ties by `_simpler`. Before MEM2-156 the
    selection accepted the bare 0.50 target, so it sat on configs with no margin (round 2: inner
    .503, held-out .499). The logistic fits are shared across the (top_k, delta) members of a
    (C, label)."""
    import numpy as np

    if select_floor is None:
        select_floor = floor
    seeds = sorted(set(seed_of.tolist()))
    sub, kinds = np.asarray(sub, dtype=object), np.asarray(kinds, dtype=object)
    per_cfg: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(grid))}
    hit_all: dict[int, Any] = {i: np.zeros(len(seed_of), dtype=bool) for i in range(len(grid))}
    groups: dict[tuple[Any, Any], list[int]] = {}
    for i, cfg in enumerate(grid):
        groups.setdefault((cfg.get("C"), cfg.get("label")), []).append(i)
    for sd in seeds:
        te = seed_of == sd
        tr, ts = _subset(d, ~te), _subset(d, te)
        for members in groups.values():
            cfg0 = grid[members[0]]
            if fit_fn is not None:
                score_tr, score_ts = fit_fn(tr, ts)
            else:
                w, b = _fit_lr(tr, cfg0)
                score_tr, score_ts = _scored(tr, w, b), _scored(ts, w, b)
            ptr, pts = _prep(score_tr, tr), _prep(score_ts, ts)
            for i in members:
                cfg = grid[i]
                theta, ok = _pick_theta(ptr, cfg["delta"], cfg["top_k"], floor)
                hit, prec, ret = _score_cut(pts, theta, cfg["delta"], cfg["top_k"])
                hit_all[i][te] = hit
                per_cfg[i].append(
                    {"seed": sd, **_fold(hit, prec, ret, ts, sub[te], kinds[te], theta, ok)}
                )
    results = []
    for i, cfg in enumerate(grid):
        folds = per_cfg[i]
        results.append({"config": cfg, "folds": folds,
                        "worst_surfacing": min(f["surfacing"] for f in folds),
                        "worst_precision": min(f["precision"] for f in folds),
                        "mean_surfacing": float(np.mean([f["surfacing"] for f in folds])),
                        "by_subtype": _by_subtype(sub[d["upd"]].tolist(), hit_all[i][d["upd"]]),
                        "paraphrase": _paraphrase_rows(kinds.tolist(), hit_all[i])})  # fmt: skip
    feasible = [r for r in results if r["worst_precision"] >= select_floor]
    pool = feasible or results
    best = max(
        pool, key=lambda r: (r["worst_surfacing"], r["worst_precision"], _simpler(r["config"]))
    )
    return best, results


def fit_conf_model(
    records: Sequence[dict[str, Any]], features: Sequence[str] | None = None,
    floor: float = TRAIN_FLOOR, grid: Sequence[dict[str, Any]] = GRID,
    select_floor: float | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Fit the scorer; estimate it honestly by NESTED leave-one-seed-out CV over the dev seeds.

    Outer loop: hold out one dev seed. On the other seeds only, choose (C, label, top_k, delta) by
    inner leave-one-seed-out CV (`_select`, train precision floor `floor` for theta, a config is
    eligible when its worst inner fold's precision >= `select_floor`, default `floor`), refit on
    those seeds, pick theta_conf on them, then score the held-out seed once. `nested` reports each
    outer fold (surfacing, precision, per subtype, paraphrase rows, the chosen config), the worst
    fold and the per-subtype worst; `feasible` is the nested worst fold reaching the target.
    The final coefficients: the same selection on all five seeds (`cv`, optimistic by construction:
    it saw every seed), then a fit and theta on all records. The raw reranker logit (the v1
    scorer, one weight, top 5, no delta cut) goes through the flat CV as the baseline."""
    import numpy as np

    from mem import confmodel

    if select_floor is None:
        select_floor = floor
    feats = list(features or confmodel.FEATURES)
    if bad := [f for f in feats if f not in confmodel.FEATURES]:
        raise ValueError(f"not scorer features: {bad}")
    for cfg in grid:
        if not 1 <= cfg["top_k"] <= confmodel.MAX_TOP_K or not cfg["delta"] > 0:
            raise ValueError(f"grid: top_k must be 1..{confmodel.MAX_TOP_K}, delta > 0: {cfg}")
    recs = [r for r in records if r["candidates"]]
    if lacking := sorted({f for r in recs for c in r["candidates"] for f in feats
                          if f not in c["features"]}):  # fmt: skip
        raise ValueError(
            f"records lack the scorer feature(s) {lacking}: scored before the feature"
            " joined FEATURES; re-run bench syn-score-dev at this commit"
        )
    seeds = sorted({r["seed"] for r in recs})
    if len(seeds) < 3:
        raise ValueError("nested CV needs records of at least three seeds (two inner folds)")
    seed_of = np.array([r["seed"] for r in recs])
    sub = [r.get("subtype") or r["kind"] for r in recs]
    kinds = [r["kind"] for r in recs]
    all_d = _ops(recs, feats)

    # nested: the outer seed takes no part in the selection, the refit or the theta pick
    outer = []
    for sd in seeds:
        te = seed_of == sd
        tr, ts = _subset(all_d, ~te), _subset(all_d, te)
        inner_best, _ = _select(tr, seed_of[~te], [s for s, t in zip(sub, te, strict=True) if not t],
                                [k for k, t in zip(kinds, te, strict=True) if not t], grid, floor,
                                select_floor=select_floor)  # fmt: skip
        cfg = inner_best["config"]
        w, b = _fit_lr(tr, cfg)
        theta, ok = _pick_theta(_prep(_scored(tr, w, b), tr), cfg["delta"], cfg["top_k"], floor)
        hit, prec, ret = _score_cut(_prep(_scored(ts, w, b), ts), theta, cfg["delta"], cfg["top_k"])
        outer.append({"seed": sd, "selected_on": sorted(set(seed_of[~te].tolist())), "chosen": cfg,
                      "inner": {"worst_surfacing": inner_best["worst_surfacing"],
                                "worst_precision": inner_best["worst_precision"]},
                      **_fold(hit, prec, ret, ts, [s for s, t in zip(sub, te, strict=True) if t],
                              [k for k, t in zip(kinds, te, strict=True) if t], theta, ok)})  # fmt: skip
    nested = {"outer_folds": outer, "worst_surfacing": min(f["surfacing"] for f in outer),
              "worst_precision": min(f["precision"] for f in outer),
              "mean_surfacing": float(np.mean([f["surfacing"] for f in outer])),
              "by_subtype_worst": _worst_by_subtype([f["by_subtype"] for f in outer]),
              "paraphrase_worst": _worst_paraphrase([f["paraphrase"] for f in outer]),
              "target": {"surfacing": TARGET_SURFACING, "precision": TARGET_PRECISION}}  # fmt: skip
    feasible = bool(nested["worst_precision"] >= TARGET_PRECISION
                    and nested["worst_surfacing"] >= TARGET_SURFACING)  # fmt: skip

    # the final model: selection on all dev seeds (flat CV), fit and theta on all records
    ri = feats.index("rerank") if "rerank" in feats else None
    baseline = None
    if ri is not None:
        base_cfg = [{"C": None, "label": "rerank", "top_k": confmodel.MAX_TOP_K, "delta": 1e9}]
        baseline, _ = _select(all_d, seed_of, sub, kinds, base_cfg, floor,
                              fit_fn=lambda tr, ts: (tr["x"][..., ri], ts["x"][..., ri]),
                              select_floor=select_floor)  # fmt: skip
    best, results = _select(all_d, seed_of, sub, kinds, grid, floor, select_floor=select_floor)
    cfg = best["config"]
    w, b = _fit_lr(all_d, cfg)
    theta, train_ok = _pick_theta(
        _prep(_scored(all_d, w, b), all_d), cfg["delta"], cfg["top_k"], floor
    )
    model = confmodel.ConfModel.build(feats, [round(float(x), 8) for x in w], round(b, 8),
                                      cfg["top_k"], cfg["delta"])  # fmt: skip
    theta = float(np.floor(theta * 1e6) / 1e6)  # a rounded cut never loses a train candidate
    final = _prep(_scored(all_d, np.array(model.weights), model.bias), all_d)
    hit, prec, ret = _score_cut(final, theta, model.delta, model.top_k)
    upd_sub = [s for s, u in zip(sub, all_d["upd"], strict=True) if u]
    return {
        "theta_conf": theta,
        "conf_model": model.to_dict(),
        "feasible": feasible,
        "train_floor": floor,
        "select_floor": select_floor,
        "grid_size": len(grid),
        "chosen": cfg,
        "ops": len(recs),
        "seeds": seeds,
        "surfacing_recall": float(hit[all_d["upd"]].mean()),
        "list_precision": prec,
        "returned": ret,
        "train_precision_floor_met": train_ok,
        "nested": nested,
        "cv": {
            "chosen": best,
            "baseline_rerank_logit": baseline,
            "grid": results,
            "target": {"surfacing": TARGET_SURFACING, "precision": TARGET_PRECISION},
            "note": "flat leave-one-seed-out CV that picks the final config: optimistic, every"
            " seed took part in the choice; `nested` is the honest estimate",
        },
        "in_sample_by_subtype": _by_subtype(upd_sub, hit[all_d["upd"]]),
        "paraphrase_rows": _paraphrase_rows(kinds, hit),
    }


def _by_subtype(sub: Sequence[str], hit: Any) -> dict[str, dict[str, float]]:
    out: dict[str, list[bool]] = {}
    for s, h in zip(sub, hit, strict=True):
        out.setdefault(s, []).append(bool(h))
    return {s: {"n": len(v), "surfacing": sum(v) / len(v)} for s, v in sorted(out.items())}


def _worst_by_subtype(
    per_fold: Sequence[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Per subtype: the lowest fold surfacing (over the folds where it occurs) and the total n."""
    out: dict[str, dict[str, float]] = {}
    for bs in per_fold:
        for s, v in bs.items():
            cur = out.setdefault(s, {"n": 0, "surfacing": 1.0})
            cur["n"] += v["n"]
            cur["surfacing"] = min(cur["surfacing"], v["surfacing"])
    return dict(sorted(out.items()))


def _paraphrase_rows(kinds: Sequence[str], hit: Any) -> dict[str, float]:
    h = [bool(x) for k, x in zip(kinds, hit, strict=True) if k == "dup_paraphrase"]
    return {"n": len(h), "surfacing": sum(h) / len(h) if h else float("nan")}


def _worst_paraphrase(per_fold: Sequence[dict[str, float]]) -> dict[str, float]:
    rows = [p for p in per_fold if p["n"]]
    return {"n": sum(p["n"] for p in rows),
            "surfacing": min(p["surfacing"] for p in rows) if rows else float("nan")}  # fmt: skip


SUBTYPES = ("normal", "late", "future", "same_instant", "value_returns")


def _cfg_str(cfg: dict[str, Any]) -> str:
    return f"C={cfg['C']:g} {cfg['label']} top{cfg['top_k']} delta={cfg['delta']:g}"


def format_cv(fit: dict[str, Any]) -> str:
    """The nested CV report as text (printed by `bench syn-fit-theta`, written next to the fit):
    the per-outer-fold table (chosen config, theta, surfacing, precision, returned, per subtype,
    paraphrase), the worst fold, the final fit, then the flat CV and the baseline for reference."""
    n, c, base = fit["nested"], fit["cv"]["chosen"], fit["cv"]["baseline_rerank_logit"]
    tgt = n["target"]
    cols = [*SUBTYPES, "paraphrase"]

    def row(label: str, f: dict[str, Any]) -> str:
        bs = {**f["by_subtype"], "paraphrase": f["paraphrase"]}
        cells = "  ".join(f"{bs[s]['surfacing']:13.3f}" if s in bs and bs[s]["n"] else f"{'-':>13}"
                          for s in cols)  # fmt: skip
        return (f"{label:>5}  {f['surfacing']:9.3f}  {f['precision']:9.3f}  {f['returned']:8d}  "
                f"{cells}")  # fmt: skip

    lines = [
        (
            f"conflict scorer nested CV: outer = held-out dev seed {fit['seeds']}, inner = leave-one-"
            f"seed-out on the other {len(fit['seeds']) - 1}; {fit['ops']} ops, train precision floor "
            f"{fit['train_floor']}, selection floor {fit.get('select_floor', TARGET_PRECISION)}, "
            f"grid {fit.get('grid_size', len(GRID))} configs, list cap {max(TOP_KS)}"
        ),
        "outer fold: config chosen on the other seeds only (theta on them too)",
    ]
    lines += [f"  seed {f['seed']}: {_cfg_str(f['chosen'])} theta={f['theta']:.4f} (selected on "
              f"{f['selected_on']}; inner worst {f['inner']['worst_surfacing']:.3f} / "
              f"{f['inner']['worst_precision']:.3f})" for f in n["outer_folds"]]  # fmt: skip
    lines.append(" seed  surfacing  precision  returned  " + "  ".join(f"{s:>13}" for s in cols))
    lines += [row(str(f["seed"]), f) for f in n["outer_folds"]]
    lines.append(row("worst", {"surfacing": n["worst_surfacing"], "precision": n["worst_precision"],
                               "returned": sum(f["returned"] for f in n["outer_folds"]),
                               "by_subtype": n["by_subtype_worst"],
                               "paraphrase": n["paraphrase_worst"]}))  # fmt: skip
    lines.append(
        f"worst outer fold: surfacing {n['worst_surfacing']:.3f}, precision "
        f"{n['worst_precision']:.3f} (target >= {tgt['surfacing']} / >= {tgt['precision']}): "
        + ("PASS" if fit["feasible"] else "MISS")
    )
    lines.append(f"final fit (all dev seeds, selected by flat CV: optimistic): {_cfg_str(fit['chosen'])}, "
                 f"theta_conf {fit['theta_conf']:g}, in-sample surfacing {fit['surfacing_recall']:.3f}, "
                 f"precision {fit['list_precision']:.3f}; feasible {fit['feasible']} (nested)")  # fmt: skip
    lines.append(f"flat CV of the final config (optimistic): worst surfacing {c['worst_surfacing']:.3f}, "
                 f"worst precision {c['worst_precision']:.3f}")  # fmt: skip
    if base:
        lines.append(
            f"baseline raw rerank logit, top 5, no delta (flat CV): worst surfacing "
            f"{base['worst_surfacing']:.3f}, worst precision {base['worst_precision']:.3f}"
        )
    return "\n".join(lines)


# ---- the fit sweep: named variants of the fit rules, each under the same nested CV ---------------


def make_grid(
    cs: Sequence[float] = CS, labels: Sequence[str] = LABELS, top_ks: Sequence[int] = TOP_KS,
    deltas: Sequence[float] = DELTAS,
) -> list[dict[str, Any]]:  # fmt: skip
    """The (C, label, top_k, delta) grid `fit_conf_model` searches; the default is `GRID`."""
    return [{"C": float(c), "label": lab, "top_k": int(k), "delta": float(dl)}
            for lab in labels for c in cs for k in top_ks for dl in deltas]  # fmt: skip


# Pre-registered order: the first feasible variant is the one to gate with; the rows after `base`
# are diagnostics. Picking a later variant because its outer folds pass is a choice made on the
# outer folds (optimistic), to be said so and confirmed by the live tune gate.
VARIANTS: dict[str, dict[str, Any]] = {
    "base": {"note": "the rules (train floor .52, selection floor .52, full grid)"},
    "select50": {"select_floor": TARGET_PRECISION,
                 "note": "pre-MEM2-156 selection at the bare target (shows what the margin changes)"},
    "floor55": {"floor": 0.55, "note": "train and selection floor .55"},
    "k3": {"top_ks": (3,), "note": "top_k 3 only"},
    "d34": {"deltas": (3.0, 4.0), "note": "delta grid {3, 4}"},
    "k3d34": {"top_ks": (3,), "deltas": (3.0, 4.0), "note": "top_k 3 and delta grid {3, 4}"},
}  # fmt: skip


def sweep(
    records: Sequence[dict[str, Any]], names: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """`fit_conf_model` once per named variant (`VARIANTS`, in that order): rows of
    {name, variant, fit}. The target is the same for every row; only the fit's rules vary."""
    rows = []
    for name in names or list(VARIANTS):
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
        v = VARIANTS[name]
        grid = make_grid(top_ks=v.get("top_ks", TOP_KS), deltas=v.get("deltas", DELTAS),
                         cs=v.get("cs", CS))  # fmt: skip
        fit = fit_conf_model(records, features, floor=v.get("floor", TRAIN_FLOOR), grid=grid,
                             select_floor=v.get("select_floor"))  # fmt: skip
        rows.append({"name": name, "variant": v, "fit": fit})
    return rows


def format_sweep(rows: Sequence[dict[str, Any]]) -> str:
    """One line per variant: nested worst outer fold, feasible, the final config and theta, and the
    configs the outer folds chose (how stable the selection is)."""
    head = (
        "fit sweep: every row is the same nested CV on the same records; only the fit rules differ\n"
        "the first feasible row in this order is the one to gate with; a later row chosen because its"
        " outer folds pass is a choice on the outer folds (optimistic): say so, confirm live\n"
        f"{'variant':>8}  {'surf':>6}  {'prec':>6}  feasible  final config / theta_conf  | outer"
        " folds' configs"
    )
    lines = [head]
    for r in rows:
        f, n = r["fit"], r["fit"]["nested"]
        outer = ", ".join(f"top{o['chosen']['top_k']} d{o['chosen']['delta']:g} C{o['chosen']['C']:g}"
                          for o in n["outer_folds"])  # fmt: skip
        lines.append(f"{r['name']:>8}  {n['worst_surfacing']:6.3f}  {n['worst_precision']:6.3f}  "
                     f"{'PASS' if f['feasible'] else 'MISS':>8}  {_cfg_str(f['chosen'])} / "
                     f"{f['theta_conf']:g}  | {outer}")  # fmt: skip
        lines.append(f"{'':>8}  {r['variant']['note']}")
    return "\n".join(lines)
