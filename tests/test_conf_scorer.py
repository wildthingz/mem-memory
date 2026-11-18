"""Conflict scorer v2 (architecture-v2 §3.1 A6, MEM2-146): coefficients in the gate, no fallback."""

import json
import math
from dataclasses import asdict

import numpy as np
import pytest

import bench.cli  # noqa: F401  (import order: suites before synth)
from bench.synth import calibration
from mem import confmodel, read
from mem.confmodel import FEATURES, ConfModel, list_cut
from mem.gate import Gate, gate_json, read_gate, write_gate
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker, stub_conf_model

# SYN-specific vocabulary a feature must never encode (rule 4: no cue lists, templates, ids, values)
SYN_WORDS = ("slot", "relation", "rel_", "subtype", "cue", "template", "value_returns", "hobby",
             "role", "food", "late", "future", "paraphrase", "counterpart", "gold")  # fmt: skip


def test_feature_list_has_no_syn_specific_feature():
    assert FEATURES and not [f for f in FEATURES if any(w in f for w in SYN_WORDS)]
    assert not [f for f, d in FEATURES.items() if any(w in d.lower() for w in ("syn", "gold"))]


def test_gate_without_coefficients_is_refused_not_scored_by_rerank():
    with pytest.raises(ValueError, match="conf_model"):
        Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.5)
    raw = json.loads(gate_json(Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.5,
                                    conf_model=stub_conf_model())))  # fmt: skip
    del raw["conf_model"]
    with pytest.raises(ValueError, match="no fallback"):
        Gate(**raw)
    Gate(1.0, 0.0, 0.5, 0.5, 0.0)  # an uncalibrated write path (theta None) is a different state


def test_tampered_coefficients_are_refused():
    m = stub_conf_model()
    with pytest.raises(ValueError, match="fingerprint"):
        Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.5, conf_model={**m, "weights": [2.0]})
    with pytest.raises(ValueError, match="unknown"):
        ConfModel.build(["rerank", "syn_relation"], [1.0, 1.0], 0.0, 5, 3.0)
    with pytest.raises(ValueError, match="top_k"):
        ConfModel.build(["rerank"], [1.0], 0.0, 6, 3.0)
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="delta"):
            ConfModel.build(["rerank"], [1.0], 0.0, 5, bad)


# ---- the adaptive list cut (MEM2-154 FIX B) ----------------------------------------------------


def test_list_cut_applies_theta_then_delta_then_top_k_with_deterministic_ties():
    scores = [1.0, 4.0, 3.5, 4.0, 0.5, 2.5, 3.9]
    # theta alone (delta wide, k wide): everything >= 2.0, best first, ties by index
    assert list_cut(scores, 2.0, 100.0, 5, None) == [1, 3, 6, 2, 5]
    # delta: keep only within 1.0 of the top (4.0): 4.0, 4.0, 3.9, 3.5 (2.5 is out)
    assert list_cut(scores, 2.0, 1.0, 5, None) == [1, 3, 6, 2]
    # top_k after the two cuts
    assert list_cut(scores, 2.0, 1.0, 3, None) == [1, 3, 6]
    # the tie between the two 4.0 is broken by the caller's keys (node ids), not by position
    assert list_cut(scores, 2.0, 1.0, 3, [70, 60, 50, 40, 30, 20, 10]) == [3, 1, 6]
    # theta above the top: empty; the delta is measured from the list's top, not from theta
    assert list_cut(scores, 4.5, 1.0, 5, None) == []
    assert list_cut(scores, 3.95, 1.0, 5, None) == [1, 3]
    assert list_cut([], 0.0, 1.0, 5, None) == []
    # top_k never exceeds the hard cap of 5, whatever the caller asks
    assert len(list_cut([float(i) for i in range(9)], -1.0, 100.0, 9, None)) == 5
    with pytest.raises(ValueError, match="delta"):
        list_cut(scores, 0.0, 0.0, 3, None)
    with pytest.raises(ValueError, match="top_k"):
        list_cut(scores, 0.0, 1.0, 0, None)


def test_gate_json_round_trips_delta_and_top_k_and_is_refused_without_them():
    cm = ConfModel.build(["rerank", "cosine"], [1.0, 2.0], -0.5, 3, 2.5)
    d = cm.to_dict()
    assert d["top_k"] == 3 and d["delta"] == 2.5 and d["version"] == 2
    gate = Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.25, conf_model=d)
    back = Gate(**json.loads(gate_json(gate)))
    assert back.model == cm and back.model.delta == 2.5 and back.model.top_k == 3
    for key in ("delta", "top_k"):
        raw = json.loads(gate_json(gate))
        del raw["conf_model"][key]
        with pytest.raises(ValueError, match=key):
            Gate(**raw)
        raw = json.loads(gate_json(gate))
        raw["conf_model"][key] = None
        with pytest.raises(ValueError, match=key):
            Gate(**raw)
    # a v1 record (no delta) is refused even when it names a fingerprint
    raw = json.loads(gate_json(gate))
    del raw["conf_model"]["delta"]
    raw["conf_model"]["version"] = 1
    with pytest.raises(ValueError):
        Gate(**raw)
    # delta is fingerprinted: changing it alone is tampering
    with pytest.raises(ValueError, match="fingerprint"):
        Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.25, conf_model={**d, "delta": 3.0})
    # the stub model used by the T0 suites carries both
    assert {"top_k", "delta"} <= set(stub_conf_model())


def test_find_applies_the_delta_cut_like_list_cut(tmp_path, clock, monkeypatch):
    """The store's list cut is `confmodel.list_cut` on the model scores: with the rerank-only model
    the candidate list keeps only those within `delta` of the best logit, then `top_k`."""
    logits = [5.0, 2.0, 4.6, 4.2, 1.0, 4.9, 3.0, 0.0]

    def fake_rerank(store, text, chunk_ids):
        return [logits[i] for i in range(len(chunk_ids))]

    monkeypatch.setattr(read, "rerank", fake_rerank)
    gate = Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=-10.0, conf_model=stub_conf_model(5, 1.0))
    s = Store.open(tmp_path / "d.db", HashEmbedder(), now=clock, reranker=OverlapReranker())
    write_gate(s.conn, gate)
    s.gate = read_gate(s.conn)
    for i in range(8):
        s.remember(text=f"Ana desk is on floor {i}.", type="fact",
                   about=["new:person:Ana"] if i == 0 else ["Ana"])  # fmt: skip
    out = s.remember(text="Ana desk is on floor 9.", type="fact", about=["Ana"]).structured
    s.close()
    got = [c["score"] for c in out["possible_conflicts"]]
    assert got == [5.0, 4.9, 4.6, 4.2]  # 3.0 and below are more than delta=1.0 under the top
    keep = list_cut(logits, -10.0, 1.0, 5, None)
    assert [logits[i] for i in keep] == got


def test_bench_cut_matches_the_store_cut_on_random_lists():
    """`bench syn-fit-theta` evaluates lists with a vectorised cut; it must be the one `find` applies."""
    rng = np.random.default_rng(3)
    recs = []
    for op in range(60):
        n = int(rng.integers(1, 9))
        pool = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.5, 3.0]  # repeated values: ties happen
        cands = [{"score": float(rng.choice(pool)), "same_slot": bool(rng.random() < 0.4),
                  "counterpart": j == 0, "features": {f: float(rng.normal()) for f in FEATURES}}
                 for j in range(n)]  # fmt: skip
        recs.append({"seed": 1, "op": op, "kind": "update", "subtype": "normal",
                     "candidates": cands})  # fmt: skip
    d = calibration._ops(recs, list(FEATURES))
    score = np.array([[c["score"] for c in r["candidates"]] + [0.0] * (8 - len(r["candidates"]))
                      for r in recs])  # fmt: skip
    for theta, delta, k in [
        (-5.0, 100.0, 5),
        (0.0, 1.0, 3),
        (0.5, 2.0, 5),
        (1.0, 0.5, 1),
        (2.6, 1.0, 5),
    ]:
        prep = calibration._prep(score, d)
        hit, ss, ret = calibration._cut(prep, theta, delta, k)
        exp_hit, exp_ss, exp_ret = [], 0, 0
        for r in recs:
            keep = list_cut([c["score"] for c in r["candidates"]], theta, delta, k, None)
            exp_hit.append(any(r["candidates"][i]["counterpart"] for i in keep))
            exp_ss += sum(r["candidates"][i]["same_slot"] for i in keep)
            exp_ret += len(keep)
        assert (hit.tolist(), ss, ret) == (exp_hit, exp_ss, exp_ret), (theta, delta, k)


def test_scorer_is_deterministic_and_loads_from_the_gate_json(tmp_path, clock):
    names = ["rerank", "cosine", "open"]
    cm = ConfModel.build(names, [0.5, 3.0, 1.0], -2.0, 3, 50.0).to_dict()
    gate = Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.25, conf_model=cm, fingerprint_hash="f")
    s = Store.open(tmp_path / "s.db", HashEmbedder(), now=clock, reranker=OverlapReranker())
    write_gate(s.conn, gate)
    s.gate = read_gate(s.conn)
    assert s.gate == gate and asdict(s.gate)["conf_model"]["fingerprint"] == cm["fingerprint"]
    x = {"rerank": 2.0, "cosine": 0.5, "open": 1.0}
    assert s.gate.model.score(x) == pytest.approx(0.5 * 2 + 3 * 0.5 + 1 - 2)
    assert [s.gate.model.score(x) for _ in range(3)] == [s.gate.model.score(x)] * 3
    for i in range(3):
        s.remember(text=f"Ana desk is on floor {i}.", type="fact",
                   about=["new:person:Ana"] if i == 0 else ["Ana"])  # fmt: skip
    r1 = s.remember(text="Ana desk is on floor 9.", type="fact", about=["Ana"]).structured
    s.close()
    assert len(r1["possible_conflicts"]) <= 3  # top_k from the coefficients


def test_unfiltered_dev_run_returns_every_pick_with_features(tmp_path, clock):
    s = Store.open(tmp_path / "u.db", HashEmbedder(), now=clock, reranker=OverlapReranker())
    s.gate = Gate(1.0, 0.0, 0.0, 0.0, 0.0, theta_conf=-math.inf)
    for i in range(8):
        s.remember(text=f"Ana desk is on floor {i}.", type="fact",
                   about=["new:person:Ana"] if i == 0 else ["Ana"])  # fmt: skip
    out = s.remember(text="Ana desk is on floor 9.", type="fact", about=["Ana"]).structured
    s.close()
    assert len(out["possible_conflicts"]) == 8  # all picks, past MAX_OUT
    assert all(set(c["features"]) == set(FEATURES) for c in out["possible_conflicts"])
    assert all(0.0 <= c["features"]["cosine"] <= 1.0 + 1e-9 for c in out["possible_conflicts"])


# ---- the fit -------------------------------------------------------------------------------


def test_fit_refuses_records_that_lack_a_scorer_feature():
    """Records scored before a feature joined `FEATURES` (MEM2-156 round 5 added `same_type`) are
    refused by name, with the re-score command, not with a KeyError deep in the arrays."""
    recs = _records()
    for r in recs:
        for c in r["candidates"]:
            del c["features"]["same_type"]
    with pytest.raises(ValueError, match=r"same_type.*syn-score-dev"):
        calibration.fit_conf_model(recs)


def _records(seed_n=5, ops=40, rng_seed=0):
    """Synthetic lists where rerank alone is a weak signal and cosine/jaccard carry the rest."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for seed in range(1, seed_n + 1):
        for op in range(ops):
            cands = []
            for j in range(8):
                cp = j == 0
                f = {n: float(rng.normal()) for n in FEATURES}
                f["rerank"] = float(rng.normal(1.0 if cp else 0.6, 2.0))
                f["cosine"] = float(rng.normal(0.9 if cp else 0.5, 0.05))
                f["open"] = float(rng.random() < 0.8)
                cands.append({"id": f"m{j}", "score": f["rerank"], "same_slot": cp or j == 1,
                              "counterpart": cp, "features": f})  # fmt: skip
            out.append({"seed": seed, "op": op, "kind": "update", "subtype": "normal",
                        "counterpart_returned": True, "candidates": cands})  # fmt: skip
    return out


def test_fit_is_reproducible_and_beats_the_raw_logit():
    recs = _records()
    a, b = calibration.fit_conf_model(recs), calibration.fit_conf_model(recs)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    cv = a["cv"]
    assert cv["chosen"]["worst_surfacing"] > cv["baseline_rerank_logit"]["worst_surfacing"]
    assert len(cv["chosen"]["folds"]) == 5 and a["chosen"]["top_k"] in (3, 5)
    assert a["chosen"]["delta"] in calibration.DELTAS and a["chosen"]["C"] in calibration.CS
    assert {1.0, 10.0} <= set(calibration.CS) and len(calibration.CS) >= 3
    assert a["train_floor"] == 0.52 and a["feasible"]
    m = ConfModel.from_dict(a["conf_model"])  # fingerprinted, loadable
    assert m.top_k == a["chosen"]["top_k"] and m.delta == a["chosen"]["delta"]
    nested = a["nested"]
    assert [f["seed"] for f in nested["outer_folds"]] == [1, 2, 3, 4, 5]
    for f in nested["outer_folds"]:
        assert f["seed"] not in f["selected_on"] and len(f["selected_on"]) == 4
        assert set(f["chosen"]) == {"C", "label", "top_k", "delta"} and "theta" in f
        assert set(f["by_subtype"]) == {"normal"} and f["paraphrase"]["n"] == 0
    assert nested["worst_surfacing"] == min(f["surfacing"] for f in nested["outer_folds"])
    assert nested["worst_precision"] == min(f["precision"] for f in nested["outer_folds"])
    assert "normal" in nested["by_subtype_worst"] and "paraphrase_worst" in nested
    assert a["feasible"] == (nested["worst_surfacing"] >= 0.95 and nested["worst_precision"] >= 0.5)
    text = calibration.format_cv(a)
    assert "nested" in text and "worst outer fold" in text and "value_returns" in text
    assert "optimistic" in text  # the flat CV that picks the final config is labelled as such


def _planted_records():
    """Seeds 1-4: the counterpart (cosine .80) leads two same-slot decoys (.70, .65) and noise (.30).
    Seed 5, the plant: the counterpart (.84) sits behind three same-slot decoys (.95, .93, .91), so
    with theta fitted on seeds 1-4 (just under .80) only `top_k` 5 keeps it; `top_k` 3 misses it."""
    rng = np.random.default_rng(7)
    recs = []
    for seed in range(1, 6):
        for op in range(30):
            if seed < 5:
                cos, same = [0.80, 0.70, 0.65] + [0.30] * 5, [1, 1, 1, 0, 0, 0, 0, 0]
            else:
                cos, same = [0.84, 0.95, 0.93, 0.91] + [0.30] * 4, [1, 1, 1, 1, 0, 0, 0, 0]
            cands = []
            for j in range(8):
                f = {n: float(rng.normal(0, 0.01)) for n in FEATURES}  # every other feature: noise
                f["cosine"] = cos[j] + float(rng.normal(0, 0.005))
                cands.append({"id": f"m{j}", "score": f["cosine"], "same_slot": bool(same[j]),
                              "counterpart": j == 0, "features": f})  # fmt: skip
            recs.append({"seed": seed, "op": op, "kind": "update", "subtype": "normal",
                         "counterpart_returned": True, "candidates": cands})  # fmt: skip
    return recs


def test_nested_cv_never_selects_on_the_outer_seed():
    """On seeds 1-4 every config surfaces at precision 1.0 (theta sits just under the counterpart),
    so the tie-break picks top_k 3. A selection that saw the planted seed 5 would find that only
    top_k 5 surfaces there and pick it for the outer fold 5, reporting surfacing 1.0; the honest one
    picks on seeds 1-4, keeps top_k 3 and reports the miss. The wide config alone shows the leak
    would have been rewarded."""
    recs = _planted_records()
    fit = calibration.fit_conf_model(recs)
    outer = {f["seed"]: f for f in fit["nested"]["outer_folds"]}
    assert all(f["selected_on"] == sorted(set(range(1, 6)) - {s}) for s, f in outer.items())
    assert all(outer[s]["surfacing"] >= 0.9 for s in range(1, 5)), outer
    assert outer[5]["chosen"]["top_k"] == 3 and outer[5]["surfacing"] == 0.0, outer[5]
    assert fit["nested"]["worst_surfacing"] == 0.0 and fit["feasible"] is False
    wide = {"C": 1.0, "label": "counterpart", "top_k": 5, "delta": max(calibration.DELTAS)}
    leaked = calibration.fit_conf_model(recs, grid=[wide])
    assert leaked["nested"]["outer_folds"][4]["surfacing"] == 1.0  # what a leak would have shown
    # the flat CV (final config choice) does see seed 5 and so prefers the wide list: optimistic
    assert fit["chosen"]["top_k"] == 5 and fit["cv"]["chosen"]["worst_surfacing"] > 0.9


def test_fit_reports_misses_instead_of_relaxing_the_target():
    recs = _records()
    rng = np.random.default_rng(1)
    for r in recs:  # no feature separates the counterpart: the target cannot be met
        for c in r["candidates"]:
            c["features"] = {n: float(rng.normal()) for n in FEATURES}
    fit = calibration.fit_conf_model(recs)
    assert fit["feasible"] is False and fit["cv"]["target"]["surfacing"] == 0.95


def test_fit_uses_only_declared_features_and_needs_three_seeds():
    with pytest.raises(ValueError, match="not scorer features"):
        calibration.fit_conf_model(_records(), features=["rerank", "slot_id"])
    with pytest.raises(ValueError, match="three seeds"):
        calibration.fit_conf_model(_records(seed_n=2))
    assert confmodel.MAX_TOP_K == 5


def _margin_records(two_noise: dict[int, int], low_ops: int = 4, ops: int = 100):
    """One score per candidate (the `cosine` feature; `fit_fn` reads it, no logistic fit). Per op:
    the counterpart (same slot) at .80, or at .50 on `low_ops` ops; a same-slot decoy at .70; one
    non-slot noise candidate at .60 on every op and a second at .55 on `two_noise[seed]` ops (the
    low ops always have both); padding at .30. Surfacing the low counterparts needs theta <= .50:
    top 5 then keeps 2 same-slot rows out of 4 (two-noise ops) or 3 (one-noise ops), so list
    precision = 200 / (300 + two_noise) per seed: .5305 at 77, .5102 at 92. Top 3 never keeps a
    low counterpart (the decoy and both noise rows outrank it) and so sits at .96 surfacing."""
    recs = []
    for seed, q in two_noise.items():
        for op in range(ops):
            low = op < low_ops
            second = op < q
            cos = [0.50 if low else 0.80, 0.70, 0.60, 0.55 if second else 0.30, 0.30, 0.30]
            same = [1, 1, 0, 0, 0, 0]
            cands = [{"id": f"m{j}", "score": cos[j], "same_slot": bool(same[j]),
                      "counterpart": j == 0, "features": {n: 0.0 for n in FEATURES} | {"cosine": cos[j]}}
                     for j in range(6)]  # fmt: skip
            recs.append({"seed": seed, "op": op, "kind": "update", "subtype": "normal",
                         "counterpart_returned": True, "candidates": cands})  # fmt: skip
    return recs


def test_selection_demands_the_train_margin_not_the_bare_target():
    """MEM2-156 round 2: outer fold 5's inner search accepted a config at precision .503 (margin
    .003 over the .50 target) because it had the best surfacing; the held-out seed came out at
    .499 and the nested fit was infeasible. Here config A (top 5) surfaces 1.0 at a worst inner
    precision of .510 and config B (top 3) surfaces .96 at 1.0. The bare target picks A; the
    selection must demand the same margin theta is fitted with (the train floor) and pick B."""
    recs = _margin_records({1: 77, 2: 77, 3: 77, 4: 77, 5: 92})
    d = calibration._ops(recs, list(FEATURES))
    seed_of = np.array([r["seed"] for r in recs])
    sub, kinds = ["normal"] * len(recs), ["update"] * len(recs)
    ci = list(FEATURES).index("cosine")
    fit_fn = lambda tr, ts: (tr["x"][..., ci], ts["x"][..., ci])
    grid = [{"C": 1.0, "label": "counterpart", "top_k": 5, "delta": 10.0},
            {"C": 1.0, "label": "counterpart", "top_k": 3, "delta": 10.0}]  # fmt: skip
    old, res = calibration._select(d, seed_of, sub, kinds, grid, 0.52, fit_fn=fit_fn,
                                   select_floor=0.50)  # fmt: skip
    a, b = res
    assert a["worst_surfacing"] == 1.0 and 0.50 <= a["worst_precision"] < 0.52, a
    assert b["worst_surfacing"] == pytest.approx(0.96) and b["worst_precision"] == 1.0, b
    assert old["config"]["top_k"] == 5  # the bare target: fragile, at the edge
    new, _ = calibration._select(d, seed_of, sub, kinds, grid, 0.52, fit_fn=fit_fn)
    assert new["config"]["top_k"] == 3  # default select floor = the train floor
    # the whole fit carries the rule and reports it; the target itself is unchanged
    fit = calibration.fit_conf_model(_records(), select_floor=0.55)
    assert fit["select_floor"] == 0.55 and fit["train_floor"] == 0.52
    assert fit["cv"]["target"] == {"surfacing": 0.95, "precision": 0.5}
    assert calibration.fit_conf_model(_records())["select_floor"] == 0.52
    assert "selection floor" in calibration.format_cv(fit)


def test_sweep_runs_named_variants_with_the_same_nested_cv():
    recs = _records()
    rows = calibration.sweep(recs)
    names = [r["name"] for r in rows]
    assert names[0] == "base" and set(names) >= {"base", "select50", "floor55", "k3", "d34"}
    base = next(r for r in rows if r["name"] == "base")["fit"]
    assert json.dumps(base, sort_keys=True) == json.dumps(calibration.fit_conf_model(recs),
                                                           sort_keys=True)  # fmt: skip
    for r in rows:
        f = r["fit"]
        assert set(r) >= {"name", "fit", "variant"} and f["feasible"] in (True, False)
        assert f["cv"]["target"] == {"surfacing": 0.95, "precision": 0.5}  # never relaxed
    k3 = next(r for r in rows if r["name"] == "k3")["fit"]
    assert all(f["chosen"]["top_k"] == 3 for f in k3["nested"]["outer_folds"])
    d34 = next(r for r in rows if r["name"] == "d34")["fit"]
    assert all(f["chosen"]["delta"] in (3.0, 4.0) for f in d34["nested"]["outer_folds"])
    assert next(r for r in rows if r["name"] == "select50")["fit"]["select_floor"] == 0.50
    assert next(r for r in rows if r["name"] == "floor55")["fit"]["train_floor"] == 0.55
    text = calibration.format_sweep(rows)
    assert "base" in text and "feasible" in text and "optimistic" in text
    with pytest.raises(ValueError, match="unknown variant"):
        calibration.sweep(recs, names=["nope"])
