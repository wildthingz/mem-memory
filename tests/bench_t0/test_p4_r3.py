"""B-R3 scoring (hand-worked values), the baselines, the DR-4 rule, and the suite on a frozen PG."""

import json
import shutil

import pytest

from bench import baselines_p4, pg_eval
from bench.gold import pg
from bench.lme_eval import Retrieved
from bench.metrics import ir
from bench.suites import b_p4
from bench.suites.base import BackendUnavailable, Context
from tests.bench_t0 import pg_world, vault_world


def item(qid, cat, gold, R, S, legs=None, nonrel=()):
    q = pg.Question(qid, cat, "q", answer="" if cat in pg.NO_ANSWER else "a", gold=gold,
                    nonrel=list(nonrel))  # fmt: skip
    return pg_eval.PgItem(q, Retrieved(R, R, S, "text " * 5, 10.0, [], legs or {}))


def tiny():
    return [
        item(
            "q1",
            "single-fact",
            {"a.md::k": 2, "b.md::k": 1},
            ["b.md::k", "x.md::k", "a.md::k"],
            ["a.md::k", "b.md::k"],
        ),
        item("q2", "single-fact", {"c.md::k": 2}, ["y.md::k", "z.md::k"], []),
        item("q3", "no-answer-near-miss", {}, ["y.md::k"], ["y.md::k"]),
        item("q4", "no-answer-off-topic", {}, [], []),
    ]


def test_ndcg_std_graded_by_hand_and_upstream_is_not_computed_for_pg():
    # ranking [rel 1, 0, rel 2] against gold {2, 1}: DCG = 1/1 + 0 + 3/2 = 2.5, IDCG = 3 + 1/log2(3)
    assert ir.ndcg_std(["b", "x", "a"], {"a": 2, "b": 1}, 10) == pytest.approx(
        2.5 / 3.6309297535714573
    )
    sc = pg_eval.score_pg(tiny(), b=100)
    assert sc.metrics["ndcg_std10_R"] == pytest.approx((2.5 / 3.6309297535714573 + 0) / 2)
    assert not any("upstream" in k or "ndcg_any" in k for k in sc.metrics)  # A6: kept separate


def test_delivered_metrics_by_hand():
    m = pg_eval.score_pg(tiny(), b=100).metrics
    assert m["coverage_any"] == 0.5 and m["coverage_all"] == 0.5
    assert m["delivered_precision"] == pytest.approx(
        2 / 3
    )  # 2 relevant of 3 rows, no-answer row counts
    assert m["ira"] == 0.5  # (0 + 1 irrelevant) over the 2 queries that returned rows
    assert m["far"] == 0.5 and m["abstention_recall"] == 0.5
    assert m["delivered_f05"] == pytest.approx(1.25 * (2 / 3) * 0.5 / (0.25 * (2 / 3) + 0.5))
    assert m["recall_any5_R"] == 0.5 and m["n_answerable"] == 2


def test_coverage_all_needs_every_rel2_chunk_and_supports_do_not_count():
    it = item(
        "q", "multi-note", {"a.md::k": 2, "b.md::k": 2, "c.md::k": 1}, [], ["a.md::k", "c.md::k"]
    )
    m = pg_eval.score_pg([it], b=50).metrics
    assert m["coverage_any"] == 1.0 and m["coverage_all"] == 0.0


def test_per_category_macro_and_ci_are_reported_with_micro():
    sc = pg_eval.score_pg(tiny(), b=200)
    assert set(sc.per_category["ndcg_std10_R"]) == {"single-fact"}
    assert sc.per_category["n"] == {"no-answer-near-miss": 1.0, "no-answer-off-topic": 1.0,
                                    "single-fact": 2.0}  # fmt: skip
    lo, hi = sc.ci95["ndcg_std10_R"]
    assert lo <= sc.metrics["ndcg_std10_R"] <= hi and sc.metrics["ndcg_std10_R_macro"] > 0


def test_empty_answerable_set_produces_no_metrics_and_never_zero():
    sc = pg_eval.score_pg([item("n", "no-answer-off-topic", {}, [], [])], b=10)
    assert sc.metrics == {} and "no answerable questions" in sc.notes


def test_path_level_ranking_dedupes_chunks_of_one_note_and_takes_best_rel():
    assert pg_eval.dedupe_paths(["a.md::1", "a.md::2", "b.md::1"]) == ["a.md", "b.md"]
    assert pg_eval.path_gold({"a.md::1": 1, "a.md::2": 2, "b.md::1": 1}) == {"a.md": 2, "b.md": 1}
    # gold {a.md: 2} ranked second: DCG = 3 / log2(3), IDCG = 3, so 1 / log2(3) = 0.63093
    assert pg_eval.ndcg_path(["b.md", "a.md"], {"a.md::2": 2}) == pytest.approx(0.6309297535714574)


def test_a_baseline_missing_questions_fails_loudly_instead_of_scoring_zero():
    items = tiny()[:2]  # q1, q2 answerable
    with pytest.raises(ValueError, match=r"baseline 'qmd' has no run for 1 of 2.*q2"):
        pg_eval.score_pg(items, baselines={"qmd": {"q1": ["a.md"]}}, b=20)
    ok = {"qmd": {"q1": ["a.md"], "q2": []}}  # an empty ranking is a real miss, not a gap
    assert "ndcg_std10_path_qmd" in pg_eval.score_pg(items, baselines=ok, b=20).metrics


def test_dr4_rule_reopens_when_qmd_point_estimate_is_at_least_ours():
    items = tiny()[:2]
    qmd_better = {"qmd": {"q1": ["a.md", "b.md"], "q2": ["c.md"]}}  # perfect on both
    m = pg_eval.score_pg(items, baselines=qmd_better, b=100).metrics
    assert m["dr4_reopen"] == 1.0 and m["ndcg_std10_path_qmd"] > m["ndcg_std10_path_R"]
    qmd_worse = {"qmd": {"q1": ["x.md"], "q2": []}}
    m = pg_eval.score_pg(items, baselines=qmd_worse, b=100).metrics
    assert m["dr4_reopen"] == 0.0 and m["dr4_delta_ours_minus_qmd"] > 0
    tie = {
        "qmd": {"q1": ["b.md", "x.md", "a.md"], "q2": ["y.md", "z.md"]}
    }  # same path ranking as ours
    assert pg_eval.score_pg(items, baselines=tie, b=100).metrics["dr4_reopen"] == 1.0  # >= reopens


def test_unjudged_share_of_top10_names_the_pooling_bias():
    its = [item("q1", "single-fact", {"a.md::k": 2}, ["a.md::k", "j.md::k", "u.md::k", "v.md::k"],
                [], nonrel=["j.md::k"])]  # fmt: skip
    m = pg_eval.score_pg(its, baselines={"qmd": {"q1": ["a.md", "u.md"]}}, b=20).metrics
    assert m["unjudged_at10_R"] == 0.5  # u and v were never judged
    assert m["unjudged_at10_path_qmd"] == 0.5  # u.md has no judged chunk


# ---- baselines -----------------------------------------------------------------------------------


def test_qmd_commands_and_json_parsing(tmp_path):
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[-1] == "--version":
            return "qmd 1.2.3\n"
        if "query" in argv:
            return json.dumps(
                [
                    {"docid": "#1", "score": 0.9, "file": "qmd://vault/work/a.md"},
                    {"docid": "#2", "score": 0.5, "file": "qmd://vault/work/a.md"},
                    {"docid": "#3", "score": 0.4, "file": "qmd://vault/personal/b c.md"},
                ]
            )
        return ""

    qs = [pg.Question("q1", "single-fact", "what is the plan?")]
    rec = baselines_p4.qmd_run(qs, tmp_path, run)
    assert rec["runs"] == {
        "q1": ["work/a.md", "personal/b c.md"]
    }  # chunk repeats collapse to a file
    assert rec["version"] == "qmd 1.2.3" and rec["index"] == "pgv1"
    assert [
        "qmd",
        "--index",
        "pgv1",
        "collection",
        "add",
        str(tmp_path),
        "--name",
        "vault",
        "--mask",
        "**/*.md",
    ] in calls
    assert ["qmd", "--index", "pgv1", "embed"] in calls
    assert ["qmd", "--index", "pgv1", "query", "--json", "-n", "10", "what is the plan?"] in calls
    assert baselines_p4.qmd_path("qmd://vault/x/y.md") == "x/y.md"


def test_rg_floor_ranks_files_by_distinct_query_tokens(tmp_path):
    for name, text in {"a.md": "harbor quartz", "b.md": "harbor", "c.md": "nothing"}.items():
        (tmp_path / name).write_text(text)

    def run(argv):
        if argv[0] == "rg" and "--version" in argv:
            return "ripgrep 14.1.0\n"
        tok = argv[argv.index("-e") + 1]
        return "".join(
            f"{tmp_path / n}\n"
            for n in ("a.md", "b.md", "c.md")
            if tok in (tmp_path / n).read_text()
        )  # what `rg -ilw` would list

    rec = baselines_p4.rg_run(
        [pg.Question("q", "single-fact", "Where is the harbor quartz?")], tmp_path, run
    )
    assert rec["runs"]["q"] == ["a.md", "b.md"]  # a.md has two tokens, b.md one, c.md none


# ---- the suite on a frozen PG --------------------------------------------------------------------


@pytest.fixture
def frozen(tmp_path):
    be, vault = pg_world.imported_world(tmp_path / "w")
    qs = pg_world.build_questions(be)
    out = tmp_path / "gold"
    pg.freeze(qs, be.index, out, snapshot_sha256="22" * 32, snapshot_name="v.tar.zst",
              snapshot_date=pg_world.DATE, routes=vault_world.ROUTES)  # fmt: skip
    be.close()
    _, _, pg_root = pg_world.pg_snapshot(vault, tmp_path)  # B-R3 runs on the PG snapshot
    return pg.load(out), pg_root


def setup_for(tmp_path, gold, vault, **kw):
    made = []

    def new_backend(name):
        b = pg_world.backend(tmp_path / "stores" / name)
        made.append(b)
        return b

    s = b_p4.P4Setup(new_backend=new_backend, pg_vault=vault, gold=gold, split="tune",
                     snapshot_routes=vault_world.ROUTES, bootstrap_b=100, **kw)  # fmt: skip
    return s, made


def ctx(tmp_path, s):
    return Context(tmp_path / "c", lambda *a, **k: None, None, {b_p4.OPT: s})


def test_b_r3_runs_the_tune_split_per_store_and_writes_every_item(tmp_path, frozen):
    gold, vault = frozen
    s, made = setup_for(tmp_path, gold, vault)
    try:
        res = b_p4.run_b_r3(ctx(tmp_path, s))
        assert res.metrics["stale_gold"] == 0 and res.violations == []
        assert res.extra["n"] == {
            "answerable": 30,
            "no_answer": 10,
            "all": 40,
        }  # tune: 40 / 30 + 10
        assert len(res.items) == 40 and all(
            "R" in i and "S" in i and "gold" in i for i in res.items
        )
        for k in (
            "ndcg_std10_R",
            "recall_any5_R",
            "coverage_any",
            "coverage_all",
            "delivered_precision",
            "ira",
            "far",
            "abstention_recall",
            "delivered_f05",
            "tokens_est_mean",
        ):
            assert k in res.metrics
        # PG-v1 holds no work-store content: nothing gold or retrieved is a work-routed note
        paths = {g.split("::")[0] for i in res.items for g in [*i["R"], *i["gold"]]}
        assert paths and not any(p.startswith(("work/", "projects/client-x/")) for p in paths)
        assert res.extra["dataset"]["name"] == "PG-v1" and res.extra["split"] == "tune"
    finally:
        s.close()
        for b in made:
            b.close()


def test_b_r3_baselines_flow_into_dr4_and_missing_gold_is_unavailable(tmp_path, frozen):
    gold, vault = frozen
    qmd = tmp_path / "qmd.json"
    qmd.write_text(
        json.dumps({"tool": "qmd", "version": "9", "runs": {q.qid: [] for q in gold.questions}})
    )
    s, made = setup_for(tmp_path, gold, vault, baselines={"qmd": qmd})
    try:
        res = b_p4.run_b_r3(ctx(tmp_path, s))
        assert res.metrics["dr4_reopen"] == 0.0 and res.extra["baselines"]["qmd"]["version"] == "9"
        s.gold = None
        with pytest.raises(BackendUnavailable):
            b_p4.run_b_r3(ctx(tmp_path, s))
        s.gold, s.pg_vault = (
            gold,
            None,
        )  # no PG snapshot: B-R3 never falls back to the fixture or the full vault
        with pytest.raises(BackendUnavailable):
            b_p4.run_b_r3(ctx(tmp_path, s))
    finally:
        s.close()
        for b in made:
            b.close()


def test_b_r3_stale_gold_fails_the_run(tmp_path, frozen):
    gold, vault = frozen
    changed = tmp_path / "changed"
    shutil.copytree(vault, changed)
    path = next(p for p in sorted(changed.rglob("*.md")) if "Facts" in p.read_text())
    path.write_text(path.read_text().replace("budget of", "budget amounting to", 1))
    s, made = setup_for(tmp_path, gold, changed)
    try:
        res = b_p4.run_b_r3(ctx(tmp_path, s))
        assert res.metrics["stale_gold"] >= 1 and res.violations
        assert "ndcg_std10_R" not in res.metrics  # no scores from a stale gold
    finally:
        s.close()
        for b in made:
            b.close()


def test_b_r3_without_a_gate_is_unavailable_never_a_pass(tmp_path, frozen):
    gold, vault = frozen

    def nogate(name):
        return pg_world.backend(tmp_path / "ng" / name, gate=False)

    s = b_p4.P4Setup(new_backend=nogate, pg_vault=vault, gold=gold, split="tune",
                     snapshot_routes=vault_world.ROUTES)  # fmt: skip
    try:
        with pytest.raises(BackendUnavailable, match="no gate"):
            b_p4.run_b_r3(ctx(tmp_path, s))
    finally:
        s.close()


def test_calibration_pool_gives_logits_with_labels_for_the_tune_split(tmp_path, frozen):
    from mem import calibrate

    gold, vault = frozen
    s, made = setup_for(tmp_path, gold, vault)
    try:
        be = s.imported("pg")
        pools = []
        for q in gold.of_split("tune"):
            pools.append(calibrate.QueryPool(
                q.qid, q.answerable,
                [calibrate.PoolItem(g, z, q.gold.get(g, 0)) for g, z in be.calibration_pool(q)],
                q.category))  # fmt: skip
        rec = tmp_path / "pg_tune.jsonl"
        calibrate.write_records(rec, pools, fp_all=be.cfg.fp_all())
        fp, back = calibrate.read_records(rec)
        assert fp == be.cfg.fp_all() and len(back) == 40
        assert (
            sum(any(i.rel for i in p.items) for p in back) >= 25
        )  # the gold chunks are in the pools
        gate = calibrate.fit_gate(back, fit_id="pg-tune-t0", fingerprint_hash=fp,
                                  limit=be.cfg.retrieval["limit"], fitted_at="2026-10-24T00:00:00Z")  # fmt: skip
        assert gate.fit_id == "pg-tune-t0"
    finally:
        s.close()
        for b in made:
            b.close()
