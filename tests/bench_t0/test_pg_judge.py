"""PG-v1 with Claude as the judge and a blind kappa check (benchmarks-v2 §3.1, spec change
2026-10-24): kappa on hand-computed tables, the seeded blind sample, the < 0.7 path, the judge
column, the pre-filter, the work-free sample and snapshot, and the work-route guard."""

import math

import pytest
import yaml
from typer.testing import CliRunner

from bench import llm
from bench.cli import app
from bench.datasets import vault as vault_ds
from bench.gold import check, judge, kappa, pg, pool, sampling, session
from tests.bench_t0 import pg_world, vault_world


def run(*args, input=None):
    return CliRunner().invoke(app, ["gold", *args], input=input)


# ---- kappa on hand-computed tables --------------------------------------------------------------


def pairs_of(table):
    """(claude, human) grade pairs of a 3x3 count table (rows claude, columns human)."""
    return [(i, j) for i in range(3) for j in range(3) for _ in range(table[i][j])]


def test_kappa_perfect_agreement_is_one_and_chance_agreement_is_zero():
    assert kappa.weighted_kappa(pairs_of([[20, 0, 0], [0, 20, 0], [0, 0, 10]])) == 1.0
    # every cell = row total x column total / n, so agreement is exactly what chance gives
    chance = [[8, 8, 4], [8, 8, 4], [4, 4, 2]]
    assert kappa.weighted_kappa(pairs_of(chance)) == pytest.approx(0.0, abs=1e-12)


def test_kappa_linear_weights_by_hand():
    # rows 25/20/5, columns 25/15/10, n = 50. Observed disagreement = (5+5+5) x 0.5 / 50 = 0.15.
    # Expected = (187.5 + 250 + 100 + 37.5 + 250 + 125) / 2500 = 0.38. kappa = 1 - 0.15/0.38 = 23/38.
    table = [[20, 5, 0], [5, 10, 5], [0, 0, 5]]
    assert kappa.weighted_kappa(pairs_of(table)) == pytest.approx(23 / 38)
    # an off-by-two disagreement costs twice an off-by-one one: swap the (0,1) for a (0,2) cell
    worse = [[20, 0, 5], [5, 10, 5], [0, 0, 5]]
    assert kappa.weighted_kappa(pairs_of(worse)) < kappa.weighted_kappa(pairs_of(table))


def test_kappa_is_undefined_not_a_pass_when_both_raters_use_one_grade_or_n_is_zero():
    assert math.isnan(kappa.weighted_kappa([(0, 0)] * 10))
    assert math.isnan(kappa.weighted_kappa([]))
    assert not kappa.result_of([(0, 0)] * 10).stands


def test_the_threshold_is_inclusive_at_0_7():
    assert kappa.KAPPA_MIN == 0.7
    assert kappa.KappaResult(0.7, 50, 0.9, [[0] * 3] * 3).stands
    assert not kappa.KappaResult(0.6999, 50, 0.9, [[0] * 3] * 3).stands


def test_the_sample_is_deterministic_for_a_seed_and_sorted_input_order_does_not_matter():
    keys = [(f"q{i % 7}", f"p{i}.md::k") for i in range(200)]
    a = kappa.draw_sample(keys, 50, seed=1)
    assert a == kappa.draw_sample(list(reversed(keys)), 50, seed=1)
    assert a != kappa.draw_sample(keys, 50, seed=2)
    assert len(a) == 50 == len(set(a)) and set(a) <= set(keys)
    assert len(kappa.draw_sample(keys[:12], 50, seed=1)) == 12  # fewer pairs than 50: all of them


# ---- the judge ------------------------------------------------------------------------------------


@pytest.fixture
def accepted(tmp_path):
    """A work dir with 120 accepted questions and a pool, over the stand-in world."""
    be, _ = pg_world.imported_world(tmp_path)
    w = session.Work(tmp_path / "work")
    truth = pg_world.build_questions(be)
    drafts = [{"draft_id": f"d{i:03d}", "unit": "u", "slot": 0, "category": q.category,
               "question": q.question, "answer": q.answer, "gold": {}, "filters": {},
               "key_terms": ["zzzz", "yyyy"], "source": q.source, "store": q.store}
              for i, q in enumerate(truth)]  # fmt: skip
    from bench.gold import draft

    draft.write_drafts(w.path("drafts.jsonl"), drafts)
    session.review_sitting(w, be.index, lambda d, k: [], minutes=None, out=lambda s: None,
                           ask=lambda p: "a")  # fmt: skip
    pool.build(w, w.accepted(), be)
    yield be, w, tmp_path
    be.close()


def test_judge_grades_every_pooled_pair_as_claude_and_a_rerun_asks_for_nothing(accepted):
    be, w, _ = accepted
    pairs = {(p["qid"], p["gold_id"]) for p in w.pool()}
    client = pg_world.ScriptedJudge()
    rep = judge.judge_all(
        w, be.index, client, llm.Ledger("claude-opus-5-5", 25.0), out=lambda s: None
    )
    assert rep.judged == len(pairs) == len(client.calls) and rep.remaining == 0
    assert rep.unparsable == 0 and rep.cost["usd"] > 0 and rep.cost["calls"] == len(pairs)
    assert set(w.judgments("claude")) == pairs and not w.judgments("human")
    assert {j for _, j in w.label_state().values()} == {"claude"}
    again = judge.judge_all(
        w, be.index, client, llm.Ledger("claude-opus-5-5", 25.0), out=lambda s: None
    )
    assert again.judged == 0 and len(client.calls) == len(pairs)


def test_judge_counts_unparsable_replies_never_guesses_and_stops_at_the_budget(accepted):
    be, w, _ = accepted
    bad = w.pool()[3]
    chunk = be.index.by_gold[bad["gold_id"]].text
    client = pg_world.ScriptedJudge(replies={chunk[:60]: "I think it is probably relevant"})
    rep = judge.judge_all(
        w, be.index, client, llm.Ledger("claude-opus-5-5", 25.0), out=lambda s: None
    )
    assert rep.unparsable >= 1 and rep.remaining >= 1
    assert (bad["qid"], bad["gold_id"]) not in w.label_state()
    capped = llm.Ledger("claude-opus-5-5", 0.02)
    w2 = session.Work(w.root.parent / "work2")
    w2.root.mkdir()
    for f in ("drafts", "review", "pool"):
        w2.path(f + ".jsonl").write_bytes(w.path(f + ".jsonl").read_bytes())
    rep2 = judge.judge_all(w2, be.index, pg_world.ScriptedJudge(), capped, out=lambda s: None)
    assert rep2.stopped == "budget" and rep2.judged < len(w2.pool())


def test_judge_cli_refuses_a_budget_above_25_and_asks_before_sending_text(accepted, monkeypatch):
    _, w, tmp = accepted
    monkeypatch.setattr(llm, "AnthropicLLM", lambda model, effort: pg_world.ScriptedJudge())
    args = ["--stores-dir", str(tmp / "stores"), "--work", str(w.root)]
    assert run("judge", "--budget-usd", "25.01", "--yes", *args).exit_code != 0
    r = run("judge", "--budget-usd", "5", *args, input="n\n")
    assert r.exit_code == 1 and "sends the text of those chunks" in r.output
    assert not w.judgments("claude")


def test_a_no_answer_question_claude_grades_answerable_is_voided_and_reopens_its_quota(accepted):
    be, w, _ = accepted
    q = next(q for q in w.accepted() if not q.answerable)
    before = session.counts(w)[q.category]
    client = pg_world.ScriptedJudge(rule=lambda prompt: 2 if q.question in prompt else 0)
    rep = judge.judge_all(
        w, be.index, client, llm.Ledger("claude-opus-5-5", 25.0), out=lambda s: None
    )
    assert rep.voided >= 1 and session.counts(w)[q.category] < before
    assert q.qid not in {x.qid for x in w.accepted()}


# ---- the blind sample, the kappa gate and the < 0.7 path ----------------------------------------


def judged(accepted):
    be, w, tmp = accepted
    judge.judge_all(w, be.index, pg_world.ScriptedJudge(), llm.Ledger("claude-opus-5-5", 25.0),
                    out=lambda s: None)  # fmt: skip
    return be, w, tmp


def sit(w, be, answers, **kw):
    it = iter(answers)
    shown: list[str] = []
    res = session.kappa_sitting(w, be.index, n=kappa.SAMPLE_N, seed=kappa.SAMPLE_SEED,
                                ask=lambda p: next(it), out=shown.append, **kw)  # fmt: skip
    return res, "\n".join(shown)


def test_the_sample_is_50_of_claudes_pooled_judgments_seeded_and_saved(accepted):
    _, w, _ = judged(accepted)
    s1 = w.kappa_sample(n=50, seed=kappa.SAMPLE_SEED)
    claude = w.judgments("claude", "pool")
    assert len(s1) == 50 == len(set(s1)) and set(s1) <= set(claude)
    assert s1 == kappa.draw_sample(list(claude), 50, kappa.SAMPLE_SEED)
    assert w.kappa_sample(n=10, seed=1) == s1  # the saved first draw wins: a resume sees the same


def test_the_blind_sitting_never_shows_claudes_label_and_records_yours_as_human(accepted):
    be, w, _ = judged(accepted)
    sample = w.kappa_sample(n=50, seed=kappa.SAMPLE_SEED)
    claude = w.judgments("claude", "pool")
    res, text = sit(w, be, [str(claude[k]) for k in sample])
    assert res.items == 50 and res.remaining == 0
    assert "claude" not in text.lower() and "rel " not in text  # nothing of the judge on screen
    assert all(k[1] in text for k in sample)  # each sampled chunk was shown
    human = w.judgments("human", "kappa")
    assert set(human) == set(sample) and all(human[k] == claude[k] for k in sample)
    result, left = w.kappa_result()
    assert left == 0 and result.n == 50 and result.kappa == 1.0 and result.stands
    # their label wins on those pairs; the rest stay Claude's
    state = w.label_state()
    assert all(state[k][1] == "human" for k in sample)
    assert sum(j == "claude" for k, (_, j) in state.items() if k in claude) == len(claude) - 50


def test_a_blind_sitting_can_stop_and_resume_and_reports_the_missing_ones(accepted):
    be, w, _ = judged(accepted)
    sample = w.kappa_sample(n=50, seed=kappa.SAMPLE_SEED)
    claude = w.judgments("claude", "pool")
    res, _ = sit(w, be, [str(claude[k]) for k in sample[:20]] + ["q"])
    assert res.stopped == "quit" and res.remaining == 30
    left = w.kappa_result()[1]
    assert left == 30 and "still to grade" in session.status_text(w)
    res2, _ = sit(w, be, [str(claude[k]) for k in sample[20:]])
    assert res2.items == 30 and w.kappa_result()[1] == 0


def disagree(claude, sample):
    """the human swaps Claude's 1 and 2 and agrees on every 0 (so no no-answer question is voided)."""
    return [str({0: 0, 1: 2, 2: 1}[claude[k]]) for k in sample]


def test_below_0_7_the_tool_says_so_and_label_hands_the_rest_to_human(accepted, monkeypatch):
    be, w, tmp = judged(accepted)
    sample = w.kappa_sample(n=50, seed=kappa.SAMPLE_SEED)
    claude = w.judgments("claude", "pool")
    args = ["--stores-dir", str(tmp / "stores"), "--work", str(w.root)]
    r = run("label", *args)  # before the check there is nothing to decide
    assert r.exit_code == 1 and "kappa-sample" in r.output
    r = run("kappa-sample", *args, input="".join(g + "\n" for g in disagree(claude, sample)))
    assert r.exit_code == 0 and "DO NOT stand" in r.output and "n = 50" in r.output
    result, _ = w.kappa_result()
    assert not result.stands and result.kappa < 0.7
    n_pool = len(w.pool())
    assert f"the other {n_pool - 50} pooled pair" in r.output  # their 50 count as done
    asked: list[str] = []
    answers = iter("0\n" * 1000)
    monkeypatch.setattr(session, "input", lambda prompt="": next(answers), raising=False)
    res = session.label_sitting(w, be.index, minutes=None, relabel_claude=True,
                                ask=lambda p: next(answers).strip(), out=asked.append)  # fmt: skip
    assert res.items == n_pool - 50 and res.remaining == 0  # none of their 50 asked again
    state = w.label_state()
    assert {j for _, j in state.values()} == {"human"}  # every pooled label is theirs now
    shown = "\n".join(x for x in asked if "by judge" not in x and "kappa" not in x)
    assert "claude" not in shown.lower()  # their own labelling never shows Claude's grade


def test_label_with_a_passing_kappa_has_nothing_to_do(accepted):
    be, w, tmp = judged(accepted)
    sample = w.kappa_sample(n=50, seed=kappa.SAMPLE_SEED)
    claude = w.judgments("claude", "pool")
    sit(w, be, [str(claude[k]) for k in sample])
    r = run("label", "--stores-dir", str(tmp / "stores"), "--work", str(w.root))
    assert r.exit_code == 0 and "nothing for you to label" in r.output


def test_human_label_wins_over_claude_whatever_the_order_of_the_rows(tmp_path):
    w = session.Work(tmp_path / "w")
    key = {"qid": "q", "gold_id": "a.md::k"}
    w.log_event("labels.jsonl", {**key, "rel": 2, "source": "kappa", "judge": "human"})
    w.log_event("labels.jsonl", {**key, "rel": 0, "source": "pool", "judge": "claude"})  # later row
    assert w.label_state()[("q", "a.md::k")] == (2, "human")
    w.log_event("labels.jsonl", {**key, "rel": -1, "source": "undo", "judge": "human"})
    assert w.label_state()[("q", "a.md::k")] == (0, "claude")  # undo falls back to Claude's
    legacy = {"qid": "q2", "gold_id": "b.md::k", "rel": 1, "source": "pool"}  # no judge column yet
    w.log_event("labels.jsonl", legacy)
    assert w.label_state()[("q2", "b.md::k")] == (1, "human")


# ---- the judge column in qrels.tsv, freeze and verify -------------------------------------------


@pytest.fixture
def frozen_parts(tmp_path):
    be, vault = pg_world.imported_world(tmp_path)
    qs = pg_world.build_questions(be)
    yield be, qs, vault, tmp_path
    be.close()


def freeze(be, qs, out, **kw):
    return pg.freeze(qs, be.index, out, snapshot_sha256="00" * 32, snapshot_name="pg.tar",
                     snapshot_date="2026-10-24", routes=vault_world.ROUTES, **kw)  # fmt: skip


def test_qrels_has_a_judge_column_that_load_and_verify_check(frozen_parts):
    be, qs, _, tmp = frozen_parts
    other = next(c.gold_id for c in be.index.chunks if c.store == "personal"
                 and c.gold_id not in qs[0].gold)  # fmt: skip
    qs[0].nonrel, qs[0].judges[other] = [other], "claude"
    freeze(be, qs, tmp / "v1", kappa=0.81, kappa_n=50)
    lines = (tmp / "v1" / "qrels.tsv").read_text().splitlines()
    assert lines[0] == "qid\tgold_id\trel\tcontent_hash\tjudge"
    assert {ln.split("\t")[4] for ln in lines[1:]} == {"claude", "human"}
    snap = pg.read_snapshot(tmp / "v1" / "SNAPSHOT")
    assert snap["judge_kappa"] == "0.8100" and snap["judge_kappa_n"] == "50"
    assert snap["judge_kappa_weights"] == "linear"
    gold = pg.load(tmp / "v1")  # B-R3 loading: the (rel, hash) tuples are unchanged
    assert gold.qrels[qs[0].qid][other][0] == 0 and len(gold.qrels[qs[0].qid][other]) == 2
    assert gold.judges[(qs[0].qid, other)] == "claude"
    # a hand edit of the column is caught by the hash
    p = tmp / "v1" / "qrels.tsv"
    p.write_text(p.read_text().replace("\tclaude", "\thuman"))
    with pytest.raises(ValueError, match="sha256"):
        pg.load(tmp / "v1")


def test_a_qrels_file_without_the_judge_column_does_not_load(frozen_parts):
    be, qs, _, tmp = frozen_parts
    freeze(be, qs, tmp / "v1")
    p = tmp / "v1" / "qrels.tsv"
    rows = [ln.rsplit("\t", 1)[0] for ln in p.read_text().splitlines()]
    p.write_text("\n".join(rows) + "\n")
    snap = tmp / "v1" / "SNAPSHOT"
    text = snap.read_text().replace(
        pg.read_snapshot(snap)["qrels_tsv_sha256"], pg.sha256_bytes(p.read_bytes())
    )
    snap.write_text(text)
    with pytest.raises(ValueError, match="no judge column"):
        pg.load(tmp / "v1")


def test_freeze_refuses_a_missing_or_unknown_judge_and_claude_without_a_passing_kappa(frozen_parts):
    be, qs, _, tmp = frozen_parts
    g = next(iter(qs[0].gold))
    qs[0].judges[g] = ""
    with pytest.raises(ValueError, match="judge ''"):
        freeze(be, qs, tmp / "a")
    qs[0].judges[g] = "gpt"
    with pytest.raises(ValueError, match="judge 'gpt'"):
        freeze(be, qs, tmp / "b")
    qs[0].judges[g] = "claude"
    for k, why in ((None, "none recorded"), (0.69, "0.690"), (float("nan"), "none recorded")):
        with pytest.raises(ValueError, match=f"blind kappa is {why}"):
            freeze(be, qs, tmp / "c", kappa=k, kappa_n=50)
    freeze(be, qs, tmp / "d", kappa=0.7, kappa_n=50)  # exactly 0.7 stands
    qs[0].judges[g] = "human"  # all human labels need no kappa
    freeze(be, qs, tmp / "e")
    assert pg.read_snapshot(tmp / "e" / "SNAPSHOT")["judge_kappa"] == "none"


def test_load_refuses_a_frozen_set_with_claude_labels_and_no_passing_kappa(frozen_parts):
    be, qs, _, tmp = frozen_parts
    qs[0].judges[next(iter(qs[0].gold))] = "claude"
    freeze(be, qs, tmp / "v1", kappa=0.8, kappa_n=50)
    snap = tmp / "v1" / "SNAPSHOT"
    snap.write_text(snap.read_text().replace("judge_kappa: 0.8000", "judge_kappa: 0.5000"))
    with pytest.raises(ValueError, match="blind kappa is 0.500"):
        pg.load(tmp / "v1")


# ---- no work notes ----------------------------------------------------------------------------------


def test_freeze_and_verify_refuse_a_gold_or_pooled_path_in_a_work_routed_folder(frozen_parts):
    be, qs, _, tmp = frozen_parts
    work_chunk = next(c for c in be.index.chunks if c.store == "work")
    bad = pg.Question("pg1-x", "single-fact", "q", gold={work_chunk.gold_id: 2}, store="work",
                      judges={work_chunk.gold_id: "human"})  # fmt: skip
    with pytest.raises(ValueError, match=f"{work_chunk.path} is in a folder routed to work"):
        freeze(be, [bad, *qs[1:]], tmp / "w1")
    pooled = pg.Question("pg1-y", "single-fact", "q", nonrel=[work_chunk.gold_id])
    assert any("routed to work" in e for e in pg.work_routed([pooled], be.index, {}))  # rel 0 too
    # an override such as projects/client-x -> work is caught by the config routes alone
    q = qs[0]
    source = q.source[0]
    prefix = source.rsplit("/", 1)[0]
    errs = pg.work_routed([q], be.index, {prefix: "work"})
    assert errs and source in errs[0]
    # verify: a frozen gold whose folder the config routes to work is refused
    freeze(be, qs, tmp / "v1")
    cfg = tmp / "config.yaml"
    cfg.write_text(yaml.safe_dump({"vault": {"routes": {"personal": "work"}}}))
    args = ["--gold-dir", str(tmp / "v1"), "--stores-dir", str(tmp / "stores")]
    assert run("verify", *args).exit_code == 0
    r = run("verify", *args, "--config", str(cfg))
    assert r.exit_code == 1 and "work-routed paths:" in r.output and "routed to work" in r.output


def test_the_pg_snapshot_holds_only_personal_routed_personal_and_projects_files(tmp_path):
    world = vault_world.make_world(tmp_path / "vault")
    (world / "personal" / "deep").mkdir()
    (world / "personal" / "deep" / "Secret.md").write_text("# s\n")
    (world / "personal" / "img.png").write_text("png")
    (world / "projects" / ".git").mkdir()
    (world / "projects" / ".git" / "x").write_text("x")
    (world / "random").mkdir()
    (world / "random" / "Other.md").write_text("# o\n")
    routes = {**vault_world.ROUTES, "personal/deep": "work"}
    kept, dropped = vault_ds.pg_members(world, routes)
    assert not any(p.startswith(("work/", "random/", "projects/client-x/", "personal/deep/"))
                   for p in kept)  # fmt: skip
    assert set(dropped) == {
        p for p in dropped if p.startswith(("projects/client-x/", "personal/deep/"))
    }
    assert "personal/deep/Secret.md" in dropped and "personal/img.png" in kept
    assert not any(".git" in p for p in kept + dropped) and kept == sorted(kept)
    sha, n_kept, n_dropped = vault_ds.make_pg_snapshot(world, tmp_path / "pg.tar", routes)
    assert (n_kept, n_dropped) == (len(kept), len(dropped)) and sha == vault_ds.sha256_file(
        tmp_path / "pg.tar"
    )
    sha2, *_ = vault_ds.make_pg_snapshot(world, tmp_path / "pg2.tar", routes)
    assert sha2 == sha  # deterministic
    root = vault_ds.extract(tmp_path / "pg.tar", tmp_path / "out")
    got = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert got == kept


def test_snapshot_make_pg_needs_the_config_and_tars_no_work_notes(tmp_path):
    world = vault_world.make_world(tmp_path / "vault")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"vault": {"routes": vault_world.ROUTES}}))
    out = tmp_path / "pgvault-2026-10-24.tar"
    base = ["snapshot-make", "--vault", str(world), "--out", str(out), "--pg"]
    r = CliRunner().invoke(app, base)
    assert r.exit_code != 0 and "--config" in r.output and not out.exists()
    r = CliRunner().invoke(app, [*base, "--config", str(cfg)])
    assert r.exit_code == 0 and "PG snapshot: 44 files" in r.output and "6 work-routed" in r.output
    assert f"sha256 {vault_ds.sha256_file(out)}" in r.output
    import subprocess

    names = subprocess.run(["tar", "-tf", str(out)], capture_output=True, text=True,
                           check=True).stdout.splitlines()  # fmt: skip
    assert names and not any("/work/" in n or "client-x" in n for n in names)
    full = tmp_path / "full.tar"
    assert (
        CliRunner()
        .invoke(app, ["snapshot-make", "--vault", str(world), "--out", str(full)])
        .exit_code
        == 0
    )
    assert vault_ds.sha256_file(full) != vault_ds.sha256_file(out)  # the full snapshot is separate


def test_a_pg_import_holds_no_work_chunk_so_none_can_reach_the_pool_or_qrels(tmp_path):
    world = vault_world.make_world(tmp_path / "vault")
    _, _, root = pg_world.pg_snapshot(world, tmp_path)
    be = pg_world.backend(tmp_path / "stores")
    try:
        be.import_vault(root, routes=vault_world.ROUTES)
        assert set(be.index.store_of_path.values()) == {"personal"}
        assert not any(p.startswith(("work/", "projects/client-x/")) for p in be.index.paths())
        units = sampling.sample_units(be.index, be.conns())
        assert len(units) == 50
        qs = pg_world.build_questions(be)
        assert pg.work_routed(qs, be.index, vault_world.ROUTES) == []
        runs = {q.qid: be.systems(q) for q in qs[:5]}
        assert all(not g.startswith(("work/", "projects/client-x/"))
                   for r in runs.values() for ids in r.values() for g in ids)  # fmt: skip
    finally:
        be.close()


# ---- folder quotas and drafts per unit -------------------------------------------------------------


def test_folder_quotas_are_proportional_with_largest_remainder_and_sum_to_40():
    assert pg.folder_quotas({"personal": 14, "projects": 30}) == {"personal": 13, "projects": 27}
    assert pg.folder_quotas({"personal": 10, "projects": 30}) == {"personal": 10, "projects": 30}
    # 7:13 of 10 = 3.5 / 6.5: floors 3 + 6, the spare one goes to the tie's larger folder
    assert pg.folder_quotas({"personal": 7, "projects": 13}, 10) == {"personal": 3, "projects": 7}
    for a in range(1, 60):
        got = pg.folder_quotas({"personal": a, "projects": 100 - a})
        assert sum(got.values()) == 40 and all(v >= 0 for v in got.values())
    with pytest.raises(ValueError, match="need 40"):
        pg.folder_quotas({"personal": 10, "projects": 20})


def test_more_drafts_per_unit_scale_the_slot_targets_and_never_lower_a_quota(accepted):
    be, _, _ = accepted
    assert sum(sampling.slot_targets(200).values()) == 200
    assert sampling.slot_targets(200) == {c: n * 5 // 3 for c, n in pg.TOTAL.items()}
    for k in (5, 6):
        units = sampling.sample_units(be.index, be.conns(), slots_per_unit=k)
        want = sampling.slot_targets(50 * k)
        got = {c: sum(s == c for u in units for s in u.slots) for c in pg.QUOTAS}
        assert got == want and all(len(u.slots) == k for u in units)
        assert all(
            got[c] >= pg.TOTAL[c] * 5 // 3 for c in pg.QUOTAS
        )  # at least the 4-per-unit plan


# ---- the pre-filter -------------------------------------------------------------------------------


def draft_of(be, question, answer="1000", cat="single-fact", gold=None):
    c = next(c for c in be.index.chunks if "Facts" in c.chunk_key and c.store == "personal")
    return {
        "draft_id": "d",
        "category": cat,
        "question": question,
        "answer": answer,
        "gold": {c.gold_id: 2} if gold is None else gold,
        "key_terms": ["zzzz", "yyyy"],
    }, c


def test_the_prefilter_drops_lexical_overlap_failures_weak_drafts_and_keeps_reworded_ones(accepted):
    be, _, _ = accepted
    d, c = draft_of(be, "")
    copied = " ".join(c.text.split()[:14])  # the chunk's own words as the question
    r = check.check_draft({**d, "question": copied}, be.index)
    assert not r["ok"] and r["overlap_median"] > 0.5
    assert any(x.startswith("lexical overlap") for x in r["prefilter"])
    assert any("pre-filter: lexical overlap" in p for p in r["problems"])
    reworded = "How much money is set aside for the programme each quarter"
    r = check.check_draft({**d, "question": reworded}, be.index)
    assert r["ok"] and r["prefilter"] == [] and r["overlap_median"] <= 0.5
    weak = check.check_draft({**d, "question": "how much?"}, be.index)
    assert not weak["ok"] and any("fewer than 3 content tokens" in x for x in weak["prefilter"])
    noanswer = check.check_draft({**d, "question": reworded, "answer": " "}, be.index)
    assert not noanswer["ok"] and any("empty short answer" in x for x in noanswer["prefilter"])


def test_the_prefilter_uses_the_median_over_a_drafts_gold_chunks(accepted):
    be, _, _ = accepted
    facts = [c for c in be.index.chunks if "Facts" in c.chunk_key and c.store == "personal"][:3]
    q = " ".join(facts[0].text.split()[:14])
    d = {"draft_id": "d", "category": "multi-note", "question": q, "answer": "x",
         "gold": {c.gold_id: 2 for c in facts}, "key_terms": []}  # fmt: skip
    r = check.check_draft(d, be.index)
    ov = sorted(pg.overlap(q, c.text) for c in facts)
    assert r["overlap_median"] == ov[1] and ov[2] > 0.5  # one chunk copied, the median decides


def test_check_hides_prefiltered_drafts_from_review_and_counts_them(accepted, tmp_path):
    be, _, tmp = accepted
    c = next(c for c in be.index.chunks if "Facts" in c.chunk_key and c.store == "personal")
    copied = " ".join(c.text.split()[:14])
    rows = [{"draft_id": "d1", "unit": "u", "slot": 0, "category": "single-fact",
             "question": copied, "answer": "a", "gold": {c.gold_id: 2}, "filters": {},
             "key_terms": [], "source": [c.path], "store": "personal"},
            {"draft_id": "d2", "unit": "u", "slot": 1, "category": "single-fact",
             "question": "How much money is set aside for the programme each quarter",
             "answer": "a", "gold": {c.gold_id: 2}, "filters": {}, "key_terms": [],
             "source": [c.path], "store": "personal"}]  # fmt: skip
    w2 = session.Work(tmp / "w2")
    from bench.gold import draft

    draft.write_drafts(w2.path("drafts.jsonl"), rows)
    r = run("check", "--stores-dir", str(tmp / "stores"), "--work", str(w2.root))
    assert "2 drafts checked, 1 flagged" in r.output
    assert "pre-filter dropped 1: 0 weak, 1 lexical-overlap failures" in r.output
    shown: list[str] = []
    session.review_sitting(w2, be.index, None, minutes=None, out=shown.append, ask=lambda p: "r")
    text = "\n".join(shown)
    assert (
        f"Q: {copied}" not in text and "set aside for the programme" in text
    )  # the human never sees d1
