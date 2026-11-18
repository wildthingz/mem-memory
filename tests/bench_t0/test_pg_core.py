"""PG-v1 core (benchmarks-v2 §3.1): quotas, split, overlap, validation, freeze and its tamper checks."""

import hashlib
from collections import Counter

import pytest

from bench.gold import chunks as ch
from bench.gold import pg
from tests.bench_t0 import pg_world


def test_quotas_sum_to_120_with_tune_exactly_a_third():
    assert pg.N_QUESTIONS == 120
    assert sum(t for t, _ in pg.QUOTAS.values()) == 40
    assert sum(s for _, s in pg.QUOTAS.values()) == 80
    assert all(pg.TOTAL[c] == 3 * t for c, (t, _) in pg.QUOTAS.items())  # exact by construction
    assert sum(pg.TOTAL[c] for c in pg.NO_ANSWER) == 30  # no-answer 10 / 20


def test_split_is_sha1_order_first_third_and_hits_every_quota():
    qs = [pg.Question(f"q{c}-{i}", c, "x") for c, n in pg.TOTAL.items() for i in range(n)]
    sp = pg.split_of(qs)
    got = Counter((q.category, sp[q.qid]) for q in qs)
    for cat, (tune, test) in pg.QUOTAS.items():
        assert (got[(cat, "tune")], got[(cat, "test")]) == (tune, test)
    # the first third of single-fact by sha1(qid) is tune, worked out independently
    ordered = sorted((q.qid for q in qs if q.category == "single-fact"),
                     key=lambda x: hashlib.sha1(x.encode()).hexdigest())  # fmt: skip
    assert {x for x in ordered[:10]} == {x for x, s in sp.items() if s == "tune" and "single" in x}


def test_overlap_is_fraction_of_question_content_tokens_in_the_chunk():
    # content tokens of the question: budget, euro, harbor (stop words and 1-char tokens dropped)
    assert pg.content_tokens("What is the budget in euro of a harbor?") == {
        "budget",
        "euro",
        "harbor",
    }
    assert pg.overlap(
        "What is the budget in euro of a harbor?", "The budget is 5 euro"
    ) == pytest.approx(2 / 3)
    assert pg.overlap("the of a", "anything") == 0.0


def test_validate_flags_every_quota_gold_and_overlap_problem(tmp_path):
    be, _ = pg_world.imported_world(tmp_path)
    try:
        qs = pg_world.build_questions(be)
        assert pg.validate(qs, be.index) == []
        assert any("quota needs exactly" in e for e in pg.validate(qs[:-1], be.index))
        bad = [
            *qs[:-1],
            pg.Question(
                "pg1-x",
                "single-fact",
                "Which?",
                answer="a",
                gold={"nope.md::k#1/p0": 2},
                store="personal",
            ),
        ]
        errs = pg.validate(bad, be.index)
        assert any("not a chunk of the imported vault" in e for e in errs)
        na = next(q for q in qs if q.category == "no-answer-off-topic")
        na.gold = {next(iter(be.index.by_gold)): 1}
        assert any("a no-answer question has gold" in e for e in pg.validate(qs, be.index))
        na.gold = {}
        for q in qs:
            q.reworded = False
        assert any("reworded 0%" in e for e in pg.validate(qs, be.index))
    finally:
        be.close()


def test_validate_median_overlap_over_half_fails_and_pair_needs_two_notes(tmp_path):
    be, _ = pg_world.imported_world(tmp_path)
    try:
        qs = pg_world.build_questions(be)
        for q in qs:
            if q.answerable:
                chunk = be.index.by_gold[next(iter(q.gold))]
                q.question = chunk.text  # a copied question: overlap 1.0
        assert any("median lexical overlap" in e for e in pg.validate(qs, be.index))
        qs = pg_world.build_questions(be)
        mn = next(q for q in qs if q.category == "multi-note")
        mn.gold = {next(iter(mn.gold)): 2}
        assert any(
            "needs gold chunks in at least two notes" in e for e in pg.validate(qs, be.index)
        )
    finally:
        be.close()


def test_freeze_writes_hashed_files_and_refuses_overwrite_and_tamper(tmp_path):
    be, _ = pg_world.imported_world(tmp_path)
    try:
        qs = pg_world.build_questions(be)
        out = tmp_path / "v1"
        hashes = pg.freeze(qs, be.index, out, snapshot_sha256="ab" * 32, snapshot_name="v.tar",
                           snapshot_date=pg_world.DATE)  # fmt: skip
        assert set(hashes) == {"questions.yaml", "qrels.tsv", "split.tsv"}
        snap = pg.read_snapshot(out / "SNAPSHOT")
        assert (
            snap["qrels_tsv_sha256"] == hashlib.sha256((out / "qrels.tsv").read_bytes()).hexdigest()
        )
        gold = pg.load(out)
        assert len(gold.questions) == 120 and len(gold.of_split("tune")) == 40
        assert len(gold.of_split("test")) == 80
        assert pg.stale_gold(gold, be.index) == []
        with pytest.raises(pg.Frozen):  # an edit is a new version, never an overwrite
            pg.freeze(qs, be.index, out, snapshot_sha256="ab" * 32, snapshot_name="v.tar",
                      snapshot_date=pg_world.DATE)  # fmt: skip
        (out / "qrels.tsv").write_text((out / "qrels.tsv").read_text() + "x\n")
        with pytest.raises(ValueError, match="frozen files are never edited"):
            pg.load(out)
    finally:
        be.close()


def test_qrels_keep_judged_not_relevant_rows_and_rel_zero_does_not_stale(tmp_path):
    be, _ = pg_world.imported_world(tmp_path)
    try:
        qs = pg_world.build_questions(be)
        other = next(
            c.gold_id
            for c in be.index.chunks
            if c.gold_id not in qs[0].gold and c.store == "personal"
        )
        qs[0].nonrel = [other]
        qs[0].judges[other] = "claude"  # a pooled pair Claude judged not relevant (kappa stood)
        out = tmp_path / "v1"
        pg.freeze(qs, be.index, out, snapshot_sha256="00" * 32, snapshot_name="v",
                  snapshot_date="d", kappa=0.8, kappa_n=50)  # fmt: skip
        gold = pg.load(out)
        first = next(q for q in gold.questions if q.qid == qs[0].qid)
        assert first.nonrel == [other] and other not in first.gold
        assert gold.qrels[qs[0].qid][other][0] == 0
        assert gold.judges[(qs[0].qid, other)] == "claude"
        assert gold.judges[(qs[0].qid, next(iter(qs[0].gold)))] == "human"
    finally:
        be.close()


def test_stale_gold_reports_changed_and_missing_chunks(tmp_path):
    be, vault = pg_world.imported_world(tmp_path)
    try:
        qs = pg_world.build_questions(be)
        out = tmp_path / "v1"
        pg.freeze(
            qs, be.index, out, snapshot_sha256="00" * 32, snapshot_name="v", snapshot_date="d"
        )
        gold = pg.load(out)
        q = next(q for q in gold.questions if q.answerable)
        path, _ = ch.split_gold_id(next(iter(q.gold)))
        f = vault / path
        f.write_text(f.read_text().replace("budget of", "budget amounting to", 1))
        from tests.bench_t0 import vault_world

        be.import_vault(vault, routes=vault_world.ROUTES)
        assert any(line.endswith("changed") for line in pg.stale_gold(gold, be.index))
        f.unlink()
        be.import_vault(vault, routes=vault_world.ROUTES)
        assert any(line.endswith("missing") for line in pg.stale_gold(gold, be.index))
    finally:
        be.close()


def test_gold_id_is_path_and_chunk_key_and_rejects_malformed():
    assert ch.gold_id("work/a.md", "Intro > Plan#2/p1") == "work/a.md::Intro > Plan#2/p1"
    assert ch.split_gold_id("work/a.md::Intro > Plan#2/p1") == ("work/a.md", "Intro > Plan#2/p1")
    with pytest.raises(ValueError):
        ch.split_gold_id("work/a.md")


def test_a_changed_judged_not_relevant_chunk_does_not_stale_the_gold(tmp_path):
    from tests.bench_t0 import vault_world

    be, vault = pg_world.imported_world(tmp_path)
    try:
        qs = pg_world.build_questions(be)
        gold_ids = {g for q in qs for g in q.gold}
        other = next(c for c in be.index.chunks if c.gold_id not in gold_ids)
        qs[0].nonrel = [other.gold_id]
        qs[0].judges[other.gold_id] = "human"
        out = tmp_path / "v1"
        pg.freeze(
            qs, be.index, out, snapshot_sha256="00" * 32, snapshot_name="v", snapshot_date="d"
        )
        gold = pg.load(out)
        f = vault / other.path
        f.write_text(f.read_text().replace("Before the budget", "Earlier the budget", 1))
        be.import_vault(vault, routes=vault_world.ROUTES)
        assert pg.stale_gold(gold, be.index) == []  # only rel >= 1 chunks can stale a question
    finally:
        be.close()


def test_load_checks_the_quotas_too(tmp_path):
    be, _ = pg_world.imported_world(tmp_path)
    try:
        qs = pg_world.build_questions(be)
        out = tmp_path / "v1"
        pg.freeze(qs, be.index, out, snapshot_sha256="ab" * 32, snapshot_name="v.tar",
                  snapshot_date=pg_world.DATE)  # fmt: skip
        # a consistent but short set: 119 questions, hashes and split re-derived, SNAPSHOT rewritten
        short = qs[1:]
        files = {"questions.yaml": pg.questions_yaml(short), "qrels.tsv": pg.qrels_tsv(short, be.index),
                 "split.tsv": pg.split_tsv(short)}  # fmt: skip
        snap = pg.read_snapshot(out / "SNAPSHOT")
        for name, data in files.items():
            (out / name).write_bytes(data)
            snap[name.replace(".", "_") + "_sha256"] = pg.sha256_bytes(data)
        (out / "SNAPSHOT").write_text("".join(f"{k}: {v}\n" for k, v in snap.items()))
        with pytest.raises(ValueError, match="quotas do not hold"):
            pg.load(out)
    finally:
        be.close()
