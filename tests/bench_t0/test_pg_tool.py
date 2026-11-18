"""The PG-v1 authoring tool: sampling, drafting (scripted LLM), checks, review and label sittings,
pooling, resumability and the full path to a frozen set (benchmarks-v2 §3.1)."""

import json
from collections import Counter

import pytest

from bench import llm
from bench.gold import check, draft, pg, pool, sampling, session
from bench.gold.bm25 import bm25_top
from bench.gold.chunks import ChunkIndex
from tests.bench_t0 import pg_world


@pytest.fixture
def world(tmp_path):
    be, vault = pg_world.imported_world(tmp_path)
    yield be, vault, tmp_path
    be.close()


# ---- sampling and slot plan --------------------------------------------------------------------


def test_sample_units_come_from_personal_routed_personal_and_projects_notes_only(world):
    be, _, _ = world
    units = sampling.sample_units(be.index, be.conns())
    singles = [u for u in units if not u.is_pair]
    pairs = [u for u in units if u.is_pair]
    assert len(singles) == 40 and len(pairs) == 10
    eligible = {f: len(ps) for f, ps in sampling.eligible_notes(be.index).items()}
    assert eligible == {"personal": 14, "projects": 30}  # projects/client-x (6) is routed to work
    by_folder = Counter(sampling.folder_of(u.paths[0]) for u in singles)
    assert by_folder == pg.folder_quotas(eligible) == {"personal": 13, "projects": 27}
    for u in units:  # no work folder, no work-routed override, in any unit
        for p in u.paths:
            assert sampling.folder_of(p) in pg.PG_FOLDERS and "client-x" not in p
            assert be.index.store_of_path[p] == "personal" == u.store
    again = sampling.sample_units(be.index, be.conns())
    assert [u.paths for u in again] == [u.paths for u in units]  # deterministic (seed 20261022)
    other = sampling.sample_units(be.index, be.conns(), seed=1)
    assert [u.paths for u in other] != [u.paths for u in units]


def test_slot_plan_is_200_drafts_four_per_unit_pair_only_categories_on_pairs(world):
    be, _, _ = world
    units = sampling.sample_units(be.index, be.conns())
    assert all(len(u.slots) == 4 for u in units)
    counts = Counter(c for u in units for c in u.slots)
    assert sum(counts.values()) == 200
    assert counts == {c: n * 5 // 3 for c, n in pg.TOTAL.items()}  # 5/3 of each quota
    for u in units:
        if not u.is_pair:
            assert not set(u.slots) & pg.PAIR_ONLY
    shifted = sampling.plan_slots(units, shift=1)
    assert {u.uid for u in shifted} == {u.uid + ".r1" for u in units}


# ---- drafting ----------------------------------------------------------------------------------


class ScriptedLLM:
    """Plays the drafting model: one valid draft per slot, optionally one bad citation."""

    model = "claude-opus-5-5"

    def __init__(self, index, bad_unit=None, garbage_unit=None):
        self.index, self.bad, self.garbage, self.calls = index, bad_unit, garbage_unit, 0

    def create(self, *, system, messages, tools, max_tokens):
        self.calls += 1
        prompt = messages[0]["content"]
        if self.garbage and self.garbage in prompt:
            return llm.Reply([{"type": "text", "text": "sorry, no JSON"}], "end_turn", USAGE)
        ids = [
            ln.split("chunk id: ")[1]
            for ln in prompt.splitlines()
            if ln.startswith("### chunk id: ")
        ]
        slots = [ln for ln in prompt.splitlines() if ln[:1].isdigit() and "category `" in ln]
        drafts = []
        for ln in slots:
            slot, cat = int(ln.split(".")[0]), ln.split("`")[1]
            gold = [] if cat in pg.NO_ANSWER else [{"id": ids[0], "rel": 2}]
            if self.bad and self.bad in prompt and slot == 0:
                gold = [{"id": "ghost.md::x#1/p0", "rel": 2}]
            drafts.append(
                {
                    "slot": slot,
                    "category": cat,
                    "question": f"What about the unrelated {cat} matter number {slot}",
                    "answer": "" if cat in pg.NO_ANSWER else "a",
                    "gold": gold,
                    "filters": {},
                    "key_terms": ["zzzz", "yyyy"],
                }
            )
        return llm.Reply(
            [{"type": "text", "text": json.dumps({"drafts": drafts})}], "end_turn", USAGE
        )


USAGE = {"input_tokens": 1000, "output_tokens": 300}


def test_draft_all_makes_200_drafts_counts_failures_and_resumes(world):
    be, _, _ = world
    units = sampling.sample_units(be.index, be.conns())
    bad, garbage = units[0].paths[0], units[1].paths[0]
    client = ScriptedLLM(be.index, bad_unit=bad, garbage_unit=garbage)
    ledger = llm.Ledger("claude-opus-5-5", 50.0)
    rep = draft.draft_all(units, be.index, client, ledger)
    assert rep.parse_failures == 1 and rep.unknown_chunk == 1
    assert rep.skipped_slots == 4 + 1  # the unparsable unit's 4 slots + the bad-citation draft
    assert len(rep.drafts) == 200 - 5
    assert all(d["store"] in ("personal", "work") for d in rep.drafts)
    assert rep.cost["calls"] == 50 and rep.cost["usd"] > 0
    assert all(d["gold"] == {} for d in rep.drafts if d["category"] in pg.NO_ANSWER)
    again = draft.draft_all(units, be.index, client, llm.Ledger("claude-opus-5-5", 50.0),
                            existing=rep.drafts)  # fmt: skip
    # units that produced drafts are not asked again; only the unparsable unit is retried
    assert client.calls == 50 + 1 and len(again.drafts) == len(rep.drafts)


def test_draft_budget_cap_stops_before_overspending(world):
    be, _, _ = world
    units = sampling.sample_units(be.index, be.conns())
    with pytest.raises(llm.BudgetExceeded):
        draft.draft_all(units, be.index, ScriptedLLM(be.index), llm.Ledger("claude-opus-5-5", 0.05))


# ---- checks ------------------------------------------------------------------------------------


def test_rg_check_flags_a_no_answer_draft_whose_key_terms_sit_in_one_chunk(world):
    be, _, _ = world
    chunk = be.index.chunks[0]
    words = [w for w in chunk.text.split() if w.isalpha() and len(w) > 5][:2]
    hit = {"draft_id": "d1", "category": "no-answer-near-miss", "question": "q", "gold": {},
           "key_terms": words}  # fmt: skip
    r = check.check_draft(hit, be.index)
    assert not r["ok"] and chunk.gold_id in r["rg_hits"]
    miss = {**hit, "draft_id": "d2", "key_terms": ["qqqq", "wwww"],
            "question": "Where did the unrelated launch party end up happening"}  # fmt: skip
    assert check.check_draft(miss, be.index)["ok"]
    assert not check.check_draft({**hit, "key_terms": ["only"]}, be.index)["ok"]  # < 2 key terms


def test_check_flags_answerable_drafts_with_missing_gold(world):
    be, _, _ = world
    d = {"draft_id": "d", "category": "single-fact", "question": "how much", "gold": {"x::y": 2}}
    r = check.check_draft(d, be.index)
    assert not r["ok"] and "gold ids not in the vault" in r["problems"][0]
    good = {**d, "gold": {be.index.chunks[0].gold_id: 2}}
    assert check.check_draft(good, be.index)["overlap"] is not None


def test_bm25_top_needs_no_model_and_respects_the_store(world):
    be, _, _ = world
    index = ChunkIndex.from_conns(be.conns())
    path = next(p for p in index.paths() if p.startswith("personal/"))
    topic = pg_world.topic_of(path)
    top = bm25_top(be.conns(), index, "personal", f"{topic} programme budget", {}, 20,
                   "2026-10-24T00:00:00Z")  # fmt: skip
    assert top and top[0].startswith(path)
    assert all(index.by_gold[g].store == "personal" for g in top)


# ---- review sittings ---------------------------------------------------------------------------


def _drafts(be, per_cat):
    out, n = [], 0
    for cat in pg.QUOTAS:
        for i in range(per_cat[cat]):
            n += 1
            c = be.index.chunks[n % len(be.index.chunks)]
            out.append({"draft_id": f"d{n:03d}", "unit": "u", "slot": i, "category": cat,
                        "question": f"question {n}", "answer": "" if cat in pg.NO_ANSWER else "ans",
                        "gold": {} if cat in pg.NO_ANSWER else {c.gold_id: 2}, "filters": {},
                        "key_terms": ["zz", "yy"], "source": [c.path], "store": c.store})  # fmt: skip
    return out


def script(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_review_accepts_edits_rejects_and_marks_reworded(world, tmp_path):
    be, _, _ = world
    w = session.Work(tmp_path / "work")
    draft.write_drafts(w.path("drafts.jsonl"), _drafts(be, {c: 2 for c in pg.QUOTAS}))
    lines: list[str] = []
    res = session.review_sitting(w, be.index, lambda d, k: [], minutes=None, out=lines.append,
                                 ask=script("a", "e", "my wording", "my answer", "r", "q"))  # fmt: skip
    assert res.items == 3 and res.stopped == "quit"
    acc = w.accepted()
    assert [q.reworded for q in acc] == [False, True]
    assert acc[1].question == "my wording" and acc[1].answer == "my answer"
    assert [e["action"] for e in w.decisions().values()] == ["accept", "edit", "reject"]
    assert {q.qid for q in acc} == {"pg1-sf-01", "pg1-mn-01"}  # qids are per category, in order
    # a draft's proposed gold is labelled rel 2 on accept, so pooling skips it
    assert w.labels()[(acc[0].qid, next(iter(acc[0].gold)))] == 2


def test_review_stops_at_the_time_limit_and_resumes_where_it_stopped(world, tmp_path):
    be, _, _ = world
    w = session.Work(tmp_path / "work")
    draft.write_drafts(w.path("drafts.jsonl"), _drafts(be, {c: 2 for c in pg.QUOTAS}))
    t = iter(range(0, 10_000, 40))  # the fake clock: 40 s per call
    r1 = session.review_sitting(w, be.index, None, minutes=2, out=lambda s: None,
                                ask=lambda p: "a", clock=lambda: next(t))  # fmt: skip
    assert r1.stopped == "time" and 0 < r1.items < 16
    done = set(w.decisions())
    r2 = session.review_sitting(w, be.index, None, minutes=None, out=lambda s: None,
                                ask=lambda p: "a")  # fmt: skip
    assert r2.stopped == "done" and r1.items + r2.items == 16
    assert done <= set(w.decisions()) and len(w.decisions()) == 16
    assert len(w.sittings()) == 2


def test_review_never_shows_a_category_at_its_quota_and_reports_a_shortfall(world, tmp_path):
    be, _, _ = world
    w = session.Work(tmp_path / "work")
    per = {c: 2 for c in pg.QUOTAS} | {"relational": 8}  # quota total for relational is 6
    draft.write_drafts(w.path("drafts.jsonl"), _drafts(be, per))
    session.review_sitting(w, be.index, None, minutes=None, out=lambda s: None, ask=lambda p: "a")
    assert session.counts(w)["relational"] == 6  # the 7th and 8th were never shown
    text = session.status_text(w)
    assert "NOT ENOUGH DRAFTS LEFT" in text and "single-fact (short by 28)" in text


def test_review_hides_flagged_no_answer_drafts_and_shows_bm25_for_the_rest(world, tmp_path):
    be, _, _ = world
    w = session.Work(tmp_path / "work")
    ds = [d for d in _drafts(be, {c: 1 for c in pg.QUOTAS}) if d["category"] in pg.NO_ANSWER]
    draft.write_drafts(w.path("drafts.jsonl"), ds)
    w.log_event("checks.jsonl", {"draft_id": ds[0]["draft_id"], "ok": False, "problems": ["hit"]})
    w.log_event("checks.jsonl", {"draft_id": ds[1]["draft_id"], "ok": True, "problems": []})
    shown: list[str] = []
    session.review_sitting(w, be.index, lambda d, k: [be.index.chunks[0].gold_id], minutes=None,
                           out=shown.append, ask=lambda p: "a")  # fmt: skip
    text = "\n".join(shown)
    assert ds[0]["question"] not in text and ds[1]["question"] in text
    assert be.index.chunks[0].gold_id in text  # the BM25 top 20 is on screen
    assert [q.qid for q in w.accepted()] == ["pg1-ot-01"]


def test_a_no_answer_draft_the_person_says_is_answered_is_rejected(world, tmp_path):
    be, _, _ = world
    w = session.Work(tmp_path / "work")
    ds = [
        d for d in _drafts(be, {c: 1 for c in pg.QUOTAS}) if d["category"] == "no-answer-near-miss"
    ]
    draft.write_drafts(w.path("drafts.jsonl"), ds)
    session.review_sitting(w, be.index, lambda d, k: [], minutes=None, out=lambda s: None,
                           ask=script("x"))  # fmt: skip
    ev = w.decisions()[ds[0]["draft_id"]]
    assert ev["action"] == "reject" and ev["why"] == "answered" and w.accepted() == []


# ---- pooling and labelling ---------------------------------------------------------------------


class FakeSystems:
    def __init__(self, runs):
        self.runs = runs

    def systems(self, q):
        return self.runs[q.qid]


def test_pool_takes_top5_of_three_systems_dedupes_and_skips_labelled(world, tmp_path):
    be, _, _ = world
    ids = [c.gold_id for c in be.index.chunks[:20]]
    w = session.Work(tmp_path / "work")
    q = pg.Question("pg1-sf-01", "single-fact", "q", answer="a", gold={ids[0]: 2})
    runs = {"pg1-sf-01": {"ours": ids[0:8], "bm25": ids[3:12], "dense": ids[10:18]}}
    w.log_event(
        "labels.jsonl", {"qid": "pg1-sf-01", "gold_id": ids[0], "rel": 2, "source": "draft"}
    )
    st = pool.build(w, [q], FakeSystems(runs))
    got = {p["gold_id"]: p["systems"] for p in w.pool()}
    # ours top5 = 0..4, bm25 top5 = 3..7, dense top5 = 10..14; chunk 0 is already judged
    assert set(got) == set(ids[1:8]) | set(ids[10:15])
    assert got[ids[3]] == ["ours", "bm25"] and got[ids[10]] == ["dense"]
    assert st.pairs == 12 and not st.over_budget
    again = pool.build(w, [q], FakeSystems(runs))
    assert again.new_pairs == 0 and len(w.pool()) == 12  # idempotent
    with pytest.raises(ValueError):
        pool.pool_for(q, FakeSystems({"pg1-sf-01": {"ours": ids}}), set())


def _accept_one(w, be, cat="single-fact", gold_cat=None):
    d = _drafts(be, {c: (1 if c == cat else 0) for c in pg.QUOTAS})[0]
    draft.write_drafts(w.path("drafts.jsonl"), [d])
    session.review_sitting(w, be.index, lambda dd, k: [], minutes=None, out=lambda s: None,
                           ask=lambda p: "a")  # fmt: skip
    return w.accepted()[0]


def test_label_sitting_writes_labels_supports_undo_skip_and_resume(world, tmp_path):
    be, _, _ = world
    w = session.Work(tmp_path / "work")
    q = _accept_one(w, be)
    ids = [c.gold_id for c in be.index.chunks if c.gold_id not in q.gold][:4]
    for g in ids:
        w.log_event("pool.jsonl", {"qid": q.qid, "gold_id": g, "systems": ["ours"]})
    # 1 for the first pair, 2 for the second, undo it, relabel 0, skip the third, quit
    r = session.label_sitting(w, be.index, minutes=None, out=lambda s: None,
                              ask=script("1", "2", "u", "0", "s", "q"))  # fmt: skip
    lab = w.labels()
    assert (lab[(q.qid, ids[0])], lab[(q.qid, ids[1])]) == (1, 0) and (q.qid, ids[2]) not in lab
    assert r.stopped == "quit" and r.remaining == 2
    r2 = session.label_sitting(w, be.index, minutes=None, out=lambda s: None,
                               ask=script("2", "0"))  # fmt: skip
    assert w.labels()[(q.qid, ids[2])] == 2 and r2.remaining == 0
    acc = w.accepted()[0]
    assert acc.gold[ids[0]] == 1 and acc.gold[ids[2]] == 2 and ids[1] in acc.nonrel


def test_a_no_answer_question_judged_answerable_is_voided_and_its_quota_reopens(world, tmp_path):
    be, _, _ = world
    w = session.Work(tmp_path / "work")
    q = _accept_one(w, be, "no-answer-near-miss")
    ids = [c.gold_id for c in be.index.chunks[:3]]
    for g in ids:
        w.log_event("pool.jsonl", {"qid": q.qid, "gold_id": g, "systems": ["ours"]})
    out: list[str] = []
    session.label_sitting(w, be.index, minutes=None, out=out.append, ask=script("0", "2"))
    assert w.accepted() == [] and any("voided" in s for s in out)
    assert (q.qid, ids[2]) not in w.labels()  # the rest of its pairs were dropped
    assert session.counts(w)["no-answer-near-miss"] == 0


def test_status_reports_real_pace_against_the_spec_budget(tmp_path):
    w = session.Work(tmp_path / "work")
    w.log_event("sittings.jsonl", {"kind": "review", "items": 60, "seconds": 3000.0})
    w.log_event("sittings.jsonl", {"kind": "label", "items": 100, "seconds": 600.0})
    text = session.status_text(w)
    assert "review: 60 items in 50 min, 50 s each (spec assumes 25 s)" in text
    assert "label: 100 items in 10 min, 6 s each (spec assumes 8 s)" in text


# ---- the whole path ----------------------------------------------------------------------------


def test_full_path_review_pool_label_freeze_then_b_r3_scores_it(world, tmp_path):
    be, _, tmp = world
    w = session.Work(tmp / "work")
    truth = {q.qid: q for q in pg_world.build_questions(be)}
    drafts = []
    for i, q in enumerate(truth.values()):  # the drafts the model "wrote": the truth + 20% extra
        drafts.append({"draft_id": f"d{i:03d}", "unit": "u", "slot": 0, "category": q.category,
                       "question": q.question, "answer": q.answer, "gold": dict(q.gold),
                       "filters": {}, "key_terms": ["zzzz", "yyyy"], "source": q.source,
                       "store": q.store})  # fmt: skip
    draft.write_drafts(w.path("drafts.jsonl"), drafts)
    for d in drafts:
        w.log_event("checks.jsonl", check.check_draft(d, be.index))
    answers: list[str] = []
    for i, d in enumerate(session.review_order(drafts, set(), set())):
        answers += ["e", d["question"] + "?", ""] if i % 2 == 0 else ["a"]
    ask = iter(answers)
    session.review_sitting(w, be.index, lambda d, k: [], minutes=None, out=lambda s: None,
                           ask=lambda p: next(ask))  # fmt: skip
    assert sum(session.counts(w).values()) == 120
    st = pool.build(w, w.accepted(), be)
    assert 0 < st.pairs <= pool.MAX_PAIRS
    gold_of = {ev["qid"]: next(d["gold"] for d in drafts if d["draft_id"] == did)
               for did, ev in w.decisions().items()}  # fmt: skip
    cur: dict[str, object] = {"gold": {}, "next": "0"}

    def out(text):  # what the person sees: the question header, then the pair's gold id and text
        if text.startswith("\n[pg1-"):
            cur["gold"] = gold_of[text.split("]")[0].lstrip("\n[")]
        elif "::" in text.split("\n")[0]:
            cur["next"] = "2" if text.split("\n")[0].strip() in cur["gold"] else "0"

    session.label_sitting(w, be.index, minutes=None, out=out, ask=lambda p: cur["next"])
    qs = w.accepted()
    assert len(qs) == 120 and pg.validate(qs, be.index) == []
    out_dir = tmp / "frozen"
    pg.freeze(qs, be.index, out_dir, snapshot_sha256="11" * 32, snapshot_name="v.tar.zst",
              snapshot_date=pg_world.DATE)  # fmt: skip
    gold = pg.load(out_dir)
    assert Counter(q.category for q in gold.questions) == pg.TOTAL
    assert any(q.nonrel for q in gold.questions)  # the pooled-and-judged-0 chunks are in qrels
