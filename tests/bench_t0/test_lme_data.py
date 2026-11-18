"""Manifest, round mapping, negatives, audit and load corpora on the official file layout."""

import hashlib
import json
from pathlib import Path

import pytest

from bench.datasets import corpora, lme, negatives
from bench.fixtures.lme import make_fixture

FIX = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme"

# benchmarks-v2 §1: (question_type, n answerable, n _abs)
OFFICIAL = [
    ("knowledge-update", 72, 6), ("temporal-reasoning", 127, 6), ("multi-session", 121, 12),
    ("single-session-user", 64, 6), ("single-session-assistant", 56, 0),
    ("single-session-preference", 30, 0),
]  # fmt: skip


def official_layout(tmp_path: Path) -> Path:
    """500 questions with the official stratum sizes, in the official file layout, tiny sessions."""
    qs = []
    for t, (qtype, n_ans, n_abs) in enumerate(OFFICIAL):
        for i in range(n_ans + n_abs):
            qid = f"t{t}-{i:03d}" + ("_abs" if i >= n_ans else "")
            qs.append({
                "question_id": qid, "question_type": qtype, "question": "q", "answer": "a",
                "question_date": "2023/05/20 (Sat) 02:21", "haystack_session_ids": ["s"],
                "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                "haystack_sessions": [[{"role": "user", "content": "hi"}]],
                "answer_session_ids": ["s"],
            })  # fmt: skip
    path = tmp_path / "longmemeval_s_cleaned.json"
    path.write_text(json.dumps(qs))
    return path


def test_manifest_counts_reproduce_the_spec_from_the_official_layout(tmp_path):
    rows = lme.build_manifest(lme.load_questions(official_layout(tmp_path)))
    c = lme.manifest_counts(rows)
    assert (c["total"], c["tune"], c["test"]) == (500, 151, 349)
    assert (c["c_ans_tune"], c["c_ans_test"]) == (96, 224)
    assert (c["br1_tune"], c["br1_test"]) == (115, 269)
    assert (c["abs_tune"], c["abs_test"]) == (10, 20)


def test_per_stratum_split_matches_the_table(tmp_path):
    rows = lme.build_manifest(lme.load_questions(official_layout(tmp_path)))
    got = {}
    for r in rows:
        got.setdefault((r.question_type, r.is_abs), [0, 0])[r.split == "test"] += 1
    want = {("knowledge-update", False): (22, 50), ("knowledge-update", True): (2, 4),
            ("temporal-reasoning", False): (38, 89), ("temporal-reasoning", True): (2, 4),
            ("multi-session", False): (36, 85), ("multi-session", True): (4, 8),
            ("single-session-user", False): (19, 45), ("single-session-user", True): (2, 4),
            ("single-session-assistant", False): (17, 39),
            ("single-session-preference", False): (9, 21)}  # fmt: skip
    assert {k: tuple(v) for k, v in got.items()} == want


def test_manifest_is_deterministic_sorted_by_sha1_and_roundtrips(tmp_path):
    qs = lme.load_questions(official_layout(tmp_path))
    a, b = lme.build_manifest(qs), lme.build_manifest(list(reversed(qs)))
    assert a == b
    stratum = [r for r in a if (r.question_type, r.is_abs) == ("multi-session", True)]
    assert [r.question_id for r in stratum] == sorted(
        (r.question_id for r in stratum), key=lambda i: hashlib.sha1(i.encode()).hexdigest()
    )
    assert [r.split for r in stratum] == ["tune"] * 4 + ["test"] * 8
    path = tmp_path / "lme_split.tsv"
    sha = lme.write_manifest(a, path)
    assert lme.read_manifest(path) == a
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    lme.verify_manifest(qs, path)


def test_verify_manifest_detects_drift(tmp_path):
    qs = lme.load_questions(official_layout(tmp_path))
    path = tmp_path / "m.tsv"
    lme.write_manifest(lme.build_manifest(qs), path)
    lines = path.read_text().splitlines()
    swapped = (
        lines[1].replace("\ttune\t", "\ttest\t")
        if "\ttune\t" in lines[1]
        else lines[1].replace("\ttest\t", "\ttune\t")
    )
    path.write_text("\n".join([lines[0], swapped, *lines[2:]]) + "\n")
    with pytest.raises(ValueError, match="differs"):
        lme.verify_manifest(qs, path)
    with pytest.raises(ValueError, match="differs"):
        lme.verify_manifest(qs[:-1], path)


def test_n_tune_rounds_half_up_without_float_ties():
    assert [lme.n_tune(n) for n in (0, 1, 2, 3, 6, 12, 30, 72, 127, 133)] == [
        0,
        0,
        1,
        1,
        2,
        4,
        9,
        22,
        38,
        40,
    ]
    assert lme.n_tune(5) == 2  # 1.5 -> 2, half up


def test_checked_in_fixture_is_the_generator_output():
    assert json.loads((FIX / "lme_s_mini.json").read_text()) == make_fixture.make_s()
    assert json.loads((FIX / "lme_m_mini.json").read_text()) == make_fixture.make_m()


def test_round_mapping_and_gold():
    qs = {q["question_id"]: q for q in lme.load_questions(FIX / "lme_s_mini.json")}
    q = qs[next(k for k in qs if qs[k]["question_type"] == "knowledge-update" and "_abs" not in k)]
    gold_s = q["answer_session_ids"][0]
    rounds = lme.rounds_of(q["haystack_sessions"][q["haystack_session_ids"].index(gold_s)])
    assert [(r.index, r.has_answer) for r in rounds] == [(0, True), (1, False)]
    assert rounds[0].assistant.startswith("Noted")
    assert lme.gold_rounds(q) == {lme.ref(gold_s, 0)}
    # a leading assistant turn is its own round 0; the first user turn starts round 1
    q1 = qs[sorted(qs)[1]]
    r1 = lme.rounds_of(q1["haystack_sessions"][0])
    assert r1[0].user == "" and r1[0].assistant == "Welcome back." and r1[1].user


def test_rounds_group_consecutive_assistant_turns_and_flag_either_turn():
    s = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
         {"role": "assistant", "content": "a1b", "has_answer": True}, {"role": "user", "content": "u2"}]  # fmt: skip
    r = lme.rounds_of(s)
    assert [(x.user, x.assistant, x.has_answer) for x in r] == [
        ("u1", "a1\na1b", True),
        ("u2", "", False),
    ]


def test_session_dedup_in_rank_order_and_haystack_dedup():
    assert lme.dedupe_sessions(["b#3", "a#0", "b#0", "c#1", "a#2"]) == ["b", "a", "c"]
    qs = {q["question_id"]: q for q in lme.load_questions(FIX / "lme_s_mini.json")}
    q = qs[sorted(qs)[2]]
    assert len(q["haystack_session_ids"]) == 7
    h = lme.haystack_of(q)
    assert len(h.sessions) == 6 and h.dropped_duplicates == 1


def test_parse_lme_date():
    assert lme.parse_lme_date("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21:00Z"


def test_evidence_removed_haystacks_and_audit_sheet(tmp_path):
    qs = lme.load_questions(FIX / "lme_s_mini.json")
    rows = lme.build_manifest(qs)
    neg = negatives.build_negatives(qs, rows)
    assert len(neg) == 12  # 4 answerable types x 3, no abs, no SSA/SSP
    by = {q["question_id"]: q for q in qs}
    for n in neg:
        q = by[n.question_id]
        rest = lme.haystack_of(q, exclude=q["answer_session_ids"])
        assert not set(q["answer_session_ids"]) & set(rest.session_ids)
        assert n.n_remaining == len(rest.sessions) and n.n_removed == len(q["answer_session_ids"])
        assert n.auto_hit_sessions == ()  # fixture fillers never contain the answer
    sample = negatives.audit_sample(neg, n=5)
    assert sample == negatives.audit_sample(list(reversed(neg)), n=5) and len(sample) == 5
    sheet = tmp_path / "audit.tsv"
    negatives.write_audit_sheet(sample, qs, sheet)
    with pytest.raises(ValueError, match="incomplete"):
        negatives.apply_audit(neg, sheet)
    lines = sheet.read_text().splitlines()
    cols = lines[0].split("\t")
    i = cols.index("still_contains_answer")
    filled = [lines[0]]
    for k, ln in enumerate(lines[1:]):
        f = ln.split("\t")
        f[i] = "y" if k == 0 else "n"
        filled.append("\t".join(f))
    sheet.write_text("\n".join(filled) + "\n")
    out, report = negatives.apply_audit(neg, sheet)
    assert report == {"audited": 5, "dropped": 1, "pairs_after": 11}
    assert [r.question_id for r in out if r.audit_dropped] == [sample[0].question_id]
    p = tmp_path / "neg.tsv"
    negatives.write_negatives(out, p)
    assert negatives.read_negatives(p) == out


def test_auto_hit_flags_answer_left_in_a_filler_session():
    q = dict(lme.load_questions(FIX / "lme_s_mini.json")[0])
    q["haystack_sessions"] = [list(s) for s in q["haystack_sessions"]]
    q["haystack_sessions"][0] = [
        *q["haystack_sessions"][0],
        {"role": "user", "content": f"fyi {q['answer']}"},
    ]
    assert negatives.auto_hits(q) == [q["haystack_session_ids"][0]]


def test_load_corpus_plan_is_exact_sorted_deduped_and_deterministic():
    m = lme.load_questions(FIX / "lme_m_mini.json")
    sessions = corpora.collect_sessions(m)
    total = sum(len(lme.rounds_of(t)) for _, t in sessions.values())
    assert total > 20
    plan = corpora.plan_corpus(m, 20)
    assert corpora.n_chunks(plan) == 20
    ids = [p.session_id for p in plan]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert corpora.corpus_fingerprint(plan) == corpora.corpus_fingerprint(
        corpora.plan_corpus(list(reversed(m)), 20)
    )
    assert plan[-1].truncated == (len(plan[-1].rounds) < len(lme.rounds_of(sessions[ids[-1]][1])))
    calls = []
    assert corpora.build_corpus(plan, lambda d, r, s: calls.append((d, len(r), s))) == 20
    assert [c[2] for c in calls] == ids
    with pytest.raises(ValueError, match="only"):
        corpora.plan_corpus(m, total + 1)


def test_conflicting_duplicate_session_id_is_an_error():
    m = json.loads((FIX / "lme_m_mini.json").read_text())
    seen: dict[str, tuple[int, int]] = {}
    for qi, q in enumerate(m):
        for si, sid in enumerate(q["haystack_session_ids"]):
            if sid in seen:
                q["haystack_sessions"][si] = [{"role": "user", "content": "different"}]
                with pytest.raises(ValueError, match="different"):
                    corpora.collect_sessions(m)
                return
            seen[sid] = (qi, si)
    pytest.fail("fixture has no session shared by two haystacks")


def test_cli_build_lme_refuses_wrong_counts_and_writes_frozen_files(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from bench import cli

    for name in ("MANIFEST", "NEGATIVES", "AUDIT"):
        monkeypatch.setattr(cli, name, tmp_path / getattr(cli, name).name)
    runner = CliRunner()
    r = runner.invoke(cli.app, ["build-lme", str(FIX / "lme_s_mini.json")])
    assert r.exit_code == 1 and "differ from benchmarks-v2 §1" in r.output
    assert not cli.MANIFEST.exists()
    # official-layout skeleton: counts reproduce, but its sessions all share one id, so no
    # negatives can be built (the answer session is removed from a 1-session haystack: still valid)
    big = official_layout(tmp_path)
    skeleton = big.with_name("official_layout.json")
    big.rename(skeleton)
    r = runner.invoke(cli.app, ["build-lme", str(skeleton)])
    assert r.exit_code == 0, r.output
    assert cli.MANIFEST.exists() and len(cli.NEGATIVES.read_text().splitlines()) == 1 + 115 + 269
    assert len(cli.AUDIT.read_text().splitlines()) == 1 + 30
    again = runner.invoke(cli.app, ["build-lme", str(skeleton)])
    assert "already frozen" in again.output


def test_tune_labels_export_only_tune_with_negatives_and_no_test_ids():
    from bench.datasets import calibration

    qs = lme.load_questions(FIX / "lme_s_mini.json")
    rows_m = lme.build_manifest(qs)
    out = calibration.tune_label_rows({q["question_id"]: q for q in qs}, rows_m,
                                      negatives.build_negatives(qs, rows_m))  # fmt: skip
    tune_ids = {r.question_id for r in lme.select(rows_m, "tune", "br1")}
    assert {r["qid"] for r in out} == tune_ids and len(out) == 2 * len(tune_ids) == 8
    neg = [r for r in out if r["kind"] == "negative"]
    ans = {r["qid"]: r for r in out if r["kind"] == "answerable"}
    assert all(r["rel"] == {} for r in neg) and all(r["rel"] for r in ans.values())
    for r in neg:
        assert not set(r["sessions"]) & set(
            next(q for q in qs if q["question_id"] == r["qid"])["answer_session_ids"]
        )
