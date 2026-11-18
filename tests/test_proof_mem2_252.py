"""Proof tests for the MEM2-252 re-review of MEM2-222 (fix branches mem2-249, mem2-250, mem2-251).

F24 fails on the fixed code; the two parity tests pass and record that MEM2-249's F13 move and
MEM2-250's F6 rewrite are behaviour-preserving (the ruling "prf-check gives the same numbers").
"""

from __future__ import annotations

import importlib.util
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from mem import listmode, read
from mem.splitq import split_endpoints
from tests.test_list_mode import Q as LIST_Q
from tests.test_list_mode import _store
from tests.test_list_paging import _ids, _paged, _walk
from tests.test_prf import store  # noqa: F401 (fixture)

ROOT = Path(__file__).resolve().parents[1]

# ---- F24 (B, new in MEM2-249): the oversize-row page's trailer line is not priced against the cap --


@pytest.mark.parametrize("budget", [50, 60, 70])
def test_f24_an_oversize_row_page_is_itself_under_the_cap(tmp_path, budget):
    """The header alone (40 tokens) fits these budgets; the appended 'is over the budget' line is
    not priced, so every page of the walk is 75 tokens, over the cap. Paging off answers in 15."""
    s = _paged(tmp_path)
    pages = _walk(s, budget=budget)
    assert not _ids(pages)  # every row alone is over the cap: six oversize pages
    for p in pages:
        head = p.text.splitlines()[0]
        assert listmode.response_tokens(head) <= budget, "the header itself fits"
        assert listmode.response_tokens(p.text) <= budget, (budget, p.text)
    s.close()


def test_f24_paging_off_answers_the_same_budget_in_one_short_line(tmp_path):
    s = _store(tmp_path, list_mode=True)
    out = read.recall(s, query=LIST_Q, budget=50)
    assert out.text.startswith("budget 50 too small for one row")
    assert listmode.response_tokens(out.text) <= 50
    s.close()


# ---- parity: F13 (prf-check numbers) and F6 (the clause scanner) ----------------------------------


def _old(name: str, rev: str, path: str):
    got = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, check=False,
                         capture_output=True, text=True)  # fmt: skip
    if got.returncode != 0:
        pytest.skip(f"needs the development history (commit {rev}), not in this checkout")
    src = got.stdout
    tmp = Path(tempfile.mkdtemp()) / f"{name}.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # the old dataclasses resolve their module at class creation
    spec.loader.exec_module(mod)
    return mod


def test_f13_expanded_pool_gives_the_pool_the_serving_expand_leg_gave(store):  # noqa: F811
    from bench.prf_check import expanded_pool

    old = _old("read_before_mem2_249", "47473482797d", "src/mem/read.py")
    t = "2023-06-01T00:00:00Z"
    for q in ["alpha one", "bravo", "echo five charlie", "zulu"]:
        for e in ["foxtrot six", "delta four alpha", "alpha one bravo two charlie three"]:
            for size in (1, 2, 3, 6):
                before, _ = old.candidate_pool(store, q, t=t, size=size, expand=e)
                assert expanded_pool(store, q, e, t=t, size=size) == before, (q, e, size)


def test_f6_the_clause_scanner_splits_as_the_old_regex_did_except_for_f2():
    old = _old("splitq_before_mem2_250", "9f2be085df88", "src/mem/splitq.py")
    words = ["I", "we", "the", "day", "my", "car", "first", "before", "after", "when", "did",
             "replace", "spark", "plugs", "harvest", "herbs", ",", "then", "went", "to", "Paris",
             "and", "bought", ",", "a", "new", "phone", "which", "was"]  # fmt: skip
    rnd = random.Random(7)
    corpus = {" ".join(rnd.choice(words) for _ in range(rnd.randint(3, 16)))
              + rnd.choice(["", "?", ".", " ,"]) for _ in range(4000)}  # fmt: skip
    corpus |= {"".join(rnd.choice("abc, \n?") for _ in range(rnd.randint(5, 60))) + " before "
               + "".join(rnd.choice("abc, \n") for _ in range(rnd.randint(1, 30))) for _ in range(500)}  # fmt: skip
    for f in ["tests/test_split_endpoints.py", "tests/test_proof_rv2_review.py"]:
        corpus |= set(re.findall(r'"([^"\n]{15,})"', (ROOT / f).read_text()))
    key = lambda s: None if s is None else (s.pattern, list(s.parts))
    diffs = []
    for q in sorted(corpus):
        a, b = old.split_endpoints(q), split_endpoints(q)
        if key(a) == key(b):
            continue
        if a is None or all(p.strip() for p in a.parts):
            diffs.append((q, key(a), key(b)))  # anything but the F2 cases (an old empty part)
    assert diffs == []
