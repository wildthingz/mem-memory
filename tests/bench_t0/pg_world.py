"""Shared builders for the PG-v1 tests: a real imported world (stand-in importer, stub models) and
a full set of 120 accepted questions over it."""

from __future__ import annotations

import hashlib
from pathlib import Path

from bench.adapters.p4_ours import OursVaultBackend
from bench.gold import pg
from mem.gate import Gate
from mem.testing import HashEmbedder, OverlapReranker
from tests.bench_t0 import fake_vault, vault_world

DATE = "2026-10-24"


def open_gate(cfg) -> Gate:
    """Stub-model gate: p = sigma(z), keep p >= 0.5, no gap rule (tests/helpers.py form)."""
    return Gate(1.0, 0.0, 0.5, 0.5, 0.0, fit_id="t0-stub", fingerprint_hash=cfg.fp_all())


def backend(root: Path, *, gate: bool = True, stores=("personal", "work")) -> OursVaultBackend:
    be = OursVaultBackend(root, DATE, models="hash", loaded=(HashEmbedder(256), OverlapReranker()),
                          importer=fake_vault.run_import, stores=stores)  # fmt: skip
    if gate:
        be.gate = open_gate(be.cfg)
    return be


def imported_world(tmp: Path) -> tuple[OursVaultBackend, Path]:
    vault = vault_world.make_world(tmp / "vault")
    be = backend(tmp / "stores")
    be.import_vault(vault, routes=vault_world.ROUTES)
    return be, vault


def pg_snapshot(vault: Path, tmp: Path, name: str = "pgvault.tar") -> tuple[Path, str, Path]:
    """The PG snapshot of `vault` (routes of the test world), extracted: (tarball, sha256, root)."""
    from bench.datasets import vault as vault_ds

    tar = tmp / name
    sha, _, _ = vault_ds.make_pg_snapshot(vault, tar, vault_world.ROUTES)
    return tar, sha, vault_ds.extract(tar, tmp / (name + ".x"))


def topic_of(path: str) -> str:
    """The unique topic word of a generated note (see vault_world.note_text)."""
    stem = Path(path).stem  # "Work note 03"
    folder, _, num = stem.lower().partition(" note ")
    i = int(num)
    return f"{vault_world.WORDS[i % len(vault_world.WORDS)]}{folder[:2]}{i}"


def build_questions(be: OursVaultBackend) -> list[pg.Question]:
    """120 questions with the exact quotas over the generated world. Answerable ones ask about the
    note's budget in words that are not in the chunk; no-answer ones ask for things nobody wrote."""
    paths = [
        p
        for p in be.index.paths()
        if "note" in p.lower() and be.index.store_of_path[p] == "personal"
    ]  # PG-v1 holds no work-routed note
    paths.sort(key=lambda p: hashlib.sha1(p.encode()).hexdigest())
    qs: list[pg.Question] = []
    n = 0
    for cat, total in pg.TOTAL.items():
        for k in range(total):
            n += 1
            qid = f"pg1-t-{n:03d}"
            path = paths[(n * 7) % len(paths)]
            store = be.index.store_of_path[path]
            facts = next(
                c
                for c in be.index.of_path(path)
                if c.chunk_key.startswith(be.index.of_path(path)[0].heading_path.split(" > ")[0])
                and "Facts" in c.chunk_key
            )
            topic = topic_of(path)
            if cat in pg.NO_ANSWER:
                qs.append(
                    pg.Question(
                        qid,
                        cat,
                        f"Which colour is the {topic}{n} mascot painted?",
                        store=store,
                        source=[path],
                        reworded=True,
                    )
                )
                continue
            gold = {facts.gold_id: 2}  # judged by the human (accepted gold of a drafted question)
            if cat in pg.PAIR_ONLY:
                other = next(
                    c
                    for p in paths
                    if p != path and be.index.store_of_path[p] == store
                    for c in be.index.of_path(p)
                    if "Facts" in c.chunk_key
                )
                gold[other.gold_id] = 2
            qs.append(pg.Question(qid, cat, f"How much money is set aside for the {topic} desk?",
                                  answer=str(1000), gold=gold, store=store, source=[path],
                                  judges=dict.fromkeys(gold, "human"),
                                  reworded=n % 2 == 0))  # fmt: skip
    return qs


class ScriptedJudge:
    """Plays Claude as the judge: a grade per passage from a fixed rule, one call per pair.
    `replies` overrides by a substring of the prompt (e.g. a garbage reply for one chunk)."""

    model = "claude-opus-5-5"

    def __init__(self, rule=None, replies=None) -> None:
        self.rule = rule or self._default

        self.replies = replies or {}
        self.calls: list[str] = []

    @staticmethod
    def _default(prompt: str) -> int:
        if "(no answer expected" in prompt:  # a no-answer question: nothing answers it
            return 0
        return int(hashlib.sha1(prompt.encode()).hexdigest(), 16) % 3

    def create(self, *, system, messages, tools, max_tokens):
        from bench import llm

        prompt = messages[0]["content"]
        self.calls.append(prompt)
        text = next((r for k, r in self.replies.items() if k in prompt), None)
        if text is None:
            text = f'{{"rel": {self.rule(prompt)}}}'
        return llm.Reply(
            [{"type": "text", "text": text}], "end_turn", {"input_tokens": 600, "output_tokens": 40}
        )
