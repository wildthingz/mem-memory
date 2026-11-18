"""LoCoMo backend for the production pipeline (MEM2-196): the LME adapter's store, gate and models,
one store per conversation instead of per question.

`open_conversation(conv)`: a fresh store at the conversation's last session date plus a day, every
session written with `mem.write.ingest_session` (its own date, `source_ref locomo:<sample>:D<N>`),
the gate written to the store; `.recall(query)` is the production `mem.read.recall` at that
date, as B-E1's mem arm reads it: recall's text plus the full text of every delivered row
(`lme_ours.delivered_bodies`, `e1.expanded_context`; the index alone has no chunk text). A row
`get` cut is recorded in `backend.truncated_rows` (sample id -> ids), never silent. The models
are the bench's own pinned `mem serve` (D5), as in `lme_ours`.
"""

from __future__ import annotations

from bench.adapters.lme_ours import OursLmeBackend, chunk_refs, delivered_bodies
from bench.datasets.locomo import Conversation
from bench.e1 import expanded_context
from bench.suites.base import BackendUnavailable


class OpenConversation:
    def __init__(self, backend: OursLocomoBackend, conv: Conversation) -> None:
        from mem import write
        from mem.gate import write_gate

        self.as_of, self.sample_id, self._backend = conv.as_of, conv.sample_id, backend
        self.dates = {s.id: s.date for s in conv.sessions}
        self._db, self.store = backend._open(f"locomo-{conv.sample_id}", self.as_of)
        try:
            write_gate(self.store.conn, backend.gate)
            self.store.gate = backend.gate
            for s in conv.sessions:
                write.ingest_session(self.store, source_ref=f"locomo:{conv.sample_id}:{s.id}",
                                     date=s.date, rounds=s.rounds)  # fmt: skip
            self.refs = chunk_refs(self.store)
        except BaseException:
            self.close()
            raise

    def recall(self, query: str) -> str:
        from mem import read

        res = read.recall(self.store, query=query, as_of=self.as_of)
        ids = [r["id"] for r in res.structured["rows"]]
        bodies, cut = delivered_bodies(self.store, ids, self.refs, self.dates)
        if cut:
            seen = self._backend.truncated_rows.setdefault(self.sample_id, [])
            seen += [c for c in cut if c not in seen]
        return expanded_context(res.text, bodies)

    def close(self) -> None:
        self.store.close()
        self._db.unlink(missing_ok=True)


class OursLocomoBackend(OursLmeBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.truncated_rows: dict[str, list[str]] = {}

    def open_conversation(self, conv: Conversation) -> OpenConversation:
        if self.gate is None:
            raise BackendUnavailable("no gate: pass --gate gate.json (from mem calibrate --out)")
        return OpenConversation(self, conv)
