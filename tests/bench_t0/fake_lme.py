"""A lexical LongMemEval backend for T0 tests: a test double for the P2 pipeline, not a result.

Rounds are scored by the fraction of the question's content words they contain; `never_abstain`
turns the gate off and returns the top 8 (the "fixed top 8" baseline).
"""

from __future__ import annotations

import re

from bench.datasets.lme import Haystack, ref
from bench.lme_eval import Retrieved

STOP = {"what", "is", "the", "in", "my", "a", "of", "to", "and", "do", "i"}


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower())) - STOP


class FakeLme:
    def __init__(self, *, tau: float = 0.5, never_abstain: bool = False, tag: str = "calibrated"):
        self.tau, self.never_abstain, self.tag = tau, never_abstain, tag

    def retrieve(self, question: dict, haystack: Haystack, *, want_legs: bool) -> Retrieved:
        q = _words(question["question"])
        scored = []
        for sid, date, rounds in haystack.sessions:
            for r in rounds:
                text = r.user + " " + r.assistant
                lex = len(q & _words(text)) / max(len(q), 1)
                scored.append((lex, -r.index, sid, date, ref(sid, r.index), text))
        scored.sort(key=lambda t: (-t[0], t[2], -t[1]))
        r_list = [s[4] for s in scored[:50]]
        shown = scored[:8] if self.never_abstain else [s for s in scored[:5] if s[0] >= self.tau]
        best = f"{scored[0][0] if scored else 0.0:.2f}".removeprefix("0")
        lines = [f"c{i} · chat_session · {s[3][:10]} · {' '.join(s[5].split()[:6])} · {s[0]:.4f}"
                 for i, s in enumerate(shown, start=1)]  # fmt: skip
        text = (f"{len(shown)} results (of 30 checked) · personal · {self.tag}\n" + "\n".join(lines)
                if shown else f"no relevant memories (best {best}) · personal")  # fmt: skip
        legs = {}
        if want_legs:
            legs = {"bm25": r_list[:], "dense": list(reversed(r_list))[:0] or r_list[3:] + r_list[:3],
                    "rrf": r_list[:]}  # fmt: skip
        return Retrieved(F=r_list, R=r_list, S=[s[4] for s in shown], text=text, latency_ms=1.0,
                         p_scores=[s[0] for s in scored[:5]], legs=legs)  # fmt: skip
