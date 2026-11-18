"""B-W1 baselines (benchmarks-v2 B-W1): resolvers run offline over the same mention stream.

- exact only: link iff the surface equals one canonical name (case-insensitive); else new.
- exact + alias: also the aliases the oracle has taught so far.
- v1 rule: link to the best entity iff max(token_set_ratio, cosine) >= 0.92 (the v1 resolver that
  shows the A8 failure: "Ana" vs "Ana Silva" has token_set_ratio 100). `token_set_ratio` is the
  rapidfuzz definition on difflib's ratio (an approximation of the Indel ratio); cosine is the
  stub `HashEmbedder` on the surfaces.

Metrics are the B-W1 ones on the same scored mentions (never gated): wrong-merge and split rate.
A baseline never asks; on a truly ambiguous mention it links to the first best candidate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from difflib import SequenceMatcher

from bench.synth.replay import OpLog
from bench.synth.score import SCORED_STYLES, rate
from mem.common import norm
from mem.testing import HashEmbedder

_EMB = HashEmbedder(256)


def token_set_ratio(a: str, b: str) -> float:
    ta, tb = set(norm(a).split()), set(norm(b).split())
    inter = " ".join(sorted(ta & tb))
    a2 = (inter + " " + " ".join(sorted(ta - tb))).strip()
    b2 = (inter + " " + " ".join(sorted(tb - ta))).strip()
    r = SequenceMatcher
    cands = [r(None, a2, b2).ratio()]
    if inter:
        cands += [r(None, inter, a2).ratio(), r(None, inter, b2).ratio()]
    return 100 * max(cands)


def _cos(a: str, b: str) -> float:
    return sum(x * y for x, y in zip(_EMB.embed_query(a), _EMB.embed_query(b), strict=True))


Resolver = Callable[[str, list[tuple[str, str, str | None]]], str | None]


def exact_only(surface: str, known: list[tuple[str, str, str | None]]) -> str | None:
    hits = [k for k, name, _ in known if norm(name) == norm(surface)]
    return hits[0] if hits else None


def exact_alias(surface: str, known: list[tuple[str, str, str | None]]) -> str | None:
    hits = [k for k, name, alias in known
            if norm(name) == norm(surface) or (alias and norm(alias) == norm(surface))]  # fmt: skip
    return hits[0] if hits else None


def v1_rule(surface: str, known: list[tuple[str, str, str | None]]) -> str | None:
    best, who = 0.0, None
    for k, name, alias in known:
        for cand in (name, alias):
            if not cand:
                continue
            s = max(token_set_ratio(surface, cand) / 100, _cos(surface, cand))
            if s > best:
                best, who = s, k
    return who if best >= 0.92 else None


BASELINES: dict[str, Resolver] = {"exact_only": exact_only, "exact_alias": exact_alias,
                                  "v1_rule": v1_rule}  # fmt: skip


def baseline_metrics(logs: list[OpLog], eid: dict[str, int]) -> dict[str, float]:
    """Replays the scored mentions of `logs` through each baseline resolver."""
    out: dict[str, float] = {}
    for name, resolve in BASELINES.items():
        known: list[tuple[str, str, str | None]] = []
        taught: dict[str, str] = {}
        wrong = split = n = 0
        for lg in logs:
            for m, d in zip(lg.op["mentions"], lg.decisions, strict=False):
                ent, style = m["entity"], d.style
                if style == "new":
                    known.append((ent, m["surface"], None))
                    continue
                if style == "alias_intro":
                    taught[ent] = m["surface"]
                    known[:] = [(k, nm, taught.get(k)) for k, nm, _ in known]
                    continue
                if style not in SCORED_STYLES:
                    continue
                n += 1
                got = resolve(m["surface"], known)
                if got is None:
                    split += 1
                elif got != ent:
                    wrong += 1
        out[f"baseline_{name}_wrong_merge"] = rate(wrong, n)
        out[f"baseline_{name}_split"] = rate(split, n)
    return {k: (v if not math.isnan(v) else math.nan) for k, v in out.items()}
