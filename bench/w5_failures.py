"""B-W5 failure analysis (offline, no network): which `about` the writer missed and which conflict
action it took instead of `supersedes`, grouped, from a `bench w5-run` file.

Mirrors the two `score_run` metrics (`about_fill`, `conflict_action`) so every counted miss is one
the scorer counts. Only gold structure (entity types, relations) is reported, never dialogue text,
so the report can be pasted back without leaking dev wording into the skill.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from bench.suites import b_w5
from bench.synth.align import find, tokens


def _remember_inputs(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """memory id (`m12`) -> the `remember` input that created it."""
    out: dict[str, dict[str, Any]] = {}
    for r in run["dialogues"]:
        for c in r["calls"]:
            mid = c["structured"].get("id") if c["tool"] == "remember" else None
            if mid:
                out[mid] = c["input"]
    return out


def about_misses(run: dict[str, Any], seed: int, size: str) -> list[dict[str, Any]]:
    syn, _ = b_w5.dialogues_for(seed, size)
    snap = run["snapshot"]
    ekeys = b_w5._entity_keys(snap, syn)
    forms = {e["key"]: [tokens(e["name"])] + ([tokens(e["alias"])] if e["alias"] else [])
             for e in syn.expected["entities"]}  # fmt: skip
    etype = {e["key"]: e["type"] for e in syn.expected["entities"]}
    inputs = _remember_inputs(run)
    out = []
    for m in snap["memories"]:
        toks = tokens(m["text"])
        named = sorted(k for k, fs in forms.items() if any(find(toks, f) for f in fs))
        if not named:
            continue
        about = {ekeys[e] for e in m["about"] if e in ekeys}
        if about & set(named):
            continue
        raw = inputs.get(f"m{m['id']}", {}).get("about")
        cause = "about_empty" if not m["about"] else (
            "about_other_gold_entity" if about else "about_not_a_gold_entity")  # fmt: skip
        out.append({"memory": f"m{m['id']}", "cause": cause, "named_types": [etype[k] for k in named],
                    "about_types": sorted(etype[k] for k in about), "n_named": len(named),
                    "about_input": raw})  # fmt: skip
    return out


def conflict_failures(run: dict[str, Any], seed: int, size: str) -> list[dict[str, Any]]:
    """Surfaced true conflicts that `score_run` does not count as acted, with what the writer did."""
    _syn, dialogues = b_w5.dialogues_for(seed, size)
    _, details = b_w5.score_run(run, seed, size)
    fact_mems: dict[tuple[int, int], set[str]] = {}
    for mid, keys in details["aligned"].items():
        for key in keys:
            fact_mems.setdefault(tuple(key), set()).add(f"m{mid}")
    calls_by_d = {r["dialogue"]: r["calls"] for r in run["dialogues"]}
    out = []
    for dl in dialogues:
        calls = calls_by_d.get(dl.index, [])
        for i, f in enumerate(dl.facts):
            if not f["conflict_with"]:
                continue
            old = fact_mems.get(tuple(f["conflict_with"]), set())
            new = fact_mems.get((dl.index, i), set())
            hit = None
            for c in calls:
                if c["tool"] == "remember" and c["structured"].get("id") in new:
                    cp = {x["id"] if isinstance(x, dict) else str(x).split()[0]
                          for x in c["structured"].get("possible_conflicts") or []}  # fmt: skip
                    if cp & old:
                        hit = (c["structured"]["id"], cp & old, c["input"].get("supersedes"))
            if not hit:
                continue
            nid, olds, inline = hit
            rels = sorted({c["input"].get("rel") for c in calls if c["tool"] == "link"
                           and nid in (c["input"].get("src"), c["input"].get("dst"))
                           and {c["input"].get("src"), c["input"].get("dst")} & olds
                           and not c["error"]})  # fmt: skip
            acted = inline in olds or "supersedes" in rels
            if acted:
                continue
            action = ("inline_supersedes_other_id" if inline else
                      "+".join(rels) if rels else "no_call")  # fmt: skip
            out.append({"dialogue": dl.index, "fact": i, "relation": f["relation"],
                        "action": action, "n_candidates_aligned": len(olds)})  # fmt: skip
    return out


def analyse(run: dict[str, Any], seed: int, size: str) -> dict[str, Any]:
    metrics, _ = b_w5.score_run(run, seed, size)
    ab = about_misses(run, seed, size)
    cf = conflict_failures(run, seed, size)
    return {
        "about_fill": metrics["about_fill"], "conflict_action": metrics["conflict_action"],
        "conflicts_surfaced": metrics["conflicts_surfaced"],
        "about_miss_n": len(ab),
        "about_miss_by_cause": dict(Counter(a["cause"] for a in ab)),
        "about_miss_by_named_types": dict(Counter("+".join(sorted(a["named_types"])) for a in ab)),
        "about_miss_by_input": dict(Counter(_input_shape(a["about_input"]) for a in ab)),
        "about_misses": ab,
        "conflict_fail_n": len(cf),
        "conflict_fail_by_action": dict(Counter(c["action"] for c in cf)),
        "conflict_fail_by_relation": dict(Counter(c["relation"] for c in cf)),
        "conflict_failures": cf,
    }  # fmt: skip


def _input_shape(about: Any) -> str:
    """Form of the writer's `about` argument, with no names: none | new | id | name (| mixed)."""
    if not about:
        return "none"
    items = about if isinstance(about, list) else [about]
    shapes = {"new" if str(a).startswith("new:") else "id" if str(a).startswith("#")
              else "arrow" if "→" in str(a) else "name" for a in items}  # fmt: skip
    return "+".join(sorted(shapes))
