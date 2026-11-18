"""SYN generator (benchmarks-v2 §3.2): `generate(seed, size) -> Synth`.

Deterministic: the only randomness is `random.Random(seed)`; no LLM, no clock, no I/O. Outputs:

- `ops.jsonl`            scenario ops, in order. The oracle driver (`bench.synth.replay`) turns each
                         into tool calls (mention styles -> `about` strings, `ref` -> a server id).
- `expected_state.json`  the entities, every slot's gold chain, every memory's final status and
                         interval (UTC), the op counts. B-W3 T0 compares the replayed graph to it.
- `queries.jsonl`        the B-R5 probes (current-value, as-of) and the B-R2 (ii) missing-slot probes.

Interpretation choices where the spec leaves room (each is also listed in bench/README.md):
- `update` "cue": half of the updates start with a change cue (`UPDATE_PREFIXES`); every text has
  its relation cue regardless.
- Update, late, future and same-instant sub-types are drawn from per-seed quotas
  (round(share * n_updates), at least 1 for late and same_instant), so a small seed still has them.
- Quotas of op kinds are consumed in random order, but a kind with no feasible target at that
  point (e.g. an update before any slot exists) is replaced by `new_fact`; `expected_state.counts`
  reports what was actually generated.
- First mentions are `new:<type>:<name>`. The first use of an alias is an introduction op written
  `alias → #e<id>` (not scored); unique first names are never used alone (only colliding ones, as
  `ambiguous_mention` ops that the server must ask about).
- Every slot memory carries a date-only `as_of`; its gold valid_from is local midnight of that
  date in `TZ`, converted to UTC.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from bench.synth import data as D

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def lev_within(a: str, b: str, k: int) -> bool:
    """Levenshtein distance(a, b) <= k (banded DP)."""
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [k + 1] * len(b)
        lo, hi = max(1, i - k), min(len(b), i + k)
        for j in range(lo, hi + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1]))
        if min(cur[lo - 1 : hi + 1]) > k:
            return False
        prev = cur
    return prev[len(b)] <= k


def local_midnight_utc(day: str) -> str:
    local = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=ZoneInfo(D.TZ))
    return local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Entity:
    key: str
    type: str
    name: str
    alias: str | None = None
    group: str | None = None  # collision group id (first-name or cross-kind)
    first: str | None = None
    introduced: bool = False
    alias_taught: bool = False


@dataclass
class Member:
    op: int
    value: str
    text: str
    as_of: str  # date-only
    vf: str  # UTC
    now: str
    status: str = "active"  # active | retracted


@dataclass
class Slot:
    key: str
    entity: str
    rel: str
    members: list[Member] = field(default_factory=list)
    dups: list[int] = field(default_factory=list)  # ops of paraphrase duplicates

    def active(self) -> list[Member]:
        return sorted((m for m in self.members if m.status == "active"),
                      key=lambda m: (m.vf, m.now, m.op))  # fmt: skip

    def intervals(self, multi: bool) -> dict[int, tuple[str, str | None]]:
        act = self.active()
        if multi:
            return {m.op: (m.vf, None) for m in act}
        return {m.op: (m.vf, act[i + 1].vf if i + 1 < len(act) else None)
                for i, m in enumerate(act)}  # fmt: skip

    def open_members(self, multi: bool) -> list[Member]:
        act = self.active()
        return act if multi else act[-1:]


@dataclass
class Synth:
    seed: int
    size: str
    ops: list[dict[str, Any]]
    expected: dict[str, Any]
    queries: list[dict[str, Any]]

    def files(self) -> dict[str, bytes]:
        def lines(rows: list[dict[str, Any]]) -> bytes:
            return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                           for r in rows).encode()  # fmt: skip

        return {"ops.jsonl": lines(self.ops),
                "expected_state.json": (json.dumps(self.expected, sort_keys=True, indent=1,
                                                   ensure_ascii=False) + "\n").encode(),
                "queries.jsonl": lines(self.queries)}  # fmt: skip

    def sha256(self) -> dict[str, str]:
        return {n: hashlib.sha256(b).hexdigest() for n, b in self.files().items()}

    def write(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        for name, blob in self.files().items():
            (out / name).write_bytes(blob)


def _forbidden_tokens() -> set[str]:
    out: set[str] = set()
    for r in D.RELATIONS.values():
        for c in r.cues:
            out |= set(tokens(c))
    for n in D.NEGATIONS:
        out |= set(tokens(n))
    for f in (*D.FILLER, *D.UPDATE_PREFIXES):
        out |= set(tokens(f))
    out |= {t for r in D.RELATIONS.values() for t in tokens(r.question)}
    out |= {"project", "org", "person", "place"}
    for vals in D.VALUES.values():
        for v in vals:
            out |= set(tokens(v))
    return out


class _Names:
    """Canonical names and aliases; no two within Levenshtein distance 2 (except the deliberate
    cross-kind pairs, which are identical)."""

    def __init__(self, rng: random.Random) -> None:
        self.rng, self.strings, self.tok = rng, [], set()

    def _ok(self, s: str) -> bool:
        n = " ".join(tokens(s))
        return all(not lev_within(n, o, 2) for o in self.strings)

    def add(self, s: str) -> None:
        self.strings.append(" ".join(tokens(s)))
        self.tok |= set(tokens(s))

    def invent(self, parts: int = 3) -> str:
        return "".join(self.rng.choice(D.SYLLABLES) for _ in range(parts)).capitalize()

    def draw(self, pool: list[str], make: Any, avoid: set[str]) -> str:
        while True:
            cand = pool.pop() if pool else make()
            if avoid & set(tokens(cand)):
                continue
            if self._ok(cand):
                self.add(cand)
                return cand


def build_world(rng: random.Random, spec: dict[str, int]) -> list[Entity]:
    names = _Names(rng)
    avoid = _forbidden_tokens()
    firsts, lasts, things = list(D.FIRST_NAMES), list(D.LAST_NAMES), list(D.THING_NAMES)
    for pool in (firsts, lasts, things):
        rng.shuffle(pool)
    ents: list[Entity] = []

    def person(first: str, group: str | None) -> Entity:
        last = names.draw(list(lasts), lambda: names.invent(3), avoid)
        full = ""
        for _ in range(1000):
            full = f"{first} {last}"
            if names._ok(full) and not (avoid & set(tokens(full))):
                break
            last = names.invent(3)
        else:  # pragma: no cover
            raise RuntimeError("cannot place a person name")
        names.strings.pop()  # `draw` registered the bare surname; register the full name instead
        names.add(full)
        e = Entity(f"P{len(ents)}", "person", full, group=group, first=first)
        ents.append(e)
        return e

    n_first = spec["first_collisions"]
    coll_firsts = [firsts.pop() for _ in range(n_first)]
    for g, first in enumerate(coll_firsts):
        for _ in range(2):
            person(first, f"F{g}")
    taken = set(coll_firsts)
    while sum(e.type == "person" for e in ents) < spec["people"]:
        first = firsts.pop() if firsts else names.invent(3)
        if first in taken or avoid & set(tokens(first)):
            continue
        taken.add(first)
        person(first, None)

    def thing(etype: str, prefix: str, group: str | None = None, name: str | None = None) -> None:
        if name is None:
            name = names.draw(things, lambda: names.invent(3), avoid)
        else:
            names.add(name)
        ents.append(Entity(f"{prefix}{sum(e.type == etype for e in ents)}", etype, name,
                           group=group))  # fmt: skip

    for g in range(spec["cross_kind"]):
        shared = names.draw(things, lambda: names.invent(3), avoid)
        thing("project", "J", f"X{g}", shared)
        thing("org", "O", f"X{g}", shared)
    for etype, prefix, n in (
        ("project", "J", spec["projects"]),
        ("org", "O", spec["orgs"]),
        ("place", "L", spec["places"]),
    ):
        while sum(e.type == etype for e in ents) < n:
            thing(etype, prefix)
    # aliases: not for the cross-kind pairs
    nicks = list(D.NICKNAMES)
    rng.shuffle(nicks)
    for e in ents:
        if e.group and e.group.startswith("X"):
            continue
        if rng.random() < (0.5 if e.type == "person" else 0.3):
            pool = nicks if e.type == "person" else []
            e.alias = names.draw(pool, lambda: names.invent(3), avoid)
    return ents


class _Values:
    """World-unique slot values: lists first (no replacement), then invented words."""

    def __init__(self, rng: random.Random, taken: set[str]) -> None:
        self.rng = rng
        self.lists = {k: rng.sample(list(v), len(v)) for k, v in D.VALUES.items()}
        self.list_tokens = {t for vs in D.VALUES.values() for v in vs for t in tokens(v)}
        self.forbidden = (_forbidden_tokens() - self.list_tokens) | taken
        self.used: set[str] = set()

    def draw(self, key: str) -> str:
        pool = self.lists[key]
        while True:
            cand = (
                pool.pop()
                if pool
                else "".join(self.rng.choice(D.SYLLABLES) for _ in range(3)).capitalize()
            )
            toks = tokens(cand)
            if len(toks) != 1 or toks[0] in self.used:
                continue
            if toks[0] in self.forbidden:
                continue
            if cand not in D.VALUES[key] and toks[0] in self.list_tokens:
                continue
            self.used.add(toks[0])
            return cand


def _typo(rng: random.Random, name: str) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    idx = [i for i in range(1, len(name) - 1) if name[i].isalpha()]
    i = rng.choice(idx)
    kind = rng.choice(("sub", "del", "ins"))
    if kind == "sub":
        c = rng.choice([x for x in letters if x != name[i].lower()])
        return name[:i] + c + name[i + 1 :]
    if kind == "del":
        return name[:i] + name[i + 1 :]
    return name[:i] + rng.choice(letters) + name[i:]


def _quota(n: int) -> dict[str, int]:
    q = {k: int(s * n) for k, s in D.SHARES.items() if k != "new_fact"}
    q["new_fact"] = n - sum(q.values())
    return q


def generate(seed: int, size: str = "S") -> Synth:
    spec = D.SIZES[size]
    rng = random.Random(seed)
    ents = build_world(rng, spec)
    by_key = {e.key: e for e in ents}
    name_tokens = {t for e in ents for t in (*tokens(e.name), *tokens(e.alias or ""))}
    vals = _Values(rng, name_tokens)
    n_ops = spec["ops"]

    # op clock: 730 days, sorted days, five minutes apart within a day
    start = datetime.strptime(D.WORLD_START, "%Y-%m-%d").replace(tzinfo=UTC)
    days = sorted(rng.randrange(D.WORLD_DAYS) for _ in range(n_ops))
    seen_per_day: dict[int, int] = {}
    nows = []
    for d in days:
        k = seen_per_day.get(d, 0)
        seen_per_day[d] = k + 1
        nows.append(_iso(start + timedelta(days=d, hours=8, minutes=5 * k)))

    quota = _quota(n_ops)
    n_upd = quota["update"]
    sub_quota = {k: max(1, round(s * n_upd)) if k in ("late", "same_instant") else round(s * n_upd)
                 for k, s in D.UPDATE_SHARES.items()}  # fmt: skip
    slots: dict[str, Slot] = {}
    ops: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    sub_counts: dict[str, int] = {}
    cue_phase = rng.randrange(2)
    memories: dict[int, dict[str, Any]] = {}

    def slot_of(ent: str, rel: str) -> Slot:
        key = f"{ent}|{rel}"
        return slots.setdefault(key, Slot(key, ent, rel))

    def render(ent_surface: str, rel: str, cue: str, value: str, filler: str,
               prefix: str = "") -> str:  # fmt: skip
        sentence = f"{ent_surface} {cue} {value}."
        if prefix:
            sentence = f"{prefix} {sentence}"
        return f"{sentence} {filler}"

    def pick_text(
        ent: Entity,
        rel: str,
        value: str,
        prefix: str = "",
        avoid_texts: set[str] | None = None,
        avoid_cue: str | None = None,
    ) -> str:
        r = D.RELATIONS[rel]
        for _ in range(100):
            cue = rng.choice([c for c in r.cues if c != avoid_cue] or list(r.cues))
            text = render(ent.name, rel, cue, value, rng.choice(D.FILLER), prefix)
            if not avoid_texts or text not in avoid_texts:
                return text
        raise RuntimeError("cannot vary text")  # pragma: no cover

    def mention(ent: Entity, *, force: str | None = None) -> dict[str, Any]:
        style = force
        if style is None:
            partner = [o for o in ents if o.group == ent.group and o is not ent and o.introduced]
            if not ent.introduced:
                style = "new"
            elif ent.group and ent.group.startswith("X") and partner:
                style = "id"
            elif ent.alias and not ent.alias_taught and rng.random() < 0.35:
                style = "alias_intro"
            elif ent.alias and ent.alias_taught and rng.random() < 0.30:
                style = "alias"
            elif rng.random() < D.TYPO_RATE and len(ent.name) >= 5 and not ent.group:
                style = "typo"
            else:
                style = "canonical"
        surface = {
            "new": ent.name,
            "id": ent.name,
            "canonical": ent.name,
            "alias_intro": ent.alias,
            "alias": ent.alias,
            "typo": _typo(rng, ent.name) if style == "typo" else None,
        }.get(style, ent.name)
        if style == "new":
            ent.introduced = True
        if style == "alias_intro":
            ent.alias_taught = True
        return {"entity": ent.key, "style": style, "surface": surface, "type": ent.type}

    def open_targets() -> list[tuple[Slot, Member]]:
        out = []
        for key in sorted(slots):
            s = slots[key]
            multi = D.RELATIONS[s.rel].multi
            out += [(s, m) for m in s.open_members(multi)]
        return out

    def free_slots(ent: Entity) -> list[str]:
        return [
            r.name
            for r in D.RELATIONS.values()
            if r.subject == ent.type
            and not any(m.status == "active" for m in slot_of_exists(ent.key, r.name))
        ]

    def slot_of_exists(ent: str, rel: str) -> list[Member]:
        s = slots.get(f"{ent}|{rel}")
        return s.members if s else []

    def emit(i: int, kind: str, args: dict[str, Any], mentions: list[dict[str, Any]],
             gold: dict[str, Any], tool: str = "remember") -> None:  # fmt: skip
        counts[kind] = counts.get(kind, 0) + 1
        ops.append({"i": i, "kind": kind, "now": nows[i], "tool": tool, "args": args,
                    "mentions": mentions, "gold": gold})  # fmt: skip

    def day_of(i: int) -> str:
        return nows[i][:10]

    def add_member(i: int, slot: Slot, ent: Entity, value: str, text: str, as_of: str) -> Member:
        m = Member(i, value, text, as_of, local_midnight_utc(as_of), nows[i])
        slot.members.append(m)
        memories[i] = {"op": i, "slot": slot.key, "value": value, "entity": ent.key,
                       "rel": slot.rel, "text": text}  # fmt: skip
        return m

    # ---- per-kind builders: return True when an op was emitted -----------------------------
    def do_new_fact(i: int, kind: str = "new_fact") -> bool:
        fresh = sorted((e for e in ents if not e.introduced), key=lambda e: e.key)
        used = sorted((e for e in ents if e.introduced and free_slots(e)), key=lambda e: e.key)
        if not fresh and not used:
            return False
        pools = [p for p in (fresh if fresh and (not used or rng.random() < 0.5) else used,
                             used, fresh) if p]  # fmt: skip
        ent = rng.choice(pools[0])
        rel = rng.choice(free_slots(ent))
        value = vals.draw(D.RELATIONS[rel].values)
        text = pick_text(ent, rel, value)
        men = mention(ent)
        as_of = day_of(i)
        slot = slot_of(ent.key, rel)
        add_member(i, slot, ent, value, text, as_of)
        emit(i, kind, {"text": text, "type": D.RELATIONS[rel].mem_type, "as_of": as_of}, [men],
             {"slot": slot.key, "value": value})  # fmt: skip
        return True

    def current_values(s: Slot) -> set[str]:
        """Values of the latest memory by valid time and by op order; a value_returns op must repeat
        neither (it would be an exact duplicate of the memory it supersedes)."""
        act = s.active()
        return {act[-1].value, max(act, key=lambda m: m.op).value}

    def update_targets(sub: str, i: int) -> list[Slot]:
        out = []
        today = day_of(i)
        for key in sorted(slots):
            s = slots[key]
            r = D.RELATIONS[s.rel]
            act = s.active()
            if r.multi or not act:
                continue
            latest = act[-1]
            latest_day = latest.as_of
            distinct = len({m.value for m in act})
            if sub == "value_returns":
                if (
                    len(act) >= 2
                    and len(act) < 5
                    and latest_day < today
                    and any(m.value not in current_values(s) for m in act[:-1])
                ):  # an earlier value differs from both current ones
                    out.append(s)
            elif distinct >= 4:
                continue
            elif sub == "normal" and latest_day < today or sub == "future":
                out.append(s)
            elif sub == "late":
                days_taken = {m.as_of for m in act}
                if any(_day_before(latest_day, k) not in days_taken for k in range(1, 40)):
                    out.append(s)
            elif sub == "same_instant" and sum(m.as_of == latest_day for m in act) == 1:
                out.append(s)
        return out

    def do_update(i: int) -> bool:
        subs = [(k, sub_quota[k] - sub_counts.get(k, 0)) for k in D.UPDATE_SHARES]
        order = [k for k, left in sorted(subs) if left > 0]
        rng.shuffle(order)
        normal_left = max(0, n_upd - sum(sub_counts.values()))
        cand_order = [*order[:1], *(["normal"] if normal_left else []), *order[1:], "normal"]
        for sub in dict.fromkeys(cand_order):
            targets = update_targets(sub, i)
            if not targets:
                continue
            slot = rng.choice(targets)
            ent = by_key[slot.entity]
            act = slot.active()
            # The counterpart is the member valid at the new as_of: the latest by valid time, which
            # the new value supersedes. Not the member written last: after a `late` update that is
            # the late member, already closed in the chain (MEM2-156; SYN_VERSION 3).
            latest = act[-1]
            has_cue = (counts.get("update", 0) + cue_phase) % 2 == 0  # exactly half
            prefix = rng.choice(D.UPDATE_PREFIXES) if has_cue else ""
            returns_to = None
            if sub == "value_returns":
                cur = current_values(slot)
                earlier = [m for m in act[:-1] if m.value not in cur]
                orig = rng.choice(earlier)
                value, text, returns_to = orig.value, orig.text, orig.op
                as_of = day_of(i)
            else:
                value = vals.draw(D.RELATIONS[slot.rel].values)
                text = pick_text(ent, slot.rel, value, prefix)
                if sub == "future":
                    as_of = _day_after(max(day_of(i), latest.as_of), rng.randint(1, 60))
                elif sub == "late":
                    taken = {m.as_of for m in act}
                    gaps = [k for k in range(1, 40) if _day_before(latest.as_of, k) not in taken]
                    as_of = _day_before(latest.as_of, rng.choice(gaps))
                elif sub == "same_instant":
                    as_of = latest.as_of
                else:
                    as_of = day_of(i)
            men = mention(ent)
            add_member(i, slot, ent, value, text, as_of)
            sub_counts[sub] = sub_counts.get(sub, 0) + 1
            emit(i, "update", {"text": text, "type": D.RELATIONS[slot.rel].mem_type,
                               "as_of": as_of}, [men],
                 {"slot": slot.key, "value": value, "subtype": sub, "has_cue": has_cue,
                  "counterpart": latest.op, "returns_to": returns_to})  # fmt: skip
            return True
        return False

    def do_dup_exact(i: int) -> bool:
        cands = open_targets()
        if not cands:
            return False
        slot, m = rng.choice(cands)
        ent = by_key[slot.entity]
        men = mention(ent)
        emit(i, "dup_exact", {"text": m.text, "type": D.RELATIONS[slot.rel].mem_type,
                              "as_of": m.as_of}, [men],
             {"slot": slot.key, "value": m.value, "target": m.op})  # fmt: skip
        return True

    def do_dup_paraphrase(i: int) -> bool:
        cands = open_targets()
        if not cands:
            return False
        slot, m = rng.choice(cands)
        ent = by_key[slot.entity]
        all_texts = {x.text for s in slots.values() for x in s.members}
        orig_cue = next((c for c in D.RELATIONS[slot.rel].cues if f" {c} " in f" {m.text} "), None)
        text = pick_text(ent, slot.rel, m.value, "", all_texts, orig_cue)
        men = mention(ent)
        slot.dups.append(i)
        memories[i] = {"op": i, "slot": slot.key, "value": m.value, "entity": ent.key,
                       "rel": slot.rel, "text": text, "duplicate_of": m.op}  # fmt: skip
        emit(i, "dup_paraphrase", {"text": text, "type": D.RELATIONS[slot.rel].mem_type}, [men],
             {"slot": slot.key, "value": m.value, "target": m.op, "counterpart": m.op})  # fmt: skip
        return True

    def do_compatible(i: int) -> bool:
        cands = [s for k, s in sorted(slots.items()) if D.RELATIONS[s.rel].multi and s.active()]
        if not cands:
            return False
        slot = rng.choice(cands)
        ent = by_key[slot.entity]
        last_written = max(slot.active(), key=lambda m: m.op)
        value = vals.draw(D.RELATIONS[slot.rel].values)
        text = pick_text(ent, slot.rel, value)
        men = mention(ent)
        as_of = day_of(i)
        add_member(i, slot, ent, value, text, as_of)
        emit(i, "compatible_add", {"text": text, "type": D.RELATIONS[slot.rel].mem_type,
                                   "as_of": as_of}, [men],
             {"slot": slot.key, "value": value, "counterpart": last_written.op})  # fmt: skip
        return True

    def do_ambiguous(i: int) -> bool:
        groups: dict[str, list[Entity]] = {}
        for e in ents:
            if e.group and e.introduced:
                groups.setdefault(e.group, []).append(e)
        opts = [(g, es) for g, es in sorted(groups.items()) if len(es) >= 2
                and any(free_slots(e) for e in es)]  # fmt: skip
        if not opts:
            return False
        _, es = rng.choice(opts)
        ent = rng.choice([e for e in es if free_slots(e)])
        rel = rng.choice(free_slots(ent))
        surface = ent.first or ent.name
        value = vals.draw(D.RELATIONS[rel].values)
        r = D.RELATIONS[rel]
        text = render(surface, rel, rng.choice(r.cues), value, rng.choice(D.FILLER))
        as_of = day_of(i)
        slot = slot_of(ent.key, rel)
        add_member(i, slot, ent, value, text, as_of)
        men = {"entity": ent.key, "style": "ambiguous", "surface": surface, "type": ent.type}
        emit(i, "ambiguous_mention", {"text": text, "type": r.mem_type, "as_of": as_of}, [men],
             {"slot": slot.key, "value": value})  # fmt: skip
        return True

    def do_retract(i: int) -> bool:
        cands = []
        for key in sorted(slots):
            s = slots[key]
            multi = D.RELATIONS[s.rel].multi
            if multi:
                cands += [(s, m) for m in s.active()]
            elif s.active():
                cands.append((s, s.active()[-1]))
        if not cands:
            return False
        slot, m = rng.choice(cands)
        m.status = "retracted"
        emit(i, "retract", {"ref": m.op, "reason": "the user said this was wrong"}, [],
             {"slot": slot.key, "target": m.op}, tool="forget")  # fmt: skip
        return True

    builders = {
        "new_fact": do_new_fact,
        "update": do_update,
        "dup_exact": do_dup_exact,
        "dup_paraphrase": do_dup_paraphrase,
        "compatible_add": do_compatible,
        "ambiguous_mention": do_ambiguous,
        "retract": do_retract,
    }
    left = dict(quota)
    for i in range(n_ops):
        placed = False
        kinds = [k for k in sorted(left) if left[k] > 0]
        # early ops: the dependent kinds are tried less often so slots exist first
        weights = [left[k] * (1.0 if k == "new_fact" or i > n_ops * 0.15 else 0.2) for k in kinds]
        tried: list[str] = []
        while kinds and not placed:
            k = rng.choices(kinds, weights)[0]
            tried.append(k)
            if builders[k](i):
                left[k] -= 1
                placed = True
            else:
                idx = kinds.index(k)
                kinds.pop(idx)
                weights.pop(idx)
        if not placed:
            raise RuntimeError(
                f"SYN {size} seed {seed}: no op is placeable at {i}: the world is "
                "too small for the op count"
            )

    return _finish(
        seed, size, spec, ents, ops, slots, memories, counts, sub_counts, nows, rng, vals
    )


def _day_before(day: str, k: int) -> str:
    return (date.fromisoformat(day) - timedelta(days=k)).isoformat()


def _day_after(day: str, k: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=k)).isoformat()


def _finish(seed, size, spec, ents, ops, slots, memories, counts, sub_counts, nows, rng, vals):
    # final gold state
    for s in slots.values():
        multi = D.RELATIONS[s.rel].multi
        iv = s.intervals(multi)
        for m in s.members:
            mem = memories[m.op]
            if m.status == "active":
                mem["status"], mem["valid_from"], mem["valid_to"] = "active", *iv[m.op]
            else:
                mem["status"], mem["valid_from"], mem["valid_to"] = "retracted", m.vf, "any"
        for d in s.dups:
            memories[d]["status"], memories[d]["valid_from"], memories[d]["valid_to"] = (
                "duplicate", "any", "any")  # fmt: skip
    all_days = [m.as_of for s in slots.values() for m in s.members] + [n[:10] for n in nows]
    end_now = _iso(datetime.strptime(max(all_days), "%Y-%m-%d").replace(tzinfo=UTC)
                   + timedelta(days=2, hours=12))  # fmt: skip
    slot_rows = {}
    for key in sorted(slots):
        s = slots[key]
        multi = D.RELATIONS[s.rel].multi
        act = s.active()
        slot_rows[key] = {"entity": s.entity, "rel": s.rel, "multi": multi,
                          "chain": [m.op for m in act],
                          "current": [m.op for m in s.open_members(multi)],
                          "members": [m.op for m in s.members], "dups": s.dups,
                          "had_update": sum(1 for m in s.members
                                            if ops[m.op]["kind"] == "update") > 0}  # fmt: skip
    expected = {
        "syn_version": D.SYN_VERSION, "template_version": D.TEMPLATE_VERSION,
        "cues_version": D.CUES_VERSION, "seed": seed, "size": size, "tz": D.TZ, "end_now": end_now,
        "counts": dict(sorted(counts.items())), "update_subtypes": dict(sorted(sub_counts.items())),
        "entities": [{"key": e.key, "type": e.type, "name": e.name, "alias": e.alias,
                      "group": e.group, "first": e.first} for e in ents],
        "slots": slot_rows, "memories": {str(k): memories[k] for k in sorted(memories)},
    }  # fmt: skip
    queries = _queries(rng, ents, slots, memories, end_now)
    return Synth(seed, size, ops, expected, queries)


def _subject_phrase(e: Entity) -> str:
    """A cross-kind pair shares one name, so a query names the type as well."""
    return f"the {e.type} {e.name}" if (e.group or "").startswith("X") else e.name


def _queries(rng, ents, slots, memories, end_now) -> list[dict[str, Any]]:
    by_key = {e.key: e for e in ents}
    out: list[dict[str, Any]] = []
    for key in sorted(slots):
        s = slots[key]
        r = D.RELATIONS[s.rel]
        if r.multi:
            continue
        act = s.active()
        if not act:
            continue
        q = r.question.format(s=_subject_phrase(by_key[s.entity]))
        slot_ops = [m.op for m in s.members] + list(s.dups)
        out.append({"id": f"cur:{key}", "type": "current", "slot": key, "query": q,
                    "now": end_now, "gold": [act[-1].op], "slot_ops": slot_ops})  # fmt: skip
        iv = s.intervals(False)
        for m in act[:-1]:
            vf, vt = iv[m.op]
            if vt is None or vt <= vf:
                continue  # empty (same-instant) interval: no midpoint
            a = datetime.strptime(vf, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            b = datetime.strptime(vt, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            out.append({"id": f"asof:{key}:{m.op}", "type": "as_of", "slot": key, "query": q,
                        "now": end_now, "as_of": _iso(a + (b - a) / 2), "gold": [m.op],
                        "slot_ops": slot_ops})  # fmt: skip
    missing = []
    for e in sorted(ents, key=lambda e: e.key):
        if not e.introduced:
            continue
        for r in D.RELATIONS.values():
            if r.subject == e.type and f"{e.key}|{r.name}" not in slots:
                missing.append((e, r))
    rng.shuffle(missing)
    for e, r in sorted(missing[:40], key=lambda p: (p[0].key, p[1].name)):
        out.append({"id": f"miss:{e.key}|{r.name}", "type": "missing_slot",
                    "slot": f"{e.key}|{r.name}", "query": r.question.format(s=_subject_phrase(e)),
                    "now": end_now, "gold": []})  # fmt: skip
    return out


def load(path: Path) -> Synth:
    """Read back what `Synth.write` wrote."""

    def rows(name: str) -> list[dict[str, Any]]:
        return [json.loads(ln) for ln in (path / name).read_text(encoding="utf-8").splitlines()
                if ln.strip()]  # fmt: skip

    expected = json.loads((path / "expected_state.json").read_text(encoding="utf-8"))
    return Synth(
        expected["seed"], expected["size"], rows("ops.jsonl"), expected, rows("queries.jsonl")
    )
