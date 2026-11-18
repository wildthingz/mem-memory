"""B-E4 daily use (benchmarks-v2 §4, gate of P6): metrics from the `events` tables, the daily
review marks, the nightly duplicate scans and the incident log. Pure functions over plain data;
`bench.p6_cli` is the command line.

Definitions (each is the spec's sentence, made exact):
- **Live events**: rows of `events` in `personal.db` and `work.db` whose host does not start with
  `bench-` (bench runs write into the live stores under that label, `bench.e5.LIVE_EVENTS_FILTER`).
- **Working day**: a calendar day (in `tz`) with at least one live event. A **work day** is a working
  day with at least one live event from a host in `work_hosts` (label prefix, default `work`).
- **Reviewed**: a recall event with a mark in the reviews file. The latest mark per event wins.
  `useful_hit` = useful / non-empty reviewed; `miss` = should-have-found / reviewed.
- **Unresolved conflict**: a `remember` that saved a new memory and returned `possible_conflicts`,
  with no `link` event naming the new id, no `forget` of it, and no `supersedes` argument, in the
  same MCP session and store (A12). A session-less event can never be resolved. The rate is over
  the remembers that returned conflicts.
- **Duplicates per 100**: distinct memory pairs flagged by any nightly scan in the window, over the
  new memories saved in the window, times 100.
- **Round trip**: the events log holds server time only. `work_p95_ms` = p95 of the server time of
  work-host recalls + `work_overhead_ms` (the p95 the network adds over local, from the B-E5
  record of the work machine; `bench e5-run`). With no overhead given it is the server p95, and the
  report says so.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from bench.e5 import BENCH_HOST_PREFIX
from bench.metrics import stats
from mem import review

STORES = ("personal", "work")
WORKING_DAYS = 10
MIN_WORK_DAYS = 3
SAMPLE_PER_DAY = 10
CONTEXT_EVENTS = 3
MARKS = ("useful", "irrelevant", "should-have-found")
NOT_STORED = "not stored"
INCIDENT_KINDS = ("crash", "data-loss", "wrong-store")
BLOCKING_KINDS = ("data-loss", "wrong-store")


# ---- events ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    store: str
    id: int
    at: str
    session: str | None
    host: str | None
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    latency_ms: int | None

    @property
    def ref(self) -> str:
        return review.event_ref(self.store, self.id)

    @property
    def ids(self) -> list[str]:
        return [str(i) for i in self.result.get("ids") or []]


def _json(text: str | None) -> dict[str, Any]:
    try:
        v = json.loads(text) if text else {}
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def load_events(stores_dir: Path, *, include_bench: bool = False) -> list[Event]:
    """Every event of the stores found under `stores_dir`, sorted by (at, store, id)."""
    out: list[Event] = []
    for store in STORES:
        db = Path(stores_dir) / f"{store}.db"
        if not db.exists():
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT id, at, session, host, tool, args, result, latency_ms"
                                " FROM events ORDER BY id").fetchall()  # fmt: skip
        finally:
            conn.close()
        for i, at, session, host, tool, args, result, lat in rows:
            if not include_bench and (host or "").startswith(BENCH_HOST_PREFIX):
                continue
            out.append(Event(store, i, at, session, host, tool, _json(args), _json(result), lat))
    return sorted(out, key=lambda e: (e.at, e.store, e.id))


def day_of(at: str, tz: str = "UTC") -> str:
    moment = datetime.fromisoformat(at)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d")


def in_window(events: Iterable[Event], since: str | None, until: str | None, tz: str = "UTC"
              ) -> list[Event]:  # fmt: skip
    """Events whose day (in tz) is in [since, until] (inclusive, YYYY-MM-DD)."""
    out = []
    for e in events:
        d = day_of(e.at, tz)
        if (since is None or d >= since) and (until is None or d <= until):
            out.append(e)
    return out


# ---- daily review ------------------------------------------------------------------------------


sha1_key = review.sample_key


def review_sample(events: Sequence[Event], day: str, n: int = SAMPLE_PER_DAY, tz: str = "UTC"
                  ) -> list[Event]:  # fmt: skip
    """Up to n of the day's recall events, in sha1(event ref) order (`mem.review.sample_key`, the
    same pick as `mem review`; the store is in the key because the two stores number their events
    independently)."""
    recalls = [e for e in events if e.tool == "recall" and day_of(e.at, tz) == day]
    return sorted(recalls, key=lambda e: sha1_key(e.ref))[:n]


def context_of(events: Sequence[Event], ev: Event, k: int = CONTEXT_EVENTS) -> list[str]:
    """The k preceding `remember`/`recall` texts of ev's session, oldest first."""
    if ev.session is None:
        return []
    prior = [e for e in events if e.session == ev.session and e.store == ev.store
             and e.tool in ("remember", "recall") and (e.at, e.id) < (ev.at, ev.id)]  # fmt: skip
    return [f"{e.tool}: " + str(e.args.get("query") or e.args.get("text") or "")[:160]
            for e in prior[-k:]]  # fmt: skip


@dataclass(frozen=True)
class Review:
    ref: str
    mark: str
    found: str | None = None  # should-have-found: the memory id, or "not stored"
    at: str = ""


def check_review(r: Review) -> None:
    if r.mark not in MARKS:
        raise ValueError(f"{r.ref}: mark {r.mark!r} (one of {', '.join(MARKS)})")
    if r.mark == "should-have-found" and not r.found:
        raise ValueError(f"{r.ref}: should-have-found needs an id, or '{NOT_STORED}'")


def append_review(path: Path, r: Review) -> None:
    check_review(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ref": r.ref, "mark": r.mark, "found": r.found, "at": r.at},
                           ensure_ascii=False, sort_keys=True) + "\n")  # fmt: skip


def load_reviews(path: Path) -> dict[str, Review]:
    """ref -> latest Review. A missing file is no reviews."""
    out: dict[str, Review] = {}
    if not Path(path).exists():
        return out
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            r = Review(d["ref"], d["mark"], d.get("found"), d.get("at", ""))
            check_review(r)
            out[r.ref] = r
    return out


# ---- nightly duplicate scan --------------------------------------------------------------------

Scorer = Callable[[str, Sequence[str]], Sequence[float]]  # (query, docs) -> raw logits


def duplicate_pairs(
    conn: sqlite3.Connection, score: Scorer, theta: float
) -> list[tuple[str, str, float]]:
    """Pairs of current memories (open interval, active) that share an `about` subject, whose
    reranker logit is >= theta (either direction; the larger), and with no `duplicate_of` /
    `supersedes` path between them. Returned as (mA, mB, logit) with A < B, sorted."""
    mem = {i: t for i, t in conn.execute(
        "SELECT id, body FROM nodes WHERE kind = 'memory' AND status = 'active'"
        " AND valid_to IS NULL")}  # fmt: skip
    by_subject: dict[int, list[int]] = defaultdict(list)
    for src, dst in conn.execute("SELECT src, dst FROM edges WHERE rel = 'about'"
                                 " AND removed_at IS NULL ORDER BY src, dst"):  # fmt: skip
        if src in mem:
            by_subject[dst].append(src)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in conn.execute("SELECT src, dst FROM edges WHERE rel IN ('duplicate_of', 'supersedes')"
                             " AND removed_at IS NULL"):  # fmt: skip
        parent[find(a)] = find(b)
    seen: dict[tuple[int, int], float] = {}
    for ids in by_subject.values():
        ids = sorted(set(ids))
        for k, a in enumerate(ids):
            rest = [b for b in ids[k + 1 :] if (a, b) not in seen and find(a) != find(b)]
            if not rest:
                continue
            fwd = score(mem[a], [mem[b] for b in rest])
            for b, z_ab in zip(rest, fwd, strict=True):
                z_ba = score(mem[b], [mem[a]])[0]
                seen[(a, b)] = max(float(z_ab), float(z_ba))
    return sorted((f"m{a}", f"m{b}", round(z, 4)) for (a, b), z in seen.items() if z >= theta)


def append_scan(path: Path, *, date: str, store: str, pairs: Sequence[tuple[str, str, float]],
                theta: float, n_current: int) -> None:  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": date, "store": store, "theta": theta, "n_current": n_current,
                            "pairs": [list(p) for p in pairs]}, sort_keys=True) + "\n")  # fmt: skip


def load_scans(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip()]  # fmt: skip


# ---- incidents ---------------------------------------------------------------------------------


def append_incident(path: Path, *, date: str, kind: str, note: str, store: str = "") -> None:
    if kind not in INCIDENT_KINDS:
        raise ValueError(f"incident kind {kind!r} (one of {', '.join(INCIDENT_KINDS)})")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": date, "kind": kind, "store": store, "note": note},
                           ensure_ascii=False, sort_keys=True) + "\n")  # fmt: skip


def load_incidents(path: Path) -> list[dict[str, Any]]:
    return load_scans(path)


# ---- metrics -----------------------------------------------------------------------------------


@dataclass
class E4Report:
    metrics: dict[str, float] = field(default_factory=dict)
    per_day: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # remember refs left with conflicts
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _is_work(e: Event, work_hosts: Sequence[str]) -> bool:
    return any((e.host or "").startswith(p) for p in work_hosts)


def unresolved_conflicts(events: Sequence[Event]) -> tuple[int, list[str]]:
    """(remembers that returned possible_conflicts, refs of those left unresolved)."""
    ordered = sorted(events, key=lambda e: (e.at, e.store, e.id))
    with_conflicts, left = 0, []
    for i, e in enumerate(ordered):
        if e.tool != "remember" or e.result.get("status") != "saved":
            continue
        if not e.result.get("possible_conflicts"):
            continue
        with_conflicts += 1
        new = str(e.result.get("id"))
        resolved = review.inline_resolution(e.args)
        if not resolved and e.session is not None:
            for later in ordered[i + 1 :]:
                if later.session != e.session or later.store != e.store:
                    continue
                if review.closes_conflict(new, later.tool, later.args):
                    resolved = True
                    break
        if not resolved:
            left.append(e.ref)
    return with_conflicts, left


def compute_e4(
    events: Sequence[Event], reviews: dict[str, Review], scans: Sequence[dict[str, Any]] = (),
    incidents: Sequence[dict[str, Any]] = (), *, since: str | None = None, until: str | None = None,
    tz: str = "UTC", work_hosts: Sequence[str] = ("work",), work_overhead_ms: float | None = None,
) -> E4Report:  # fmt: skip
    """Every B-E4 metric over the live events in [since, until]."""
    rep = E4Report()
    ev = [
        e
        for e in in_window(events, since, until, tz)
        if not (e.host or "").startswith(BENCH_HOST_PREFIX)
    ]
    m = rep.metrics
    days = sorted({day_of(e.at, tz) for e in ev})
    work_days = sorted({day_of(e.at, tz) for e in ev if _is_work(e, work_hosts)})
    recalls = [e for e in ev if e.tool == "recall"]
    per_day_n = {d: 0 for d in days}
    for e in recalls:
        per_day_n[day_of(e.at, tz)] += 1
    m["working_days"] = float(len(days))
    m["work_days"] = float(len(work_days))
    m["recalls"] = float(len(recalls))
    m["recalls_per_day_median"] = float(statistics.median(per_day_n.values())) if days else 0.0
    m["empty_rate"] = (sum(not e.ids for e in recalls) / len(recalls)) if recalls else 0.0
    tokens = [
        e.result["tokens"] for e in recalls if isinstance(e.result.get("tokens"), (int, float))
    ]
    if tokens:
        m["tokens_per_recall_mean"] = sum(tokens) / len(tokens)
        m["tokens_per_recall_p95"] = float(stats.percentile_nearest_rank(tokens, 0.95))
    else:
        rep.notes.append("no recall event carries `tokens` (events from before MEM2-46): tokens "
                         "per recall not computed")  # fmt: skip

    by_ref = {e.ref: e for e in recalls}
    stray = sorted(r for r in reviews if r not in by_ref)
    reviewed = [(by_ref[r.ref], r) for r in reviews.values() if r.ref in by_ref]
    nonempty = [(e, r) for e, r in reviewed if e.ids]
    useful = sum(r.mark == "useful" for _, r in nonempty)
    shf = sum(r.mark == "should-have-found" for _, r in reviewed)
    m["reviewed"] = float(len(reviewed))
    m["reviewed_nonempty"] = float(len(nonempty))
    m["useful_hit"] = useful / len(nonempty) if nonempty else float("nan")
    m["miss"] = shf / len(reviewed) if reviewed else float("nan")
    expected = sum(min(SAMPLE_PER_DAY, n) for n in per_day_n.values())
    m["review_coverage"] = len(reviewed) / expected if expected else 0.0
    if stray:
        rep.warnings.append(f"{len(stray)} review(s) match no recall event in the window "
                            f"(first: {stray[0]}): ignored")  # fmt: skip
    for e, r in reviewed:
        if r.mark == "useful" and not e.ids:
            rep.warnings.append(f"{r.ref}: marked useful but returned nothing: counted as reviewed "
                                "only (miss denominator), not in useful-hit")  # fmt: skip

    for name, host_pred in (("work", lambda e: _is_work(e, work_hosts)),
                            ("home", lambda e: not _is_work(e, work_hosts))):  # fmt: skip
        lat = [e.latency_ms for e in recalls if host_pred(e) and e.latency_ms is not None]
        if lat:
            m[f"{name}_server_p95_ms"] = float(stats.percentile_nearest_rank(lat, 0.95))
    if "work_server_p95_ms" in m:
        m["work_p95_ms"] = m["work_server_p95_ms"] + (work_overhead_ms or 0.0)
        if work_overhead_ms is None:
            rep.notes.append("work_p95_ms is server time only: pass the work machine's B-E5 "
                             "overhead (`--work-overhead-ms`) for the client round trip")  # fmt: skip
    else:
        m["work_p95_ms"] = float("nan")

    n_conf, left = unresolved_conflicts(ev)
    m["remembers_with_conflicts"] = float(n_conf)
    m["unresolved_conflicts"] = len(left) / n_conf if n_conf else 0.0
    rep.unresolved = left
    new_mem = sum(e.tool == "remember" and e.result.get("status") == "saved" for e in ev)
    lo, hi = (days[0], days[-1]) if days else (since or "", until or "")
    pairs = {tuple(sorted(p[:2])) for s in scans if lo <= s["date"] <= (until or hi)
             for p in s["pairs"]}  # fmt: skip
    m["new_memories"] = float(new_mem)
    m["duplicate_pairs"] = float(len(pairs))
    m["duplicates_per_100"] = len(pairs) / new_mem * 100 if new_mem else (
        0.0 if not pairs else float("inf"))  # fmt: skip
    m["scan_nights"] = float(len({s["date"] for s in scans if lo <= s["date"] <= (until or hi)}))
    if not scans:
        rep.warnings.append("no duplicate scan recorded: run `bench dup-scan` nightly")

    inc = [i for i in incidents if lo <= i["date"] <= (until or hi)]
    m["blocking_incidents"] = float(sum(i["kind"] in BLOCKING_KINDS for i in inc))
    m["crashes"] = float(sum(i["kind"] == "crash" for i in inc))
    for d in days:
        rep.per_day.append({"day": d, "recalls": per_day_n[d], "work": d in work_days,
                            "reviewed": sum(day_of(e.at, tz) == d for e, _ in reviewed)})  # fmt: skip
    return rep


def window_start(days: Sequence[str], n: int = WORKING_DAYS) -> str | None:
    """First day of the last n working days (the gate window)."""
    return sorted(days)[-n] if len(days) >= n else None


def render_report(rep: E4Report) -> str:
    m = rep.metrics
    lines = ["| metric | value |", "|---|---|"]
    for k, v in m.items():
        lines.append(f"| {k} | {v:.4g} |")
    lines += ["", "| day | recalls | reviewed | work |", "|---|---|---|---|"]
    lines += [f"| {d['day']} | {d['recalls']} | {d['reviewed']} | {'yes' if d['work'] else ''} |"
              for d in rep.per_day]  # fmt: skip
    if rep.unresolved:
        lines += ["", "Unresolved conflicts (remember events): " + ", ".join(rep.unresolved)]
    lines += [f"\nNote: {n}" for n in rep.notes] + [f"\nWARNING: {w}" for w in rep.warnings]
    return "\n".join(lines) + "\n"
