"""A small in-test stand-in for the P3 server (resolver, conflicts, `link`, `forget`, `subjects`).

Like `fake_serve.py` it is used ONLY to prove that the P3 suites can pass and can fail (red-green
for the gates): it implements architecture-v2 §3.1/§4.1 on the real schema v2 with FTS5 and no
vectors. It is not the product. `mem.serve` has no `link` until the P3 write path lands, so the
gate against the real code reports UNAVAILABLE; the same suites run here at T0.

Behaviour flags let a test break one property at a time (a mutant).
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from bench.suites.base import ToolResult
from bench.synth import data as D
from bench.synth.align import find, tokens
from bench.synth.generator import lev_within
from mem.common import MemError, content_hash, norm
from tests.bench_t0.fake_serve import FakeClient, Flags

_ARROW = re.compile(r"\s*(?:→|->)\s*")
_ID = re.compile(r"^([mecnt])(\d+)$")
_CUES = {r.name: [tokens(c) for c in r.cues] for r in D.RELATIONS.values()}
MAX_CONFLICTS = 5


class P3Flags(Flags):
    def __init__(self, **kw: bool) -> None:
        super().__init__(**{k: v for k, v in kw.items() if k in _P1_FLAGS})
        for k in _P3_FLAGS:
            setattr(self, k, kw.get(k, False))


_P1_FLAGS = {"cross_store", "utc_date_only", "start_of_day_as_of", "ignore_valid_to", "ignore_valid_from",
             "closed_boundary", "lossy_get", "rewrite", "no_about", "clock_recorded",
             "nondeterministic", "chatty_empty", "no_p1_header", "edge_overlap"}  # fmt: skip
_P3_FLAGS = {
    "oracle_recall",  # SYN probes: a retriever that understands the relation questions
    "never_ask",  # resolves ambiguous names to the first candidate (wrong merges)
    "no_typo_link",  # a typo creates a new entity (splits)
    "always_ask",  # asks on every non-exact mention
    "no_conflicts",  # possible_conflicts always empty
    "exists_for_closed",  # `exists` for a superseded original (N2)
    "no_exists",  # an exact duplicate is saved again
    "no_derive",  # link adds the edge but does not touch valid_to
    "invalidate_on_add",  # a compatible add closes the previous memory
    "lossy_history",  # a link rewrites the superseded memory's text
    "no_swap",  # link in the wrong direction is not swapped
    "link_not_idempotent",  # a repeated link raises
    "needs_choice_writes",  # needs_choice leaves a new subject behind
    "same_as_crash",  # same_as with a shared neighbour fails
    "link_imported",  # supersedes may close an imported node
    "busy",  # a concurrent write surfaces SQLITE_BUSY
    "dup_not_hidden",  # duplicate_of leaves the duplicate active
    "forget_keeps",  # forget does not re-derive the chain
    "no_abstain",  # oracle_recall: a slot with no memory still returns the subject's other memories
    "ask_gold_missing",  # needs_choice never offers the right candidate
}  # fmt: skip


def relation_of(text: str) -> str | None:
    toks = tokens(text)
    for rel, cues in _CUES.items():
        if any(find(toks, c) for c in cues):
            return rel
    return None


_OPEN: dict[Path, int] = {}  # clients open per store file: `busy` fails only under concurrency


class FakeP3Client(FakeClient):
    def __init__(self, db_path: Path, *a: Any, **kw: Any) -> None:
        super().__init__(db_path, *a, **kw)
        _OPEN[self.db_path] = _OPEN.get(self.db_path, 0) + 1

    def close(self) -> None:
        _OPEN[self.db_path] = _OPEN.get(self.db_path, 1) - 1
        super().close()

    # -- entities --------------------------------------------------------------------------
    def _entity_row(self, num: int) -> tuple | None:
        return self.conn.execute("SELECT id, type, name FROM nodes WHERE id=? AND kind='entity' "
                                 "AND status='active'", (num,)).fetchone()  # fmt: skip

    def _by_alias(self, alias_norm: str) -> list[tuple]:
        return self.conn.execute(
            "SELECT n.id, n.type, n.name FROM aliases a JOIN nodes n ON n.id=a.node_id "
            "WHERE a.alias_norm=? AND n.kind='entity' AND n.status='active' ORDER BY n.id",
            (alias_norm,)).fetchall()  # fmt: skip

    def _fuzzy(self, entry: str) -> list[tuple]:
        e = norm(entry)
        rows = self.conn.execute("SELECT id, type, name FROM nodes WHERE kind='entity' "
                                 "AND status='active' ORDER BY id").fetchall()  # fmt: skip
        typo = [r for r in rows if len(norm(r[2])) >= 5 and lev_within(e, norm(r[2]), 1)
                and norm(r[2]) != e]  # fmt: skip
        if len(typo) == 1 and not self.f.no_typo_link:
            return [("typo", *typo[0])]  # type: ignore[list-item]
        et = set(tokens(entry))
        sub = [r for r in rows if et and (et < set(tokens(r[2])) or set(tokens(r[2])) < et)]
        return [("subset", *r) for r in sub]  # type: ignore[misc]

    def _resolve(self, entry: str) -> dict[str, Any] | list[tuple]:
        entry = entry.strip()
        mention = None
        if _ARROW.search(entry):
            mention, entry = (p.strip() for p in _ARROW.split(entry, maxsplit=1))
        if entry.startswith("#"):
            m = _ID.match(entry[1:])
            row = self._entity_row(int(m.group(2))) if m and m.group(1) == "e" else None
            if row is None:
                raise MemError(f"{entry!r} is not an active entity: use #e<id> from `subjects`")
            return {"id": row[0], "name": row[2], "type": row[1], "mention": mention,
                    "state": "alias learned" if mention else "matched"}  # fmt: skip
        if mention:
            raise MemError(f"{mention} → {entry}: the target after → must be #e<id>")
        if entry.startswith("new:"):
            _, _, rest = entry.partition(":")
            etype, _, name = rest.partition(":")
            hit = self.conn.execute(
                "SELECT id FROM nodes WHERE kind='entity' AND type=? AND name=? COLLATE NOCASE "
                "AND status='active'", (etype, name.strip())).fetchone()  # fmt: skip
            return {"id": hit[0] if hit else None, "name": name.strip(), "type": etype,
                    "mention": None, "state": "matched" if hit else "new"}  # fmt: skip
        found = self._by_alias(norm(entry))
        if len(found) > 1:
            return found
        if found and self.f.always_ask:
            return found
        if found:
            return {"id": found[0][0], "name": found[0][2], "type": found[0][1],
                    "mention": None, "state": "matched"}  # fmt: skip
        fuzzy = self._fuzzy(entry)
        if self.f.always_ask and fuzzy:
            return [r[1:] for r in fuzzy]
        if fuzzy and fuzzy[0][0] == "typo":
            r = fuzzy[0]
            return {"id": r[1], "name": r[3], "type": r[2], "mention": None, "state": "matched"}
        if fuzzy:
            if self.f.never_ask:
                r = fuzzy[0]
                return {"id": r[1], "name": r[3], "type": r[2], "mention": None,
                        "state": "matched"}  # fmt: skip
            return [r[1:] for r in fuzzy]
        return {"id": None, "name": entry, "type": "topic", "mention": None, "state": "new"}

    def _new_entity(self, name: str, etype: str) -> int:
        h = content_hash(f"{etype}:{name}")
        cur = self.conn.execute(
            "INSERT INTO nodes(kind,type,name,valid_from,recorded_at,origin,content_hash,host) "
            "VALUES ('entity',?,?,?,?,'agent',?,'bench')",
            (etype, name, self.now, self.now, h),
        )
        self.conn.execute("INSERT OR IGNORE INTO aliases(alias,alias_norm,node_id,origin) "
                          "VALUES (?,?,?,'agent')", (name, norm(name), cur.lastrowid))  # fmt: skip
        return cur.lastrowid

    # -- remember ----------------------------------------------------------------------------
    def _remember(self, a: dict[str, Any]) -> ToolResult:
        text, about = a["text"], a["about"]
        if not about:
            raise MemError("about needs at least one subject")
        resolved, asks = [], []
        for entry in about:
            r = self._resolve(entry)
            (resolved if isinstance(r, dict) else asks).append(r)
        if asks:
            if self.f.needs_choice_writes:
                self._new_entity("Leftover", "topic")
            cands = []
            for group in asks:
                if self.f.ask_gold_missing:
                    group = group[:0] or [("e", "topic", "Nobody")]
                cands += [f"e{i} · {t} · {n}" for i, t, n in group]
            s = {"status": "needs_choice", "id": None, "row": None, "subjects": [],
                 "possible_conflicts": [], "candidates": cands}  # fmt: skip
            return ToolResult("needs_choice · nothing was written\n" + "\n".join(cands), s)
        valid_from = self._utc(a["as_of"]) if a.get("as_of") else self.now
        chash = content_hash(text)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = None
            if not self.f.no_exists:
                q = (
                    "SELECT id FROM nodes WHERE kind='memory' AND content_hash=? AND "
                    "status='active'"
                    + ("" if self.f.exists_for_closed else " AND valid_to IS NULL")
                )
                row = self.conn.execute(q, (chash,)).fetchone()
            if row:
                mid, status = row[0], "exists"
            else:
                try:
                    cur = self.conn.execute(
                        "INSERT INTO nodes(kind,type,name,body,date,valid_from,recorded_at,origin,"
                        "content_hash,session,host) VALUES ('memory',?,?,?,?,?,?,'agent',?,'s1','bench')",
                        (a["type"], " ".join(text.split()[:10]) + (uuid.uuid4().hex[:4] if self.f.nondeterministic else ""), text, valid_from[:10], valid_from,
                         self.now,
                         chash if not self.f.no_exists else chash + str(self.counter)))  # fmt: skip
                except sqlite3.IntegrityError:
                    mid = self.conn.execute(
                        "SELECT id FROM nodes WHERE kind='memory' AND content_hash=? AND "
                        "status='active' AND valid_to IS NULL", (chash,)).fetchone()[0]  # fmt: skip
                    cur, status = None, "exists"
                if cur is not None:
                    mid, status = cur.lastrowid, "saved"
            self.counter += 1
            if self.f.busy and _OPEN.get(self.db_path, 0) > 1 and self.counter % 3 == 2:
                raise sqlite3.OperationalError("database is locked")
            subjects = []
            for r in resolved:
                if r["id"] is None:
                    if status == "exists":
                        continue
                    r["id"] = self._new_entity(r["name"], r["type"])
                if r["mention"]:
                    self.conn.execute("INSERT OR IGNORE INTO aliases(alias,alias_norm,node_id,"
                                      "origin) VALUES (?,?,?,'agent')",
                                      (r["mention"], norm(r["mention"]), r["id"]))  # fmt: skip
                if not self.f.no_about:
                    self.conn.execute("INSERT OR IGNORE INTO edges(src,dst,rel,origin,recorded_at) "
                                      "VALUES (?,?,'about','agent',?)", (mid, r["id"], self.now))  # fmt: skip
                subjects.append(f"e{r['id']} {r['name']} ({r['state']})")
            conflicts = []
            if status == "saved":
                self.conn.execute(
                    "INSERT INTO chunks(node_id,chunk_key,ord,key_kind,ctx,text,tokens,content_hash)"
                    " VALUES (?, 'body', 0, 'body', ?, ?, ?, ?)",
                    (mid, text[:40], text, len(text.split()), chash))  # fmt: skip
                if self.f.rewrite:
                    self.conn.execute(
                        "UPDATE nodes SET name = name || '!' WHERE id = "
                        "(SELECT min(id) FROM nodes WHERE kind='memory')"
                    )
                if not self.f.no_conflicts:
                    conflicts = self._conflicts(mid, text, [r["id"] for r in resolved])
                if (
                    self.f.invalidate_on_add
                    and conflicts
                    and D.RELATIONS.get(relation_of(text) or "", D.RELATIONS["speaks"]).multi
                ):
                    self.conn.execute("UPDATE nodes SET valid_to=? WHERE id=?",
                                      (valid_from, int(conflicts[0]["id"][1:])))  # fmt: skip
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        sid = f"m{mid}"
        s = {"status": status, "id": sid, "row": f"{sid} · {a['type']}", "subjects": subjects,
             "possible_conflicts": conflicts, "candidates": []}  # fmt: skip
        text = f"{status} {sid}" + (
            "\npossible_conflicts:"
            + "".join(f"\n  {c['id']} · score {c['score']} · {c['hint']}" for c in conflicts)
            if conflicts
            else ""
        )
        return ToolResult(text, s)

    def _conflicts(self, mid: int, text: str, ents: list[int]) -> list[dict[str, Any]]:
        rel = relation_of(text)
        mine = set(tokens(text))
        out = []
        marks = ",".join("?" * len(ents))
        rows = self.conn.execute(
            f"SELECT DISTINCT n.id, n.body, n.valid_to FROM nodes n JOIN edges g ON g.src=n.id "
            f"AND g.rel='about' AND g.removed_at IS NULL WHERE n.kind='memory' AND "
            f"n.status='active' AND n.id<>? AND g.dst IN ({marks})", (mid, *ents)).fetchall()  # fmt: skip
        for nid, body, vto in rows:
            same = rel is not None and relation_of(body) == rel
            toks = set(tokens(body))
            jac = len(mine & toks) / len(mine | toks)
            score = (4.0 if same else 0.0) + jac + (0.5 if vto is None and same else 0.0)
            if same or jac >= 0.5:
                out.append({"id": f"m{nid}", "row": f"m{nid} · {body[:40]}", "score": round(score, 3),
                            "hint": "same_subjects" + (", older" if vto is None else ", closed"),
                            "_n": nid})  # fmt: skip
        out.sort(key=lambda c: (-c["score"], -c["_n"]))
        for c in out:
            c.pop("_n")
        return out[:MAX_CONFLICTS]

    # -- recall ------------------------------------------------------------------------------
    def _oracle_rows(self, query: str, t: str, history: bool) -> list[dict[str, Any]] | None:
        """A stand-in retriever for the SYN probes: the query is one of the relation questions
        (`bench.synth.data`); return the memories of that subject and relation valid at t, newest
        first; nothing when there are none (an abstention)."""
        for r in D.RELATIONS.values():
            pat = re.escape(r.question).replace(re.escape("{s}"), "(?P<s>.+)")
            m = re.fullmatch(pat, query)
            if not m:
                continue
            subject = m["s"]
            typed = re.fullmatch(r"the (\w+) (.+)", subject)
            ents = self._by_alias(norm(typed[2] if typed else subject))
            ents = [e for e in ents if e[1] == r.subject] or ents
            if not ents:
                return []
            rows = self.conn.execute(
                "SELECT n.id, n.type, n.date, n.name, n.body, n.valid_from, n.valid_to FROM nodes n "
                "JOIN edges g ON g.src=n.id AND g.rel='about' AND g.removed_at IS NULL "
                "WHERE n.kind='memory' AND n.status='active' AND g.dst=? ORDER BY n.valid_from DESC, n.id DESC",
                (ents[0][0],)).fetchall()  # fmt: skip
            out = []
            for nid, typ, date, name, body, vf, vt in rows:
                if relation_of(body) != r.name or (vf > t and not self.f.ignore_valid_from):
                    continue
                if not history and vt is not None and t >= vt and not self.f.ignore_valid_to:
                    continue
                out.append({"id": f"m{nid}", "kind": "memory", "type": typ, "date": date,
                            "title": name, "about": [ents[0][2]], "p": 0.9, "flags": []})  # fmt: skip
            if self.f.no_abstain and not out:
                out = [{"id": f"m{rows[0][0]}", "kind": "memory", "type": rows[0][1],
                        "date": rows[0][2], "title": rows[0][3], "about": [], "p": 0.4,
                        "flags": []}] if rows else []  # fmt: skip
            return out
        return None

    # -- recall: `about` filter (entries that resolve to nothing are dropped, nothing is created) ---
    def _recall(self, a: dict[str, Any]) -> ToolResult:
        if self.f.oracle_recall:
            t = self._as_of(a["as_of"]) if a.get("as_of") else self.now
            rows = self._oracle_rows(a["query"], t, bool(a.get("include_history")))
            if rows is not None:
                text = (
                    "no relevant memories · personal"
                    if not rows
                    else "\n".join(
                        [
                            f"{len(rows)} results · personal",
                            *(f"{r['id']} · {r['title']}" for r in rows),
                        ]
                    )
                )
                return ToolResult(text, {"rows": rows, "abstained": not rows, "checked": len(rows),
                                         "filters_dropped": []})  # fmt: skip
        about = a.get("about") or []
        res = super()._recall({k: v for k, v in a.items() if k != "about"})
        dropped = [e for e in about if not self._by_alias(norm(e)) and not self._fuzzy(e)]
        res.structured["filters_dropped"] = dropped
        return res

    # -- chains --------------------------------------------------------------------------------
    def _component(self, start: int) -> list[int]:
        seen, todo = {start}, [start]
        while todo:
            n = todo.pop()
            for a, b in self.conn.execute(
                    "SELECT src, dst FROM edges WHERE rel='supersedes' AND removed_at IS NULL "
                    "AND (src=? OR dst=?)", (n, n)):  # fmt: skip
                for x in (a, b):
                    if x not in seen:
                        seen.add(x)
                        todo.append(x)
        return sorted(seen)

    def _derive(self, start: int) -> list[str]:
        if self.f.no_derive:
            return []
        comp = self._component(start)
        marks = ",".join("?" * len(comp))
        act = self.conn.execute(
            f"SELECT id, valid_from, valid_to FROM nodes WHERE id IN ({marks}) AND "
            f"status='active' ORDER BY valid_from, recorded_at, id", comp).fetchall()  # fmt: skip
        effects = []
        for i, (nid, vf, vto) in enumerate(act):
            want = act[i + 1][1] if i + 1 < len(act) else None
            if want != vto:
                try:
                    self.conn.execute("UPDATE nodes SET valid_to=? WHERE id=?", (want, nid))
                except sqlite3.IntegrityError as err:
                    raise MemError(
                        f"m{nid} would reopen while another open memory has the same "
                        "text: link supersedes or duplicate_of first"
                    ) from err
                effects.append(f"m{nid} valid_to → {want}")
        return effects

    # -- link / forget / subjects ------------------------------------------------------------
    def _node(self, ident: str) -> tuple:
        m = _ID.match(ident.strip())
        if not m:
            raise MemError(f"bad id {ident!r}")
        row = self.conn.execute("SELECT id, kind, status, origin, valid_from, recorded_at "
                                "FROM nodes WHERE id=?", (int(m.group(2)),)).fetchone()  # fmt: skip
        if not row or row[1][0] != m.group(1) and not (m.group(1) == "e" and row[1] == "entity"):
            raise MemError(f"{ident} not found")
        return row

    def _link(self, a: dict[str, Any]) -> ToolResult:
        rel, remove = a["rel"], a.get("remove", False)
        src, dst = self._node(a["src"]), self._node(a["dst"])
        if src[0] == dst[0]:
            raise MemError("src and dst are the same node")
        if rel not in {r[0] for r in self.conn.execute("SELECT rel FROM rels")}:
            raise MemError(f"unknown rel {rel!r}")
        if (src[2] != "active" or dst[2] != "active") and not (remove or (False)):
            raise MemError("both ends must be active")
        effects: list[str] = []
        if rel == "supersedes":
            if src[1] != "memory" or dst[1] != "memory":
                raise MemError("supersedes links memory → memory")
            if (src[3] == "import" or dst[3] == "import") and not self.f.link_imported:
                raise MemError("an imported node cannot be invalidated: edit the vault")
            if remove:
                self.conn.execute("UPDATE edges SET removed_at=? WHERE rel='supersedes' AND "
                                  "removed_at IS NULL AND ((src=? AND dst=?) OR (src=? AND dst=?))",
                                  (self.now, src[0], dst[0], dst[0], src[0]))  # fmt: skip
                effects += self._derive(src[0]) + self._derive(dst[0])
                return self._ok(f"removed {a['src']} supersedes {a['dst']}", effects)
            newer, older = src, dst
            if (src[4], src[5], src[0]) < (dst[4], dst[5], dst[0]) and not self.f.no_swap:
                newer, older = dst, src
                effects.append("direction swapped")
            live = self.conn.execute(
                "SELECT 1 FROM edges WHERE rel='supersedes' AND removed_at IS NULL AND "
                "((src=? AND dst=?) OR (src=? AND dst=?))",
                (newer[0], older[0], older[0], newer[0])).fetchone()  # fmt: skip
            if live:
                if self.f.link_not_idempotent:
                    raise MemError("edge exists")
                return self._ok("already linked", [])
            self.conn.execute("INSERT INTO edges(src,dst,rel,origin,recorded_at) "
                              "VALUES (?,?,'supersedes','agent',?)", (newer[0], older[0], self.now))  # fmt: skip
            if self.f.lossy_history:
                self.conn.execute("UPDATE nodes SET body=body||' ' WHERE id=?", (older[0],))
            effects += self._derive(newer[0])
        elif rel == "duplicate_of":
            if src[1] != "memory" or dst[1] != "memory":
                raise MemError("duplicate_of links memory → memory")
            if src[3] == "import":
                raise MemError("an imported node cannot be invalidated: edit the vault")
            live = self.conn.execute("SELECT 1 FROM edges WHERE src=? AND dst=? AND "
                                     "rel='duplicate_of' AND removed_at IS NULL",
                                     (src[0], dst[0])).fetchone()  # fmt: skip
            if live:
                return self._ok("already linked", [])
            self.conn.execute("INSERT INTO edges(src,dst,rel,origin,recorded_at) VALUES "
                              "(?,?,'duplicate_of','agent',?)", (src[0], dst[0], self.now))  # fmt: skip
            if not self.f.dup_not_hidden:
                self.conn.execute("UPDATE nodes SET status='duplicate', status_at=? WHERE id=?",
                                  (self.now, src[0]))  # fmt: skip
            for (ent,) in self.conn.execute("SELECT dst FROM edges WHERE src=? AND rel='about' "
                                            "AND removed_at IS NULL", (src[0],)).fetchall():  # fmt: skip
                self.conn.execute("INSERT OR IGNORE INTO edges(src,dst,rel,origin,recorded_at) "
                                  "VALUES (?,?,'about','agent',?)", (dst[0], ent, self.now))  # fmt: skip
            effects.append(f"m{src[0]} status → duplicate")
        elif rel == "same_as":
            if src[1] != "entity" or dst[1] != "entity":
                raise MemError("same_as links entity → entity")
            effects += self._same_as(src[0], dst[0])
        else:
            self.conn.execute("INSERT OR IGNORE INTO edges(src,dst,rel,origin,recorded_at) "
                              "VALUES (?,?,?,'agent',?)", (src[0], dst[0], rel, self.now))  # fmt: skip
        return self._ok(f"{a['src']} {rel} {a['dst']}", effects)

    def _same_as(self, src: int, dst: int) -> list[str]:
        if self.f.same_as_crash:
            raise sqlite3.IntegrityError("UNIQUE constraint failed: edges")
        for alias, an in self.conn.execute("SELECT alias, alias_norm FROM aliases WHERE node_id=?",
                                           (src,)).fetchall():  # fmt: skip
            self.conn.execute("INSERT OR IGNORE INTO aliases(alias,alias_norm,node_id,origin) "
                              "VALUES (?,?,?,'agent')", (alias, an, dst))  # fmt: skip
        self.conn.execute("DELETE FROM aliases WHERE node_id=?", (src,))
        for eid, s_, d_, rel in self.conn.execute(
                "SELECT id, src, dst, rel FROM edges WHERE removed_at IS NULL AND "
                "(src=? OR dst=?)", (src, src)).fetchall():  # fmt: skip
            if {s_, d_} == {src, dst}:
                self.conn.execute("UPDATE edges SET removed_at=? WHERE id=?", (self.now, eid))
                continue
            n_s, n_d = (dst if s_ == src else s_), (dst if d_ == src else d_)
            if self.conn.execute("SELECT 1 FROM edges WHERE src=? AND dst=? AND rel=? AND "
                                 "removed_at IS NULL", (n_s, n_d, rel)).fetchone():  # fmt: skip
                self.conn.execute("UPDATE edges SET removed_at=? WHERE id=?", (self.now, eid))
            else:
                self.conn.execute("UPDATE edges SET src=?, dst=? WHERE id=?", (n_s, n_d, eid))
        self.conn.execute("UPDATE nodes SET status='duplicate', status_at=? WHERE id=?",
                          (self.now, src))  # fmt: skip
        return [f"e{src} merged into e{dst}"]

    def _ok(self, what: str, effects: list[str]) -> ToolResult:
        return ToolResult("ok · " + what + "".join(f"\n{e}" for e in effects),
                          {"ok": True, "effects": effects})  # fmt: skip

    def _forget(self, a: dict[str, Any]) -> ToolResult:
        node = self._node(a["id"])
        if node[1] != "memory":
            raise MemError("forget works on memories only")
        if node[3] == "import":
            raise MemError("an imported node cannot be forgotten: edit the vault")
        if not a.get("reason", "").strip():
            raise MemError("a reason is required")
        self.conn.execute("UPDATE nodes SET status='retracted', status_at=? WHERE id=?",
                          (self.now, node[0]))  # fmt: skip
        effects = [] if self.f.forget_keeps else self._derive(node[0])
        return self._ok(f"m{node[0]} retracted", effects)

    def _subjects(self, a: dict[str, Any]) -> ToolResult:
        q = norm(a.get("query") or "")
        rows = self.conn.execute("SELECT id, type, name FROM nodes WHERE kind='entity' AND "
                                 "status='active' ORDER BY id").fetchall()  # fmt: skip
        out = [f"e{i} · {t} · {n}" for i, t, n in rows if q in norm(n)][: a.get("limit", 20)]
        return ToolResult("\n".join(out), {"subjects": out})


def factory_p3(flags: P3Flags | None = None):
    flags = flags or P3Flags()

    def factory(
        db_path: Path, *, scope: str = "personal", tz: str = D.TZ,
        now: str = "2026-10-22T09:00:00Z",
    ) -> FakeP3Client:  # fmt: skip
        return FakeP3Client(db_path, scope, tz, now, flags)

    return factory
