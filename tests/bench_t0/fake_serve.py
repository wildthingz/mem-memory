"""A small in-test stand-in for the P1 server, used ONLY to prove the suites can pass and fail.

It implements just enough of remember/recall/get on the real schema v2 (FTS5 only, no vectors).
It is not the product: the gate tests in test_gate_p1.py run against `mem.serve` and xfail until
p1-serve lands. Behaviour flags let a test break one property at a time (a mutant).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from bench.suites.base import ToolResult
from mem import contracts, schema

PAGE_EDGES = 10
_PREFIX = {"memory": "m", "entity": "e", "note": "n", "tag": "t"}


class Flags:
    def __init__(self, **kw: bool) -> None:
        self.cross_store = kw.get("cross_store", False)  # the work store opens the personal file
        self.utc_date_only = kw.get("utc_date_only", False)  # date-only read as UTC midnight
        self.start_of_day_as_of = kw.get("start_of_day_as_of", False)  # recall: 00:00, not 23:59:59
        self.ignore_valid_to = kw.get("ignore_valid_to", False)  # leaks closed intervals
        self.ignore_valid_from = kw.get("ignore_valid_from", False)  # leaks future-dated
        self.closed_boundary = kw.get("closed_boundary", False)  # valid_to inclusive
        self.lossy_get = kw.get("lossy_get", False)  # drops one char at each page cut
        self.rewrite = kw.get("rewrite", False)  # second remember rewrites an earlier row
        self.no_about = kw.get("no_about", False)  # memories get no about edge
        self.clock_recorded = kw.get("clock_recorded", False)  # recorded_at = wall clock
        self.nondeterministic = kw.get("nondeterministic", False)
        self.chatty_empty = kw.get("chatty_empty", False)  # empty result is more than one line
        self.no_p1_header = kw.get("no_p1_header", False)
        self.edge_overlap = kw.get("edge_overlap", False)  # next edge page repeats one edge


class FakeClient:
    def __init__(self, db_path: Path, scope: str, tz: str, now: str, flags: Flags) -> None:
        if flags.cross_store and scope == "work":  # the 6p/6w pair of b_w6 item 6
            db_path = Path(db_path).with_name(Path(db_path).name.replace("6w", "6p"))
        self.db_path, self.scope, self.tz, self.now, self.f = Path(db_path), scope, tz, now, flags
        self.conn = schema.connect(db_path)
        self.offset = 0
        self.counter = 0

    # -- helpers ------------------------------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def _shown(self, kind: str, nid: int) -> str:
        return f"{_PREFIX[kind]}{nid + self.offset}"

    def _utc(self, value: str, *, end_of_day: bool = False) -> str:
        """Date-only: local midnight, or with `end_of_day` (recall's as_of, ADR 0028) 23:59:59."""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            zone = UTC if self.f.utc_date_only else ZoneInfo(self.tz)
            local = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=zone)
            if end_of_day and not self.f.start_of_day_as_of:
                local = local.replace(hour=23, minute=59, second=59)
            return local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return value

    def _as_of(self, value: str) -> str:
        return self._utc(value, end_of_day=True)

    def _event(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.conn.execute("INSERT INTO events(at, session, host, tool, args, result, latency_ms)"
                          " VALUES (?, 's1', 'bench', ?, ?, ?, 0)",
                          (self.now, tool, repr(args), repr(result)))  # fmt: skip

    def call(self, tool: str, args: dict[str, Any], *, now: str | None = None) -> ToolResult:
        if now:
            self.now = now
        with self.conn:
            res = getattr(self, f"_{tool}")(args)
            self._event(tool, args, res.structured)
        return res

    # -- remember -----------------------------------------------------------------------------
    def _subject(self, spec: str) -> int:
        if spec.startswith("new:"):
            _, typ, name = spec.split(":", 2)
        else:
            typ, name = None, spec
        row = self.conn.execute("SELECT id FROM nodes WHERE kind='entity' AND name=? COLLATE NOCASE",
                                (name,)).fetchone()  # fmt: skip
        if row:
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO nodes(kind,type,name,valid_from,recorded_at,origin,content_hash,host)"
            " VALUES ('entity',?,?,?,?,'agent',?,'bench')",
            (typ or "topic", name, self.now, self.now, hashlib.sha256(name.encode()).hexdigest()))  # fmt: skip
        return cur.lastrowid

    def _remember(self, a: dict[str, Any]) -> ToolResult:
        text = a["text"]
        h = hashlib.sha256(text.encode()).hexdigest()
        dup = self.conn.execute("SELECT id FROM nodes WHERE kind='memory' AND status='active' AND "
                                "valid_to IS NULL AND content_hash=?", (h,)).fetchone()  # fmt: skip
        subjects = [s.split(":")[-1] for s in a["about"]]
        if dup:
            s = {"status": "exists", "id": self._shown("memory", dup[0]), "row": None,
                 "subjects": subjects, "possible_conflicts": [], "candidates": []}  # fmt: skip
            return ToolResult("exists", s)
        if self.f.rewrite:
            self.conn.execute(
                "UPDATE nodes SET name = name || '!' WHERE id = "
                "(SELECT min(id) FROM nodes WHERE kind='memory')"
            )
        eids = [self._subject(x) for x in a["about"]]
        recorded = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if self.f.clock_recorded \
            else self.now  # fmt: skip
        title = a.get("title") or " ".join(text.split()[:12])
        if self.f.nondeterministic:
            title += uuid.uuid4().hex[:4]
        cur = self.conn.execute(
            "INSERT INTO nodes(kind,type,name,body,date,valid_from,recorded_at,origin,content_hash,"
            "session,host) VALUES ('memory',?,?,?,?,?,?,'agent',?,?,'bench')",
            (a["type"], title, text, self._utc(a["as_of"])[:10] if a.get("as_of") else self.now[:10],
             self._utc(a["as_of"]) if a.get("as_of") else self.now, recorded, h,
             f"s{self.counter}" if self.f.nondeterministic else "s1"))  # fmt: skip
        self.counter += 1
        mid = cur.lastrowid
        if not self.f.no_about:
            for e in eids:
                self.conn.execute("INSERT INTO edges(src,dst,rel,origin,recorded_at) "
                                  "VALUES (?,?, 'about','agent',?)", (mid, e, self.now))  # fmt: skip
        self.conn.execute("INSERT INTO chunks(node_id,chunk_key,ord,key_kind,ctx,text,tokens,"
                          "content_hash) VALUES (?, 'body', 0, 'body', ?, ?, ?, ?)",
                          (mid, title, text, -(-len(text.encode()) // 3), h))  # fmt: skip
        sid = self._shown("memory", mid)
        s = {"status": "saved", "id": sid, "row": f"{sid} · {a['type']}", "subjects": subjects,
             "possible_conflicts": [], "candidates": []}  # fmt: skip
        return ToolResult(f"saved {sid}", s)

    # -- recall -------------------------------------------------------------------------------
    def _recall(self, a: dict[str, Any]) -> ToolResult:
        t = self._as_of(a["as_of"]) if a.get("as_of") else self.now
        words = re.findall(r"\w+", a["query"].lower())
        where = "n.status='active'"
        if not self.f.ignore_valid_from:
            where += " AND n.valid_from <= :t"
        if not (self.f.ignore_valid_to or a.get("include_history")):
            where += (
                " AND (n.valid_to IS NULL OR :t "
                + ("<=" if self.f.closed_boundary else "<")
                + " n.valid_to)"
            )
        rows = []
        if words:
            match = " OR ".join(f'"{w}"' for w in words)
            rows = self.conn.execute(
                "SELECT c.id, n.id, n.type, n.date, n.name FROM chunks_fts JOIN chunks c ON "
                f"c.id = chunks_fts.rowid JOIN nodes n ON n.id = c.node_id WHERE {where} AND "
                "n.kind='memory' AND chunks_fts MATCH :m ORDER BY bm25(chunks_fts) LIMIT 5",
                {"t": t, "m": match}).fetchall()  # fmt: skip
        out = []
        for rank, (_cid, nid, typ, date, name) in enumerate(rows, start=1):
            about = [r[0] for r in self.conn.execute(
                "SELECT e.name FROM edges g JOIN nodes e ON e.id=g.dst WHERE g.src=? AND "
                "g.rel='about' AND g.removed_at IS NULL ORDER BY e.id", (nid,))]  # fmt: skip
            out.append({"id": self._shown("memory", nid), "kind": "memory", "type": typ,
                        "date": date, "title": name, "about": about,
                        "p": round(1 / (60 + rank), 4), "flags": []})  # fmt: skip
        if not out:
            return ToolResult("no results · " + self.scope + ("\nTry other words." if self.f.chatty_empty else ""),
                              {"rows": [], "abstained": False, "checked": 0, "filters_dropped": []})  # fmt: skip
        budget, kept = a.get("budget", 600), []
        for r in out:
            p = f"{r['p']:.4f}".lstrip("0")
            line = (f"{r['id']} · {r['type']} · {r['date']} · {r['title']} · about: "
                    f"{', '.join(r['about'])} · {p}")  # fmt: skip
            head = f"{len(kept) + 1} results (of {len(out)} checked) · {self.scope} · {'' if self.f.no_p1_header else 'uncalibrated (P1 build)'}"
            if -(-len("\n".join([head, *[ln for _, ln in kept], line]).encode()) // 3) > budget:
                break
            kept.append((r, line))
        head = f"{len(kept)} results (of {len(out)} checked) · {self.scope} · {'' if self.f.no_p1_header else 'uncalibrated (P1 build)'}"
        return ToolResult("\n".join([head, *[ln for _, ln in kept]]),
                          {"rows": [r for r, _ in kept], "abstained": False, "checked": len(out),
                           "filters_dropped": []})  # fmt: skip

    # -- get ----------------------------------------------------------------------------------
    def _get(self, a: dict[str, Any]) -> ToolResult:
        budget = a.get("max_tokens", 2000) * 3  # bytes, whole response
        ids = a["ids"]
        share = max(budget // max(len(ids), 1) - 40, 20)
        items, texts = [], []
        for token in ids:
            item = self._get_one(token, share)
            items.append(item)
            texts.append(f"{item['id']}: {item.get('text', '')}"
                         + (f"\n… more: get {item['continuation']}" if item.get("continuation") else ""))  # fmt: skip
        return ToolResult("\n\n".join(texts), {"items": items})

    def _get_one(self, token: str, share: int) -> dict[str, Any]:
        m = re.fullmatch(r"([mecn])(\d+)(?:@(.+))?", token)
        if not m:
            return {"id": token, "found": False, "text": "bad id"}
        kind, num, cont = m.group(1), int(m.group(2)) - self.offset, m.group(3)
        if kind == "c":
            row = self.conn.execute(
                "SELECT text, node_id FROM chunks WHERE id=?", (num,)
            ).fetchone()
            if not row:
                return {"id": token, "found": False,
                        "text": f"c{num} no longer exists (changed): get its node"}  # fmt: skip
            return self._page(f"c{num}", row[0], int(cont or 0), share)
        if kind == "m":
            row = self.conn.execute(
                "SELECT body FROM nodes WHERE id=? AND kind='memory'", (num,)
            ).fetchone()
            if not row:
                return {"id": token, "found": False, "text": f"{token} not found"}
            return self._page(
                f"m{num}", row[0], int(cont or 0), share, cont_id=f"m{num + self.offset}"
            )
        if kind == "e":
            start = int(cont.split(":")[1]) if cont else 0
            ids = [r[0] for r in self.conn.execute(
                "SELECT src FROM edges WHERE dst=? AND removed_at IS NULL ORDER BY src", (num,))]  # fmt: skip
            page = ids[max(start - (1 if self.f.edge_overlap else 0), 0) : start + PAGE_EDGES]
            more = start + PAGE_EDGES < len(ids)
            text = "\n".join(f"about ← {self._shown('memory', i)}" for i in page)
            return {"id": f"e{num + self.offset}", "found": True, "text": text,
                    "continuation": f"e{num + self.offset}@edges:{start + PAGE_EDGES}" if more else None}  # fmt: skip
        return {"id": token, "found": False}

    def _page(
        self, id_: str, text: str, off: int, share: int, cont_id: str | None = None
    ) -> dict[str, Any]:
        raw = text.encode()
        end = min(off + share, len(raw))
        while end < len(raw) and (raw[end] & 0xC0) == 0x80:  # never split a UTF-8 character
            end -= 1
        piece = raw[off:end].decode()
        if self.f.lossy_get and end < len(raw):
            piece = piece[:-1]
        more = end < len(raw)
        shown = cont_id or id_
        return {"id": shown, "found": True, "header": shown, "text": piece,
                "continuation": f"{shown}@{end}" if more else None}  # fmt: skip


def factory_with(flags: Flags | None = None):
    flags = flags or Flags()

    def factory(
        db_path: Path,
        *,
        scope: str = "personal",
        tz: str = "Europe/Amsterdam",
        now: str = "2026-10-22T09:00:00Z",
    ) -> FakeClient:
        return FakeClient(db_path, scope, tz, now, flags)

    return factory


def list_tools() -> list[dict[str, Any]]:
    return contracts.mcp_tools()["tools"]
