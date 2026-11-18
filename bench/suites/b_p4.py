"""The P4 gate suites (benchmarks-v2 §4 P4 row): B-I1, B-I2, B-I3, B-I4, B-I5, B-C4, B-R3 and the V5k
forms of B-C1 / B-C3.

Everything a suite needs from the outside is in `P4Setup`: a factory of fresh vault stores (the
production importer on the bench's models, `bench.adapters.p4_ours.OursVaultBackend`; in the T0 tests
the stand-in importer), the vault directories (the mini-vault fixture and the extracted snapshot),
the frozen gold, and the recorded runs of the human-run legs. A leg whose input is missing raises
`BackendUnavailable` (a gate then says UNAVAILABLE) or leaves its metric out (FAIL, "not produced"):
never a pass. Definitions: docs/decisions/0010-p4-benchmarks-pg-importer-checks.md.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from bench import i5, latency, pg_eval, records, ref_parse
from bench.gold import pg
from bench.gold.chunks import ChunkIndex
from bench.suites.base import BackendUnavailable, Context, SuiteResult

OPT = "p4"
TABLES = ("nodes", "edges", "chunks", "link_occurrences", "aliases", "props", "source_files")
DUMP_DROP = {"nodes": {"session"}}


class VaultBackend(Protocol):
    """The vault stores (personal.db, work.db) and the importer over them."""

    dbs: dict[str, Path]

    def import_vault(
        self, vault_root: Path, *, routes: dict[str, str] | None = None, now: str | None = None
    ) -> dict[str, int]: ...
    def conns(self) -> dict[str, sqlite3.Connection]: ...
    def store(self, name: str = "personal", *, now: str | None = None) -> Any: ...
    @property
    def index(self) -> ChunkIndex: ...
    def close(self) -> None: ...


@dataclass
class P4Setup:
    new_backend: Callable[[str], VaultBackend]  # name -> a fresh, empty vault store
    fixture_vault: Path | None = None  # the mini-vault (T0 fixture)
    snapshot_vault: Path | None = None  # the extracted full snapshot (T1): B-I1, B-I3, B-C4, V5k
    pg_vault: Path | None = None  # the extracted PG snapshot (no work notes): B-R3 and PG baselines
    lifecycle: Path | None = None  # B-I2 descriptor yaml (vault-mini.lifecycle.yaml)
    gold: pg.Gold | None = None
    split: str = "tune"
    dataset_meta: dict[str, Any] = field(default_factory=dict)
    routes: dict[str, str] = field(
        default_factory=dict
    )  # vault.routes of the fixture's config.yaml
    snapshot_routes: dict[str, str] = field(
        default_factory=dict
    )  # vault.routes of the person's config
    render: Callable[[list[Path], Path], None] | None = None  # (store dbs, out dir) -> `mem render`
    i5_sheet: Path | None = None
    c4_run: Path | None = None
    baselines: dict[str, Path] = field(default_factory=dict)  # "qmd" / "rg" -> run file
    latency_runs: dict[str, list[Path]] = field(default_factory=dict)  # "V5k" / "V50k"
    c3_runs: dict[str, list[Path]] = field(default_factory=dict)
    bootstrap_b: int = 10_000
    cache: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        for v in self.cache.values():
            if hasattr(v, "close"):
                v.close()
        self.cache.clear()

    def vault(self, which: str) -> Path | None:
        return {
            "fixture": self.fixture_vault,
            "snapshot": self.snapshot_vault,
            "pg": self.pg_vault,
        }[which]

    def routes_of(self, which: str) -> dict[str, str]:
        return self.routes if which == "fixture" else self.snapshot_routes

    def imported(self, which: str) -> VaultBackend:
        """The store of `which` vault, imported once and shared by the suites of one gate run."""
        key = f"imported:{which}"
        if key not in self.cache:
            root = self.vault(which)
            if root is None:
                raise BackendUnavailable(f"no {which} vault given")
            be = self.new_backend(f"{which}-main")
            be.import_vault(root, routes=self.routes_of(which))
            self.cache[key] = be
        return self.cache[key]


def setup_of(ctx: Context) -> P4Setup:
    s = ctx.options.get(OPT)
    if s is None:
        raise BackendUnavailable("no P4 setup: needs the mini-vault, the vault snapshot, the frozen "
                                 "PG-v1 and a backend; see deliverables/runbooks/p4-gate.md")  # fmt: skip
    return s


# ---- dumps ---------------------------------------------------------------------------------------


def store_dump(conns: dict[str, sqlite3.Connection]) -> list[str]:
    """Canonical rows of every table the import writes (B-I1: nodes, edges, occurrences, chunks,
    aliases, props, plus source_files) in every store, and the sha256 of each chunk's fp32
    embedding bytes."""
    import hashlib

    lines: list[str] = []
    for name, conn in sorted(conns.items()):
        conn.row_factory = sqlite3.Row
        try:
            for t in TABLES:
                cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({t})")]
                keep = [c for c in cols if c not in DUMP_DROP.get(t, ())]
                for row in conn.execute(f"SELECT {', '.join(keep)} FROM {t} ORDER BY 1, 2"):
                    lines.append(json.dumps({"store": name, "t": t,
                                             **dict(zip(keep, tuple(row), strict=True))},
                                            sort_keys=True, ensure_ascii=False))  # fmt: skip
            (vec,) = conn.execute("SELECT value FROM meta WHERE key = 'vec_table'").fetchone()
            for cid, blob in conn.execute(
                f"SELECT chunk_id, embedding FROM {vec} ORDER BY chunk_id"
            ):
                lines.append(json.dumps({"store": name, "t": "vec", "chunk_id": cid,
                                         "sha256": hashlib.sha256(bytes(blob)).hexdigest()},
                                        sort_keys=True))  # fmt: skip
        finally:
            conn.row_factory = None
    return lines


def diff_lines(a: list[str], b: list[str]) -> list[str]:
    return [ln for ln in difflib.unified_diff(a, b, lineterm="", n=0) if ln[:1] in "+-"
            and ln[:3] not in ("+++", "---")]  # fmt: skip


# ---- B-I1 ----------------------------------------------------------------------------------------


def run_b_i1(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    res = SuiteResult("B-I1", tier="t1", mode="import")
    any_target = False
    for which in ("fixture", "snapshot"):
        root = s.vault(which)
        if root is None:
            res.notes.append(f"no {which} vault: diff_lines_{which} not produced")
            continue
        any_target = True
        a = s.imported(which)
        b = s.new_backend(f"{which}-second")
        try:
            b.import_vault(root, routes=s.routes_of(which))
            da, db = store_dump(a.conns()), store_dump(b.conns())
        finally:
            b.close()
        d = diff_lines(da, db)
        res.metrics[f"diff_lines_{which}"] = float(len(d))
        res.metrics[f"dump_rows_{which}"] = float(len(da))
        res.check(f"B-I1 {which}: two fresh imports give one dump", not d, "; ".join(d[:6]))
    if not any_target:
        raise BackendUnavailable("B-I1 needs the mini-vault and the snapshot")
    return res


# ---- B-I2 ----------------------------------------------------------------------------------------


def _read_spec(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _edit(root: Path, edit: dict[str, str]) -> None:
    p = root / edit["path"]
    text = p.read_text(encoding="utf-8")
    if edit["find"] not in text:
        raise ValueError(f"lifecycle spec: {edit['find']!r} not in {edit['path']}")
    p.write_text(text.replace(edit["find"], edit["replace"], 1), encoding="utf-8")


def _find(be: VaultBackend, path: str) -> tuple[sqlite3.Connection, tuple[Any, ...]] | None:
    """The connection of the store that holds the import node of `path`, and the node row."""
    for conn in be.conns().values():
        row = conn.execute(
            "SELECT id, status, content_hash, source_ref FROM nodes"
            " WHERE origin = 'import' AND source_ref = ?",
            (path,),
        ).fetchone()
        if row:
            return conn, row
    return None


def _conn(be: VaultBackend, path: str) -> sqlite3.Connection:
    hit = _find(be, path)
    if hit is None:
        raise ValueError(f"{path} is in no store")
    return hit[0]


def _chunks_of(be: VaultBackend, path: str) -> dict[str, tuple[int, str, str]]:
    return {k: (i, h, t) for i, k, h, t in _conn(be, path).execute(
        "SELECT c.id, c.chunk_key, c.content_hash, c.text FROM chunks c JOIN nodes n"
        " ON n.id = c.node_id WHERE n.source_ref = ?", (path,))}  # fmt: skip


def _node(be: VaultBackend, path: str) -> tuple[Any, ...] | None:
    hit = _find(be, path)
    return hit[1] if hit else None


def _vec_ids(conn: sqlite3.Connection) -> set[int]:
    (vec,) = conn.execute("SELECT value FROM meta WHERE key = 'vec_table'").fetchone()
    return {r[0] for r in conn.execute(f"SELECT chunk_id FROM {vec}")}


def _agent_rows(be: VaultBackend, path: str) -> list[Any]:
    """Seed rows the importer must never touch: an agent alias, an agent prop, an agent edge."""
    conn, node = _find(be, path)  # type: ignore[misc]
    nid = node[0]
    conn.execute("INSERT OR IGNORE INTO aliases(alias, alias_norm, node_id, origin)"
                 " VALUES ('agent alias', 'agent alias', ?, 'agent')", (nid,))  # fmt: skip
    conn.execute("INSERT OR IGNORE INTO props(node_id, key, value, origin)"
                 " VALUES (?, 'agent_key', 'agent value', 'agent')", (nid,))  # fmt: skip
    other = conn.execute("SELECT id FROM nodes WHERE id <> ? AND kind IN ('note', 'entity')"
                         " ORDER BY id LIMIT 1", (nid,)).fetchone()  # fmt: skip
    if other:
        conn.execute("INSERT OR IGNORE INTO edges(src, dst, rel, origin, recorded_at)"
                     " VALUES (?, ?, 'relates_to', 'agent', '2026-10-22T09:00:00Z')",
                     (other[0], nid))  # fmt: skip
    return _agent_snapshot(be, path)


def _agent_snapshot(be: VaultBackend, path: str) -> list[Any]:
    conn, node = _find(be, path)  # type: ignore[misc]
    nid = node[0]
    return [
        conn.execute("SELECT alias, alias_norm FROM aliases WHERE node_id=? AND origin='agent'"
                     " ORDER BY 1", (nid,)).fetchall(),
        conn.execute("SELECT key, value FROM props WHERE node_id=? AND origin='agent' ORDER BY 1",
                     (nid,)).fetchall(),
        conn.execute("SELECT src, dst, rel, removed_at FROM edges WHERE dst=? AND origin='agent'"
                     " ORDER BY 1, 2", (nid,)).fetchall(),
    ]  # fmt: skip


def run_b_i2(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    if s.lifecycle is None or s.fixture_vault is None:
        raise BackendUnavailable("B-I2 needs the mini-vault and its lifecycle descriptor "
                                 "(bench/fixtures/vault-mini.lifecycle.yaml)")  # fmt: skip
    spec = _read_spec(s.lifecycle)
    res = SuiteResult("B-I2", tier="t0", mode="import")
    for name, fn in _CASES.items():
        with tempfile.TemporaryDirectory(prefix=f"bench-i2-{name}-") as tmp:
            vault = Path(tmp) / "vault"
            shutil.copytree(s.fixture_vault, vault)
            be = s.new_backend(f"i2-{name}")
            case = _Case(res, be, vault, spec, s.routes)
            try:
                rep = be.import_vault(vault, routes=s.routes)
                if rep.get("added", 0) == 0:
                    res.check(f"{name}: fixture imports", False, f"first import report {rep}")
                    continue
                try:
                    fn(case)
                except (KeyError, ValueError, AssertionError, StopIteration, TypeError) as e:
                    res.check(f"{name}: case runs", False, f"{type(e).__name__}: {e}")
            finally:
                be.close()
    res.metrics["case_failures"] = float(len({v.split(":")[0] for v in res.violations}))
    res.metrics["checks"] = float(len(res.items))
    return res


@dataclass
class _Case:
    res: SuiteResult
    be: VaultBackend
    vault: Path
    spec: dict[str, Any]
    routes: dict[str, str]

    def importing(self) -> dict[str, int]:
        return self.be.import_vault(self.vault, routes=self.routes)

    def dump(self) -> list[str]:
        return store_dump(self.be.conns())

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.res.check(name, ok, detail)


def _only(rep: dict[str, int], **expect: int) -> bool:
    return all(rep.get(k, 0) == v for k, v in expect.items()) and all(
        v == 0 for k, v in rep.items() if k not in expect
    )


def _case_a(c: _Case) -> None:
    before = c.dump()
    rep = c.importing()
    c.check(
        "a: re-import unchanged reports only unchanged",
        rep.get("unchanged", 0) > 0 and _only(rep, unchanged=rep.get("unchanged", 0)),
        str(rep),
    )
    c.check(
        "a: re-import unchanged writes nothing", not diff_lines(before, c.dump()), "dump changed"
    )


def _case_b(c: _Case) -> None:
    import os
    import time

    before = c.dump()
    new = time.time() + 3600
    os.utime(c.vault / c.spec["edit"]["path"], (new, new))
    rep = c.importing()
    c.check(
        "b: mtime-only touch reports only unchanged",
        _only(rep, unchanged=rep.get("unchanged", 0)),
        str(rep),
    )
    c.check("b: mtime-only touch writes nothing", not diff_lines(before, c.dump()), "dump changed")


def _case_c(c: _Case) -> None:
    e = c.spec["edit"]
    conn = _conn(c.be, e["path"])
    before, node_before = _chunks_of(c.be, e["path"]), _node(c.be, e["path"])
    others_before = {p: _chunks_of(c.be, p) for cn in c.be.conns().values() for (p,) in cn.execute(
        "SELECT source_ref FROM nodes WHERE origin='import' AND source_ref IS NOT NULL"
        " AND kind IN ('note', 'entity') AND source_ref <> ?", (e["path"],))}  # fmt: skip
    _edit(c.vault, e)
    rep = c.importing()
    after, vec_after = _chunks_of(c.be, e["path"]), _vec_ids(conn)
    key = e["chunk_key"]
    c.check("c: one changed note", rep.get("changed") == 1, str(rep))
    c.check("c: the edited chunk is replaced (new id, new hash)", key in before and key in after
            and after[key][0] != before[key][0] and after[key][1] != before[key][1],
            f"{before.get(key, ('-',))[0]} -> {after.get(key, ('-',))[0]}")  # fmt: skip
    c.check("c: the new chunk is re-embedded, the old vector is gone", after[key][0] in vec_after
            and before[key][0] not in vec_after, "vec rows")  # fmt: skip
    kept = {k: v[0] for k, v in after.items() if k != key}
    c.check("c: other chunk ids of the note are unchanged",
            kept == {k: v[0] for k, v in before.items() if k != key}, f"{kept}")  # fmt: skip
    c.check(
        "c: node content hash updated", _node(c.be, e["path"])[2] != node_before[2], "same hash"
    )
    c.check("c: other notes untouched", all(_chunks_of(c.be, p) == ch for p, ch in others_before.items()),
            "another note's chunks changed")  # fmt: skip
    c.check("c: unchanged chunk vectors survive", {v[0] for k, v in before.items() if k != key}
            <= vec_after, "a kept chunk lost its vector")  # fmt: skip


def _case_d(c: _Case) -> None:
    before = set(c.dump())
    add = c.spec["add"]
    (c.vault / add["path"]).parent.mkdir(parents=True, exist_ok=True)
    (c.vault / add["path"]).write_text(add["text"], encoding="utf-8")
    rep = c.importing()
    after = set(c.dump())
    new_node = _find(c.be, add["path"])
    c.check("d: one added note", rep.get("added") == 1, str(rep))
    c.check("d: no existing row changed or removed", before <= after,
            f"{len(before - after)} rows changed")  # fmt: skip
    new_chunks = [json.loads(ln) for ln in after - before if '"t": "chunks"' in ln]
    c.check("d: every new chunk belongs to the new note", new_node is not None and all(
        r["node_id"] == new_node[1][0] for r in new_chunks) and bool(new_chunks), "foreign chunk rows")  # fmt: skip


def _case_e(c: _Case) -> None:
    r = c.spec["rename"]
    node, chunks = _node(c.be, r["from"]), _chunks_of(c.be, r["from"])
    (c.vault / r["to"]).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(c.vault / r["from"], c.vault / r["to"])
    rep = c.importing()
    new = _node(c.be, r["to"])
    c.check("e: reported as one rename", rep.get("renamed") == 1 and rep.get("deleted", 0) == 0
            and rep.get("added", 0) == 0, str(rep))  # fmt: skip
    c.check("e: same node id, new source_ref", new is not None and new[0] == node[0], str(new))
    c.check("e: chunk ids unchanged", {k: v[0] for k, v in _chunks_of(c.be, r["to"]).items()}
            == {k: v[0] for k, v in chunks.items()}, "chunk ids moved")  # fmt: skip
    c.check("e: node still active", new is not None and new[1] == "active", str(new))


def _table_counts(be: VaultBackend) -> dict[str, int]:
    return {t: sum(cn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for cn in be.conns().values())
            for t in ("nodes", "chunks", "edges", "aliases", "props")}  # fmt: skip


def _case_f(c: _Case) -> None:
    path = c.spec["delete"]["path"]
    node, counts = _node(c.be, path), _table_counts(c.be)
    (c.vault / path).unlink()
    rep = c.importing()
    after = _table_counts(c.be)
    c.check("f: reported as one deletion", rep.get("deleted") == 1, str(rep))
    c.check("f: status deleted, same node", _node(c.be, path)[:2] == (node[0], "deleted"),
            str(_node(c.be, path)))  # fmt: skip
    c.check("f: nothing removed", after == counts, f"{counts} -> {after}")


def _case_g(c: _Case) -> None:
    path = c.spec["delete"]["path"]
    node = _node(c.be, path)
    chunks = {k: v[0] for k, v in _chunks_of(c.be, path).items()}
    agent = _agent_rows(c.be, path)
    text = (c.vault / path).read_text(encoding="utf-8")
    (c.vault / path).unlink()
    c.importing()
    (c.vault / path).write_text(text, encoding="utf-8")
    rep = c.importing()
    c.check("g: reported as one restore", rep.get("restored") == 1 and rep.get("added", 0) == 0,
            str(rep))  # fmt: skip
    c.check("g: the same node is reactivated", _node(c.be, path)[:2] == (node[0], "active"),
            str(_node(c.be, path)))  # fmt: skip
    c.check("g: chunk ids unchanged", {k: v[0] for k, v in _chunks_of(c.be, path).items()} == chunks,
            "chunk ids changed")  # fmt: skip
    c.check("g: agent rows intact", _agent_snapshot(c.be, path) == agent, "agent rows changed")


def _case_h(c: _Case) -> None:
    h = c.spec["remove"]
    path = h["path"]
    conn, node = _find(c.be, path)  # type: ignore[misc]
    agent = _agent_rows(c.be, path)
    for key in ("alias_edit", "tag_edit", "prop_edit", "link_edit"):
        _edit(c.vault, {"path": path, **h[key]})
    rep = c.importing()
    c.check("h: one changed note", rep.get("changed") == 1, str(rep))
    alias = h["alias"].casefold()
    gone_alias = not conn.execute("SELECT 1 FROM aliases WHERE node_id=? AND alias_norm=?"
                                  " AND origin='import'", (node[0], alias)).fetchone()  # fmt: skip
    fts = conn.execute("SELECT count(*) FROM aliases_tri WHERE aliases_tri MATCH ?",
                       (f'"{alias}"',)).fetchone()[0] if len(alias) >= 3 else 0  # fmt: skip
    c.check("h: the removed alias is gone from aliases and its FTS", gone_alias and fts == 0,
            f"alias row gone={gone_alias}, fts hits={fts}")  # fmt: skip
    tag_edge = conn.execute(
        "SELECT e.removed_at FROM edges e JOIN nodes t ON t.id=e.dst WHERE e.src=? AND e.rel='tagged'"
        " AND t.name=? ORDER BY e.id DESC", (node[0], h["tag"])).fetchall()  # fmt: skip
    c.check("h: the removed tag edge has removed_at and is not live", bool(tag_edge)
            and all(r[0] for r in tag_edge), f"{tag_edge}")  # fmt: skip
    c.check("h: the removed prop is gone", not conn.execute(
        "SELECT 1 FROM props WHERE node_id=? AND key=? AND origin='import'",
        (node[0], h["prop_key"])).fetchone(), "prop row still there")  # fmt: skip
    link = conn.execute(
        "SELECT e.removed_at, (SELECT count(*) FROM link_occurrences o WHERE o.edge_id=e.id)"
        " FROM edges e JOIN nodes d ON d.id=e.dst WHERE e.src=? AND e.origin='import' AND e.rel=?"
        " AND (d.source_ref = ? OR d.name = ? COLLATE NOCASE)",
        (node[0], h["link_rel"], h["link_target"], Path(h["link_target"]).stem)).fetchall()  # fmt: skip
    c.check("h: the removed link edge has removed_at (row kept), no live occurrences",
            bool(link) and all(r[0] for r in link), f"{link}")  # fmt: skip
    c.check("h: agent rows untouched", _agent_snapshot(c.be, path) == agent, "agent rows changed")


def _case_i(c: _Case) -> None:
    ins = c.spec["insert_section"]
    before = _chunks_of(c.be, ins["path"])
    p = c.vault / ins["path"]
    text = p.read_text(encoding="utf-8")
    if ins["before"] not in text:
        raise ValueError(f"lifecycle spec: {ins['before']!r} not in {ins['path']}")
    p.write_text(text.replace(ins["before"], ins["text"].rstrip("\n") + "\n\n" + ins["before"], 1),
                 encoding="utf-8")  # fmt: skip
    rep = c.importing()
    after = _chunks_of(c.be, ins["path"])
    c.check("i: one changed note", rep.get("changed") == 1, str(rep))
    c.check("i: existing chunk ids are unchanged", all(k in after and after[k][0] == v[0]
                                                       for k, v in before.items()),
            "an existing chunk id moved or vanished")  # fmt: skip
    c.check("i: old ids still resolve to the same text", all(after[k][2] == v[2] for k, v in before.items()
                                                             if k in after), "text changed")  # fmt: skip
    c.check("i: a new chunk exists", len(after) > len(before), f"{len(before)} -> {len(after)}")


_CASES: dict[str, Callable[[_Case], None]] = {
    "a": _case_a, "b": _case_b, "c": _case_c, "d": _case_d, "e": _case_e, "f": _case_f,
    "g": _case_g, "h": _case_h, "i": _case_i}  # fmt: skip


# ---- B-I3 ----------------------------------------------------------------------------------------


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def store_view(conns: dict[str, sqlite3.Connection]) -> dict[str, Any]:
    """What the stores hold, in the reference parser's terms (`ref_parse.RefVault`)."""
    notes: dict[str, tuple[sqlite3.Connection, int, str]] = {}
    for conn in conns.values():
        for nid, p, name in conn.execute(
                "SELECT id, source_ref, name FROM nodes WHERE origin='import' AND source_ref IS NOT NULL"
                " AND kind IN ('note', 'entity') AND status='active' AND coalesce(type, '') <> 'file'"):  # fmt: skip
            notes[p] = (conn, nid, name)
    dst_key = lambda ref, name: ref if ref else f"stub:{name.casefold()}"
    occ: list[tuple[str, str, str]] = []
    edges: set[tuple[str, str, str]] = set()
    tags: set[tuple[str, str]] = set()
    fm_rows: set[str] = set()
    for p, (conn, nid, name) in notes.items():
        for eid, rel, dref, dname in conn.execute(
                "SELECT e.id, e.rel, d.source_ref, d.name FROM edges e JOIN nodes d ON d.id=e.dst"
                " WHERE e.src=? AND e.origin='import' AND e.removed_at IS NULL", (nid,)):  # fmt: skip
            if rel == "tagged":
                tags.add((p, dname))
            elif rel != "part_of":
                edges.add((p, dst_key(dref, dname), rel))
                n = conn.execute("SELECT count(*) FROM link_occurrences WHERE edge_id=?",
                                 (eid,)).fetchone()[0]  # fmt: skip
                occ += [(p, dst_key(dref, dname), rel)] * n
        stem = Path(p).stem
        extra_alias = {
            a
            for (a,) in conn.execute(
                "SELECT alias_norm FROM aliases WHERE node_id=? AND origin='import'", (nid,)
            )
        } - {_norm(name), _norm(stem), _norm(re.sub(r"^\d{4}-\d{2}-\d{2}[_ -]", "", stem))}
        has_prop = conn.execute("SELECT 1 FROM props WHERE node_id=? AND origin='import'",
                                (nid,)).fetchone()  # fmt: skip
        has_field = conn.execute(
            "SELECT 1 FROM link_occurrences o JOIN edges e ON e.id=o.edge_id WHERE e.src=?"
            " AND o.field IS NOT NULL AND e.removed_at IS NULL", (nid,)).fetchone()  # fmt: skip
        if extra_alias or has_prop or has_field:
            fm_rows.add(p)
    return {"notes": set(notes), "occurrences": occ, "edges": edges, "tag_edges": tags,
            "frontmatter_row_notes": fm_rows}  # fmt: skip


def compare_to_reference(
    conns: dict[str, sqlite3.Connection], root: Path
) -> tuple[dict[str, Any], list[str]]:
    ref = ref_parse.parse_vault(root)
    st = store_view(conns)
    mism: list[str] = []
    counts: dict[str, Any] = {}

    def setdiff(name: str, a: set, b: set) -> None:
        for x in sorted(a - b):
            mism.append(f"{name}: only in the reference parser: {x}")
        for x in sorted(b - a):
            mism.append(f"{name}: only in the store: {x}")

    setdiff("notes", set(ref.notes), st["notes"])
    setdiff("edges", ref.edges, st["edges"])
    setdiff("tag edges", ref.tag_edges, st["tag_edges"])
    ref_fm = {p for p, n in ref.notes.items() if n.has_frontmatter_rows}
    setdiff("frontmatter notes", ref_fm, st["frontmatter_row_notes"])
    ro, so = Counter(ref.occurrences), Counter(st["occurrences"])
    for k in sorted(set(ro) | set(so)):
        if ro[k] != so[k]:
            mism.append(f"link occurrences {k}: reference {ro[k]}, store {so[k]}")
    counts = {"notes": (len(ref.notes), len(st["notes"])),
              "frontmatter_notes (any, reference only)": (ref.counts()["frontmatter_notes"], None),
              "frontmatter notes with props/aliases/links": (len(ref_fm), len(st["frontmatter_row_notes"])),
              "link occurrences": (len(ref.occurrences), len(st["occurrences"])),
              "logical edges": (len(ref.edges), len(st["edges"])),
              "tag edges": (len(ref.tag_edges), len(st["tag_edges"]))}  # fmt: skip
    for name, (a, b) in counts.items():
        if b is not None and a != b:
            mism.append(f"count {name}: reference {a}, store {b}")
    return {"counts": counts, "edges_by_rel": ref.edges_by_rel()}, mism


def run_b_i3(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    res = SuiteResult("B-I3", tier="t1", mode="import")
    produced = False
    for which in ("fixture", "snapshot"):
        if s.vault(which) is None:
            res.notes.append(f"no {which} vault: mismatches_{which} not produced")
            continue
        produced = True
        be = s.imported(which)
        info, mism = compare_to_reference(be.conns(), s.vault(which))  # type: ignore[arg-type]
        res.metrics[f"mismatches_{which}"] = float(len(mism))
        res.items.append({"check": f"B-I3 {which}", "ok": not mism, "counts": {
            k: list(v) for k, v in info["counts"].items()}, "edges_by_rel": info["edges_by_rel"],
            "mismatches": mism[:200]})  # fmt: skip
        if mism:
            res.violations.append(f"{which}: {len(mism)} mismatch(es), first: {mism[0]}")
    if not produced:
        raise BackendUnavailable("B-I3 needs the mini-vault and the snapshot")
    res.notes.append("expected on the 2026-10-22 snapshot (benchmarks-v2 B-I): 354 notes, 1,059 "
                     "link occurrences, 340 frontmatter notes; the gate compares importer and "
                     "reference parser on the snapshot actually used")  # fmt: skip
    return res


# ---- B-I4 ----------------------------------------------------------------------------------------


def _ws(s: str) -> str:
    return " ".join(s.split())


def source_fidelity(conns: dict[str, sqlite3.Connection], root: Path) -> list[str]:
    """Per note, the chunk texts in order, joined, equal the normalised body (whitespace runs
    collapsed: chunk boundaries trim blank lines, nothing else may differ)."""
    bad = []
    for path, text in sorted(ref_parse.scan(root).items()):
        hit = next(((cn, r) for cn in conns.values() if (r := cn.execute(
            "SELECT id FROM nodes WHERE origin='import' AND source_ref=? AND status='active'",
            (path,)).fetchone())), None)  # fmt: skip
        if hit is None:
            bad.append(f"{path}: not imported")
            continue
        conn, row = hit
        got = _ws("\n".join(t for (t,) in conn.execute(
            "SELECT text FROM chunks WHERE node_id=? ORDER BY ord, id", (row[0],))))  # fmt: skip
        want = _ws(ref_parse.normalised_body(text))
        if got != want:
            n = next((i for i, (x, y) in enumerate(zip(got, want, strict=False)) if x != y),
                     min(len(got), len(want)))  # fmt: skip
            bad.append(f"{path}: chunks differ from the normalised body at char {n} "
                       f"(chunks {len(got)} chars, body {len(want)})")  # fmt: skip
    return bad


def tree_files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}  # fmt: skip


def run_b_i4(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    if s.render is None:
        raise BackendUnavailable("B-I4 needs `mem render` (MEM2-33): no renderer given")
    res = SuiteResult("B-I4", tier="t1", mode="import")
    for which in ("fixture", "snapshot"):
        root = s.vault(which)
        if root is None:
            res.notes.append(f"no {which} vault: source_fidelity_{which} not produced")
            continue
        bad = source_fidelity(s.imported(which).conns(), root)
        res.metrics[f"source_fidelity_failures_{which}"] = float(len(bad))
        res.check(f"B-I4 (1) {which}: chunks reconstruct the normalised source", not bad,
                  "; ".join(bad[:5]))  # fmt: skip
    be = s.new_backend("i4-render")
    try:
        if s.fixture_vault is None:
            raise BackendUnavailable("B-I4 (2)/(3) run on the mini-vault: none given")
        be.import_vault(s.fixture_vault, routes=s.routes)
        store = be.store("personal")
        with tempfile.TemporaryDirectory(prefix="bench-i4-") as tmp:
            out = Path(tmp)
            store.remember(text="Bench render probe one is a durable fact about the render check.",
                           type="fact", about=["new:topic:Render probe one"])  # fmt: skip
            m2 = store.remember(text="Bench render probe two is another fact for the render check.",
                                type="fact", about=["new:topic:Render probe two"])  # fmt: skip
            s.render(list(be.dbs.values()), out / "a1")
            s.render(list(be.dbs.values()), out / "a2")
            nondet = diff_trees(tree_files(out / "a1"), tree_files(out / "a2"))
            res.metrics["render_nondeterminism"] = float(len(nondet))
            res.check(
                "B-I4 (2): `mem render` twice is byte-identical", not nondet, "; ".join(nondet[:5])
            )
            victim = m2.structured["id"]
            store.forget(id=victim, reason="B-I4 probe: one memory changes")
            s.render(list(be.dbs.values()), out / "b")
            changed = diff_trees(tree_files(out / "a1"), tree_files(out / "b"))
            res.metrics["render_unconfined_files"] = float(max(len(changed) - 1, 0))
            res.check("B-I4 (3): one memory change differs in exactly that memory's file",
                      len(changed) == 1 and str(victim) in changed[0], "; ".join(changed[:5]) or
                      "no file changed")  # fmt: skip
    finally:
        be.close()
    res.metrics["source_fidelity_failures"] = float(sum(
        v for k, v in res.metrics.items() if k.startswith("source_fidelity_failures_")))  # fmt: skip
    return res


def diff_trees(a: dict[str, bytes], b: dict[str, bytes]) -> list[str]:
    out = [f"only in A: {p}" for p in sorted(set(a) - set(b))]
    out += [f"only in B: {p}" for p in sorted(set(b) - set(a))]
    out += [f"differs: {p}" for p in sorted(set(a) & set(b)) if a[p] != b[p]]
    return out


# ---- B-I5, B-C4 ----------------------------------------------------------------------------------


def run_b_i5(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    if s.i5_sheet is None:
        raise BackendUnavailable("B-I5 needs the labelled sheet (`bench i5 label`, ~30 min)")
    res = SuiteResult("B-I5", tier="t1", mode="import")
    rows = i5.read_sheet(s.i5_sheet)
    sc = i5.score(rows)
    res.metrics |= {"precision": sc.precision, "rel_precision_min": sc.min_rel_precision,
                    "n_labelled": float(sc.n), "unlabelled": float(sc.unlabelled)}  # fmt: skip
    res.items = [
        {"check": "B-I5", "rel": r, "correct": c, "n": n} for r, (c, n) in sc.by_rel.items()
    ]
    res.extra["wrong"] = sc.wrong
    if s.snapshot_vault is not None:
        fresh = i5.sample(s.imported("snapshot").conns())
        same = [r["edge_id"] for r in fresh] == [r["edge_id"] for r in rows]
        res.metrics["sheet_matches_store"] = float(same)
        res.check("B-I5: the sheet is this store's seed-7 sample", same,
                  "edge ids differ: re-sample with `bench i5 sample` and label again")  # fmt: skip
    return res


def run_b_c4(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    if s.c4_run is None:
        raise BackendUnavailable(
            "B-C4 needs a `bench c4-run` record (cold + warm import on the host)"
        )
    rec = json.loads(Path(s.c4_run).read_text(encoding="utf-8"))
    res = SuiteResult("B-C4", tier="t1", mode="import")
    _anchor_c4_record(s, rec)
    res.metrics |= {"cold_s": float(rec["cold_s"]), "warm_s": float(rec["warm_s"]),
                    "warm_all_unchanged": float(rec["warm_all_unchanged"])}  # fmt: skip
    res.items = [rec]
    res.notes.append(f"device {rec.get('device')}, threads {rec.get('omp_num_threads')}; cold "
                     "includes embedding, excludes the model load of `mem serve`")  # fmt: skip
    return res


def _anchor_c4_record(s: P4Setup, rec: dict[str, Any]) -> None:
    """A B-C4 record counts only for this snapshot, this commit and this import (its chunk count):
    a record from another snapshot, an older commit or a different vault is rejected, not scored."""
    want_sha = s.dataset_meta.get("snapshot_sha256")
    if not want_sha:
        raise BackendUnavailable("B-C4 needs the snapshot (--snapshot, --snapshot-sha256) to check "
                                 "that the c4-run record was measured on it")  # fmt: skip
    be = s.imported("snapshot" if s.snapshot_vault is not None else "fixture")
    want = {"snapshot_sha256": want_sha, "git_sha": records.git_sha(), "chunks": len(be.index)}
    bad = [
        f"{k}: record {rec.get(k)!r}, this run {v!r}" for k, v in want.items() if rec.get(k) != v
    ]
    if bad:
        raise ValueError("the B-C4 record does not match this run (" + "; ".join(bad) + "): "
                         "re-run `bench c4-run` on this snapshot and commit")  # fmt: skip


def c4_measure(
    be: VaultBackend, vault: Path, clock: Callable[[], float], routes: dict[str, str] | None = None
) -> dict[str, Any]:
    """B-C4: cold full import with embeddings, then the warm no-op re-import, on one store."""
    import os

    be.conns()  # models loaded and the stores created before the clock starts (serve start is out)
    t0 = clock()
    cold = be.import_vault(vault, routes=routes)
    t1 = clock()
    warm = be.import_vault(vault, routes=routes)
    t2 = clock()
    return {"cold_s": t1 - t0, "warm_s": t2 - t1, "cold_report": cold, "warm_report": warm,
            "warm_all_unchanged": _only(warm, unchanged=warm.get("unchanged", 0))
            and warm.get("unchanged", 0) > 0, "chunks": len(be.index),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}  # fmt: skip


# ---- B-R3 ----------------------------------------------------------------------------------------


def _load_baselines(s: P4Setup) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    runs, meta = {}, {}
    for name, path in s.baselines.items():
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        runs[name] = rec["runs"]
        meta[name] = {k: v for k, v in rec.items() if k != "runs"}
    return runs, meta


def run_b_r3(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    if s.gold is None or s.pg_vault is None:
        raise BackendUnavailable(
            "B-R3 needs the frozen PG-v1 and the PG snapshot it was built on (no work notes)"
        )
    be = s.imported("pg")
    res = SuiteResult("B-R3", tier="t1", mode="import")
    stale = pg.stale_gold(s.gold, be.index)
    if stale:  # §2 B-R3 step 3: an unmapped gold id fails the run
        res.check("B-R3: every gold id maps to a chunk of the import, unchanged", False,
                  f"{len(stale)} stale: {'; '.join(stale[:4])}")  # fmt: skip
        res.metrics["stale_gold"] = float(len(stale))
        return res
    res.metrics["stale_gold"] = 0.0
    questions = s.gold.of_split(s.split)
    items = pg_eval.run_pg(be, questions)  # type: ignore[arg-type]
    baselines, bmeta = _load_baselines(s)
    sc = pg_eval.score_pg(items, baselines=baselines, b=s.bootstrap_b)
    res.items = [i.record() for i in items]
    res.metrics |= sc.metrics
    res.notes += sc.notes
    res.extra |= {"ci95": {k: list(v) for k, v in sc.ci95.items()}, "per_category": sc.per_category,
                  "n": sc.n, "split": s.split, "baselines": bmeta,
                  "dataset": {"name": "PG-v1", "version": s.gold.version, **s.dataset_meta}}  # fmt: skip
    res.notes.append("DR-4 compares path-level nDCG@10 (std); qmd and rg are descriptive")
    return res


# ---- B-C1 / B-C3 at V5k -----------------------------------------------------------------------------


def _aggregate_corpora(res: SuiteResult, runs: dict[str, list[Path]], what: str) -> None:
    for corpus in ("V5k", "V50k"):
        paths = runs.get(corpus, [])
        if len(paths) < latency.RUNS_REQUIRED:
            if corpus == "V5k":
                raise BackendUnavailable(
                    f"{what} needs {latency.RUNS_REQUIRED} runs at V5k, got {len(paths)}"
                )
            res.notes.append(f"V50k: {len(paths)} runs, not recorded")
            continue
        loaded = latency.load_runs(paths)
        agg = latency.aggregate(loaded)
        suffix = "" if corpus == "V5k" else "_v50k"
        for k in ("warm_p95_ms_median", "warm_p95_ms_max", "warm_p95_4clients_ms_median",
                  "cold_ms_median"):  # fmt: skip
            res.metrics[f"{k}{suffix}"] = agg[k]
        res.items += [{"corpus": corpus, "run": str(p), "cold_ms": r["cold_ms"],
                       "p95": r["one_client"]["total"]["p95"]} for p, r in zip(paths, loaded, strict=True)]  # fmt: skip


def run_b_c1(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    res = SuiteResult("B-C1", tier="t1", mode="import")
    _aggregate_corpora(res, s.latency_runs, "B-C1")
    res.notes.append("B-C1 at V5k (PG queries) on the serving host; median of 3 runs' warm p95")
    return res


def run_b_c3(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    res = SuiteResult("B-C3", tier="t1", mode="oracle_client")
    _aggregate_corpora(res, s.c3_runs, "B-C3")
    res.notes.append("B-C3 at V5k (SYN-M remember replay) on the serving host")
    return res
