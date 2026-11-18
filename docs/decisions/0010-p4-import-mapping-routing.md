# 0010: P4 import mapping, routing (Q3) and the frozen mini-vault

**Status:** accepted for the fixture (MEM2-31); the importer (next P4 tasks) must meet it.

**Decision (the person's Q3 answer, 2026-10-24, on MEM2-22)**
1. `personal/` and `projects/` → `personal.db`; `work/` → `work.db`, as architecture-v2 §2.4.
2. Per-folder overrides: a folder can be mapped to another store in `config.yaml`; the override
   wins over the default. Rejected: no overrides (the person asked for them); a `.mem` file
   inside each folder (scatters routing across the vault, not reviewable in one place).
3. The fixture proves it: `projects/client-x/` is mapped to `work` and its note lands in `work.db`.

**Readings I froze where the spec is silent** (each is checkable in `tests/test_p4_vault_fixture.py`
and `tests/fixtures/minivault/expected_s1.json`; change them only with a new record)
- Override syntax: top-level `vault: {routes: {<folder>: personal|work}}` in `config.yaml`, exposed as
  `Config.vault_routes`. `config.load` rejects unknown keys today, so the loader must learn `vault`.
  The longest matching folder prefix wins.
- Importer API: `mem.importer.run_import(vault, stores: {"personal": Store, "work": Store}, *, routes,
  render_dir="export") -> report` with int fields `added/changed/renamed/deleted/restored/unchanged`
  summed over both stores. Dates use the store's `tz`; the `mtime` fallback comes from the file.
- `heading_path` = headings H1..Hn joined by `" > "` (no title is added; a note with no H1 starts at
  its first heading); text before the first heading has an empty path. A tiny section (under 40
  tokens, including an empty H1 body) merges into the next one and the merged chunk takes the
  **next** section's key. A last section that is tiny has nothing to merge into and stays.
- Nested tag `a/b`: tag nodes `a/b` and `a`, edge `part_of` from `a/b` to `a`. Frontmatter `tags:` and
  inline `#tag` both give `tagged` edges. `created`, `aliases` and `tags` are not `props`.
- Any frontmatter value that is a wikilink gives an edge with the key as rel and a `field`
  occurrence (`owner`, `project`, `reports_to`); `attendees` → rel `attendee`.
- A link to an unindexed file that exists in the vault (`.png`, `.canvas`) is a stub
  `note`/`file` node carrying its `source_ref`; an unresolved name is a `topic` stub without one.
- Render folder default name `export/`.
- **Future-dated `created` (2099-01-01) is kept as stated**; the spec has no clamp. Open question for
  Maintainer: confirm, or say what the importer should do instead.

**Not covered here:** links across stores (a note in `work/` linking to one in `personal/`); the
fixture has none and §2.4 does not say. Left for the importer task to ask about.

**Tests state:** all new tests are strict-xfail (they fail now with `ModuleNotFoundError: mem.importer` (the config test with
`ConfigError: unknown config key 'vault'`): the feature does not exist yet.
