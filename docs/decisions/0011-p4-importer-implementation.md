# 0011: P4 importer: parser, ids, link resolution, edge cases

**Status:** accepted for MEM2-32. Implements architecture-v2 §2.4 and decision 0010; no design change.
Items marked **Q** are open questions for the maintainer (answers are cheap to apply).

**Decisions**
1. **`markdown-it-py` is a direct dependency** (`pyproject.toml`, `uv.lock`; the spec's component
   table names it, and it was already in the lock through typer/rich). It gives headings, top-level
   blocks and code ranges with line maps, so fences, callouts and setext headings are not
   re-implemented. Rejected: a regex line scanner (misreads `#` in indented code and html blocks).
2. **Two modules.** `mem.vaultparse` is pure (note text → `ParsedNote`); `mem.importer` is the scan,
   classification and per-file transaction. `mem import` (cli) embeds through the running
   `mem serve` (D5, `models_http.connect`); `--embedder hash` is the test stub.
3. **A replaced chunk is inserted before the old row is deleted** (the key is `…\0new` until then).
   `chunks.id` is an `INTEGER PRIMARY KEY` without `AUTOINCREMENT`, so deleting first hands the old
   id to the new row when it was the highest one, and a stale id would resolve to a different
   section, which §2.3 forbids. `test_stale_chunk_id_reported_gone` fails if the order is swapped.
   **Q1:** the same reuse happens when a chunk is only deleted and a later write (importer or
   `remember`) takes the highest id. Fixing it needs `AUTOINCREMENT` on `chunks` (and `nodes`), a
   table-rebuild migration 0002; I did not make a schema change without asking.
4. **Occurrences (A3 step 4.5):** all body occurrences of the node's chunks and all frontmatter
   occurrences are deleted and re-inserted from the new plan, so a kept chunk cannot hit `occ_body`.
5. **A10 (`#n` shift on duplicate headings):** kept as specified, no design change. Inserting a
   section that repeats an earlier heading path renumbers the later ones; the chunks whose key now
   holds other text get new ids and their old ids are gone, never reassigned
   (`test_inserting_a_duplicate_heading_never_reuses_a_stale_chunk_id`). Moving a row to its new key
   when the text hash matches would keep the ids, but it changes the §2.4 rule ("same key and hash").
   **Q2:** want that?
6. **Future-dated `created`** is kept as stated (0010; test). **Q3:** confirmed? A note dated 2099 is
   hidden from default `recall` until then (review "noted, not raised").
7. **Chunk text is the section body without heading lines**, merged tiny sections included (the
   frozen test requires 250 words per essay chunk). B-I4(1) says "concatenated chunk texts == the
   normalised body", which cannot hold with headings dropped. **Q4:** B-I4(1) should compare
   `heading_path` + text, or the test should accept the heading lines; the renderer task decides.
8. **Unchanged means same hash *and* same resolved link targets.** Every note is parsed each run
   (cheap); if the hash is equal but the set of (target, rel) differs from the live import edges
   (a page was created for an unresolved link, a target was renamed), the note is re-applied and
   counted `changed`. Otherwise a link would stay on its stub forever. Zero writes still holds when
   nothing differs.
9. **Link resolution** is per store (a note links only to files of its own store): exact path,
   then unique name, then shortest path, case-insensitive; a note whose file is gone (deleted node)
   answers only when no file does, so links to a deleted note keep their edge. Links to a note of the
   *other* store become a stub topic in this store. **Q5:** is that right for `work/` ↔ `personal/`?
10. **Targets that do not exist yet** (a note later in the same run) get a shell node (`kind=note`,
    no hash) that the later file fills in, so one transaction per file still sees every target id.
11. **Frontmatter link keys** become rels by singularising the key (`attendees` → `attendee`) and
    are registered in `rels`; a key that equals a core rel (`about`, `tagged`, …) becomes `fm_<key>`.
12. **Skip list** (matches §2.4): `.obsidian/ .smart-env/ .stfolder/ node_modules/` at any depth,
    `Templates/` and the render dir at the vault root, `CLAUDE.md` and `obsidian-instructions.md` at
    any depth. **Files under no route** (a root-level or unknown folder) are not imported and are
    listed in the report and by `mem import` (**Q6:** route them to personal instead?).
13. **`chunks.props`** holds callout types and block ids; the line range of §2.3 is not stored
    (comment stripping makes file line numbers unreliable, and nothing reads it yet).
14. Unreadable files (bad frontmatter YAML, not UTF-8) are reported, left as they were (never
    marked deleted), and make `mem import` exit 1 after importing the rest.
