# Obsidian vault: import and render

mem can read an [Obsidian](https://obsidian.md) vault into its stores (so notes are searchable next
to memories) and write the stores back out as Markdown (a one-way mirror you can browse or keep in
git). The importer never writes to your vault.

## Import

The importer embeds new chunks through the running `mem serve` (it loads no model itself), so start
the service first.

```bash
uv run --extra models mem import ~/path/to/vault
```

Notes are routed to a store by folder: `personal/` and `projects/` go to `personal`, `work/` to
`work`. Add or change routes in `config.yaml`; the longest folder prefix wins:

```yaml
vault:
  routes:
    projects/client-x: work
```

A note outside every route is skipped with `not imported, folder has no route`.

It prints one line, `added N  changed N  renamed N  deleted N  restored N  unchanged N`. Then:

- Run it again straight away: the line must read `unchanged <all>` and nothing else. `mem import` is
  idempotent and safe to interrupt; each note is one transaction.
- Edit, rename or delete a note and run it again: `changed`, `renamed` or `deleted` is 1.
- A note it cannot read prints `[FAIL] path: reason` and the exit code is 1; the rest is imported.

It ignores `export/` (the render folder), `.obsidian/`, `Templates/`, `CLAUDE.md` and
`obsidian-instructions.md`, and it never touches memories saved through Claude.

Imported notes appear in `recall` with kind `note` and are read-only through MCP: to change one,
edit the file and import again. `link` and `forget` refuse a note (a note can still be the target of
a `relates_to` link).

## Render

```bash
uv run --extra models mem render \
  --store ~/.local/share/mem/personal.db --store ~/.local/share/mem/work.db \
  --out ~/mem-export --html --commit
```

This writes Markdown pages for the memories and subjects (and HTML with `--html`) under
`~/mem-export/<store>/`, and with `--commit` makes a git commit of the rendered folders. The output
is byte-identical for an identical store: a second run prints `0 written, N unchanged, 0 removed` and
`nothing to commit`, and each day's changes are one commit. A daily mirror is one cron line:

```
17 3 * * * cd /path/to/mem && uv run --extra models mem render --store ... --out ~/mem-export --html --commit
```

Do not put `--out` inside your vault unless it is the `export/` folder, which the importer skips.
