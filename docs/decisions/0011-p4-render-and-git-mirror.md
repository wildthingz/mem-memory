# 0011: P4 render layout, jinja2, and the git mirror

Status: accepted · Task MEM2-33 · Spec: architecture-v2 §2.1/§2.2, DR-5, A14, B-I4

## Decision
- `mem render --store PATH… --out DIR [--html] [--commit]` writes `DIR/<scope>/` only: `index`, `memories/m<id>`,
  `entities/e<id>`, `notes/<source_ref without .md>` (`.md`, plus `.html` with `--html`). Pages are
  jinja2 templates in `src/mem/templates/`; the store is opened read-only.
- Deterministic: ordered by id/path, no render time or host in any file, LF, UTF-8. A file is rewritten only when
  its bytes change; files no longer rendered are pruned, but only under `DIR/<scope>/` (so `DIR/.git` and
  other content survive). Note paths drop `..`/empty parts and are checked to stay inside the scope dir.
- Status lives only on a node's own page (index and entity pages list ids and titles), so one memory change
  changes only that memory's pages (B-I4 (3)).
- A note page is `---` front matter `---` + the chunk texts concatenated in `ord` order, verbatim:
  stripping the front matter gives the normalised source (B-I4 (1)).
- `--commit` = `git init` if needed, `git add -A`, one commit "mem render" (fixed message, identity `mem`) only
  when something changed. It is a readable mirror, not a backup (§5).
- `jinja2>=3.1` becomes a direct dependency (named by the spec; already in uv.lock transitively).

## Rejected
- Rendering Markdown → HTML with markdown-it-py: new dependency and a parser version in the output bytes.
  HTML shows the escaped source in `<pre>` with the metadata and links instead.
- A commit message with a timestamp or counts: breaks byte-identical history for an identical store.
- Deleting everything in `--out` before rendering: would remove `.git` and rewrite every mtime.

## Consequences
Wikilinks inside note bodies are not rewritten (the render is a mirror, not a site). Edges other than
`about`/`supersedes` are not shown yet.
