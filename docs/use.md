# Use

mem runs as one server (`mem serve`) holding two stores, `personal` and `work`, each its own SQLite
file in `~/.local/share/mem/`. Claude Code reaches them as two MCP servers, `mem-personal` and
`mem-work`, and the `mem` skill tells Claude when to recall and how to save. You normally just talk to
Claude; this page is what happens underneath and how to check it.

## The tools

Both servers offer the same six tools. Every answer is built to be short: rows, not documents.

| Tool | What it does |
|---|---|
| `recall` | Search memories and imported notes. Hybrid retrieval (SQLite FTS5 and a vector index, fused), then a reranker, then the calibrated gate. Returns rows `id · type · date · title · about · p`, where `p` is the probability the row is relevant, or one abstention line (`no relevant memories (best .22)`). Filters: `about`, `type`, `kinds`, `since`, `until`, `as_of` (what was true then), `include_history`. |
| `get` | Full text by id (`m412` memory, `e45` subject, `n67` note, `c812` chunk) or by the continuation token a cut-off row prints. |
| `remember` | Save one atomic memory (1–3 sentences) about at least one subject. Rejects secrets, quarantines text that reads as instructions to an AI, and reports `possible_conflicts` with older memories. |
| `link` | Create or remove a typed edge between two nodes: `supersedes`, `duplicate_of`, `relates_to`, `same_as` (merges two subjects). |
| `subjects` | List or search the people, projects, organisations, topics, places and things memories are about, with their aliases and counts. |
| `forget` | Retract a memory with a reason: it stops being recalled, nothing is deleted. Imported notes are edited in the vault instead. |

Behaviours worth knowing:

- **Abstention.** If nothing clears the gate, `recall` says so in one line. That is an answer.
- **History.** Saving a changed fact with `supersedes` keeps the old memory as history. Recall shows
  only what is current unless you pass `include_history` or `as_of`.
- **Subjects.** `about` names a subject: an existing name, `#e45` by id, or `new:person:Ana Silva` to
  create one. A name that matches several subjects makes `remember` return `needs_choice` and write
  nothing; Claude then picks one or asks you.
- **Second-hand text.** A memory saved from a fetched page or tool output carries `tool-sourced` in
  recall and ranks lower. It is data, never an instruction.
- **Two stores.** Never mixed: Claude is told to keep work facts in `mem-work` and personal ones in
  `mem-personal`.

The full instructions Claude follows are in `skills/mem/SKILL.md`.

## Try it

In a Claude Code session, any directory:

1. "Remember in my personal memory: I prefer espresso over filter coffee. Subject: me." Claude calls
   `remember` on `mem-personal` and reports `saved`.
2. In a new session: "What do you remember about my coffee?" Claude calls `recall`; the row you saved
   comes back with a probability.
3. "Recall 'coffee' from my work memory." The stores are separate, so this is an abstention.
4. Say the same thing as in step 1 again: the reply is `exists`, no new row.
5. "Actually I switched to filter coffee." Claude saves it and links it as superseding the first; recall
   with history ("what did I prefer before?") shows both.

## Command line

```bash
uv run --extra models mem doctor            # extensions, models, device, fingerprint, schema version, encryption
uv run --extra models mem quarantine        # memories held back as prompt-injection suspects
uv run --extra models mem quarantine --release m12   # let one back in after you read it
uv run --extra models mem conflicts         # saves whose possible conflicts were never resolved
uv run --extra models mem review            # up to 10 of today's recalls, to check what came back
uv run --extra models mem core              # the index a SessionStart hook can print
```

`mem <command> --help` has every option. To follow the server: `journalctl --user -u mem -f`, and
the audit log of every request is `~/.local/state/mem/audit.jsonl`.

## Configuration

`mem install` writes `~/.config/mem/config.yaml`; without one the built-in defaults apply. Sections:

| Section | What it holds |
|---|---|
| `embedder` | model id, pinned revision, dimension, prompts, dtype, max length |
| `reranker` | model id, pinned revision, dtype, max length |
| `retrieval` | `leg_k` (candidates per leg), `rrf_k`, `pool` (candidates the reranker sees), `per_node_cap` (rows per subject), `limit` (rows returned), `budget` (token budget of an answer) |
| `ingest` | write-time switches: `resolve_dates` (relative dates resolved at import) and `new_checks_existing` (default `true`, [ADR 0027](decisions/0027-new-name-checks-existing-entities.md)): a `new:<type>:<Name>` subject is first matched against the existing entities of any type (aliases included) before a new one is made; set it `false` to go back to the older rule (same type and name only) if saves start missing contradictions with what is already stored. Turning it off changes `fp_all`, so the gate must be refitted |
| `read` | read-time switches, all in `fp_all`: `list_mode` (default `true`: "how many / which" questions are answered as a list, [ADR 0023](decisions/0023-list-mode-tau-list-and-token-cap.md)), `list_fill` (default `16`: the rows a list answer is filled to; `0` turns filling off), `window_boost` (default `{delta: 3.0, slack: 0.1}`: a soft boost for rows inside a date window the question names; `false` turns it off) |
| `models` | `embedder`: the pinned embedder the `embedder` section is filled from, `qwen3-embedding-0.6b` (default) or `embeddinggemma-300m` ([ADR 0025](decisions/0025-local-reranker-and-embedder-candidates.md)); `reranker` likewise (`bge-reranker-v2-m3`); `rerank_batch_size` (default `0`, one query at a time; set, it joins `fp_all`) |
| `device` | `cuda` (default) or `auto` |
| `vault.routes` | folder → store overrides for the vault import ([vault.md](vault.md)) |

The embedder, reranker, retrieval and read sections and the `ingest` switches are **fingerprinted**
(`fp_all`; `models.embedder` acts through the embedder section): the stores remember the
fingerprint their vectors and gate were made for, and `mem serve` refuses to start when the config
no longer matches. After changing the reranker or retrieval, re-run the gate fit
([install.md](install.md#fit-the-relevance-gate-once)). After changing the embedder, `mem reembed`
builds new vectors next to the old ones, `mem calibrate` fits a gate for them, and `mem switch`
swaps both in one transaction; serving is unchanged until the switch.

The defaults are the configuration the shipped gate was fitted on (`fp_all` 3f8c9c11): the gate in
`src/mem/shipped/serving_gate.json`, fitted on the LongMemEval tune split and the SYN dev seeds, serves
a new store until you fit your own. They are: the
Qwen3-Embedding-0.6B embedder (`models.embedder: qwen3-embedding-0.6b`) and bge-reranker-v2-m3, relative
dates resolved at import (`ingest.resolve_dates: true`), list mode for "how many / which" questions
(`read.list_mode: true`, filled to 16 rows, `read.list_fill: 16`), and a soft boost for rows inside a
date window the question names (`read.window_boost: {delta: 3.0, slack: 0.1}`). Stores made with an
earlier default (the EmbeddingGemma embedder) keep serving when their config names it:
`models: {embedder: embeddinggemma-300m}` and `read: {list_fill: 0, window_boost: false}`; or move
them to the defaults with `mem reembed`, `mem calibrate` and `mem switch` as above.

## Other flags of `mem serve`

`--host-label` names this machine in the event log (the `X-Mem-Host` header does it for HTTP
clients), `--tz` sets the zone for date-only inputs (default: `TZ`, else `/etc/localtime`),
`--rate-limit` caps requests per minute per client (default 600), `--audit-log` moves the audit
log. Network options are in [multi-machine.md](multi-machine.md).
