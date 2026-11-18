# mem

Local long-term memory for [Claude Code](https://claude.com/claude-code), served over MCP.

mem keeps what you tell Claude (decisions, preferences, facts about people and projects) in SQLite
files on your own machine, and gives it back when it matters, in as few tokens as possible. Nothing
leaves the machine: the models run on your GPU and the stores are local files.

- **Hybrid retrieval.** SQLite FTS5 (keywords) and a vector index ([sqlite-vec](https://github.com/asg017/sqlite-vec),
  Qwen3-Embedding-0.6B embeddings) are fused, then a cross-encoder reranker (bge-reranker-v2-m3) orders the
  candidates.
- **A calibrated gate.** The reranker's score is turned into a probability; rows below the cut are
  dropped, and when nothing clears it the whole answer is one line: `no relevant memories (best .22)`.
  mem would rather say "I don't know" than return plausible filler.
- **History, not overwrites.** A changed fact *supersedes* the old one; the old row stays as history
  and recall can answer "what did I think then?" (`as_of`).
- **A typed graph.** Memories are about subjects (people, projects, organisations, topics, places);
  edges say `supersedes`, `duplicate_of`, `relates_to`, `same_as`.
- **Cleaner saving.** `remember` asks when a name is ambiguous, shows possible conflicts with older
  memories, rejects secrets, and quarantines text that reads like instructions to an AI.
- **Obsidian.** Import a vault so notes are searchable next to memories; render the stores back to
  Markdown (and git) as a one-way mirror.
- **Encrypted backups,** a bearer-token-protected endpoint, and an optional Tailscale setup for a second machine.

## Requirements

- Linux with a systemd user session.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU with CUDA and **at least 4 GB of free VRAM**. Serving on the CPU is not supported.
- Nothing to sign up for: both models (Qwen3-Embedding-0.6B and bge-reranker-v2-m3) are Apache-2.0 and
  not gated on Hugging Face, so the download needs no account and no `hf auth login`.
- A coding harness: Claude Code (`claude`), or Codex CLI, Gemini CLI, Cursor or VS Code ([docs/harnesses.md](docs/harnesses.md)).
  Plain `mem setup` and `mem setup --uninstall` need `claude` on PATH; without Claude Code, add `--no-mcp` to
  both (the other harnesses are still set up and undone).

## Quickstart

```bash
git clone https://github.com/wildthingz/mem-memory.git mem && cd mem   # 1. get it
uv tool install ".[models]"                        # 2. puts `mem` on your PATH
mem setup                                          # 3. shows the plan, asks once, runs it, then checks itself
claude                                             # 4. say: "Remember in my personal memory: I prefer espresso. Subject: me."
```

No Claude Code? Step 3 is `mem setup --no-mcp`, and step 4 opens your harness (`codex`, `gemini`, Cursor or VS Code).

`mem setup` checks the GPU, downloads the two models, writes the config, makes a token, starts the
systemd user service, registers the MCP servers and the skill in Claude Code (and in Codex CLI, Gemini CLI,
Cursor and VS Code when they are installed), runs `mem doctor` and a self-test, and prints `self-test: PASS`.
Run it again any time: it skips what is done. `mem setup --dry-run` shows the plan and changes nothing;
`mem setup --uninstall` (with `--no-mcp` if you set up with it) puts every edited config back byte for byte. Then, in another session, ask
*"What do you remember about my coffee?"*

The package carries a relevance gate fitted on public data, so there is nothing to calibrate. If
your checkout has no `src/mem/shipped/serving_gate.json`, `mem doctor` says so and gives the one-off
gate fit ([docs/install.md](docs/install.md#fit-the-relevance-gate-once)). Model downloads depend on
your connection. Something failed? [docs/troubleshooting.md](docs/troubleshooting.md) maps each
`mem doctor` failure to its fix. A second machine: [docs/two-machines.md](docs/two-machines.md).

**Plugin instead of `mem setup`'s MCP step:** run `mem setup --no-mcp`, set `MEM_TOKEN` from
`~/.config/mem/token`, then `claude plugin marketplace add <path to this checkout>` and
`claude plugin install mem`. Details in [docs/install.md](docs/install.md).

## The MCP tools

Two servers, `mem-personal` and `mem-work` (separate stores), each with the same six tools:

| Tool | Purpose |
|---|---|
| `recall` | search memories and notes; filters `about`, `type`, `kinds`, `since`, `until`, `as_of`, `include_history` |
| `get` | full text by id or continuation token |
| `remember` | save one atomic memory about a subject |
| `link` | add or remove a typed edge (`supersedes`, `duplicate_of`, `relates_to`, `same_as`) |
| `subjects` | list or search people, projects and other subjects |
| `forget` | retract a memory (kept as history, never recalled) |

## The skill

`skills/mem/SKILL.md` teaches Claude when to recall (before answering about people, projects, past
decisions or your preferences), what to save and what never to save, how to word a memory, and how to
resolve ambiguous names and conflicts. `mem setup` copies it to `~/.claude/skills/mem`; the plugin
carries it as well.

## Configuration

`~/.config/mem/config.yaml`, written by `mem setup`. Sections `embedder`, `reranker`, `retrieval`
(candidate counts, rows returned, token budget), `ingest` (`resolve_dates`, `new_checks_existing`),
`read` (`list_mode`, `list_fill`, `window_boost`), `models` (`embedder`: `qwen3-embedding-0.6b` or
`embeddinggemma-300m`), `device`, and `vault.routes`. All but the last two are fingerprinted: change
them and `mem serve` refuses to start until the vectors and gate match again. The defaults are the
configuration the shipped gate was fitted on. See [docs/use.md](docs/use.md#configuration).

## Benchmark results (summary)

End-to-end question answering at the shipped configuration: `claude-sonnet-5-5` answers from what
`recall` returns and grades with each benchmark's judge prompt. One run each, on one machine (RTX 4080
SUPER), October 2026. Per-type tables, the component measurements, the targets that were missed and how
to reproduce everything: [docs/benchmarks.md](docs/benchmarks.md).

| Benchmark | Score | 95% CI |
|---|---|---|
| LongMemEval-S (500 questions) | **82.6%** (no memory: 7.6%; gold sessions only: 92.4%) | 79.2–85.6 |
| LoCoMo (categories 1–4, 1,540 questions) | **82.4%** | 80.5–84.2 |
| LongMemEval-M (500 questions, ~500 sessions each) | **73.8%** | 70.0–77.2 |

Recall latency, warm p95: 452 ms on a 5,000-chunk store, 540 ms on 50,000 chunks. A typical answer is
about 90 tokens. Weakest areas: preference questions (67% on S) and evidence spread over hundreds of
sessions (multi-session 59% on M).

## Limitations

- **GPU only.** Linux and systemd only; no macOS or Windows. The models need about 4 GB of VRAM.
- **One writer.** One server owns the SQLite files. A second machine connects to it
  ([docs/multi-machine.md](docs/multi-machine.md)); syncing the files between machines is not supported.
- **Two fixed stores,** `personal` and `work`. One person, not a team.
- **Changing models or retrieval settings needs a gate fit** (the shipped gate belongs to the shipped
  config; a one-off LongMemEval download and about ten minutes of GPU, in [docs/install.md](docs/install.md)).
- **Encryption at rest is your disk's job** (LUKS or gocryptfs); `mem doctor` checks it. Backups are
  encrypted by mem ([docs/backups.md](docs/backups.md)).
- **Young.** Measured on one machine by one person; see [docs/benchmarks.md](docs/benchmarks.md) for the
  component targets it still misses.
- **History references.** Code comments, tests and decision records cite the original design documents
  (`architecture-v2 §…`, `benchmarks-v2 …`, phases P1–P7), the task ids they came from (`MEM2-…`) and
  runbooks that are not part of this repository. They are kept as provenance; the
  [architecture overview](docs/architecture.md) is the public map.

## Contributing

Read [AGENTS.md](AGENTS.md) first (written for human contributors and coding agents alike): layout,
commands, the invariants the tests guard, and how a change is made. The short version: `uv sync`,
`uv run pytest -q` (CPU only, no network, no GPU), `uv run ruff check` and `ruff format`, and a decision
record in [docs/decisions/](docs/decisions/) for any change to behaviour or defaults.

## Docs

[Architecture](docs/architecture.md) · [Install](docs/install.md) · [Use](docs/use.md) · [Vault import and render](docs/vault.md) ·
[Backups and encryption](docs/backups.md) · [Two machines](docs/two-machines.md) · [Multiple machines (server side)](docs/multi-machine.md) · [Other harnesses](docs/harnesses.md) ·
[Troubleshooting](docs/troubleshooting.md) ·
[Benchmarks](docs/benchmarks.md) · [Design decisions](docs/decisions/)

## Licence

MIT, see [LICENSE](LICENSE). Models and datasets have their own terms, listed in
[THIRD_PARTY.md](THIRD_PARTY.md). Both default models are Apache-2.0 and are downloaded by you, from
Hugging Face; no weights are in this repository. (EmbeddingGemma, the embedder before, is still
selectable with `models.embedder: embeddinggemma-300m`; its Gemma terms then apply, and its repository is
gated.)
