# AGENTS.md

Guide for anyone changing this repository, human or coding agent. Users want the
[README](README.md) instead. The system is described in [docs/architecture.md](docs/architecture.md):
read it before touching `src/mem/read.py`, `write.py` or the gate.

## What this is

`mem` is a local long-term memory server for coding agents: an MCP server (`mem serve`) over two SQLite
stores (`personal`, `work`) with hybrid FTS5 + vector retrieval, a cross-encoder reranker and a
calibrated relevance gate. Python 3.12, managed with [uv](https://docs.astral.sh/uv/). Linux only. It
serves on an NVIDIA GPU, but **development and tests need no GPU, no network and no model download.**

## Commands

```bash
uv sync                                   # dev install (no torch); add --extra models for the real models
uv run pytest -q                          # the whole suite, CPU stand-in models, offline (~10 min)
uv run pytest -q tests/test_gate.py       # one file; use -k to pick tests
uv run ruff check src tests bench         # lint (line length 100)
uv run ruff format src tests bench        # format; CI runs `ruff format --check`
uv run mem --help                         # the CLI; `uv run bench --help` for the benchmark harness
```

CI (`.github/workflows/ci.yml`) runs `uv sync --locked`, ruff and the offline suite, plus
`scripts/fresh-install-test.sh` (a clean `mem setup` → doctor → self-test → uninstall with faked GPU,
systemd and Claude Code). A change is done when all of that is green locally.

## Layout

| Path | What |
|---|---|
| `src/mem/` | the package; one module per concern (map in [docs/architecture.md](docs/architecture.md#around-the-core)) |
| `src/mem/cli.py` | the `mem` CLI (Typer), thin wrappers over library code |
| `src/mem/contracts/mcp_tools.json` | **frozen** MCP tool schemas; `tests/test_mcp_contract.py` pins them |
| `src/mem/schema/migrations/` | numbered SQL migrations; never edit a shipped one, add `NNNN_<name>.sql` |
| `src/mem/shipped/serving_gate.json` | the gate fitted for the default config; written by `bench package-gate`, never by hand |
| `skills/mem/SKILL.md` | the agent skill; `docs/instructions.md` is generated from it (`mem instructions`) |
| `bench/` | benchmark harness (`uv run bench`), datasets manifests, suites, adapters into `src/mem` |
| `tests/` | unit and integration tests; `tests/bench_t0/` benchmark self-checks; `tests/e2e/` fresh install |
| `docs/` | user docs, `architecture.md`, `benchmarks.md`, and `decisions/` (ADRs) |
| `deploy/` | systemd user units the installer copies |

## Invariants the tests guard

Break one of these and something downstream silently goes wrong, which is why a test pins each.

- **Fingerprints.** The `embedder`, `reranker`, `retrieval` and `read` config sections are hashed into
  `fp_embed` / `fp_all` (`config.py`). A new setting that changes results must join the fingerprint
  *only when set*: off or absent leaves `fp_all` as it was, so existing stores and the shipped gate stay
  valid. `tests/test_config.py` checks the default fingerprint; if you change a default on purpose, the
  shipped gate has to be refitted on the GPU host, so say so in the PR.
- **The gate decides, later steps only drop.** After the gate, read-path steps may remove rows or add
  flags, never add rows. An abstention is one line: `no relevant memories (best .NN)`.
- **History, not overwrites.** Writes never update a memory's text; supersession closes `valid_to`.
  `write.py` phase B and `link.py` are single `BEGIN IMMEDIATE` transactions.
- **Only `mem serve` loads models.** Other commands borrow them through `models_http.py`. Tests use
  `mem.testing` (`HashEmbedder`, stand-in reranker) and must not import torch or reach the network.
- **Deterministic output.** `mem render` and the store dump are pure functions of the store; golden files
  live in `tests/golden/`.
- **Nothing personal in the public paths.** `tests/test_public_clean.py` scans everything `release.py`
  ships against an encoded denylist. Don't add real names, home paths or personal data, even in
  fixtures.
- **Benchmark honesty.** `bench/ledger.jsonl` records every look at a test split (`--ledger "<why>"`);
  thresholds come only from the benchmark spec. Tune on `tune`/`dev` splits, never on `test`.

## Making a change

1. Write the test first. Most modules have a matching `tests/test_<module>.py`; regression tests for a
   reviewed defect are the `test_*_proofs.py` files.
2. Keep diffs small and in the module that owns the concern; don't put logic in `cli.py`.
3. Behaviour, default or schema change → add a decision record `docs/decisions/NNNN-<slug>.md` (next
   free number; Status, Date, Context, Decision, Consequences, as the existing ones do) and update the
   user doc it affects.
4. New config switch → off by default, documented in `docs/use.md#configuration`, fingerprinted only
   when set.
5. Run ruff and the full suite before calling it done.

## Gotchas

- **Scratch files go in `.tmp/`** under the checkout, not `/tmp` (`tests/conftest.py`); it is git-ignored.
- **Provenance references.** Comments, docstrings and ADRs cite `architecture-v2 §…`, `benchmarks-v2 …`,
  phases `P1`–`P7`, task ids `MEM2-NNN` and runbooks under `deliverables/`. Those are the original
  private design documents and tracker, not in this repository. Treat them as history; the public map
  is `docs/architecture.md`. Tests that need those files skip when they are absent.
- **Two `0007`, `0011`, `0014`, `0017` and `0018` decision records** exist (numbered in parallel); keep both,
  take the next free number for a new one.
- **Real-model runs need the GPU host:** `uv sync --extra models`, `mem doctor --fetch`, and ~4 GB VRAM.
  Paid end-to-end benchmarks (`bench e1-run`, LoCoMo) need `ANTHROPIC_API_KEY` and take a `--budget-usd`.
- The CLI has two entry points: `mem` (product) and `bench` (harness). Both are in `pyproject.toml`.
