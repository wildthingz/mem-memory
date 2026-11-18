# Install

## Requirements

- Linux with a systemd user session (`systemctl --user` works).
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU with CUDA and at least 4 GB of free VRAM. `mem serve` runs the embedder and the
  reranker on the GPU; serving on the CPU is not supported (a CPU is fine for the test suite, which
  uses stand-in models).
- No Hugging Face account. The default models,
  [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and
  [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3), are Apache-2.0 and not
  gated (checked on the Hugging Face API, 2026-10-30). mem downloads them to your Hugging Face cache and
  never copies the weights anywhere else. Only if you select EmbeddingGemma
  (`models.embedder: embeddinggemma-300m`) do you need an account that has accepted the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) on its gated page, and `hf auth login`.
- [Claude Code](https://claude.com/claude-code) (`claude` on your PATH), unless you use the plugin
  path below, which needs it as well to install the plugin.

## Install

```bash
git clone <this repository> mem && cd mem
uv sync --extra models
uv run --extra models mem install --dry-run   # every action it would take; nothing is changed
uv run --extra models mem install
```

`mem setup` is the same installer with the checks around it, and `mem install` is an alias of
`mem setup --yes`:

```bash
uv tool install "mem[models] @ ./mem"        # or: uvx --from "mem[models] @ ./mem" mem setup
mem setup --dry-run                          # every action, nothing touched
mem setup                                    # shows the plan, asks once, then runs it (--yes: no question)
mem setup --uninstall [--purge]              # reverse it; edited harness configs come back byte for byte
mem setup --remote work.tailnet.ts.net --code <code>   # the other machine's servers; or paste its token
```

After the install steps it runs `mem doctor` (which fails when no gate can serve a store) and a
self-test: remember, recall and forget through the service it just started, on
`http://127.0.0.1:<port>/personal/mcp` with the installed token, printed `self-test: PASS` or `FAIL`;
the memory is retracted at the end. Errors say what is
wrong and the fix: no NVIDIA GPU, a CUDA mismatch between driver and torch, a gated model
(only when you select one; it names the page to accept the terms on), a busy port, no systemd user session. An interrupted model
download resumes where it stopped. On the other machine, `mem token add <label> --code --host <its
tailnet name>` prints the code. `mem doctor` then also checks each remote: reachability, token, latency.
Without a checkout (`uv tool install`) the unit runs the installed `mem` binary; the wheel carries
`deploy/mem.service` and the skill. Nothing is sent anywhere except to a remote you named.

`mem install --hooks` is the one form that is not `mem setup`: it only registers the optional
auto-capture Stop hook in `~/.claude/settings.json` (off until asked for; `--settings` names another
file, `--dry-run` writes nothing, `--uninstall` removes it). The hook asks a running `mem serve --http`
over a Unix socket and does nothing when serve is down.

Always pass `--extra models` to `uv sync` and `uv run` on the machine that serves: a bare `uv sync`
removes torch and sentence-transformers again.

`mem install` does these steps, each skipped when it is already in place (a second run prints
`nothing to do`):

1. checks the GPU (`mem doctor`'s device check);
2. downloads the two pinned models into the Hugging Face cache (`mem doctor --fetch`);
3. writes a default `~/.config/mem/config.yaml`;
4. creates a bearer token: its hash goes to `~/.config/mem/tokens` (read by the server) and the token
   itself to `~/.config/mem/token` (mode 600, read by the clients);
5. renders `deploy/mem.service` for this machine into `~/.config/systemd/user/mem.service`, with
   this checkout as the working directory, then enables and starts it. The server listens on
   `127.0.0.1:8765` only;
6. registers the MCP servers `mem-personal` and `mem-work` in Claude Code (user scope, with the
   bearer header) and copies the skill to `~/.claude/skills/mem`.

Options: `--no-mcp` skips step 6 (use it with the plugin), `--host-label NAME` sets the label this
machine sends as `X-Mem-Host` (default: the hostname), `--port N` changes the port. `--allow-cpu` is
for tests and CI only.

Two things to know about steps 5 and 6:

- `mem install` only writes, rewrites or removes a `mem.service` that it wrote itself: the first
  line of such a unit is the marker `# Written by `mem install``. A unit you wrote by hand (for
  example `deploy/mem.service` copied and given `--tls` or other flags, see
  [multi-machine.md](multi-machine.md)) is kept as is, by `mem install` and by
  `mem install --uninstall` alike, and the output says so: `[ok  ] service: ... was not written by
  mem install ...: kept as is`. To hand the service over to `mem install`, stop it and move your
  unit away first.
- In step 6 the bearer token is passed to `claude mcp add` as a `--header` argument, because that
  command accepts headers no other way. For the second or so that the command runs, the token is
  visible in the process list to every process of your user (and to root). `mem install` never
  prints it. On a shared machine, prefer the plugin path below, which reads the token from
  `MEM_TOKEN` in your environment instead.

## Fit the relevance gate (once)

`mem serve` refuses to start until each store holds a **gate**: the calibration (a Platt fit and
three thresholds) that turns the reranker's scores into the probability shown on every recalled row,
and decides when the answer is "no relevant memories". The gate is fitted on the tune split of
LongMemEval, which is public; you do it once per machine and again whenever the models or the
retrieval settings in `config.yaml` change. Until then `mem install` prints
`[note] mem serve will not start until: ... no gate ...`, and the service restarts in a loop.

Stop the service, then, from the repository:

```bash
systemctl --user stop mem
export OMP_NUM_THREADS=4
uv run --extra models bench fetch --which s --dest data/lme          # downloads LongMemEval-S (277 MB)
uv run --extra models bench build-lme data/lme/longmemeval_s_cleaned.json
uv run --extra models bench export-tune data/lme/longmemeval_s_cleaned.json --out data/lme/tune_labels.jsonl
```

In a second terminal, start a server that only holds the models (no store); `mem calibrate` scores
through it. Leave it running for the next command:

```bash
uv run --extra models mem serve --http --models-only
```

Back in the first terminal:

```bash
uv run --extra models mem calibrate --labels data/lme/tune_labels.jsonl \
  --lme data/lme/longmemeval_s_cleaned.json \
  --records-out data/lme/tune_records.jsonl --out data/lme/gate.json --fit-id lme-s-tune-v1
```

Stop the models-only server (Ctrl-C), then:

```bash
uv run --extra models mem calibrate --records data/lme/tune_records.jsonl --fit-id lme-s-tune-v1 \
  --store personal=$HOME/.local/share/mem/personal.db --store work=$HOME/.local/share/mem/work.db
systemctl --user start mem
```

The long step is the first `mem calibrate`: it embeds every tune haystack through the GPU. On an
RTX 4080 SUPER it took about 9 minutes. The second `mem calibrate` takes seconds and loads no model.
It prints a tune report and, as the last line, the gate as JSON. `bench build-lme` prints the split
counts and checks them against the frozen ones; if they differ, the dataset file is not the one mem
was built against.

Check the result:

```bash
systemctl --user status mem --no-pager | head -5       # active (running)
uv run --extra models mem doctor                       # every line [ok] (see the note on encryption below)
```

`mem doctor` also reports whether the folder holding the stores is on an encrypted disk
(`at-rest`); see [backups.md](backups.md). On a throwaway machine, `mem doctor --allow-unencrypted`
opts out and logs that it did.

These are the commands the project's own host runbook uses; they have not yet been re-run from a
fresh clone of the public copy.

## Use the Claude Code plugin instead of `mem install`'s MCP step

The plugin brings the skill and the two MCP servers; it does not start the server.

```bash
uv run --extra models mem install --no-mcp          # models, config, token, service; no MCP, no skill
export MEM_TOKEN="$(cat ~/.config/mem/token)"       # put this in your shell profile; MEM_HOST is optional
claude plugin marketplace add /path/to/this/checkout
claude plugin install mem
```

## Uninstall

```bash
uv run --extra models mem install --uninstall           # removes the service, the MCP servers and the skill
uv run --extra models mem install --uninstall --purge   # also deletes the config, the tokens and the stores
```

Without `--purge` the config, the token files and the stores stay. The models stay in the Hugging
Face cache (`~/.cache/huggingface`); delete them there if you want the disk back. If you used the
plugin: `claude plugin uninstall mem`. A `mem.service` that `mem install` did not write (no marker
line, see above) is neither stopped nor removed; the output says it was kept.
