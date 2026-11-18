# Troubleshooting

`mem doctor` prints one line per check, `[ok  ]`, `[warn]` or `[FAIL]`; any FAIL makes it exit 1. `mem setup` runs the same
checks at its end and says `doctor: FAIL` when one fails. Find the line's name below. After the fix, run `mem setup`
again: it skips what is done.

| Line | Fails when | Fix |
|---|---|---|
| `sqlite-vec` | the SQLite vector extension or FTS5 does not load | Reinstall: `uv tool install --reinstall ".[models]"` (or `uv sync --extra models`). A Python built without loadable extensions (some distro and macOS Pythons) needs a uv-managed Python 3.12: `uv python install 3.12`. |
| `Qwen/Qwen3-Embedding-0.6B` or `BAAI/bge-reranker-v2-m3` | `revision … not in cache`, or `incomplete … missing …` | `mem doctor --fetch` (or `mem setup`). An interrupted download resumes. Neither model is gated. If you selected `models.embedder: embeddinggemma-300m` and it says gated or 401/403: open <https://huggingface.co/google/embeddinggemma-300m>, accept the Gemma terms with your account, run `hf auth login`, fetch again. An incomplete folder with only `README.md` is what a refused download leaves. |
| `device` | `device is cpu`, `torch … sees no CUDA`, `torch not installed`, or under 4 GiB free VRAM | mem serves only on an NVIDIA GPU. No GPU: use another machine, or connect to one ([two-machines.md](two-machines.md)). Driver and torch disagree (`nvidia-smi` shows CUDA *X*, torch was built for *Y*): update the driver until `nvidia-smi` reports at least *Y*, or reinstall torch for *X*. torch missing or a CPU build: install with the models extra, `uv tool install --reinstall ".[models]"`. Not enough VRAM: close what holds it (`nvidia-smi`), then check again. Tests and CI only: `mem setup --allow-cpu`. |
| `fingerprint` | never fails; it prints the embedder fingerprint | Nothing to fix. If `gate` fails after you changed the config, see below. |
| `schema <store>.db` | `warn`: the store does not exist yet (normal before the first save). `FAIL`: unreadable or `meta.schema_version` missing | Warn: nothing, the first `remember` creates it. FAIL: the file is damaged or not a mem store. Restore it from a backup ([backups.md](backups.md)); do not point `--store` at another database. |
| `gate <store>.db` | `no local gate and none shipped`, or `fitted for X, the config is Y` | The line ends with the command that fixes it: `mem calibrate --records … --config … --store name=path`. The shipped gate only fits the shipped config; after you change the embedder, reranker or retrieval settings, refit ([install.md](install.md#fit-the-relevance-gate-once)) or restore the config. |
| `at-rest` | the folder holding the stores is not on an encrypted disk, or mem cannot tell | Put `~/.local/share/mem` on LUKS or gocryptfs ([backups.md](backups.md)). On a test machine only: `mem doctor --allow-unencrypted` (a warning, logged to `~/.local/state/mem/doctor.jsonl`). |

## Setup failures

`mem setup` stops at the first failing step, changes nothing before its checks pass, and ends with `Fix that and run it
again`. The usual ones:

| Message starts with | Fix |
|---|---|
| `gpu: no NVIDIA GPU found` | As `device` above. |
| `gpu: the driver supports CUDA … but torch …` | As `device` above. |
| `systemd: no systemd user session` | Log in with a full session (ssh or a desktop login, not `su`), or `sudo loginctl enable-linger $USER` and log in again. |
| `port 8765 is in use` | Stop what holds it (`ss -ltnp \| grep 8765`), or `mem setup --port 8766`. |
| `… is gated and its terms are not accepted` | Accept the terms on the model page, `hf auth login`, rerun. |
| `downloading … failed` | Check the connection and disk space in `~/.cache/huggingface`; rerun, it resumes. |
| `claude: not on PATH` | Plain `mem setup` (and `--uninstall`) registers the servers through Claude Code. Install it, or run `mem setup --no-mcp` (and later `mem setup --uninstall --no-mcp`): Codex CLI, Gemini CLI, Cursor and VS Code are still set up. |
| `mem is not on PATH, so the service cannot run it` | `uv tool install ".[models]"` from the checkout, and make sure `~/.local/bin` is on PATH. |
| `… was not written by mem install … kept as is` | You have your own `mem.service`. It is left alone. To let `mem setup` manage it, stop it and move it away first. |
| `self-test: FAIL` | `mem doctor` and `journalctl --user -u mem.service`, then `mem setup` again. |
| `no harness to add the server to` (with `--remote`) | Install Claude Code, Codex CLI, Gemini CLI, Cursor or VS Code on this machine first. |
| `… does not parse` or `… cannot change this file safely` (a harness config) | The file has a syntax error (comments are fine), or holds a `mem-*` entry in a shape mem does not edit. `mem setup` skips that harness (`[warn]`) and does the others. Fix the file, or merge the entry `mem mcp-config --harness <name>` prints by hand; mem does not write a file it cannot parse. |

## Remote failures

`mem doctor` shows a `remote <name>` line for each machine added with `mem setup --remote`.

| Line | Fix |
|---|---|
| `… is unreachable (…)` | Is that machine up and on the tailnet? `tailscale status`, `tailscale ping <host>`; on it, `mem serve --tls` (or `tailscale serve`) must run. A `421` from the host: its `ExecStart` needs `--allow-host <name>` ([multi-machine.md](multi-machine.md#check-it)). |
| `… refused the token (401)` | The token was revoked or mistyped. On that machine: `mem token add <label> --code --host <host>`, then here `mem setup --remote <host> --code <code>`. |
| `answered HTTP … to initialize` | The server is up but unhealthy: `systemctl --user status mem` and `journalctl --user -u mem` there. |
| `no token file` | `mem setup --remote <host>` again. |
| `warn … ms round trip (slow)` | Over 500 ms: a relayed (DERP) Tailscale path, or the home machine is busy. `tailscale ping <host>` shows whether the path is direct. |
| `not a mem code` | Paste the whole line `mem token add … --code` printed, starting with `memc1_`. |

## Harness does not list the tools

Run the harness's own list command (`claude mcp list`, `codex mcp list`, `gemini mcp list`, `/mcp` in Cursor and VS Code's
chat). Missing or `failed`: `systemctl --user status mem` (is the server up, on `127.0.0.1:8765`?), then
`mem setup`, which re-adds what is missing. The details per harness are in [harnesses.md](harnesses.md).
