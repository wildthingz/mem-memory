# 0006: model access for CLI tools — `/models` on `mem serve`, and the reembed exception to D5

**Status:** accepted (MEM2-67, lead ruling on review finding F7 of MEM2-66). Amends architecture-v2
D5 (§2.1) for `mem reembed` only.

**Context.** D5: `mem import`, `mem calibrate` and `mem reembed` never load models; they call the
running `mem serve` over its local HTTP endpoint, so models load once and the ≈ 5 GB VRAM budget
holds. P2 shipped `mem calibrate --labels` and `mem reembed` loading the models in their own process
(F7). Two facts shape the fix:

- Calibration scores a pool with the full read path (ingest a haystack into a fresh store, legs,
  RRF, rerank), not with a single model call, and the tools need the same numbers the serve would
  produce.
- `mem reembed` exists to move stores to a *new* embedder. The serve holds the *old* one (its stores'
  fingerprint), so the new model cannot come from it.

**Decisions**

1. **`mem serve --http` exposes its loaded models at `/models`** (`src/mem/models_http.py`):
   `GET /models` (the config sections the models were loaded for, `fp_embed`, `dim`),
   `POST /models/embed` (`{"kind": "query"|"docs", "texts": [...]}` → `{"vectors": [...]}`) and
   `POST /models/rerank` (`{"query", "docs"}` → `{"logits": [...]}`, raw). Loopback clients only
   (403 otherwise: the port binds to Tailscale from P5); ≤ 512 items per request; 400 with
   `{"error"}` for a bad body. Calls run on the event loop, serialised with MCP tool calls.
   The client side (`connect(port, cfg)`) returns `RemoteEmbedder` / `RemoteReranker`, which satisfy
   the `mem.testing` protocols, so the store, read path and bench adapter run unchanged around them.
   Rejected: a high-level "score this haystack" endpoint. It would put bench/LME knowledge (haystack
   formats, chunk refs, fresh stores) into the server and duplicate the read path there; model-level
   calls keep one read-path implementation and are what the LME adapter (MEM2 Harness) needs too.
2. **`mem serve --http --models-only`** serves `/models` with no store mounted, so it skips the store
   preflight. Needed because calibration is exactly when no store is gated for the config yet (the
   first gate in P2, or a new config after `mem reembed`), and the normal serve refuses to start then.
   Rejected: letting `mem serve` start with an ungated store (breaks A2 / §3.3 refusal).
3. **`mem calibrate --labels`** scores through `connect(--port, config)`; it constructs no model. If
   nothing answers, or the serve holds models other than the config's (embedder or reranker section
   differ), it fails with one `[FAIL]` line naming the serve to start. `--device` moved to the serve.
4. **`mem reembed` is the one exception to D5**: it loads the new embedder in its own process, and
   only when no serve runs: nothing listens on `--port` (default 8765) and it can take an exclusive
   advisory lock on `<store>.serve.lock` for every store. Every `mem serve` holds a shared lock on each
   store it mounts, for as long as it serves; a serve that cannot take it (reembed running) refuses to
   start. Otherwise reembed refuses before loading anything: "Stop mem serve first". So at any time at
   most one process holds models for these stores, and the VRAM budget of D5 holds.
   Rejected: (a) the serve loading the new embedder on request (two embedders resident in the serve,
   the budget D5 protects is exceeded, and a long batch job runs inside the latency-sensitive
   process); (b) a reembed endpoint on a `--models-only` serve of the new config (works, but adds an
   endpoint and a write path to the server for a rare, offline operation; the lock gives the same
   guarantee with less surface).

**Consequences.** Calibration needs a running serve; the runbooks (`p2-gate.md` §4, `p2-read.md`
"Changing a model later") stop the systemd unit and start a `--models-only` serve for that step.
Lock files `<store>.serve.lock` appear next to the stores' real files (symlinks, `..` and relative
paths are resolved first, so every spelling of one database meets the same lock; MEM2-71 F7); they
are empty and safe to delete when no `mem` process runs. Remote scoring returns the in-process numbers exactly (JSON doubles carry the
models' float32 outputs; `test_pools_scored_through_serve_equal_in_process_scoring`).
