# 0001: packaging of the P1 spike and `mem doctor`

**Status:** accepted (the maintainer confirmed decisions 1-5, MEM2-8). Decision 4 as written: the limit is judged on average pairs; the worst case warns.

**Decisions**
1. `torch` and `sentence-transformers` are an optional extra (`uv sync --extra models`), not a base
   dependency. Architecture-v2 DR-1 lists them as dependencies; they stay required for `mem serve`,
   but unit tests and `mem doctor` need not install ~3 GB of CUDA torch. Rejected: base dependency
   (slow CI, tests would then be tempted to load models).
2. Pinned revisions live in `src/mem/pins.py` as full 40-char SHAs, resolved from the spec's
   prefixes `57c266a7…` and `953dc6f6…` via the HF API (2026-10-23; both are also `main`).
   `pins.py` is a stand-in for the `config.yaml` section of §3.3 and is replaced in P2.
3. The embedder section hashed into `fp_embed` uses the EmbeddingGemma model-card prompts
   (`task: search result | query: `, `title: none | text: `) and `max_len` 512. These are provisional:
   the spec does not state them; `doctor` labels the fingerprint provisional.
4. Spike judging: the 1,000 ms limit applies to the p95 of the average-length pair set (≈ 225
   tokens); a worst-case 512-token p95 over the limit is a warning. Reported p95 across runs is the
   worst run's p95. The spec says only "GPU rerank p95 > 1,000 ms"; tell me if the worst case should
   decide.
5. CPU rerank defaults to 3 × 5 calls (spec: 3 × 50) because a call is estimated at 4–34 s;
   `--cpu-calls 50` restores the spec.
