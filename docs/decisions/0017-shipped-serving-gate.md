# 0017: The serving gate fitted on public data ships as package data

- **Status:** Accepted (requested by the owner, MEM2-1 comments of 2026-10-26 16:20; setup approved in MEM2-247, shipped with the
  final config in MEM2-274). The shipped gate is the host's `data/r9/r10-all/gate-list.json` (`fp_all` 3f8c9c11), packaged by
  `bench package-gate` (setup-host.md step 1).
- **Date:** 2026-10-26 · **Task:** MEM2-232 · **By:** Vault
- Spec: easy-setup-integrations §2.1 (git-ignored research note). Code: `src/mem/gate.py`, `src/mem/serve.py` `preflight`, `src/mem/doctor.py`.

## Context
`mem serve` refuses a store that has no fitted gate (architecture-v2 §3.3), and a gate is fitted by `mem calibrate` on
the user's own data. A new install therefore cannot serve until the user has run a calibration, which needs labelled
data they do not have. The gate is fitted for one config fingerprint (`fingerprint_hash` in `gate.py`): embedder,
reranker, retrieval settings.

## Decision
1. The gate fitted on public data (the LongMemEval tune split plus SYN) ships inside the package as data, next to the
   code, for the pinned config only.
2. Its metadata records: the fingerprint it was fitted for; the dataset name, version and sha256 of each source;
   LongMemEval's MIT licence (name and copyright line); the fit date and mem version. Redistributing a fit derived from
   MIT data needs the notice, so it travels with the file.
3. `mem serve` loads the local gate when one exists, else the shipped one if its fingerprint equals the running
   config's. A local gate always wins; the shipped one is never written over it.
4. On a fingerprint mismatch (the user changed embedder, reranker or retrieval), serving stays refused as today and
   `mem doctor` prints the exact `mem calibrate` command for the current config, with the paths filled in.
5. Hugging Face gating: the pinned embedder (EmbeddingGemma) is gated, so setup keeps the step where the user accepts
   its licence and logs in. If F adopts an Apache-2.0 embedder, that step disappears and the shipped gate is refitted
   for it. The reranker is not gated.
   **Update (MEM2-274):** F adopted Qwen3-Embedding-0.6B (Apache-2.0); the default config needs no Hugging Face
   account or login any more. Checked on the Hugging Face API on 2026-10-30: `Qwen/Qwen3-Embedding-0.6B` and
   `BAAI/bge-reranker-v2-m3` are `gated: false`, licence `apache-2.0`; `google/embeddinggemma-300m` is `gated: manual`,
   licence `gemma`. The licence step is gone from the README and docs; it stays documented only for a user who selects
   EmbeddingGemma. The shipped gate is the one fitted on the new default (`r10-all`).

## Alternatives considered
- **Make the user calibrate first:** correct but blocks first use; the owner asked for easy setup.
- **Fit a default on the user's first data automatically:** too little data to fit thresholds; a bad gate is worse than
  a refusal.
- **Download the gate at install time:** needs the network and a host to run; the everything-local rule says no.

## Implementation (MEM2-233)
`src/mem/shipped.py` reads `src/mem/shipped/serving_gate.json` (gate plus metadata); `serve.preflight`, `Store`'s
`fallback_gate` and `mem doctor --store` use it; `bench package-gate` writes it and refuses a fit that saw an LME test
question. Host step: `deliverables/runbooks/setup-gate.md`.

## Consequences
- A config change always costs a recalibration; doctor says exactly how. No silent use of a gate fitted for other models.
- Each release that changes the pinned config must refit and re-ship the gate; a test fails when the shipped
  fingerprint differs from the default config's.
- The package carries one small data file and a licence notice.
