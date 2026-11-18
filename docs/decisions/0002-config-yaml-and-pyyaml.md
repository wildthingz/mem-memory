# 0002: `config.yaml` loader, PyYAML, and contract files

**Status:** accepted (MEM2-8, within the spec §3.3 and pvd-roadmap §2.3).

**Decisions**
1. `src/mem/config.py` builds on `pins.py` (kept; `doctor` uses it). `pins.py` stays the one place
   model revisions are written. Rejected: folding `pins.py` into `config.py` (churn in `doctor`).
2. `fp_embed` = sha256 of the canonical embedder section (unchanged). `fp_all` = sha256 of the
   canonical JSON of the list `[embedder, reranker, retrieval]`. The spec says only "embedder +
   reranker + retrieval"; this is the concrete encoding. `gate` and `device` are outside both.
3. Reranker section fields beyond the spec's list: `backend: sentence-transformers`,
   `pair_format`, `activation: none` (raw logit). Provisional like decision 3 of 0001; changing one
   is a new fingerprint.
4. New dependency `pyyaml>=6` for `config.yaml` (the spec's format). Rejected: TOML (stdlib, but
   not the spec's format).
5. MCP tool contract is `src/mem/contracts/mcp_tools.json` (names, input schemas with defaults,
   `structuredContent` schemas). `test_mcp_tool_contracts` compares it with
   `mem.serve.list_tool_schemas()`; it is strict-xfail until p1-serve provides that function.
   The structured shapes for `get`, `link`, `subjects` and `forget` are not spelled out in §4.1;
   they are my reading and are frozen here for review.
6. Fakes (`mem.testing`: `HashEmbedder`, `OverlapReranker`) are the T0 `HashEmbed-256` stub; they
   live in the package because `bench` T0 uses them too.
