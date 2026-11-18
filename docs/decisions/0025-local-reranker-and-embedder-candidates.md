# 0025: Local reranker and embedder candidates behind config (option F)

- **Status:** **The embedder candidate is adopted (MEM2-274): `models.embedder` defaults to `qwen3-embedding-0.6b`** (Apache-2.0,
  not gated), as part of the host's final config Round 10 `r10-all` (`fp_all` 3f8c9c11). Round 10 tune evidence (the host's runs, reported in MEM2-274 from the owner's MEM2-1 comment of 30 Oct): B-E1 tune **0.832** for `r10-all` against the Round 10 base arm 0.792 and Round 9's 9c 0.776. The reranker stays
  `bge-reranker-v2-m3`; the reranker candidates are not adopted. `embeddinggemma-300m` stays an entry, so stores fitted on it keep
  their fingerprint when the config names it. Was: proposed (defaults unchanged).
- **Date:** 2026-10-26 · **Task:** MEM2-219 (option F; report §5, scope note §0 and §2)
- **Amended:** 2026-10-28, MEM2-264: the Qwen3-Reranker score sanity checked against the card (§4), VRAM recalibrated on the measured default set (§8, VRAM)

## Decision
1. **Catalog** (`mem.catalog`): `models.embedder` / `models.reranker` in `config.yaml` name a pinned entry. Defaults
   are the current models, so with no `models:` section (or the defaults named) every section, `fp_embed`, `fp_all`
   and `vec_<fp8>` is byte for byte what stores carry today (`tests/test_config.py` golden hashes still hold).
2. **Candidates** (licence read off each model card's front matter, 2026-10-26; revision = hub commit sha, pinned
   like the defaults; `mem doctor --fetch` downloads exactly these):

   | name | repo | revision | licence | params | dtype in mem |
   |---|---|---|---|---|---|
   | `qwen3-reranker-4b` | Qwen/Qwen3-Reranker-4B | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | Apache-2.0 | 4.02 B | bf16 |
   | `mxbai-rerank-large-v2` | mixedbread-ai/mxbai-rerank-large-v2 | `ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6` | Apache-2.0 | 1.54 B | fp32 |
   | `qwen3-embedding-0.6b` | Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | Apache-2.0 | 0.60 B | fp32 |

   Not entries, and refused by name: Qwen3-Reranker-0.6B, any Gemma-licensed reranker. (EmbeddingGemma, the
   default embedder, is Gemma-licensed; it is the existing choice, not a candidate.) There is no `THIRD_PARTY.md`
   on this branch, so no entry was added there; the table above is the licence record.
3. **Local and in-process.** All three load through sentence-transformers inside the one `mem serve` process (D5);
   no vLLM and no new server. (`mem serve --http` already exposes `/models` on 127.0.0.1 to local tools; nothing
   changes there.) If the 4B reranker cannot meet latency in-process, the runbook reports it and **stops to ask**;
   a local server would have to be optional and bound to 127.0.0.1.
4. **Raw score for Platt** (`a·z + b` on `z`, any monotone real score works):
   - bge-reranker-v2-m3: classifier logit (unchanged).
   - Qwen3-Reranker-4B: `logit(yes) − logit(no)` at the last token. The card's probability is the softmax over those
     two logits, so `z` is exactly the log-odds of the card's p(yes) and Platt starts near slope 1. The card's prompt
     template (system judge text, `<Instruct>/<Query>/<Document>`, default instruction "Given a web search query,
     retrieve relevant passages that answer the query") is applied by the model's own sentence-transformers config
     (`1_LogitScore`). The card reads 6.44 / −14.38 for its two example documents: the runbook checks them.
     **Checked against the card (MEM2-264).** The host got [6.3125, −11.8125] for [6.4375, −14.375] because the runbook fed
     only the first sentence of the card's second document ("Gravity is a force that attracts two bodies towards each other.");
     the card's is two sentences longer. A different input, not a scoring fault: the relevant pair, the same input on both
     sides, is 0.125 off (bf16 rounding on another GPU/kernel). Offline, at revision 22e68366 and the locked
     sentence-transformers 6.2.0: the repo's `chat_template.jinja` rendered with the messages ST builds for a `(query, doc)`
     pair (system = the config's `query` prompt, roles `query`/`document`) is byte-identical to the card's transformers
     recipe, `prefix + format_instruction(task, q, d) + suffix`, for both card documents (system judge text, `<Instruct>: …
     \n<Query>: …\n<Document>: …`, the default instruction, `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`).
     ST's `Transformer` forces left padding and checks it per batch (`_verify_left_padding`), keeps the logits of the last
     position only (`logits_to_keep=1`), and `LogitScore` returns `logit[9693] − logit[2152]` ("yes" − "no", from the repo's
     `1_LogitScore/config.json`); with left padding the last position is each row's last real token, so unequal lengths are
     safe. bf16 is the checkpoint's `torch_dtype`, which is what the card's own ST snippet loads too. `max_length` is 512 in
     mem (8192 in the card's transformers recipe): no effect on the example (well under 100 tokens); for longer pairs ST
     truncates and then restores the template suffix (`restore_suffix`), so the last token is still the `\n\n` after
     `</think>`. Nothing in mem's path changes. Token ids were not compared here (no tokenizer in the sandbox's cache, and
     none is downloaded); the runbook's sanity step prints them against the card's recipe on the host before scoring.
   - mxbai-rerank-large-v2: the raw unbounded score `CrossEncoder.predict` returns (the card shows ≈12 for the
     relevant and ≈0.5 for an irrelevant document). **The card states no formula**, so nothing about its scale is
     assumed beyond "monotone raw"; Platt fits whatever scale it has.
   Identity activation for all; `max_len` 512 for all (pairs are ctx + chunk text, as for bge), so pools compare.
5. **Batching.** `models.rerank_batch_size` (pairs per forward; default per model: bge 0 = whole pool as before,
   Qwen3-4B 4, mxbai ~~8~~ 2 since MEM2-264). ~~It is outside the fingerprint (a runtime knob like `device`; a smaller batch may move
   half-precision scores in the last digits, which the runbook's repeat runs would show).~~ **Amended (MEM2-251,
   review MEM2-222 F23):** a set batch size (non-zero) is in `fp_all`, since a batch that moves half-precision scores
   voids a gate fitted at another; at 0 (bge's default) `fp_all` is unchanged.
6. **Embedder prompts.** Qwen3-Embedding-0.6B: query `Instruct: Given a web search query, retrieve relevant passages
   that answer the query\nQuery:` (the card's `get_detailed_instruct`), no document prompt (card: "No need to add
   instruction for retrieval documents"); mem's own context header goes in front of the chunk (`{ctx}\n{text}`,
   bare `{text}` when empty); last-token pooling, left padding, 1024-d, normalised. All of it is in the section,
   so in the fingerprint. The prompts were hard-coded to EmbeddingGemma's in `Store.query_vector`, `doc_text`,
   `subjectmask` and `relphrase`; they now come from the embedder (`Embedder.prompts`, default Gemma's; the
   `/models` client reads them from the served section), so a candidate embedder is never fed Gemma prompts.
7. **Fingerprint and refusals.** Each model's section is in `fp_all`; the embedder's also names `vec_<fp8>`. Existing
   guards apply unchanged and are now tested per model: `mem serve` preflight refuses a store whose vectors or gate
   belong to another model; `mem switch` refuses a gate fitted for another `fp_all` and a missing/partial vec table;
   `mem reembed` fills one `vec_<fp8>` per embedder fingerprint beside the old one. The loaders refuse a section
   that is not a catalog entry.
8. **`mem doctor`** reports `vram budget`: weights (params × dtype bytes) + activations (embedder 1.0 GiB; reranker per
   pair × pairs per forward, the batch or, at 0, the pool) + ~~0.5 GiB CUDA context per model~~ **3.0 GiB CUDA context and
   allocator overhead per process (MEM2-264, fitted to the measured default set, below)**, and **fails over 12 GiB**.
   `mem serve` refuses to start over it too. `--fetch` and the model checks follow the config.

## VRAM (recalibrated on one measured set, MEM2-264)
The first estimate (weights from the hub's parameter counts at the dtype above, 0.5 GiB context per model, 1.0 GiB embedder
activations, 0.05 GiB per pair for bge, 0.25 GiB per pair for the two LLM-based rerankers) said **6.8 GiB** for the default set.
The host measured it (Round 9 · F, 2d50067, nvidia-smi over a whole `f_measure`): **9,461 MiB peak, 450 idle: 8.80 GiB**, 2.0 GiB
over. Weights + activations as assumed are 5.76 GiB, so 3.04 GiB is CUDA context, the caching allocator's reserved-but-free
blocks and whatever the activation allowances miss. One measurement cannot say how that splits between the process and each
model, so it is one per-process constant, `PROCESS_GIB = 3.0`: the default set's estimate is now **8.76 GiB, 0.5% under the
measurement**. Activation allowances and parameter counts are unchanged.

| serving set (pool 30) | weights GiB | doctor estimate GiB (was) | fits 12? |
|---|---|---|---|
| default: gemma-300m (fp32) + bge-reranker-v2-m3 (fp32, whole pool) | 1.1 + 2.1 | **8.8** (6.8); measured 8.8 | yes |
| gemma-300m + qwen3-reranker-4b (bf16, batch 4) | 1.1 + 7.5 | **13.6** (11.6) | **no** |
| gemma-300m + qwen3-reranker-4b (bf16, batch 1) | 1.1 + 7.5 | **12.9** | **no** |
| gemma-300m + mxbai-rerank-large-v2 (fp32, batch 8) | 1.1 + 5.8 | **12.9** (10.9) | **no** |
| gemma-300m + mxbai-rerank-large-v2 (fp32, batch 2, the new default) | 1.1 + 5.8 | **11.4** | yes, 0.6 headroom |
| qwen3-embedding-0.6b (fp32) + bge-reranker-v2-m3 | 2.2 + 2.1 | **9.8** (7.8) | yes |
| qwen3-embedding-0.6b (fp32) + qwen3-reranker-4b (bf16, batch 4) | 2.2 + 7.5 | **14.7** (12.7) | **no** |

**What it means for rr4b.** Its bf16 weights (7.5 GiB) + the embedder's (1.1) + the overhead (3.0) are 11.6 GiB before any
activation. A smaller batch does not rescue it (batch 1 is 12.9 GiB) and neither does a smaller pool: once a batch is set, the
pool does not enter the activations (only the pairs of one forward are live). So on the estimate the 4B reranker does not fit a
12 GiB budget at all, and `mem doctor` / `mem serve` refuse it. The estimate is conservative only if the 3.0 GiB is really per
process: if most of it is bge's whole-pool forward (30 fp32 pairs at 512 tokens), rr4b at batch 4 would be near the old 11.6.
That is what the runbook's **rr4b probe** measures (default embedder + the 4B in one process, a 32-doc embed batch and a 30-doc
pool at batch 4, 512-token inputs, nvidia-smi and torch peaks). The constants are refitted from it before any rr4b A/B run;
if the probe runs out of memory or its peak − idle is over 12 GiB, the 4B reranker needs a 16 GB card. mxbai's default batch
drops from 8 to 2 (11.4 GiB; nothing was gated on 8).

**Minimum GPU for the work machine:** a 12 GB card runs the default, mxbai at batch 2 and the embedder candidate. The 4B
reranker needs a 16 GB card unless the probe shows the overhead is per model, not per process.

## Adoption rule (B-R4 model A/B; the runbook step "Round 9 · F")
A candidate replaces a default only if, on TUNE, against the 9a baseline: **B-C1 holds** (warm p95 ≤ 1,500 ms at L5k,
≤ 1,200 ms median at L50k, three runs each) **and retrieval improves** (recall_any@5 and nDCG@10 not lower with the
paired-bootstrap CI lower bound of the difference > 0 on at least one, pool recall not lower; B-R5 current@1 not
lower), with VRAM peak ≤ 12 GiB and RSS inside the memory rules. A cost-saving swap (smaller or faster model) instead
needs non-inferiority at m = 0.02 with ≥ 80% power (benchmarks-v2 B-R4). Then the gate is re-calibrated on tune and
the affected rows re-run. If the 4B reranker misses latency, the runbook tries a smaller batch and a smaller pool and
reports both runs; nothing is assumed. Otherwise the defaults stay.

## Not measured here
Latency, the candidates' VRAM, retrieval quality and the exact scores of any candidate: the sandbox never loads these models (the
default set's VRAM peak above is the host's measurement).
