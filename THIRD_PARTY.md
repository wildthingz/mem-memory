# Third-party licences

mem itself is MIT-licensed (see `LICENSE`). It uses the software, models and data below, each under
its own terms. "Checked" gives where the licence was read and when. Anything marked **not verified**
was not checked at its source and should be checked before you rely on it.

## Benchmark data and code

### LongMemEval

- Used by: `bench/fixtures/lme/` (`lme_s_mini.json`, `lme_m_mini.json`: small slices of the dataset
  for tests), the `bench/adapters/` and `bench/datasets/` code that reads the dataset, and
  `bench/metrics/upstream_lme.py`, which is vendored from LongMemEval's
  `src/retrieval/eval_utils.py` (one change, described in the file).
- Licence: MIT, "Copyright (c) 2024 Di Wu". The dataset card of `xiaowu0162/longmemeval-cleaned`
  on Hugging Face also declares `mit`.
  Checked: `github.com/xiaowu0162/LongMemEval` (LICENSE) and the Hugging Face dataset card, 2026-10-25.
- The MIT notice that travels with the vendored file:

  > MIT License
  >
  > Copyright (c) 2024 Di Wu
  >
  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software
  > and associated documentation files (the "Software"), to deal in the Software without
  > restriction, including without limitation the rights to use, copy, modify, merge, publish,
  > distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
  > Software is furnished to do so, subject to the following conditions: the above copyright notice
  > and this permission notice shall be included in all copies or substantial portions of the
  > Software. The Software is provided "as is", without warranty of any kind.

  The full text is in the upstream repository's LICENSE file.
- Cite the paper if you report numbers from the benchmark:

  ```bibtex
  @article{wu2024longmemeval,
    title   = {LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
    author  = {Di Wu and Hongwei Wang and Wenhao Yu and Yuwei Zhang and Kai-Wei Chang and Dong Yu},
    year    = {2024},
    eprint  = {2410.10813},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
    url     = {https://arxiv.org/abs/2410.10813}
  }
  ```

  (The upstream README says the paper was accepted at ICLR 2025.)
- **Not verified:** the licences of the sources LongMemEval's own sessions were built from. The full
  dataset is not shipped here; `bench fetch` downloads it from the official location.

## Models (downloaded by you; no weights are redistributed)

### Qwen3-Embedding-0.6B (`Qwen/Qwen3-Embedding-0.6B`), the default embedder

- Licence: Apache-2.0. Not gated. Checked: the model card front matter (`license: apache-2.0`),
  2026-10-26, and the Hugging Face API (`gated: false`), 2026-10-30.

### EmbeddingGemma (`google/embeddinggemma-300m`), optional (`models.embedder: embeddinggemma-300m`)

- Licence: the Gemma Terms of Use, <https://ai.google.dev/gemma/terms>, with the Gemma Prohibited
  Use Policy, <https://ai.google.dev/gemma/prohibited_use_policy>. The Hugging Face repository is
  gated: you accept the terms on its page with your own Hugging Face account, then `hf auth login`.
  mem downloads the weights to your Hugging Face cache and never copies them into this repository.
  Checked: the model card, 2026-10-25. The terms are not MIT: read them.

### bge-reranker-v2-m3 (`BAAI/bge-reranker-v2-m3`)

- Licence: Apache-2.0. Not gated. Checked: the model card (`License: apache-2.0`), 2026-10-25, and the
  Hugging Face API (`gated: false`), 2026-10-30.

## Python dependencies

Declared in `pyproject.toml`, locked in `uv.lock`. Licences read from the package metadata of the
locked versions, and, for the first four, from the project's repository on 2026-10-25.

| Package | Licence | Notes |
|---|---|---|
| sqlite-vec | MIT or Apache-2.0 (dual: `LICENSE-MIT`, `LICENSE-APACHE`) | vector index in SQLite; checked at the repository |
| torch | BSD-3-Clause | the `models` extra, via sentence-transformers; checked at the repository |
| sentence-transformers | Apache-2.0 | the `models` extra; checked at the repository |
| mcp (MCP Python SDK) | MIT | checked at the repository |
| transformers | Apache-2.0 (**not verified**) | pulled in by sentence-transformers; not checked this time |
| huggingface-hub | Apache-2.0 | package metadata |
| numpy | BSD-3-Clause, with bundled 0BSD, MIT, Zlib, CC0-1.0 parts | package metadata |
| scikit-learn | BSD-3-Clause | package metadata |
| jsonschema | MIT | package metadata |
| jinja2 | BSD-3-Clause | package metadata |
| markdown-it-py | MIT | package metadata |
| pyyaml | MIT | package metadata |
| ijson | BSD-3-Clause and ISC | package metadata |
| typer | MIT | package metadata |
| uvicorn | BSD-3-Clause | package metadata |
| cryptography | Apache-2.0 or BSD-3-Clause | encrypted backups; a declared dependency; package metadata |
| anthropic | MIT (**not verified**) | optional `llm` extra, used only by paid benchmark legs; not checked this time |
| pytest, ruff | MIT | development only; package metadata |

The lock file also pins their transitive dependencies (`uv.lock` lists every one with its version);
those were not reviewed one by one.
