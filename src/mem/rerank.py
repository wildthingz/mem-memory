"""The real reranker (a pinned catalog entry, `mem.catalog`) behind the `mem.testing.Reranker`
protocol: raw scores, no activation (architecture-v2 §3.2 step 5, §3.3).

Heavy: needs the `models` extra. bge-reranker-v2-m3 is loaded exactly as the day-1 spike measured it
(MEM2-2, verdict `pass`): fp32, `device: auto` = CUDA when torch sees it, TF32 off so GPU logits
match CPU (D1). The Qwen3 and mxbai candidates (ADR 0025) go through the same sentence-transformers
CrossEncoder with the identity activation; their model cards give the score:

- bge-reranker-v2-m3: the classifier logit.
- Qwen3-Reranker-4B: `logit(yes) - logit(no)` at the last token (the card's p(yes) is the softmax
  of those two, so this is its log-odds). sentence-transformers renders the repo's chat template
  (the card's prefix, `<Instruct>/<Query>/<Document>` and suffix, byte for byte), pads left and reads
  the last position, as the card's own recipe does (checked in MEM2-264, ADR 0025 §4). Loaded bf16
  (the checkpoint's dtype; fp32 would not fit 12 GB).
- mxbai-rerank-large-v2: the raw unbounded score `CrossEncoder.predict` returns (the card shows ~12
  for a relevant and ~0.5 for an irrelevant document and states no formula).
Platt takes `a * z + b` on any of them, so only monotone raw scores are required.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mem import catalog, pins
from mem.embedder import TORCH_DTYPES


class CrossEncoderReranker:
    def __init__(self, section: dict[str, Any], device: str = "auto", batch_size: int = 0) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        catalog.reranker_entry(section)  # a pinned entry, nothing else loads
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False  # fp32 results (D1)
            torch.backends.cudnn.allow_tf32 = False
        self.device, self.batch_size = device, batch_size
        self._model = CrossEncoder(
            str(pins.snapshot_dir(pins.Pin(section["id"], section["revision"]))),
            device=device,
            max_length=section["max_len"],
            activation_fn=torch.nn.Identity(),  # raw scores
            local_files_only=True,
            model_kwargs={"torch_dtype": getattr(torch, TORCH_DTYPES[section["dtype"]])},
        )

    def _predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        return self._model.predict(pairs, batch_size=len(pairs), convert_to_numpy=True)

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        return catalog.score_in_batches(self._predict, query, docs, self.batch_size)
