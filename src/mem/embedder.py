"""The real embedder (a pinned catalog entry, `mem.catalog`) behind the `mem.testing.Embedder` protocol.

Heavy: needs the `models` extra (sentence-transformers). The pinned snapshot must already be in the
HF hub cache (`mem doctor --fetch`); loading never touches the network (D2).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mem import catalog, pins

TORCH_DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


class SentenceTransformerEmbedder:
    """Callers pass the text already prefixed (`Store.query_vector`, `doc_text`): the prompt
    strings are part of the fingerprint, so they are applied there, once, not by the model.
    `prompts` tells those callers which strings the section documents."""

    def __init__(self, section: dict[str, Any], device: str = "auto") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        catalog.embedder_entry(section)  # a pinned entry, nothing else loads
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False  # fp32 results (D1)
        self.device = device
        self._model = SentenceTransformer(
            str(pins.snapshot_dir(pins.Pin(section["id"], section["revision"]))),
            device=device,
            local_files_only=True,
            model_kwargs={"torch_dtype": getattr(torch, TORCH_DTYPES[section["dtype"]])},
            tokenizer_kwargs={"padding_side": section.get("padding_side", "right")},
        )
        self._model.max_seq_length = section["max_len"]
        self.dim = section["dim"]
        self.prompts = catalog.prompts_of(section)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, prompt=""
        )
        return [[float(x) for x in row] for row in out]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_docs(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)
