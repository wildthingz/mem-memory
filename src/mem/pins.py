"""Pinned model revisions (D2) and the embedder fingerprint (architecture-v2 §3.3).

Provisional until `config.yaml` lands in P2: the section below is the single place the
revisions are written, so the P2 loader replaces it without touching callers.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Pin:
    id: str
    revision: str

    @property
    def cache_dir_name(self) -> str:
        return "models--" + self.id.replace("/", "--")


EMBEDDER = Pin("google/embeddinggemma-300m", "57c266a740f537b4dc058e1b0cda161fd15afa75")
RERANKER = Pin("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")

EMBEDDER_PROMPTS = {
    "query_prompt": "task: search result | query: ",
    "doc_prompt": "title: none | text: ",
}
RERANK_MAX_LEN = 512


def embedder_section() -> dict[str, Any]:
    return {
        "id": EMBEDDER.id,
        "revision": EMBEDDER.revision,
        "dim": 768,
        "normalize": True,
        **EMBEDDER_PROMPTS,
        "backend": "sentence-transformers",
        "dtype": "fp32",
        "max_len": 512,
    }


def canonical_json(section: dict[str, Any]) -> str:
    return json.dumps(section, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fp_embed(section: dict[str, Any] | None = None) -> str:
    text = canonical_json(section if section is not None else embedder_section())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hub_cache_dir() -> Path:
    """The HF hub cache, resolved at call time so HF_HOME / HF_HUB_CACHE changes are honoured."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def snapshot_dir(pin: Pin) -> Path:
    return hub_cache_dir() / pin.cache_dir_name / "snapshots" / pin.revision
