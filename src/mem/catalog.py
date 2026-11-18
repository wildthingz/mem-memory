"""The pinned model catalog (MEM2-219, ADR 0025): every embedder and reranker `models.embedder` /
`models.reranker` in config.yaml can name, with its pinned revision, licence, documented prompts and
the numbers behind the VRAM check.

The defaults are `qwen3-embedding-0.6b` (since MEM2-274, Round 10 r10-all; `DEFAULT_EMBEDDER`) and
`bge-reranker-v2-m3`. `embeddinggemma-300m`, the default before, produces exactly the section `mem.pins`
always produced, so stores and gates fitted on it keep their fingerprints when a config names it. A candidate's section is its fingerprint: id, revision, dtype, prompts, max length and
the way its score is read are all inside it.

Licences were read off each model card's front matter on 2026-10-26 (gating re-checked on the hub API
on 2026-10-30: the two defaults are not gated, EmbeddingGemma is). Qwen3-Reranker-0.6B and any Gemma-licensed reranker are not
entries and `models.reranker` refuses them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mem import pins
from mem.pins import Pin

GIB = 2**30
# CUDA context + caching-allocator overhead of the one serving process, fitted (MEM2-264) to the
# default set's host peak: 9,461 MiB over 450 idle = 8.80 GiB, of which weights + activations below
# are 5.76. One measurement: how it splits between per-process and per-model is not known, so it is
# counted once per process (p3-gate.md Round 9 F's rr4b probe measures a second set).
PROCESS_GIB = 3.0
BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}

QWEN_DOC_FORMAT = "{ctx}\n{text}"
QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query"
    "\nQuery:"
)  # the model card's `get_detailed_instruct` / the `query` prompt of its sentence-transformers config
RERANK_PAIR_FORMAT = "query, ctx + \n + text"


@dataclass(frozen=True)
class Prompts:
    """How text reaches an embedder. `doc_format` takes `ctx` and `text`; `doc_format_no_ctx` (when
    set) is used for a chunk without context. The default is the EmbeddingGemma one."""

    query_prompt: str = pins.EMBEDDER_PROMPTS["query_prompt"]
    doc_format: str = "title: {ctx} | text: {text}"
    doc_format_no_ctx: str | None = None

    def doc(self, ctx: str, text: str) -> str:
        if not ctx and self.doc_format_no_ctx is not None:
            return self.doc_format_no_ctx.format(text=text)
        return self.doc_format.format(ctx=ctx, text=text)


DEFAULT_PROMPTS = Prompts()


def prompts_of(section: dict[str, Any]) -> Prompts:
    """The prompts an embedder section documents. A section without `doc_format` is the Gemma one."""
    if "doc_format" not in section:
        return Prompts(query_prompt=section.get("query_prompt", DEFAULT_PROMPTS.query_prompt))
    return Prompts(section["query_prompt"], section["doc_format"], section.get("doc_format_no_ctx"))


def query_prompt(embedder: object) -> str:
    return getattr(embedder, "prompts", DEFAULT_PROMPTS).query_prompt


def doc_prompted(ctx: str, text: str, embedder: object | None = None) -> str:
    return getattr(embedder, "prompts", DEFAULT_PROMPTS).doc(ctx, text)


@dataclass(frozen=True)
class Entry:
    name: str
    kind: str  # "embedder" | "reranker"
    pin: Pin
    licence: str
    params: int  # parameters, from the model card / hub safetensors metadata
    build: Callable[[], dict[str, Any]]
    act_gib: float = 0.0  # embedder: activations at the reembed batch (assumed)
    act_gib_per_item: float = (
        0.0  # reranker: activations per (query, doc) pair in a batch (assumed)
    )
    default_batch: int = 0  # reranker: pairs per forward; 0 = the whole pool

    def section(self) -> dict[str, Any]:
        return self.build()

    @property
    def dtype(self) -> str:
        return self.section()["dtype"]

    @property
    def weights_gib(self) -> float:
        return self.params * BYTES[self.dtype] / GIB


def _gemma() -> dict[str, Any]:
    return pins.embedder_section()


def _qwen_embedding() -> dict[str, Any]:
    pin = EMBEDDERS["qwen3-embedding-0.6b"].pin
    return {
        "id": pin.id,
        "revision": pin.revision,
        "dim": 1024,
        "normalize": True,
        "query_prompt": QWEN_QUERY_PROMPT,
        "doc_prompt": "",  # the card: "No need to add instruction for retrieval documents"
        "doc_format": QWEN_DOC_FORMAT,
        "doc_format_no_ctx": "{text}",
        "backend": "sentence-transformers",
        "dtype": "fp32",
        "max_len": 512,
        "pooling": "last-token",
        "padding_side": "left",
    }


def _reranker(pin: Pin, dtype: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": pin.id,
        "revision": pin.revision,
        "backend": "sentence-transformers",
        "dtype": dtype,
        "max_len": pins.RERANK_MAX_LEN,
        "pair_format": RERANK_PAIR_FORMAT,
        "activation": "none",
        **extra,
    }


EMBEDDERS: dict[str, Entry] = {}
RERANKERS: dict[str, Entry] = {}

EMBEDDERS["embeddinggemma-300m"] = Entry(
    "embeddinggemma-300m", "embedder", pins.EMBEDDER, "gemma", 308_000_000, _gemma, act_gib=1.0
)
EMBEDDERS["qwen3-embedding-0.6b"] = Entry(
    "qwen3-embedding-0.6b",
    "embedder",
    Pin("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "apache-2.0",
    595_776_512,
    _qwen_embedding,
    act_gib=1.0,
)
DEFAULT_EMBEDDER = "qwen3-embedding-0.6b"  # MEM2-274: the host's final config (Round 10 r10-all)
RERANKERS["bge-reranker-v2-m3"] = Entry(
    "bge-reranker-v2-m3", "reranker", pins.RERANKER, "apache-2.0", 567_754_752,
    lambda: _reranker(pins.RERANKER, "fp32"), act_gib_per_item=0.05,
)  # fmt: skip
RERANKERS["qwen3-reranker-4b"] = Entry(
    "qwen3-reranker-4b",
    "reranker",
    Pin("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5"),
    "apache-2.0",
    4_021_784_576,
    lambda: _reranker(
        RERANKERS["qwen3-reranker-4b"].pin, "bf16", score="logit(yes) - logit(no) at the last token"
    ),
    act_gib_per_item=0.25,
    default_batch=4,
)
RERANKERS["mxbai-rerank-large-v2"] = Entry(
    "mxbai-rerank-large-v2",
    "reranker",
    Pin("mixedbread-ai/mxbai-rerank-large-v2", "ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6"),
    "apache-2.0",
    1_543_714_304,
    lambda: _reranker(
        RERANKERS["mxbai-rerank-large-v2"].pin, "fp32", score="raw sentence-transformers score"
    ),
    act_gib_per_item=0.25,
    default_batch=2,  # 8 was 12.9 GiB on the recalibrated estimate (MEM2-264), 2 is 11.4
)


def lookup(kind: str, name: str) -> Entry:
    table = EMBEDDERS if kind == "embedder" else RERANKERS
    if name not in table:
        raise ValueError(f"models.{kind} {name!r} is not a pinned model: choose one of "
                         + ", ".join(table))  # fmt: skip
    return table[name]


def _entry_for(table: dict[str, Entry], kind: str, section: dict[str, Any]) -> Entry:
    for entry in table.values():
        if entry.section() == section:
            return entry
    raise ValueError(f"the config's {kind} section ({section.get('id')!r}, revision "
                     f"{str(section.get('revision'))[:8]}) is not a pinned catalog entry: select one "
                     f"with models.{kind}: <name> ({', '.join(table)})")  # fmt: skip


def embedder_entry(section: dict[str, Any]) -> Entry:
    return _entry_for(EMBEDDERS, "embedder", section)


def reranker_entry(section: dict[str, Any]) -> Entry:
    return _entry_for(RERANKERS, "reranker", section)


# ---- VRAM ---------------------------------------------------------------------------------------


def vram_parts(cfg: Any) -> list[tuple[str, float]]:
    """(part, GiB) of the serving set `cfg` loads: each model's weights at the section's dtype
    plus its activations (embedder: a fixed allowance; reranker: per pair x the pairs of one
    forward, the batch size or, at 0, the whole pool), then the process's CUDA context and
    allocator overhead (`PROCESS_GIB`, fitted to the default set's measured peak)."""
    emb, rr = embedder_entry(cfg.embedder), reranker_entry(cfg.reranker)
    items = cfg.rerank_batch_size or int(cfg.retrieval["pool"])
    return [
        (emb.pin.id, emb.weights_gib + emb.act_gib),
        (rr.pin.id, rr.weights_gib + rr.act_gib_per_item * items),
        ("CUDA context + allocator", PROCESS_GIB),
    ]


def vram_gib(cfg: Any) -> float:
    return sum(gib for _, gib in vram_parts(cfg))


# ---- scoring ------------------------------------------------------------------------------------


def score_in_batches(
    predict: Callable[[list[tuple[str, str]]], Sequence[float]],
    query: str,
    docs: Sequence[str],
    batch_size: int,
) -> list[float]:
    """Raw scores of (query, doc) pairs, `batch_size` pairs per `predict` call (0 = one call), in
    `docs` order. The scores are what the model's head returns with no activation, so `fit_platt`
    gets the same kind of number for every reranker: bge-reranker-v2-m3's logit, Qwen3-Reranker's
    logit(yes) - logit(no) (the log-odds of the card's p(yes)), mxbai-rerank's raw score."""
    if not docs:
        return []
    step = batch_size if batch_size > 0 else len(docs)
    out: list[float] = []
    for start in range(0, len(docs), step):
        out += [float(x) for x in predict([(query, d) for d in docs[start : start + step]])]
    if len(out) != len(docs):
        raise ValueError(f"reranker returned {len(out)} scores for {len(docs)} documents")
    return out
