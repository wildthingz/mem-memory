"""P1 day-1 spike: measure the serving host (pvd-roadmap §1.5, architecture-v2 §3.3 D1).

A measurement, not a test. Needs `uv sync --extra models` and the pinned models in the HF hub
cache (`mem doctor --fetch`). Runs offline. Writes one JSON record.

    uv run --extra models python bench/spike.py [--out PATH] [--runs 3] [--calls 50] [--cpu-calls 5]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from mem import doctor
from mem.pins import EMBEDDER, EMBEDDER_PROMPTS, RERANK_MAX_LEN, RERANKER, snapshot_dir
from mem.spike import cosine_parity, logit_parity, summarize_ms, verdict

N_PAIRS = 30
N_PARITY_CHUNKS = 200
N_PARITY_PAIRS = 50
AVG_CHUNK_TOKENS = 190  # pvd-roadmap §1.5: query 15 + prefix 15 + chunk ~190 = ~225 per pair
QUERY_TOKENS = 15
VOCAB = [
    "memory",
    "note",
    "project",
    "meeting",
    "decision",
    "budget",
    "vault",
    "server",
    "model",
    "query",
    "result",
    "schedule",
    "plan",
    "review",
    "release",
    "garden",
    "recipe",
    "travel",
    "invoice",
    "contract",
    "sprint",
    "latency",
    "index",
    "table",
    "store",
    "entity",
    "alias",
    "fact",
    "event",
    "source",
    "claim",
    "topic",
    "person",
    "place",
    "date",
    "weekly",
    "summary",
    "draft",
]


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def synth_text(rng: np.random.Generator, n_tokens: int, tokenizer) -> str:
    """Deterministic pseudo-prose cut to exactly n_tokens model tokens (no special tokens)."""
    words = " ".join(rng.choice(VOCAB, size=n_tokens * 2))
    ids = tokenizer(words, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids)


def timed_ms(fn) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000


def rerank_ms(rr, pairs) -> float:
    return timed_ms(lambda: rr.predict(pairs, batch_size=N_PAIRS))


def load_models(device: str):
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False  # keep fp32 results (D1)
        torch.backends.cudnn.allow_tf32 = False
    emb = SentenceTransformer(str(snapshot_dir(EMBEDDER)), device=device, local_files_only=True)
    emb.max_seq_length = 512
    rr = CrossEncoder(
        str(snapshot_dir(RERANKER)),
        device=device,
        max_length=RERANK_MAX_LEN,
        activation_fn=torch.nn.Identity(),  # raw logits
        local_files_only=True,
    )
    return emb, rr


def make_workload(emb, rr, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    etok, rtok = emb.tokenizer, rr.tokenizer
    chunks = [synth_text(rng, AVG_CHUNK_TOKENS, etok) for _ in range(N_PARITY_CHUNKS)]
    query = synth_text(rng, QUERY_TOKENS, rtok)
    avg_pairs = [(query, synth_text(rng, AVG_CHUNK_TOKENS, rtok)) for _ in range(N_PAIRS)]
    worst_pairs = [(query, synth_text(rng, RERANK_MAX_LEN, rtok)) for _ in range(N_PAIRS)]
    parity_pairs = [(query, synth_text(rng, AVG_CHUNK_TOKENS, rtok)) for _ in range(N_PARITY_PAIRS)]
    return {
        "chunks": chunks,
        "avg_pairs": avg_pairs,
        "worst_pairs": worst_pairs,
        "parity_pairs": parity_pairs,
        "chunk_tokens": sum(len(etok(c)["input_ids"]) for c in chunks),
    }


def bench_device(device: str, wl_seed: int, runs: int, calls: int) -> dict:
    import torch

    t0 = time.perf_counter()
    emb, rr = load_models(device)
    load_s = time.perf_counter() - t0
    wl = make_workload(emb, rr, wl_seed)
    docs = [EMBEDDER_PROMPTS["doc_prompt"] + c for c in wl["chunks"]]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    emb.encode(docs[:8], normalize_embeddings=True)  # warm-up
    t = time.perf_counter()
    vectors = emb.encode(docs, batch_size=32, normalize_embeddings=True)
    embed_s = time.perf_counter() - t
    q_ms = [
        timed_ms(lambda: emb.encode([EMBEDDER_PROMPTS["query_prompt"] + "meeting budget decision"]))
        for _ in range(20)
    ]

    rr.predict(wl["avg_pairs"][:4])  # warm-up
    rerank = {}
    for key in ("avg", "worst"):
        pairs = wl[f"{key}_pairs"]
        per_run = [summarize_ms([rerank_ms(rr, pairs) for _ in range(calls)]) for _ in range(runs)]
        rerank[key] = {
            "per_run": per_run,
            "p50": float(np.median([r["p50"] for r in per_run])),
            "p95": float(max(r["p95"] for r in per_run)),  # worst run: conservative
            "runs": runs,
            "calls_per_run": calls,
        }
    logits = [float(x) for x in rr.predict(wl["parity_pairs"], batch_size=25)]

    out = {
        "device": device,
        "load_s": load_s,
        "embed": {
            "chunks": len(docs),
            "tokens": wl["chunk_tokens"],
            "seconds": embed_s,
            "tokens_per_s": wl["chunk_tokens"] / embed_s,
            "query_ms": summarize_ms(q_ms),
        },
        "rerank_30_pairs_ms": rerank,
        "peak_rss_mib": peak_rss_mib(),
        "_vectors": vectors,
        "_logits": logits,
    }
    if device == "cuda":
        out["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def host_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "sqlite": sqlite3.sqlite_version,
    }
    try:
        import sentence_transformers
        import torch

        info |= {
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        }
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info |= {"gpu": p.name, "gpu_mib": p.total_memory // 2**20, "cuda": torch.version.cuda}
    except ImportError:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="JSON record (default bench/results/)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--calls", type=int, default=50, help="calls per run on the GPU")
    ap.add_argument("--cpu-calls", type=int, default=5, help="calls per run on the CPU (slow)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch

    record: dict = {
        "kind": "p1-spike",
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": host_info(),
        "models": {"embedder": EMBEDDER.__dict__, "reranker": RERANKER.__dict__},
        "sqlite_vec": doctor.check_sqlite_vec().__dict__,
    }
    missing = [c for c in doctor.check_models() if c.status != "ok"]
    if missing or record["sqlite_vec"]["status"] != "ok":
        record["error"] = [c.__dict__ for c in missing] or [record["sqlite_vec"]]
        print(json.dumps(record, indent=2))
        return 2

    results = {"cpu": bench_device("cpu", args.seed, args.runs, args.cpu_calls)}
    if torch.cuda.is_available():
        results["cuda"] = bench_device("cuda", args.seed, args.runs, args.calls)

    parity = None
    if "cuda" in results:
        cpu, gpu = results["cpu"], results["cuda"]
        parity = {
            "embedding": cosine_parity(cpu["_vectors"], gpu["_vectors"]),
            "rerank": logit_parity(cpu["_logits"], gpu["_logits"]),
            "chunks": N_PARITY_CHUNKS,
            "pairs": N_PARITY_PAIRS,
        }
    for r in results.values():
        r.pop("_vectors")
        r.pop("_logits")
    record["devices"] = results
    record["parity"] = parity
    gpu_rr = results["cuda"]["rerank_30_pairs_ms"] if "cuda" in results else None
    record["verdict"] = verdict(
        gpu_rr,
        parity["embedding"]["ok"] if parity else None,
        parity["rerank"]["ok"] if parity else None,
    )

    out = args.out or Path("bench/results") / f"spike-{record['at'].replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({"verdict": record["verdict"], "parity": parity, "written": str(out)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
