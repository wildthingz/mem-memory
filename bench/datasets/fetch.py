"""`bench fetch`: download the LongMemEval files and pin their sha256 (benchmarks-v2 §1).

The S file is pinned by its full sha256 and size (benchmarks-v2 §1; the spec's earlier `…c442`
suffix was a typo for `…a442`). Every fetched digest is written into `checksums.sha256`, which is
committed after the first fetch; a later fetch must reproduce the committed digest exactly.
"""

from __future__ import annotations

from pathlib import Path

from bench.datasets.lme import sha256_file

REPO = "xiaowu0162/longmemeval-cleaned"
FILES = {"s": "longmemeval_s_cleaned.json", "m": "longmemeval_m_cleaned.json"}
SPEC_S_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
SPEC_S_BYTES = 277_383_467
CHECKSUMS = Path(__file__).with_name("checksums.sha256")


def read_checksums(path: Path = CHECKSUMS) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = (ln.split(maxsplit=1) for ln in path.read_text().splitlines() if ln.strip())
    return {name.strip(): digest for digest, name in pairs}


def verify(name: str, path: Path, checksums: Path = CHECKSUMS) -> str:
    """Return the file's sha256 after checking it against the spec and the committed pin."""
    if name == FILES["s"] and (size := path.stat().st_size) != SPEC_S_BYTES:
        raise ValueError(f"{name}: {size} bytes, the spec's file has {SPEC_S_BYTES} bytes")
    digest = sha256_file(path)
    if name == FILES["s"] and digest != SPEC_S_SHA256:
        raise ValueError(f"{name}: sha256 {digest} != the spec's {SPEC_S_SHA256}")
    pinned = read_checksums(checksums).get(name)
    if pinned and pinned != digest:
        raise ValueError(f"{name}: sha256 {digest} != pinned {pinned}")
    return digest


def fetch(which: list[str], dest: Path, checksums: Path = CHECKSUMS) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for key in which:
        name = FILES[key]
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=dest))
        out[name] = verify(name, path, checksums)
    known = read_checksums(checksums) | out
    checksums.write_text("".join(f"{d}  {n}\n" for n, d in sorted(known.items())), encoding="utf-8")
    return out
