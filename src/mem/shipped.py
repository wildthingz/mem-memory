"""The serving gate fitted on public data, shipped as package data (ADR 0017, easy-setup §2.1).

One file, `src/mem/shipped/serving_gate.json`, written by `bench package-gate` on the host that fitted
it: `{"gate": <Gate fields>, "metadata": {...}}`. `mem serve` uses it for a store that has no gate of
its own, and only when its fingerprint is the running config's; a local gate always wins and the
shipped one is never written into a store. A mismatch is refused as for any gate, and the message
names the `mem calibrate` command that fixes it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from mem.common import MemError
from mem.config import Config
from mem.gate import Gate

GATE_PATH = Path(__file__).parent / "shipped" / "serving_gate.json"

LME_LICENCE = {
    "name": "LongMemEval", "spdx": "MIT", "copyright": "Copyright (c) 2024 Di Wu",
    "source": "https://github.com/xiaowu0162/LongMemEval",
}  # fmt: skip
DEV_SEEDS = (1, 2, 3, 4, 5)  # SYN tune seeds; 101-105 are held out

_LME_KEYS = ("name", "version", "sha256", "split")
_SYN_KEYS = ("name", "version", "seeds")
_LICENCE_KEYS = ("name", "spdx", "copyright")


@dataclass(frozen=True)
class Shipped:
    gate: Gate
    metadata: dict[str, Any]


def _mem_version() -> str:
    try:
        return importlib_metadata.version("mem")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def metadata(
    gate: Gate, *, lme: dict[str, Any], syn: dict[str, Any],
    licences: list[dict[str, str]] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """The metadata block for `gate`: fingerprint, fit date, mem version, the datasets (LME tune
    version and sha256, SYN version and seeds) and their licences. Checked by `validate`."""
    meta = {
        "fingerprint": gate.fingerprint_hash,
        "fit_date": gate.fitted_at,
        "mem_version": _mem_version(),
        "datasets": {"lme": lme, "syn": syn},
        "licences": licences if licences is not None else [LME_LICENCE],
    }
    validate(meta, gate)
    return meta


def _missing(block: Any, keys: tuple[str, ...], where: str) -> list[str]:
    if not isinstance(block, dict):
        return [where]
    return [f"{where}.{k}" for k in keys if block.get(k) in (None, "", [])]


def validate(meta: dict[str, Any], gate: Gate) -> None:
    """Refuse an incomplete block, one that disagrees with the gate, or any dataset that is not tune."""
    datasets = meta.get("datasets") or {}
    missing = [k for k in ("fingerprint", "fit_date", "mem_version") if not meta.get(k)]
    missing += _missing(datasets.get("lme"), _LME_KEYS, "datasets.lme")
    missing += _missing(datasets.get("syn"), _SYN_KEYS, "datasets.syn")
    lics = meta.get("licences")
    if not isinstance(lics, list) or not lics:
        missing.append("licences")
    else:
        for i, lic in enumerate(lics):
            missing += _missing(lic, _LICENCE_KEYS, f"licences[{i}]")
    if missing:
        raise MemError(f"shipped gate metadata is incomplete, missing: {', '.join(missing)}")
    if meta["fingerprint"] != gate.fingerprint_hash:
        raise MemError(
            f"shipped gate metadata fingerprint {meta['fingerprint'][:8]} is not the gate's "
            f"{gate.fingerprint_hash[:8]}"
        )
    if datasets["lme"]["split"] != "tune":
        raise MemError(
            f"shipped gate: the LME split must be tune, not {datasets['lme']['split']!r}"
        )
    held_out = [s for s in datasets["syn"]["seeds"] if s not in DEV_SEEDS]
    if held_out:
        raise MemError(f"shipped gate: SYN seeds {held_out} are not dev seeds {list(DEV_SEEDS)}")


def write(path: Path, gate: Gate, meta: dict[str, Any]) -> None:
    validate(meta, gate)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"gate": asdict(gate), "metadata": meta}
    path.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")  # fmt: skip


def load(path: Path | None = None) -> Shipped | None:
    """The shipped gate, or None when this build has none. A file that exists but is unreadable or
    incomplete raises: serving must not fall back silently past a broken package."""
    path = path or GATE_PATH
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        gate = Gate(**doc["gate"])
        meta = doc["metadata"]
    except (OSError, ValueError, TypeError, KeyError) as err:
        raise MemError(f"{path}: not a shipped gate file ({type(err).__name__}: {err})") from err
    validate(meta, gate)
    return Shipped(gate, meta)


def gate_for(cfg: Config) -> Gate | None:
    """The shipped gate when it was fitted for `cfg`'s fingerprint, else None."""
    s = load()
    return s.gate if s is not None and s.gate.fingerprint_hash == cfg.fp_all() else None


def calibrate_fix(name: str, path: Path | str) -> str:
    """The command that fits a gate for the config in force (the one fix for any gate mismatch)."""
    return (f"mem calibrate --records <tune records> --config <config.yaml> "
            f"--store {name}={path}")  # fmt: skip


def mismatch(cfg: Config, s: Shipped) -> str:
    return (
        f"the shipped gate was fitted for fingerprint {s.gate.fingerprint_hash[:8]} (embedder + "
        f"reranker + retrieval of mem {s.metadata['mem_version']}), the config's is "
        f"{cfg.fp_all()[:8]}"
    )
