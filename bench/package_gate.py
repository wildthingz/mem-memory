"""`bench package-gate`: put a host-fitted serving gate where the wheel carries it (ADR 0017).

The gate is fitted on public data only: the LongMemEval tune split (`mem calibrate --labels
--records-out`) and the SYN dev seeds (θ_conf). This refuses a gate whose LME fit saw any question
of the frozen test split, whose records are not the ones it was fitted on, or whose SYN seeds are
held out; nothing is written then.

The LME split is the frozen manifest (`bench/datasets/lme_split.tsv`) when it is there. That file
is generated (`bench build-lme`) and not committed, so when it is absent the split is derived from
the pinned LME file by the same rule (`lme.build_manifest`); the result is identical, as
`lme.verify_manifest` asserts wherever the frozen file is used (MEM2-277).
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.datasets import fetch, lme
from bench.synth import data as syn_data
from mem import calibrate, shipped
from mem.common import MemError
from mem.gate import Gate


def split_of(manifest: Path, lme_file: Path) -> tuple[dict[str, str], str]:
    """Question id -> split, and where it came from: the frozen manifest, else derived from the
    pinned LME file by the rule that froze it."""
    if manifest.exists():
        rows = lme.read_manifest(manifest)
        source = f"{manifest} (frozen)"
    else:
        rows = lme.build_manifest(lme.load_questions(lme_file))
        source = f"derived from {lme_file} ({manifest} is absent; `bench build-lme` writes it)"
    return {r.question_id: r.split for r in rows}, source


def _fit_ids(records: Path, gate: Gate, split: dict[str, str]) -> int:
    """Check `records` are this gate's tune fit and contain no test question; return their count."""
    fp_all, pools = calibrate.read_records(records)
    if fp_all != gate.fingerprint_hash:
        raise MemError(f"{records}: scored with fp_all {fp_all[:8]}, the gate is for "
                       f"{gate.fingerprint_hash[:8]}")  # fmt: skip
    test = sorted(p.qid for p in pools if split.get(p.qid) == "test")
    if test:
        raise MemError(
            f"the gate was fitted on LME test questions ({len(test)}: "
            f"{', '.join(test[:5])}): a gate fitted on the test split is never shipped"
        )
    fitted = gate.tune.get("queries")
    if fitted != len(pools):
        raise MemError(
            f"{records} holds {len(pools)} queries, the gate was fitted on {fitted}: "
            "these records are not its fit; pass the --records-out of that mem calibrate"
        )
    return len(pools)


def package(
    gate_file: Path, *, records: Path, lme_file: Path, manifest: Path, dest: Path,
    syn_seeds: list[int], lme_version: str,
) -> dict:  # fmt: skip
    gate = Gate(**json.loads(gate_file.read_text(encoding="utf-8")))
    sha = fetch.verify(lme_file.name, lme_file)
    split, split_source = split_of(manifest, lme_file)
    n = _fit_ids(records, gate, split)
    meta = shipped.metadata(
        gate,
        lme={"name": "LongMemEval", "version": lme_version, "sha256": sha, "split": "tune"},
        syn={"name": "SYN", "version": syn_data.SYN_VERSION, "seeds": syn_seeds},
    )  # fmt: skip
    shipped.write(dest, gate, meta)
    meta["tune_queries"] = n  # not in the file: for the command's report
    meta["split_source"] = split_source
    return meta
