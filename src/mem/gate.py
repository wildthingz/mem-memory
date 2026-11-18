"""The relevance gate (architecture-v2 §3.2 steps 5-6) and its home in `meta.gate` (§2.3, §3.3).

`p = σ(a·z + b)` maps the reranker's raw logit `z` to a probability with the store's Platt fit.
The gate is an invariant of the final set (A2): rows are the pool members with `p ≥ τ_keep`, and
the whole answer abstains when `p1 < τ_keep`, or when `p1 < τ_high` and `p1 − p2 < g`.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mem.confmodel import ConfModel
from mem.listmode import ListModel
from mem.readguard import ReadGuard
from mem.readmodel import ReadModel

NO_READ_MODEL = (
    "read.p3_model is on but the gate has no read_model (rerank, cosine, gap; MEM2-155): refit it"
    " (mem calibrate --objective p3, p3-gate.md) or set read.p3_model: false in config.yaml"
    " (the default, ADR 0020; a new fingerprint)"
)
REQUIRED = ("platt_a", "platt_b", "tau_keep", "tau_high", "gap")


@dataclass(frozen=True)
class Gate:
    platt_a: float
    platt_b: float
    tau_keep: float
    tau_high: float
    gap: float
    theta_conf: float | None = None  # write path, fitted from P3: cut on the conf_model score
    conf_model: dict[str, Any] | None = None  # coefficients of the conflict scorer (confmodel.py)
    read_model: dict[str, Any] | None = (
        None  # P3 (MEM2-155): p = σ(logistic over rerank, cosine, gap)
    )
    read_guard: dict[str, Any] | None = None  # P3 (ADR 0018): theta_read of the read-time guard
    list_model: dict[str, Any] | None = None  # MEM2-193 (ADR 0023): tau_list of the list mode
    fit_id: str = ""
    fitted_at: str = ""
    fingerprint_hash: str = ""  # fp_all of the config the fit belongs to
    tune: dict[str, Any] = field(default_factory=dict)  # achieved metrics on the tune split

    def __post_init__(self) -> None:
        if self.read_model is not None:
            ReadModel.from_dict(self.read_model)  # incomplete or tampered coefficients: refused
        if self.conf_model is not None:
            ConfModel.from_dict(self.conf_model)
        elif self.theta_conf is not None and self.theta_conf != -math.inf:
            raise ValueError(
                "gate has theta_conf but no conf_model coefficients: refused, there is no"
                " fallback scorer (refit: bench syn-fit-theta, bench gate-add-theta)"
            )
        if self.list_model is not None:
            lm = ListModel.from_dict(self.list_model)  # missing tau_list / tampered: refused
            if (lm.platt_a, lm.platt_b) != (self.platt_a, self.platt_b):
                raise ValueError("list_model was fitted against another Platt (a, b): refit"
                                 " tau_list (bench fit-tau-list)")  # fmt: skip
        if self.read_guard is not None:
            guard = ReadGuard.from_dict(self.read_guard)  # missing theta_read / tampered: refused
            model = self.model
            if model is None:
                raise ValueError("gate has a read_guard but no conf_model: the guard scores pairs"
                                 " with the conflict model (bench gate-add-theta first)")  # fmt: skip
            if guard.conf_model_fingerprint != model.fingerprint():
                raise ValueError(
                    "read_guard was fitted against another conf_model (fingerprint"
                    f" {guard.conf_model_fingerprint[:12]} ≠ {model.fingerprint()[:12]}): refit"
                    " theta_read (bench syn-fit-read-guard)"
                )

    @property
    def model(self) -> ConfModel | None:
        return None if self.conf_model is None else ConfModel.from_dict(self.conf_model)

    @property
    def lister(self) -> ListModel | None:
        return None if self.list_model is None else ListModel.from_dict(self.list_model)

    @property
    def guard(self) -> ReadGuard | None:
        return None if self.read_guard is None else ReadGuard.from_dict(self.read_guard)

    def needs_read_guard(self) -> bool:
        """A P3 gate (a conflict model in force) without the read-time cut: refused when
        `read.conflict_guard` is on. The dev export (theta_conf = -inf) has no scorer in force."""
        return (self.theta_conf is not None and self.theta_conf != -math.inf
                and self.read_guard is None)  # fmt: skip

    @property
    def reader(self) -> ReadModel | None:
        return None if self.read_model is None else ReadModel.from_dict(self.read_model)

    def probs(
        self,
        zs: Sequence[float],
        cosines: Sequence[float | None] | None,
        masked: Mapping[str, Sequence[float | None]] | None = None,
    ) -> list[float]:
        """p of each pool member: the read model when the gate has one (it needs the cosines),
        else the Platt logit of `z` alone (P2 gates)."""
        if self.read_model is None:
            return [self.p(z) for z in zs]
        return ReadModel.from_dict(self.read_model).probs(zs, cosines, masked)

    def p(self, z: float) -> float:
        if self.read_model is not None:
            raise ValueError("this gate has a read_model: p depends on more than z, use probs()")
        return self.platt(z)

    def platt(self, z: float) -> float:
        """σ(a·z + b): the P2 serving gate, whatever else the gate file carries."""
        x = self.platt_a * z + self.platt_b
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def abstains(self, p1: float, p2: float) -> bool:
        return p1 < self.tau_keep or (p1 < self.tau_high and p1 - p2 < self.gap)

    def select(self, ps: Sequence[float]) -> tuple[list[int], bool]:
        """Indices of the kept pool members, by p descending (ties: pool order), and whether the
        answer abstains. With one candidate p2 is 0; with none the answer abstains."""
        order = sorted(range(len(ps)), key=lambda i: (-ps[i], i))
        if not order:
            return [], True
        p1 = ps[order[0]]
        p2 = ps[order[1]] if len(order) > 1 else 0.0
        if self.abstains(p1, p2):
            return [], True
        return [i for i in order if ps[i] >= self.tau_keep], False


def read_gate(conn: sqlite3.Connection) -> Gate | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'gate'").fetchone()
    if row is None:
        return None
    data = json.loads(row[0])
    missing = [k for k in REQUIRED if data.get(k) is None]
    if missing:
        raise ValueError(
            f"meta.gate is incomplete (missing {', '.join(missing)}): run mem calibrate"
        )
    return Gate(**{k: v for k, v in data.items() if k in Gate.__dataclass_fields__})


def gate_json(gate: Gate) -> str:
    return json.dumps(asdict(gate), sort_keys=True, ensure_ascii=False)


def write_gate(conn: sqlite3.Connection, gate: Gate) -> None:
    """Upsert `meta.gate`. Inside the caller's transaction it is part of it; otherwise it commits."""
    outer = conn.in_transaction
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('gate', ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (gate_json(gate),),
    )
    if not outer and conn.in_transaction:
        conn.commit()
