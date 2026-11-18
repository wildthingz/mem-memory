"""`config.yaml` (architecture-v2 §3.3) and the embedder / overall fingerprints.

`fp_embed` = sha256 of the canonical `embedder` section; `fp_all` = sha256 of the canonical
embedder + reranker + retrieval + read sections (`read.conflict_guard`, the read-time conflict guard
of ADR 0018, is a setting of the read path, so a gate fitted with it on is void with it off).
`ingest.resolve_dates` (ADR 0021) joins `fp_all` only when on, and so does `retrieval.session_leg`
(ADR 0022: the key is added to the `retrieval` section only when on). `ingest.new_checks_existing`
(ADR 0027 "Switch") is the reverse: on by default and then absent; written false, the key is in
`ingest` and the section joins `fp_all` whatever `resolve_dates` is.
Since MEM2-261 (Round 9's final config, F1 + F2+F3) `ingest.resolve_dates` and `read.list_mode` are on
by default. Since MEM2-274 the default is the host's final config, Round 10 `r10-all` (`fp_all` 3f8c9c11):
also `read.list_fill` 16, `read.window_boost` {delta 3.0, slack 0.1} and the Qwen3-Embedding-0.6B embedder;
its gate (with `list_model`) is the one fitted on it. `all_off_config` is the Round 8 default (Gemma
embedder, every switch off), whose fingerprint older stores carry.
`models.rerank_batch_size` joins `fp_all` when it is set (non-zero: a batch moves half-precision
scores, MEM2-222 F23); at 0, the default, `fp_all` is as it was. `gate` and `device` are outside both.
`models.embedder` / `models.reranker` pick a pinned entry of `mem.catalog` (ADR 0025).
Pinned model revisions live in `mem.pins`, which this module builds on.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mem import catalog, pins, window
from mem.pins import canonical_json

__all__ = [
    "Config",
    "ConfigError",
    "all_off_config",
    "canonical_json",
    "default_config",
    "load",
    "stub_config",
]

DEVICES = ("auto", "cpu", "cuda")
OPTIONAL = {
    "read": {"list_mode", "list_paging", "split_endpoints", "list_fill", "window_boost"},
    "retrieval": {"session_leg"},
}  # set to false (list_fill: 0), the key is dropped (off leaves fp_all as it was); only list_mode is in the default
# the reverse (ADR 0027 "Switch"): on by default, so set to true the key is dropped
OPTIONAL_ON = {"ingest": {"new_checks_existing"}}
FINGERPRINTED = ("embedder", "reranker", "retrieval", "read")
SECTIONS = (*FINGERPRINTED, "ingest", "gate", "security")


class ConfigError(ValueError):
    pass


def _reranker_section() -> dict[str, Any]:
    return catalog.RERANKERS["bge-reranker-v2-m3"].section()


def _retrieval_section() -> dict[str, Any]:
    return {
        "fts_tokenizer": "porter unicode61 remove_diacritics 2",
        "leg_k": 50,
        "rrf_k": 60,
        "pool": 30,
        "per_node_cap": 4,
        "limit": 8,
        "budget": 600,
    }


def _embedder_section() -> dict[str, Any]:
    """The default embedder (ADR 0025): Qwen3-Embedding-0.6B since MEM2-274 (Round 10 r10-all, Apache-2.0,
    not gated). EmbeddingGemma, the default before, is `models.embedder: embeddinggemma-300m`."""
    return catalog.EMBEDDERS[catalog.DEFAULT_EMBEDDER].section()


def _read_section() -> dict[str, Any]:
    """The read path's switches (architecture-v2 §3.2 step 7b, ADR 0018): `conflict_guard` hides,
    in recall's final set, the earlier of two same-subject memories the conflict model scores as a
    conflict at `theta_read`. Default off (the owner, 2026-10-25, ADR 0018: no cut reaches leakage 0 on SYN
    dev without hiding co-valid rows); when on, a P3 gate without `read_guard` is refused.
    `p3_model` (the owner, 2026-10-25, ADR 0020): p comes from the P3 read model (rerank + cosine +
    masked features) instead of the Platt logit on the rerank. Default off (B-E1: the P2 gate
    answers 78%, the P3 read model 31%); when on, a gate without a `read_model` is refused.
    `list_mode` (ADR 0023): on by default since MEM2-261 (F2+F3 adopted in Round 9, the owner 2026-10-28);
    tau_list comes from the gate's `list_model`, so a gate without one is refused.
    `list_fill` 16 (ADR 0030) and `window_boost` {delta 3.0, slack 0.1} (ADR 0031): on by default since
    MEM2-274 (Round 10 r10-all, the host's final config)."""
    return {
        "conflict_guard": False, "p3_model": False, "list_mode": True, "list_fill": 16,
        "window_boost": {"delta": 3.0, "slack": 0.1},
    }  # fmt: skip


def _ingest_section() -> dict[str, Any]:
    """`resolve_dates` (ADR 0021): relative time expressions are resolved at import and ingest into
    a derived annotation. On by default since MEM2-261 (F1 adopted in Round 9, the owner 2026-10-28). It
    changes what is indexed, so it is inside `fp_all` when on; off, the section is left out and the
    fingerprint is the one stores built without it carry."""
    return {"resolve_dates": True}


def _security_section() -> dict[str, Any]:
    """`at_rest_required`: `mem doctor` FAILS (not warns) on stores that are not on an encrypted
    volume. Default false: encryption at rest is deferred (owner's backlog; p7-backup.md). A policy
    switch, so outside `fp_all`."""
    return {"at_rest_required": False}


def _gate_section() -> dict[str, Any]:
    return {
        "platt_a": None, "platt_b": None, "tau_keep": None, "tau_high": None,
        "gap": None, "theta_conf": None, "fit_id": None,
    }  # fmt: skip


def _sha(*sections: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(list(sections)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Config:
    embedder: dict[str, Any]
    reranker: dict[str, Any]
    retrieval: dict[str, Any]
    gate: dict[str, Any] = field(default_factory=_gate_section)
    read: dict[str, Any] = field(default_factory=_read_section)
    security: dict[str, Any] = field(default_factory=_security_section)
    ingest: dict[str, Any] = field(default_factory=_ingest_section)
    device: str = "cuda"
    rerank_batch_size: int = (
        0  # pairs per reranker forward, 0 = the whole pool; in fp_all only when set
    )
    vault_routes: dict[str, str] = field(default_factory=dict)  # folder -> store (ADR 0010)

    def fp_embed(self) -> str:
        return pins.fp_embed(self.embedder)

    def fp_all(self) -> str:
        sections = [self.embedder, self.reranker, self.retrieval, self.read]
        batch = [{"rerank_batch_size": self.rerank_batch_size}] if self.rerank_batch_size else []
        ingest = [self.ingest] if self.resolve_dates or not self.new_checks_existing else []
        return _sha(*sections, *ingest, *batch)

    @property
    def session_leg(self) -> bool:
        """`retrieval.session_leg` (ADR 0022). Absent from the default section, so off is the
        fingerprint stores already carry; on, the key is in `retrieval` and so in `fp_all`."""
        return bool(self.retrieval.get("session_leg", False))

    @property
    def resolve_dates(self) -> bool:
        return bool(self.ingest.get("resolve_dates", False))

    @property
    def new_checks_existing(self) -> bool:
        """`ingest.new_checks_existing` (ADR 0027 "Switch", MEM2-272). On by default and then absent
        (written true, the key is dropped), so the default `fp_all` is as it was; written false, it is in
        `ingest` and the section joins `fp_all` even with `resolve_dates` off."""
        return bool(self.ingest.get("new_checks_existing", True))

    @property
    def conflict_guard(self) -> bool:
        return bool(self.read.get("conflict_guard", False))

    @property
    def list_mode(self) -> bool:
        """`read.list_mode` (ADR 0023). On by default (MEM2-261); off drops the key, as before."""
        return bool(self.read.get("list_mode", False))

    @property
    def list_paging(self) -> bool:
        """`read.list_paging` (MEM2-212): list mode pages every gated row behind a continuation token."""
        return bool(self.read.get("list_paging", False))

    @property
    def split_endpoints(self) -> bool:
        """`read.split_endpoints` (ADR 0026). Not in the default section, so off leaves `fp_all` as it was."""
        return bool(self.read.get("split_endpoints", False))

    @property
    def list_fill(self) -> int:
        """`read.list_fill` (MEM2-268, ADR 0030): in list mode, after the `tau_list` abstention test,
        deliver the top N rows by p (N = this value) instead of stopping at the cut. 0 (the default) is
        off: the key is absent and `fp_all` is as it was; set, it joins the fingerprint."""
        return int(self.read.get("list_fill", 0) or 0)

    @property
    def window_boost(self) -> dict[str, float] | None:
        """`read.window_boost` (ADR 0031) as `{delta, slack}`, or None when off. Not in the default
        section, so off leaves `fp_all` as it was; on, the values are in it."""
        value = self.read.get("window_boost", False)
        return None if value is False else window.settings(value)

    @property
    def p3_model(self) -> bool:
        return bool(self.read.get("p3_model", False))

    @property
    def at_rest_required(self) -> bool:
        return bool(self.security.get("at_rest_required", False))

    def vec_table(self) -> str:
        return f"vec_{self.fp_embed()[:8]}"

    def replace(self, section: str, **fields: Any) -> Config:
        if section not in SECTIONS:
            raise ConfigError(f"unknown section {section!r}")
        merged = {**getattr(self, section), **fields}
        # an OPTIONAL switch set to false is its default: absent, so fp_all stays as it was
        # (ADR 0022 / 0023: the key joins the fingerprint only when on)
        for name in OPTIONAL.get(section, ()):
            if merged.get(name) is False or (name == "list_fill" and merged.get(name) == 0):
                del merged[name]
        for name in OPTIONAL_ON.get(section, ()):
            if merged.get(name) is True:
                del merged[name]
        return Config(**{**self.__dict__, section: merged})


def default_config() -> Config:
    return Config(
        embedder=_embedder_section(),
        reranker=_reranker_section(),
        retrieval=_retrieval_section(),
        read=_read_section(),
        ingest=_ingest_section(),
    )


def all_off_config() -> Config:
    """The default config with every switch off and the EmbeddingGemma embedder: the Round 8 default
    (MEM2-183 to MEM2-260), whose `fp_all` older stores and gates carry."""
    base = default_config().replace("ingest", resolve_dates=False)
    base = base.replace("read", list_mode=False, list_fill=0, window_boost=False)
    return Config(**{**base.__dict__, "embedder": pins.embedder_section()})


def stub_config(base: Config | None = None) -> Config:
    """The deterministic CPU stand-ins (`mem.testing`: HashEmbedder 256-d, OverlapReranker) as a
    config with its own fingerprints, so stub stores get their own vec table and gate (§3.3)."""
    base = base or default_config()
    embedder = {**pins.embedder_section(), "id": "HashEmbed-256", "revision": "stub", "dim": 256,
                "backend": "stub"}  # fmt: skip
    reranker = {**_reranker_section(), "id": "OverlapReranker", "revision": "stub",
                "backend": "stub"}  # fmt: skip
    return Config(embedder=embedder, reranker=reranker, retrieval=base.retrieval, gate=base.gate,
                  read=base.read, ingest=base.ingest, device=base.device, vault_routes=base.vault_routes)  # fmt: skip


def _vault_routes(section: Any) -> dict[str, str]:
    unknown = set(section or {}) - {"routes"}
    if unknown:
        raise ConfigError(f"unknown field(s) in vault: {sorted(unknown)}")
    routes = (section or {}).get("routes") or {}
    if not isinstance(routes, dict):
        raise ConfigError("vault.routes must map a folder to personal or work")
    for folder, store in routes.items():
        if store not in ("personal", "work"):
            raise ConfigError(f"vault.routes[{folder!r}] must be personal or work, got {store!r}")
    return {str(folder).strip("/"): store for folder, store in routes.items()}


MODELS_KEYS = ("embedder", "reranker", "rerank_batch_size")


def _models(cfg: Config, section: Any) -> Config:
    """`models:` (ADR 0025): name a pinned embedder and/or reranker; their sections replace the
    default ones (so the fingerprints follow), `rerank_batch_size` is the pairs per forward."""
    if not isinstance(section, dict):
        raise ConfigError("models must be a mapping")
    if unknown := set(section) - set(MODELS_KEYS):
        raise ConfigError(f"unknown field(s) in models: {sorted(unknown)}")
    fields: dict[str, Any] = {}
    try:
        for kind in ("embedder", "reranker"):
            if kind in section:
                fields[kind] = catalog.lookup(kind, str(section[kind])).section()
    except ValueError as err:
        raise ConfigError(str(err)) from err
    batch = section.get("rerank_batch_size")
    if batch is None and "reranker" in section:
        batch = catalog.lookup("reranker", str(section["reranker"])).default_batch
    if batch is not None:
        if (
            isinstance(batch, bool)
            or not isinstance(batch, int)
            or batch < 0
            or (batch == 0 and "rerank_batch_size" in section)
        ):
            raise ConfigError("models.rerank_batch_size must be a positive integer")
        fields["rerank_batch_size"] = batch
    return Config(**{**cfg.__dict__, **fields})


def load(path: str | Path) -> Config:
    """Defaults overlaid with the YAML file; unknown keys are an error, never ignored."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = default_config()
    if "models" in raw:  # first: the other sections overlay the selected models
        cfg = _models(cfg, raw["models"])
    for key, value in copy.deepcopy(raw).items():
        if key == "models":
            continue
        if key == "device":
            if value not in DEVICES:
                raise ConfigError(f"device must be one of {DEVICES}, got {value!r}")
            cfg = Config(**{**cfg.__dict__, "device": value})
        elif key == "vault":
            cfg = Config(**{**cfg.__dict__, "vault_routes": _vault_routes(value)})
        elif key in SECTIONS:
            unknown = (
                set(value)
                - set(getattr(cfg, key))
                - OPTIONAL.get(key, set())
                - OPTIONAL_ON.get(key, set())
            )
            if unknown:
                raise ConfigError(f"unknown field(s) in {key}: {sorted(unknown)}")
            for sec, name in (("read", "conflict_guard"), ("read", "p3_model"), ("read", "list_mode"), ("read", "list_paging"), ("read", "split_endpoints"),
                              ("security", "at_rest_required"), ("ingest", "resolve_dates"), ("ingest", "new_checks_existing"), ("retrieval", "session_leg")):  # fmt: skip
                if key == sec and not isinstance(value.get(name, True), bool):
                    raise ConfigError(f"{sec}.{name} must be true or false")
            fill = value.get("list_fill", 0) if key == "read" else 0
            if isinstance(fill, bool) or not isinstance(fill, int) or fill < 0:
                raise ConfigError("read.list_fill must be a non-negative integer (0 = off)")
            if key == "read" and value.get("window_boost", False) is not False:
                try:  # `true` is written out as the tuned values, so fp_all names them
                    value["window_boost"] = window.settings(value["window_boost"])
                except ValueError as err:
                    raise ConfigError(str(err)) from err
            cfg = cfg.replace(key, **value)
            if cfg.list_paging and not cfg.list_mode:
                raise ConfigError("read.list_paging needs read.list_mode")
            if cfg.list_fill and not cfg.list_mode:
                raise ConfigError("read.list_fill needs read.list_mode")
        else:
            raise ConfigError(f"unknown config key {key!r}")
    return cfg
