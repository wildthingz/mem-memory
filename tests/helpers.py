"""Shared test helpers."""

import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

import uvicorn

from mem import config
from mem.gate import Gate, write_gate
from mem.serve import build_app, open_inprocess, uvicorn_config
from mem.store import Store
from mem.testing import HashEmbedder

STUB = config.stub_config()


def gate_stub_store(path: Path, cfg: config.Config = STUB, scope: str = "personal",
                    **gate: object) -> None:  # fmt: skip
    """Create (or open) a store for the stub models with a gate `mem serve --embedder hash`
    accepts: p = σ(z), keep p ≥ .5, no gap rule (OverlapReranker: full overlap is p .95)."""
    s = Store.open(path, HashEmbedder(), scope=scope, fp=cfg.fp_embed())
    fields = {"platt_a": 1.0, "platt_b": 0.0, "tau_keep": 0.5, "tau_high": 0.5, "gap": 0.0,
              "fingerprint_hash": cfg.fp_all(), **gate}  # fmt: skip
    write_gate(s.conn, Gate(**fields))  # type: ignore[arg-type]
    s.close()


def serve_thread(tmp_path: Path, access: object = None) -> tuple[str, Callable[[], object]]:
    """A real uvicorn `build_app` on 127.0.0.1 (the stores open in its thread, as in run_http)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uv: list[uvicorn.Server] = []

    def run():
        servers = {n: open_inprocess(tmp_path / f"{n}.db", scope=n, embedder=HashEmbedder(256))
                   for n in ("personal", "work")}  # fmt: skip
        uv.append(
            uvicorn.Server(
                uvicorn_config(
                    build_app(servers, access=access),
                    host="127.0.0.1",
                    port=port,
                    log_level="error",
                )
            )
        )
        uv[0].run()
        for s in servers.values():
            s.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for _ in range(100):
        if uv and uv[0].started:
            break
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", lambda: (setattr(uv[0], "should_exit", True), thread.join(5))
