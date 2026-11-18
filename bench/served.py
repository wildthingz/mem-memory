"""The bench's own pinned `mem serve` (architecture-v2 §2.1 D5).

Only `mem serve` loads models. The bench starts its **own** instance, separate from the person's
long-lived one: `mem serve --http --models-only` as a subprocess, on its own loopback port, with
the bench's config (`--config`, else the built-in pinned defaults: the child gets an empty
`XDG_CONFIG_HOME`, so a `~/.config/mem/config.yaml` on the host never leaks in). The bench's
stores (one fresh store per LME question, the L5k/L50k load stores) stay bench-owned files; their
write and read paths run the production `mem.write` / `mem.read` code with the served models
(`mem.models_http.connect`: `/models/embed`, `/models/rerank`).

`PinnedServe` refuses a port that is already in use, waits until `/models` answers for the
config's models, and always stops the process (terminate, then kill) when the bench is done.
"""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Self

from bench.suites.base import BackendUnavailable

READY_TIMEOUT_S = 600.0  # loading both pinned models on CPU; the stand-ins take ~1 s
STOP_TIMEOUT_S = 10.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class PinnedServe:
    """`mem serve --http --models-only` for `cfg`, in a child process. Use as a context manager,
    or `start()` / `stop()`; `models` are (embedder, reranker) clients of the served models."""

    def __init__(
        self, cfg: Any, *, models: str = "st", config_path: Path | None = None,
        device: str | None = None, port: int | None = None,
        ready_timeout: float = READY_TIMEOUT_S,
    ) -> None:  # fmt: skip
        if models not in ("st", "hash"):
            raise ValueError(f"models must be st or hash, got {models!r}")
        self.cfg, self._models, self._config_path = cfg, models, config_path
        self._device, self._port, self._timeout = device, port, ready_timeout
        self.port: int | None = None
        self.models: tuple[Any, Any] | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._dir: Path | None = None

    def command(self, port: int) -> list[str]:
        cmd = [sys.executable, "-m", "mem.cli", "serve", "--http", "--models-only",
               "--host", "127.0.0.1", "--port", str(port), "--embedder", self._models]  # fmt: skip
        if self._models == "hash":
            cmd.append("--allow-cpu")  # stub models run on the CPU (T0)
        if self._config_path is not None:
            cmd += ["--config", str(Path(self._config_path).resolve())]
        if self._device is not None:
            cmd += ["--device", self._device]
        return cmd

    def start(self) -> Self:
        from mem import models_http

        if self._proc is not None:
            return self
        port = self._port if self._port is not None else free_port()
        if models_http.serve_listening(port):
            raise BackendUnavailable(f"127.0.0.1:{port} is in use: the bench starts its own mem "
                                     "serve and will not share one; pick a free --serve-port")  # fmt: skip
        self._dir = Path(tempfile.mkdtemp(prefix="bench-serve-"))
        (self._dir / "config").mkdir()
        env = {**os.environ, "XDG_CONFIG_HOME": str(self._dir / "config")}
        self._log = (self._dir / "serve.log").open("wb")
        self._proc = subprocess.Popen(self.command(port), env=env, stdin=subprocess.DEVNULL,
                                      stdout=self._log, stderr=subprocess.STDOUT)  # fmt: skip
        self.port = port
        atexit.register(self.stop)  # a bench that dies on an exception still stops its serve
        try:
            self.models = self._wait_ready(port)
        except BaseException:
            self.stop()
            raise
        return self

    def _wait_ready(self, port: int) -> tuple[Any, Any]:
        from mem import models_http

        assert self._proc is not None
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise BackendUnavailable(f"the bench's mem serve exited with code "
                                         f"{self._proc.returncode}: {self.log_tail()}")  # fmt: skip
            if models_http.serve_listening(port):
                try:
                    return models_http.connect(port, self.cfg)
                except models_http.ServeUnavailable as err:
                    raise BackendUnavailable(str(err)) from err
            time.sleep(0.1)
        raise BackendUnavailable(f"the bench's mem serve did not answer on 127.0.0.1:{port} within "
                                 f"{self._timeout:.0f} s: {self.log_tail()}")  # fmt: skip

    def log_tail(self, lines: int = 8) -> str:
        if self._dir is None or not (self._dir / "serve.log").exists():
            return "(no log)"
        self._log.flush()
        text = (self._dir / "serve.log").read_text(encoding="utf-8", errors="replace")
        return " | ".join(text.strip().splitlines()[-lines:]) or "(empty log)"

    def stop(self) -> None:
        proc, self._proc, self.models = self._proc, None, None
        atexit.unregister(self.stop)
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            self._log.close()
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
