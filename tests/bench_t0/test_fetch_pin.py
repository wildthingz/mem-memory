"""H1 (MEM2-75): `bench fetch` pins the full LongMemEval_S sha256 and size (benchmarks-v2 §1)."""

from __future__ import annotations

import hashlib

import pytest

from bench.datasets import fetch

# The real longmemeval_s_cleaned.json, as hashed on the host (review/astra-run.log:5724).
REAL_S_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
REAL_S_BYTES = 277_383_467


def test_pin_is_the_full_digest_of_the_real_file():
    assert fetch.SPEC_S_SHA256 == REAL_S_SHA256
    assert fetch.SPEC_S_BYTES == REAL_S_BYTES


def _pin(monkeypatch, data: bytes) -> None:
    monkeypatch.setattr(fetch, "SPEC_S_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(fetch, "SPEC_S_BYTES", len(data))


def test_file_with_the_pinned_digest_is_accepted(tmp_path, monkeypatch):
    f = tmp_path / fetch.FILES["s"]
    f.write_bytes(b"[]")
    _pin(monkeypatch, b"[]")
    assert fetch.verify(f.name, f, tmp_path / "checksums.sha256") == fetch.SPEC_S_SHA256


def test_file_with_another_digest_is_refused(tmp_path, monkeypatch):
    f = tmp_path / fetch.FILES["s"]
    f.write_bytes(b"[1]")  # same size as the pin, different digest
    _pin(monkeypatch, b"[2]")
    with pytest.raises(ValueError, match=fetch.SPEC_S_SHA256):
        fetch.verify(f.name, f, tmp_path / "checksums.sha256")


def test_file_with_another_size_is_refused_before_hashing(tmp_path, monkeypatch):
    f = tmp_path / fetch.FILES["s"]
    f.write_bytes(b"[1, 2]")
    _pin(monkeypatch, b"[2]")
    monkeypatch.setattr(fetch, "sha256_file", lambda _: pytest.fail("hashed a wrong-size file"))
    with pytest.raises(ValueError, match="bytes"):
        fetch.verify(f.name, f, tmp_path / "checksums.sha256")


def test_old_suffix_typo_is_gone():
    assert not fetch.SPEC_S_SHA256.endswith("c442")
