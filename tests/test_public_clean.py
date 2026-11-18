"""The public paths hold nothing personal (MEM2-137). The denylist is stored encoded so this file
does not match itself."""

import base64
from pathlib import Path

from mem.release import EXCLUDED, PUBLIC_PATHS, public_files

ROOT = Path(__file__).parents[1]

_DENY = [  # base64, matched case-insensitively
    "ZGFya3Zpc2lvbg==", "aGF0ZWY=", "a2hhZGl2aQ==", "L2hvbWUv",
    "UFBOZXVlTW9udHJlYWw=", "UFAgTmV1ZSBNb250cmVhbA==",
    "UFBFZGl0b3JpYWxOZXc=", "UFAgRWRpdG9yaWFsIE5ldw==",
]  # fmt: skip
_FONTS_DIR = base64.b64decode("YnVpbGQvZm9udHM=").lower()  # the private report build's fonts
DENY = [base64.b64decode(w).lower() for w in _DENY]


def _scan(rel: Path, data: bytes) -> list[str]:
    """Denied words in the file's path (a folder or file named after someone ships too) and in
    each line of its content. The one allowed hit: the copyright line of the root LICENSE."""
    words = (*DENY, _FONTS_DIR)
    path = rel.as_posix().lower().encode()
    hits = [f"{rel}: path: {w.decode()}" for w in words if w in path]
    for n, line in enumerate(data.splitlines(), 1):
        low = line.lower()
        if rel == Path("LICENSE") and low.startswith(b"copyright"):
            continue
        hits += [f"{rel}:{n}: {w.decode()}" for w in words if w in low]
    return hits


def test_public_paths_hold_nothing_personal():
    hits = [h for rel in public_files(ROOT) for h in _scan(rel, (ROOT / rel).read_bytes())]
    assert not hits, "personal content in public paths:\n" + "\n".join(hits)


def test_scan_flags_each_denied_word_and_the_fonts_dir():
    for word in (*DENY, _FONTS_DIR):
        assert _scan(Path("src/x.py"), b"# see " + word.upper() + b" here")


def test_scan_exempts_only_the_copyright_line_of_license():
    name = base64.b64decode(_DENY[1]).decode()
    assert not _scan(Path("LICENSE"), f"Copyright (c) 2026 {name.title()}".encode())
    assert _scan(Path("LICENSE"), f"Notice: {name}".encode())
    assert _scan(Path("README.md"), f"Copyright (c) 2026 {name}".encode())
    assert _scan(Path("bench/vendor/LICENSE"), f"Copyright (c) 2026 {name}".encode()), (
        "only the root LICENSE is exempt, not a LICENSE at any depth"
    )


def test_scan_flags_a_denied_word_in_the_path():
    name = base64.b64decode(_DENY[1]).decode()
    assert _scan(Path(f"docs/{name}-notes.md"), b"nothing personal\n") == [
        f"docs/{name}-notes.md: path: {name}"
    ]
    assert _scan(Path("src/x.py"), b"") == []


def test_excluded_paths_are_not_public(tmp_path):
    for rel in ("src/a.py", "deliverables/b.md", "BRIEF-v3.md", "bench/results/r.json",
                "bench/gold/personal/v2/q.tsv", "bench/gold/cli.py", "src/__pycache__/a.pyc",
                "data/x.db", "review/r.md", "research/n.md", "graphify-out/g.json", "uv.lock"):  # fmt: skip
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    got = {p.as_posix() for p in public_files(tmp_path)}
    assert got == {"src/a.py", "bench/gold/cli.py", "uv.lock"}


def test_ledger_ships_empty():
    ledger = ROOT / "bench" / "ledger.jsonl"
    assert not ledger.exists() or ledger.stat().st_size == 0


def test_public_path_list_names_real_things():
    assert "src" in PUBLIC_PATHS and "deliverables" in EXCLUDED
    assert not set(PUBLIC_PATHS) & set(EXCLUDED)
