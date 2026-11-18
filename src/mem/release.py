"""What ships in the public copy of mem: the one list the export command and the
public-clean test both read, and `mem release export`, which builds that copy (MEM2-139)."""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

from mem.common import MemError

REPO = Path(__file__).resolve().parents[2]

PUBLIC_PATHS = (
    "src", "bench", "tests", "scripts", "skills", "deploy", "docs", "pyproject.toml", "uv.lock", "LICENSE",
)  # fmt: skip

# Also shipped, and scanned like the rest, but written for the public copy and not in the list above.
EXTRA_PATHS = (
    "README.md", "AGENTS.md", "CLAUDE.md", "THIRD_PARTY.md", ".gitignore", ".github", ".claude-plugin",
    ".mcp.json",
)  # fmt: skip

# Never public, even inside a public path: working papers, deliverables, results, personal data.
EXCLUDED = (
    "BRIEF*.md", "deliverables", "review", "research", "graphify-out", "data", "bench/results",
    "bench/gold/personal", "__pycache__", ".venv", ".cache", "node_modules", "*.db", "*.db-*",
)  # fmt: skip


def _excluded(rel: Path) -> bool:
    text = rel.as_posix()
    for pattern in EXCLUDED:
        if "/" in pattern:
            if text == pattern or text.startswith(pattern + "/"):
                return True
        elif any(fnmatch(part, pattern) for part in rel.parts):
            return True
    return False


def public_files(root: Path) -> Iterator[Path]:
    """Every file under `root` that ships, as a path relative to `root`, sorted."""
    out: list[Path] = []
    for name in (*PUBLIC_PATHS, *EXTRA_PATHS):
        top = root / name
        found = [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file())
        out += [p.relative_to(root) for p in found if not _excluded(p.relative_to(root))]
    yield from sorted(out)


COMMIT_MESSAGE = "Initial public release"
IDENTITY = ("-c", "user.name=mem", "-c", "user.email=mem@localhost")


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if done.returncode:
        raise MemError(f"{' '.join(cmd[:3])} failed: {(done.stdout + done.stderr).strip()}")
    return done.stdout


def _empty_dir(target: Path) -> bool:
    return not target.exists() or (target.is_dir() and not any(target.iterdir()))


def _extract_head(repo: Path, ref: str, target: Path) -> None:
    """`git archive` of `ref`, only the shipped top-level names that exist in it: the committed
    tree, never the working tree."""
    present = set(_run(["git", "-C", str(repo), "ls-tree", "--name-only", ref]).splitlines())
    names = [n for n in (*PUBLIC_PATHS, *EXTRA_PATHS) if n in present]
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "head.tar"
        _run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tar), ref, *names])
        with tarfile.open(tar) as archive:
            archive.extractall(target, filter="data")


def _prune(target: Path) -> None:
    """Drop what is excluded even inside a public path; the ledger ships empty."""
    for full in sorted(p for p in target.rglob("*") if p.is_file()):
        if _excluded(full.relative_to(target)):
            full.unlink()
    results = target / "bench" / "results"
    if results.exists():
        shutil.rmtree(results)
    ledger = target / "bench" / "ledger.jsonl"
    if ledger.exists():
        ledger.write_bytes(b"")


def _public_clean(target: Path) -> None:
    """The public-clean test, run inside the copy against the copy's own list."""
    test = target / "tests" / "test_public_clean.py"
    if not test.exists():
        raise MemError("public-clean: tests/test_public_clean.py is not in the copy")
    env = {**os.environ, "PYTHONPATH": str(target / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(target),
         str(test)], cwd=target, env=env, capture_output=True, text=True, check=False)  # fmt: skip
    if done.returncode:
        raise MemError(
            "public-clean test failed in the copy:\n" + (done.stdout + done.stderr)[-2000:]
        )


def _clear(target: Path) -> None:
    for child in target.iterdir():
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()


def export(target: Path, repo: Path = REPO, ref: str = "HEAD") -> int:
    """Build the public copy of `repo` at `ref` in `target`: one `git init`, one commit, no remote.
    Refuses a non-empty target. On any failure the target is emptied again. Returns the number of
    files committed."""
    if not _empty_dir(target):
        raise MemError(f"{target} is not empty: export into a new or empty folder")
    created = not target.exists()
    target.mkdir(parents=True, exist_ok=True)
    try:
        _extract_head(repo, ref, target)
        _prune(target)
        _public_clean(target)
        for stray in (*target.rglob("__pycache__"), *target.rglob(".pytest_cache")):
            shutil.rmtree(stray, ignore_errors=True)
        _run(["git", "init", "-q", "-b", "main"], cwd=target)
        _run(["git", "add", "-A"], cwd=target)
        _run(["git", *IDENTITY, "commit", "-q", "-m", COMMIT_MESSAGE], cwd=target)
        return len(_run(["git", "ls-files"], cwd=target).splitlines())
    except BaseException:
        if created:
            shutil.rmtree(target, ignore_errors=True)
        else:
            _clear(target)
        raise
