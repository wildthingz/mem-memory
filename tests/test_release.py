"""`mem release export`: the public copy is one clean commit made from the committed tree (MEM2-139).

Every test builds a throwaway git repo that looks like mem's, with private paths committed next to
public ones, so a wrong copy shows up as a path that should not be there. The denylist words are
decoded at run time so this file does not match its own scan."""

import base64
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mem import release
from mem.cli import app
from mem.common import MemError

ROOT = Path(__file__).parents[1]
HIT = base64.b64decode("aGF0ZWY=").decode()  # one denied word (the owner's first name)
PUBLIC = {
    "src/mem/__init__.py": "",
    "src/mem/a.py": "x = 1\n",
    "bench/b.py": "y = 2\n",
    "bench/gold/cli.py": "z = 3\n",
    "tests/test_a.py": "def test_a(): pass\n",
    "skills/mem/SKILL.md": "# skill\n",
    "deploy/mem.service": "[Unit]\n",
    "docs/guide.md": "# guide\n",
    "docs/decisions/0001-x.md": "# adr\n",
    "pyproject.toml": "[project]\nname='mem'\n",
    "uv.lock": "# lock\n",
    "LICENSE": "MIT License\n\nCopyright (c) 2026 Someone\n",
    "README.md": "# mem\n",
    "THIRD_PARTY.md": "# third party\n",
    ".gitignore": "__pycache__/\n",
    ".github/workflows/ci.yml": "name: ci\n",
    ".claude-plugin/plugin.json": "{}\n",
    ".mcp.json": "{}\n",
}
PRIVATE = {
    "BRIEF.md": "brief\n",
    "BRIEF-v3.md": "brief\n",
    "deliverables/report.md": "report\n",
    "review/r.md": "review\n",
    "research/n.md": "notes\n",
    "data/x.db": "db\n",
    "bench/results/r.json": "{}\n",
    "bench/gold/personal/v2/q.tsv": "q\n",
}
LEDGER = "bench/ledger.jsonl"


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return done.stdout


def make_repo(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    repo = tmp_path / "private"
    files = {**PUBLIC, **PRIVATE, LEDGER: '{"look": 1}\n', **(extra or {})}
    for rel, text in files.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    # the two files the export runs in the target: the real scan and the list it reads
    for name in ("release.py", "common.py"):
        (repo / "src/mem" / name).write_text((ROOT / "src/mem" / name).read_text())
    shutil.copy(ROOT / "tests/test_public_clean.py", repo / "tests/test_public_clean.py")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-f", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "private history")
    return repo


def tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
            and ".git" not in p.relative_to(root).parts}  # fmt: skip


def test_export_makes_exactly_one_commit_and_nothing_else(tmp_path):
    repo = make_repo(tmp_path)
    target = tmp_path / "public"
    release.export(target, repo)
    log = git(target, "log", "--format=%an|%ae|%s").splitlines()
    assert log == ["mem|mem@localhost|Initial public release"]
    assert git(target, "status", "--porcelain") == ""
    assert git(target, "remote") == ""
    assert git(target, "branch", "--show-current").strip() == "main"
    assert set(git(target, "ls-files").splitlines()) == tree(target)


def test_export_ships_the_public_paths_and_extras_only(tmp_path):
    repo = make_repo(tmp_path)
    target = tmp_path / "public"
    release.export(target, repo)
    shipped = tree(target)
    wanted = {p for p in PUBLIC} | {
        "src/mem/release.py",
        "src/mem/common.py",
        "tests/test_public_clean.py",
        LEDGER,
    }
    assert shipped == wanted
    for rel in PRIVATE:
        assert rel not in shipped
    assert not (target / "bench/results").exists()


def test_export_ships_an_empty_ledger(tmp_path):
    repo = make_repo(tmp_path)
    target = tmp_path / "public"
    release.export(target, repo)
    assert (target / LEDGER).read_bytes() == b""
    assert (repo / LEDGER).read_text() != ""  # the source is untouched


def test_export_reads_the_committed_tree_not_the_working_tree(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "README.md").write_text(f"# edited, uncommitted by {HIT}\n")
    (repo / "src/mem/untracked.py").write_text("secret = 1\n")
    target = tmp_path / "public"
    release.export(target, repo)
    assert (target / "README.md").read_text() == "# mem\n"
    assert not (target / "src/mem/untracked.py").exists()


def test_export_refuses_a_non_empty_target(tmp_path):
    repo = make_repo(tmp_path)
    target = tmp_path / "public"
    target.mkdir()
    (target / "keep.txt").write_text("mine")
    with pytest.raises(MemError, match="not empty"):
        release.export(target, repo)
    assert tree(target) == {"keep.txt"}


def test_export_accepts_an_existing_empty_target(tmp_path):
    repo = make_repo(tmp_path)
    target = tmp_path / "public"
    target.mkdir()
    release.export(target, repo)
    assert (target / "README.md").exists()


def test_export_fails_on_a_planted_hit_and_leaves_no_copy(tmp_path):
    repo = make_repo(tmp_path, {"src/mem/leak.py": f"# written by {HIT}\n"})
    target = tmp_path / "public"
    with pytest.raises(MemError, match="public-clean"):
        release.export(target, repo)
    assert not target.exists() or not any(target.iterdir())


def test_export_fails_on_a_hit_in_an_extra_path(tmp_path):
    repo = make_repo(tmp_path, {"README.md": f"by {HIT}\n"})
    target = tmp_path / "public"
    with pytest.raises(MemError, match="public-clean"):
        release.export(target, repo)
    assert not target.exists() or not any(target.iterdir())


def test_export_ref_picks_the_commit(tmp_path):
    repo = make_repo(tmp_path)
    first = git(repo, "rev-parse", "HEAD").strip()
    (repo / "docs/guide.md").write_text("# newer\n")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "newer")
    target = tmp_path / "public"
    release.export(target, repo, ref=first)
    assert (target / "docs/guide.md").read_text() == "# guide\n"


def test_cli_export(tmp_path):
    repo = make_repo(tmp_path)
    target = tmp_path / "public"
    res = CliRunner().invoke(app, ["release", "export", str(target), "--repo", str(repo)])
    assert res.exit_code == 0, res.output
    assert "1 commit" in res.output
    assert git(target, "rev-list", "--count", "HEAD").strip() == "1"


def test_cli_export_reports_a_failure_without_a_traceback(tmp_path):
    repo = make_repo(tmp_path)
    target = tmp_path / "public"
    target.mkdir()
    (target / "keep.txt").write_text("mine")
    res = CliRunner().invoke(app, ["release", "export", str(target), "--repo", str(repo)])
    assert res.exit_code == 1
    assert "[FAIL]" in res.output and "not empty" in res.output
