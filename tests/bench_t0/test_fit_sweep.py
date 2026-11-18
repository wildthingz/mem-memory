"""T0: `bench syn-fit-theta` rule options and `bench syn-fit-sweep` (MEM2-156 round 3) on stub
records. Stub numbers are smoke only; the rules (target, nested CV) are what is checked."""

import json

from typer.testing import CliRunner

from bench.cli import app
from tests.test_conf_scorer import _records


def _write(tmp_path):
    p = tmp_path / "conflicts.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in _records()), encoding="utf-8")
    return p


def test_fit_theta_takes_the_rule_options_and_records_them(tmp_path):
    conf, out = _write(tmp_path), tmp_path / "theta.json"
    r = CliRunner().invoke(app, ["syn-fit-theta", str(conf), "--out", str(out), "--floor", "0.55",
                                 "--select-floor", "0.56", "--top-ks", "3", "--deltas", "3,4"])  # fmt: skip
    assert r.exit_code in (0, 1), r.output  # 1 = honest MISS on the stub, never a relaxed target
    fit = json.loads(out.read_text(encoding="utf-8"))
    assert fit["train_floor"] == 0.55 and fit["select_floor"] == 0.56 and fit["grid_size"] == 12
    assert {c["top_k"] for c in [f["chosen"] for f in fit["nested"]["outer_folds"]]} == {3}
    assert fit["cv"]["target"] == {"surfacing": 0.95, "precision": 0.5}
    assert "selection floor 0.56" in out.with_suffix(".cv.txt").read_text(encoding="utf-8")
    bad = CliRunner().invoke(app, ["syn-fit-theta", str(conf), "--top-ks", "7"])
    assert bad.exit_code == 2 and "top_k" in bad.output


def test_fit_sweep_writes_one_fit_per_variant_and_a_table(tmp_path):
    conf, out = _write(tmp_path), tmp_path / "sweep"
    r = CliRunner().invoke(app, ["syn-fit-sweep", str(conf), "--out-dir", str(out),
                                 "--variants", "base,k3"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "fit sweep" in r.output and "optimistic" in r.output and " base " in r.output
    for name in ("base", "k3"):
        fit = json.loads((out / f"theta_{name}.json").read_text(encoding="utf-8"))
        assert fit["cv"]["target"] == {"surfacing": 0.95, "precision": 0.5}
        assert (out / f"theta_{name}.cv.txt").exists()
    assert (out / "sweep.txt").read_text(encoding="utf-8").startswith("fit sweep")
    bad = CliRunner().invoke(app, ["syn-fit-sweep", str(conf), "--out-dir", str(out),
                                   "--variants", "nope"])  # fmt: skip
    assert bad.exit_code == 2 and "unknown variant" in bad.output
