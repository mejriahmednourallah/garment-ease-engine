"""Tests for the candidate EaseLab evaluation script."""
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]


def _run(out):
    subprocess.run(
        [sys.executable, str(HERE / "evaluate_candidate_easelab.py"), "--out", str(out)],
        check=True, cwd=str(HERE), capture_output=True, text=True,
    )


def test_candidate_runs_and_cohort_counts(tmp_path):
    out = tmp_path / "cand.json"
    _run(out)
    d = json.loads(out.read_text())
    assert d["n_pairs"] == 603
    assert d["n_pairs_with_preferred_size"] == 369
    assert d["n_pairs_none"] == 234


def test_candidate_beats_baselines(tmp_path):
    out = tmp_path / "cand.json"
    _run(out)
    d = json.loads(out.read_text())
    m = d["metrics"]
    assert m["candidate"]["top1_all_pairs_count"] > m["real_scorer"]["top1_all_pairs_count"]
    assert m["candidate"]["top1_all_pairs_count"] > m["majority"]["top1_all_pairs_count"]
    assert m["candidate"]["top1_all_pairs_count"] > 234


def test_reproducibility(tmp_path):
    o1 = tmp_path / "a.json"
    o2 = tmp_path / "b.json"
    _run(o1)
    _run(o2)
    d1 = json.loads(o1.read_text())
    d2 = json.loads(o2.read_text())
    # metrics and hyperparameters identical
    assert d1["metrics"] == d2["metrics"]
    assert d1["hyperparameters"] == d2["hyperparameters"]
    assert d1["abstain_gate"] == d2["abstain_gate"]
    p1 = o1.with_name(o1.stem + "_predictions.csv")
    p2 = o2.with_name(o2.stem + "_predictions.csv")
    assert p1.read_text() == p2.read_text()


def test_leakage_feature_guard():
    import fit_experiments as fe
    for col in fe.FEATURE_SOURCE_COLUMNS:
        assert not fe.is_forbidden_feature(col), col
    # candidate's derived score column must be derivable from visible fields only
    assert "preferred" not in "easelab_minscore"
    assert "latent" not in "easelab_minscore"


def test_predictions_aligned(tmp_path):
    out = tmp_path / "cand.json"
    _run(out)
    pred = out.with_name(out.stem + "_predictions.csv")
    import csv
    rows = list(csv.DictReader(pred.open()))
    assert len(rows) == 603
    assert set(rows[0].keys()) >= {"pair_id", "truth", "real_scorer_pred", "candidate_pred"}
    # every truth is either NONE or a size label
    for r in rows:
        assert r["truth"] in {"NONE", "XXS", "XS", "S", "M", "L", "XL", "XXL", "3XL"}