#!/usr/bin/env python3
"""Seed-controlled synthetic ablation for EaseLab: scorer alone, abstain gate alone,
learned model alone, learned model + gate. Synthetic data only; no app changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ERS_SRC = (HERE / "evaluate_real_scorer.py").read_text().replace(
    "from app.main import rank_sizes  # noqa: E402", "def rank_sizes(*a, **k): return []")
_NS = {"__file__": str((HERE / "evaluate_real_scorer.py").resolve())}
exec(compile(ERS_SRC, "evaluate_real_scorer", "exec"), _NS)  # noqa: S102
complete_pairs, load = _NS["complete_pairs"], _NS["load"]

TARGETS = {"slim": (2, 1, 2), "regular": (8, 5, 8), "loose": (14, 12, 14)}
WEIGHTS = (0.4, 0.3, 0.3)


def app_rank(body, garments, pref, stretch):
    scores = []
    for g in garments:
        s = 0.0
        for value, target, weight in zip(("chest", "waist", "hip"), TARGETS[pref], WEIGHTS):
            required = body[value] + target - body[value] * stretch / 100.0
            s += weight * (g[value] - required) ** 2
        scores.append((s, g["size"]))
    scores.sort()
    return scores[0]


def pair_inputs(pair):
    pair_id, cid, gid, merged, c_row, g_row = pair
    body = {k: float(c_row[f"{k}_cm"]) for k in ("chest", "waist", "hip")}
    garments = [{"size": r.size_label, "chest": float(r.garment_chest_cm),
                 "waist": float(r.garment_waist_cm), "hip": float(r.garment_hip_cm)}
                for r in merged.itertuples()]
    return pair_id, body, garments, str(c_row.fit_preference), float(g_row.stretch_crosswise_pct)


def metrics(preds, truths):
    preds, truths = np.asarray(preds), np.asarray(truths)
    non = truths != "NONE"
    tp = int(((truths == "NONE") & (preds == "NONE")).sum())
    fp = int((non & (preds == "NONE")).sum())
    fn = int(((truths == "NONE") & (preds != "NONE")).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "total_top1": float((preds == truths).mean()),
        "conditional_top1": float((preds[non] == truths[non]).mean()) if non.any() else None,
        "coverage": float((preds != "NONE").mean()),
        "none_precision": prec, "none_recall": rec,
        "none_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "balanced_accuracy_binary": float(balanced_accuracy_score((truths == "NONE").astype(int),
                                                                  (preds == "NONE").astype(int))),
        "kappa": float(cohen_kappa_score(truths, preds)),
    }


def choose_threshold(scores, truths, scorer_preds):
    """Maximise balanced accuracy between NONE/not-NONE on train+val."""
    best = (None, -1.0)
    for t in np.unique(scores):
        preds = np.where(scores > t, "NONE", scorer_preds)
        m = metrics(preds, truths)
        h = m["balanced_accuracy_binary"]
        if h > best[1]:
            best = (float(t), h)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    customers, garments, sizes, observations, splits = load(args.data_dir)
    truth = observations.groupby("pair_id")["pair_preferred_size"].first().astype(str)
    parts = {p: complete_pairs(customers, garments, sizes, observations, splits, p)
             for p in ("train", "val", "test")}
    rows = {}
    for part, pairs in parts.items():
        built = []
        for pair in pairs:
            pid, body, gl, pref, stretch = pair_inputs(pair)
            score, size = app_rank(body, gl, pref, stretch)
            built.append((pid, truth[pid], score, size))
        rows[part] = built
    tv = rows["train"] + rows["val"]
    threshold, hm = choose_threshold(np.array([r[2] for r in tv]), np.array([r[1] for r in tv]),
                                     np.array([r[3] for r in tv]))
    test = rows["test"]
    scores = np.array([r[2] for r in test])
    truths = np.array([r[1] for r in test])
    scorer = np.array([r[3] for r in test])
    majority = pd.Series([r[1] for r in tv]).value_counts().idxmax()
    preds = {
        "scorer": scorer,
        "scorer_abstain_gate": np.where(scores > threshold, "NONE", scorer),
        "fixed_M": np.full(len(test), "M"),
        "majority": np.full(len(test), majority),
    }
    result = {"data_dir": str(args.data_dir.resolve()), "threshold": threshold,
              "threshold_balanced_accuracy_trainval": hm, "majority_label": majority,
              "n_test_pairs": len(test), "n_none": int((truths == "NONE").sum()),
              "metrics": {k: metrics(v, truths) for k, v in preds.items()},
              "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(args.data_dir.glob("*.csv"))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
