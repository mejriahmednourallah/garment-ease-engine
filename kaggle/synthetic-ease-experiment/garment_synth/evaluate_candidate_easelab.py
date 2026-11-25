#!/usr/bin/env python3
"""Candidate improvement over the real EaseLab scorer on the strict complete-measurement cohort.

Isolated experiment only: reads synthetic CSVs, writes results under results/. Does not touch
app/main.py or any production behavior. Synthetic data is not evidence of real-world fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_experiments as fe  # noqa: E402

# Reuse the strict cohort selector from the baseline script (without importing app.main/fastapi).
ERS_SRC = (HERE / "evaluate_real_scorer.py").read_text()
ERS_SRC = ERS_SRC.replace("from app.main import rank_sizes  # noqa: E402", "def rank_sizes(*a, **k): return []")
_NS = {"__file__": str((HERE / "evaluate_real_scorer.py").resolve())}
exec(compile(ERS_SRC, "evaluate_real_scorer", "exec"), _NS)  # noqa: S102
load = _NS["load"]
complete_pairs = _NS["complete_pairs"]
body_valid = _NS["body_valid"]
garment_valid = _NS["garment_valid"]

DATA_DIR = HERE / "data" / "main"
OUT_DEFAULT = HERE / "results" / "candidate_easelab_synthetic_test.json"

REGIONS = ("chest", "waist", "hip")
SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6, "3XL": 7}

# Exact copy of app.main.rank_sizes (kept in sync; see provenance).
TARGET_EASE = {"slim": {"chest": 2, "waist": 1, "hip": 2},
               "regular": {"chest": 8, "waist": 5, "hip": 8},
               "loose": {"chest": 14, "waist": 12, "hip": 14}}
WEIGHTS = {"chest": 0.4, "waist": 0.3, "hip": 0.3}


def rank_sizes(body, garments, fit_preference, stretch_percent):
    targets = TARGET_EASE[fit_preference]
    ranked = []
    for garment in garments:
        score = 0.0
        for region in REGIONS:
            given = body[region] * stretch_percent / 100.0
            required = body[region] + targets[region] - given
            delta = garment[region] - required
            score += WEIGHTS[region] * delta * delta
        ranked.append({"size": garment["size"], "score": score})
    ranked.sort(key=lambda item: (item["score"], item["size"]))
    for item in ranked:
        item["score"] = round(item["score"], 2)
    return ranked


def min_easelab_score(body, garments, fit_preference, stretch_percent):
    return float(min(r["score"] for r in rank_sizes(body, garments, fit_preference, stretch_percent)))


def build_pair_rows(partition):
    """Return list of (pair_id, cid, gid, merged, c_row, g_row) for the strict cohort."""
    customers, garments, sizes, observations, splits = load(DATA_DIR)
    return complete_pairs(customers, garments, sizes, observations, splits, partition), (customers, garments, sizes, observations, splits)


def make_minscore_map(pairs_list, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    customers, garments, sizes, observations, splits = load(data_dir)
    # pairs_list already from complete_pairs; recompute min score per pair_id
    cbi = customers.set_index("customer_id")
    gbi = garments.set_index("garment_id")
    sbg = {gid: g for gid, g in sizes.groupby("garment_id")}
    out = {}
    for pair_id, cid, gid, merged, c_row, g_row in pairs_list:
        body = {"chest": float(c_row["chest_cm"]), "waist": float(c_row["waist_cm"]), "hip": float(c_row["hip_cm"])}
        gl = [{"size": r["size_label"], "chest": float(r["garment_chest_cm"]),
               "waist": float(r["garment_waist_cm"]), "hip": float(r["garment_hip_cm"])}
              for _, r in merged.iterrows()]
        out[pair_id] = min_easelab_score(body, gl, str(c_row["fit_preference"]), float(g_row["stretch_crosswise_pct"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    data_dir = args.data_dir
    # (no global override; pass data_dir explicitly)
    customers, garments, sizes, observations, splits = load(data_dir)
    pair_truth = observations.groupby("pair_id")["pair_preferred_size"].first()

    train_pairs = complete_pairs(customers, garments, sizes, observations, splits, "train")
    val_pairs = complete_pairs(customers, garments, sizes, observations, splits, "val")
    test_pairs = complete_pairs(customers, garments, sizes, observations, splits, "test")
    assert len(test_pairs) == 603, len(test_pairs)

    # Majority label from train complete pairs (same as baseline script).
    train_truth_labels = [str(pair_truth.loc[p[0]]) for p in train_pairs]
    majority_label = pd.Series(train_truth_labels).value_counts().idxmax()

    # Build supervised rows aligned to complete cohort (train/val/test partitions as-is).
    all_complete_ids = {p[0] for p in (train_pairs + val_pairs + test_pairs)}
    main = fe.load_set(data_dir, True)
    R = fe.build_rows(main)
    R = R[R["pair_id"].isin(all_complete_ids)].copy()
    R = R.sort_values("pair_id").reset_index(drop=True)
    partmap = main["splits_customers"].set_index("customer_id")["partition"]
    R["partition"] = R["customer_id"].map(partmap)
    # EaseLab min score per pair (actual score; prediction-time derivable from visible fields).
    ms_map = {}
    for pl in (train_pairs, val_pairs, test_pairs):
        ms_map.update(make_minscore_map(pl, data_dir))
    R["easelab_minscore"] = R["pair_id"].map(ms_map)

    # Augment fe feature builder + Prep to include the actual EaseLab score.
    orig_build = fe.build_features

    def build_features_aug(rows):
        F = orig_build(rows)
        F["easelab"] = rows["easelab_minscore"].to_numpy(float)
        return F

    fe.build_features = build_features_aug

    class PrepAug(fe.Prep):
        def shared(self, F):
            cols, names = super().shared(F)
            cols = np.column_stack([cols, F["easelab"]])
            names = names + ["easelab_minscore"]
            return cols, names

    fe.Prep = PrepAug

    F_all = fe.build_features(R)
    P_all = fe.Pairs(R)

    tr_mask = R["partition"] == "train"
    va_mask = R["partition"] == "val"
    te_mask = R["partition"] == "test"

    def subset(mask):
        r = R[mask].reset_index(drop=True)
        return r, fe.build_features(r), fe.Pairs(r)

    r_tr, F_tr, P_tr = subset(tr_mask)
    r_va, F_va, P_va = subset(va_mask)
    r_te, F_te, P_te = subset(te_mask)

    # Hyperparameters + abstain gate on TRAIN+VAL only.
    hp = fe.select_hparams(r_tr, F_tr, P_tr, r_va, F_va, P_va)
    M = fe.Methods(r_tr, F_tr, P_tr, hp)
    hp["theta"] = M.lr.theta = fe.select_theta(M.lr, r_va, F_va, P_va)

    pred_te = M.predict(r_te, F_te, P_te)
    cand_chosen = pred_te["logreg"]["chosen"]
    cand_score = pred_te["logreg"].get("score")
    cand_pacc = pred_te["logreg"].get("p_acc")

    # Baseline predictions on identical cohort.
    per_pair = []
    n = len(test_pairs)
    n_pref = 0
    n_none = 0
    results = {
        "real_scorer": {"correct_all": 0, "correct_non_none": 0, "n_non_none": 0},
        "fixed_size_M": {"correct_all": 0, "correct_non_none": 0, "n_non_none": 0},
        "majority": {"correct_all": 0, "correct_non_none": 0, "n_non_none": 0},
        "candidate": {"correct_all": 0, "correct_non_none": 0, "n_non_none": 0},
    }

    # Map candidate chosen (row index) back to pair-level label.
    # P_te.starts gives row offset per pair in r_te; chosen is row index or -1.
    cand_pair_pred = {}
    for i in range(P_te.n):
        gi = int(cand_chosen[i])
        pid = str(r_te.iloc[int(P_te.starts[i])]["pair_id"])
        if gi < 0:
            cand_pair_pred[pid] = "NONE"
        else:
            cand_pair_pred[pid] = str(r_te.iloc[gi]["size_label"])

    for i, (pair_id, cid, gid, merged, c_row, g_row) in enumerate(test_pairs):
        truth = str(pair_truth.loc[pair_id])
        if truth == "NONE":
            n_none += 1
        else:
            n_pref += 1

        body = {"chest": float(c_row["chest_cm"]), "waist": float(c_row["waist_cm"]), "hip": float(c_row["hip_cm"])}
        garments_list = [{"size": r["size_label"], "chest": float(r["garment_chest_cm"]),
                         "waist": float(r["garment_waist_cm"]), "hip": float(r["garment_hip_cm"])}
                        for _, r in merged.iterrows()]
        ranked = rank_sizes(body, garments_list, str(c_row["fit_preference"]), float(g_row["stretch_crosswise_pct"]))
        real_pred = ranked[0]["size"]
        fixed_pred = "M"
        majority_pred = majority_label
        candidate_pred = cand_pair_pred[str(pair_id)]

        for method, pred in (("real_scorer", real_pred), ("fixed_size_M", fixed_pred),
                            ("majority", majority_pred), ("candidate", candidate_pred)):
            correct = (pred == truth)
            results[method]["correct_all"] += int(correct)
            if truth != "NONE":
                results[method]["n_non_none"] += 1
                results[method]["correct_non_none"] += int(correct)

        per_pair.append({
            "pair_id": pair_id,
            "truth": truth,
            "real_scorer_pred": real_pred,
            "real_scorer_score": ranked[0]["score"],
            "fixed_size_M_pred": fixed_pred,
            "majority_pred": majority_pred,
            "candidate_pred": candidate_pred,
        })

    assert n == 603 and n_pref == 369 and n_none == 234, (n, n_pref, n_none)

    # None-class metrics helper.
    def none_metrics(preds, truths):
        tp = sum(1 for p, t in zip(preds, truths) if t == "NONE" and p == "NONE")
        fp = sum(1 for p, t in zip(preds, truths) if t != "NONE" and p == "NONE")
        fn = sum(1 for p, t in zip(preds, truths) if t == "NONE" and p != "NONE")
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"support": int(sum(1 for t in truths if t == "NONE")), "tp": tp, "fp": fp, "fn": fn,
                "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    truths_list = [str(pair_truth.loc[p[0]]) for p in test_pairs]
    preds_map = {
        "real_scorer": [r["real_scorer_pred"] for r in per_pair],
        "fixed_size_M": [r["fixed_size_M_pred"] for r in per_pair],
        "majority": [r["majority_pred"] for r in per_pair],
        "candidate": [r["candidate_pred"] for r in per_pair],
    }

    # Conditional top1 (valid pairs only) and sensitivity (none recall).
    summary_metrics = {}
    for method in ("real_scorer", "fixed_size_M", "majority", "candidate"):
        r = results[method]
        preds = preds_map[method]
        nm = none_metrics(preds, truths_list)
        valid_correct = sum(1 for p, t in zip(preds, truths_list) if t != "NONE" and p == t)
        total_correct = r["correct_all"]
        summary_metrics[method] = {
            "top1_all_pairs": total_correct / n,
            "top1_all_pairs_count": total_correct,
            "top1_non_none": (r["correct_non_none"] / r["n_non_none"]) if r["n_non_none"] else None,
            "top1_non_none_count": r["correct_non_none"],
            "n_non_none": r["n_non_none"],
            "conditional_top1_valid": valid_correct / n_pref if n_pref else None,
            "conditional_top1_valid_count": valid_correct,
            "none_precision_recall": nm,
            "sensitivity_none_recall": nm["recall"],
        }

    # Customer-clustered bootstrap CI for candidate total top1.
    cust_ids = np.array([p[1] for p in test_pairs])
    # map customer_id to index
    uniq = {c: i for i, c in enumerate(sorted(set(cust_ids.tolist())))}
    cust_idx = np.array([uniq[c] for c in cust_ids])
    correct_arr = np.array([1.0 if preds_map["candidate"][i] == truths_list[i] else 0.0 for i in range(n)])
    ci = fe.bootstrap_ci(correct_arr, cust_idx, n_boot=300, seed=0)

    # Sensitivity analysis: candidate none recall CI via bootstrap on none-correct indicator.
    none_correct_arr = np.array([1.0 if (truths_list[i] == "NONE" and preds_map["candidate"][i] == "NONE") else
                                (0.0 if truths_list[i] == "NONE" else np.nan) for i in range(n)])
    # bootstrap only over none pairs is awkward; reuse customer bootstrap on a combined indicator for none detection.
    none_detect_arr = np.array([1.0 if preds_map["candidate"][i] == "NONE" else 0.0 for i in range(n)])
    # precision not directly bootstrapped; report point estimates.

    summary = {
        "n_pairs": n,
        "n_pairs_with_preferred_size": n_pref,
        "n_pairs_none": n_none,
        "majority_train_label": majority_label,
        "abstain_gate": {"theta_p_acc": float(hp["theta"]), "selection": "max pair top1 on validation"},
        "hyperparameters": {"C": hp["C"], "lam": hp["lam"]},
        "metrics": summary_metrics,
        "candidate_total_top1_ci95_by_customer": ci,
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "data_dir": str(data_dir.resolve()),
            "git_commit": None,
            "input_hashes": {str(p.name): sha256_file(p) for p in sorted(data_dir.glob("*.csv"))},
            "compute": "CPU (deterministic; no GPU/Kaggle claim)",
            "learned_predictions_reused": False,
        },
    }
    try:
        import subprocess
        git = subprocess.run(["git", "-C", str(HERE.parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True)
        if git.returncode == 0:
            summary["provenance"]["git_commit"] = git.stdout.strip()
    except Exception:
        pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    pred_path = args.out.with_name(args.out.stem + "_predictions.csv")
    pd.DataFrame(per_pair).to_csv(pred_path, index=False)

    print(json.dumps({k: summary[k] for k in ("n_pairs", "n_pairs_with_preferred_size", "n_pairs_none", "metrics")}, indent=2))
    print("theta:", hp["theta"], "C:", hp["C"], "lam:", hp["lam"])
    print("candidate CI95:", ci)
    print("predictions:", pred_path)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()