#!/usr/bin/env python3
"""Evaluate the real EaseLab scorer (app.main.rank_sizes) on the synthetic test pairs.

Scope: only pairs whose customers and all candidate sizes have complete body/garment
measurements and stretch within app.main's real schema. No imputation, no oracle gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # garment-ease-engine
sys.path.insert(0, str(REPO_ROOT))
from app.main import rank_sizes  # noqa: E402

DATA_DIR = HERE / "data" / "main"
OUT_DEFAULT = HERE / "results" / "real_scorer_synthetic_test.json"

REGIONS = ("chest", "waist", "hip")
SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6, "3XL": 7}


def body_valid(customer_row: pd.Series) -> bool:
    for col in ("chest_cm", "waist_cm", "hip_cm"):
        v = customer_row[col]
        if pd.isna(v) or not (30 <= float(v) <= 250):
            return False
    if customer_row["fit_preference"] not in ("slim", "regular", "loose"):
        return False
    return True


def garment_valid(garment_row: pd.Series, size_rows: pd.DataFrame) -> bool:
    stretch = garment_row["stretch_crosswise_pct"]
    if pd.isna(stretch) or not (0 <= float(stretch) <= 50):
        return False
    for col in ("garment_chest_cm", "garment_waist_cm", "garment_hip_cm"):
        if size_rows[col].isna().any():
            return False
        if ((size_rows[col] < 30) | (size_rows[col] > 300)).any():
            return False
    return True


def load(data_dir):
    customers = pd.read_csv(data_dir / "customers.csv")
    garments = pd.read_csv(data_dir / "garments.csv")
    sizes = pd.read_csv(data_dir / "garment_sizes.csv")
    observations = pd.read_csv(data_dir / "observations.csv")
    splits = pd.read_csv(data_dir / "splits_customers.csv")
    return customers, garments, sizes, observations, splits


def complete_pairs(customers, garments, sizes, observations, splits, partition):
    customer_by_id = customers.set_index("customer_id")
    garment_by_id = garments.set_index("garment_id")
    size_by_garment = {gid: g for gid, g in sizes.groupby("garment_id")}
    test_customers = set(splits.loc[splits["partition"] == partition, "customer_id"])
    out = []
    for pair_id, grp in observations.groupby("pair_id", sort=False):
        first = grp.iloc[0]
        cid, gid = first["customer_id"], first["garment_id"]
        if cid not in test_customers:
            continue
        c_row = customer_by_id.loc[cid]
        g_row = garment_by_id.loc[gid]
        s_rows = size_by_garment[gid]
        if not body_valid(c_row):
            continue
        if not garment_valid(g_row, s_rows):
            continue
        # all candidate sizes must have complete measurements (already enforced by garment)

        # map observation rows to garment size measurements
        merged = grp.merge(s_rows, on=["garment_id", "size_label"], how="left", validate="one_to_one")
        if merged[["garment_chest_cm", "garment_waist_cm", "garment_hip_cm"]].isna().any().any():
            continue
        out.append((pair_id, cid, gid, merged, c_row, g_row))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    customers, garments, sizes, observations, splits = load(args.data_dir)

    train_pairs = complete_pairs(customers, garments, sizes, observations, splits, "train")
    test_pairs = complete_pairs(customers, garments, sizes, observations, splits, "test")
    assert len(test_pairs) == 603, len(test_pairs)


    # Re-read truth from observations grouped
    pair_truth = observations.groupby("pair_id")["pair_preferred_size"].first()

    # majority baseline modal from train complete pairs
    train_truth_labels = [pair_truth.loc[p[0]] for p in train_pairs]
    majority_label = pd.Series(train_truth_labels).value_counts().idxmax()

    n = len(test_pairs)
    n_pref = 0
    n_none = 0
    results = {
        "real_scorer": {"correct_all": 0, "correct_non_none": 0, "n_non_none": 0},
        "fixed_size_M": {"correct_all": 0, "correct_non_none": 0, "n_non_none": 0},
        "majority": {"correct_all": 0, "correct_non_none": 0, "n_non_none": 0},
    }
    per_pair = []

    for pair_id, cid, gid, merged, c_row, g_row in test_pairs:
        truth = str(pair_truth.loc[pair_id])
        if truth == "NONE":
            n_none += 1
        else:
            n_pref += 1

        body = {"chest": float(c_row["chest_cm"]), "waist": float(c_row["waist_cm"]), "hip": float(c_row["hip_cm"])}
        garments_list = [
            {"size": r["size_label"], "chest": float(r["garment_chest_cm"]),
             "waist": float(r["garment_waist_cm"]), "hip": float(r["garment_hip_cm"])}
            for _, r in merged.iterrows()
        ]
        ranked = rank_sizes(body, garments_list, str(c_row["fit_preference"]), float(g_row["stretch_crosswise_pct"]))
        real_pred = ranked[0]["size"]

        fixed_pred = "M"
        majority_pred = majority_label

        for method, pred in (("real_scorer", real_pred), ("fixed_size_M", fixed_pred), ("majority", majority_pred)):
            correct = (pred == truth)
            results[method]["correct_all"] += int(correct)
            if truth != "NONE":
                results[method]["n_non_none"] += 1
                results[method]["correct_non_none"] += int(correct)

        per_pair.append({
            "pair_id": pair_id,
            "truth": truth,
            "real_scorer_pred": real_pred,
            "fixed_size_M_pred": fixed_pred,
            "majority_pred": majority_pred,
        })

    assert n == 603 and n_pref == 369 and n_none == 234, (n, n_pref, n_none)
    # finalize
    summary = {
        "n_pairs": n,
        "n_pairs_with_preferred_size": n_pref,
        "n_pairs_none": n_none,
        "majority_train_label": majority_label,
        "metrics": {},
    }
    for method in ("real_scorer", "fixed_size_M", "majority"):
        r = results[method]
        summary["metrics"][method] = {
            "top1_all_pairs": r["correct_all"] / n,
            "top1_all_pairs_count": r["correct_all"],
            "top1_non_none": (r["correct_non_none"] / r["n_non_none"]) if r["n_non_none"] else None,
            "top1_non_none_count": r["correct_non_none"],
            "n_non_none": r["n_non_none"],
        }

    # provenance
    def sha256_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    summary["provenance"] = {
        "script": str(Path(__file__).resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "git_commit": None,
        "input_hashes": {str(p.name): sha256_file(p) for p in sorted(args.data_dir.glob("*.csv"))},
        "compute": "CPU (deterministic; no GPU/Kaggle claim)",
        "learned_predictions_reused": False,
        "learned_predictions_reason": "Existing learned experiment predictions are aggregate-only and not aligned to this complete-measurement dress subset; not reused.",
    }
    try:
        import subprocess
        git = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        if git.returncode == 0:
            summary["provenance"]["git_commit"] = git.stdout.strip()
    except Exception:
        pass

    summary["command"] = f"{sys.executable} {Path(__file__).resolve()} --data-dir {args.data_dir} --out {args.out}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    # also save per-pair predictions
    pred_path = args.out.with_name(args.out.stem + "_predictions.csv")
    pd.DataFrame(per_pair).to_csv(pred_path, index=False)

    print(json.dumps(summary, indent=2))
    print("predictions:", pred_path)


if __name__ == "__main__":
    main()
