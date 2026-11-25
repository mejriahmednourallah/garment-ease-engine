#!/usr/bin/env python3
"""Learned-only sizing model vs EaseLab scorer on fresh synthetic seeds.

Question: does a learned size-choice model improve sizing *without* abstention?
Reports conditional top-1 (NONE cases scored wrong) and total, so a model cannot
win by abstaining. Synthetic-only fields (sim_region, region_match) are removed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_experiments as fe  # noqa: E402

ERS = (HERE / "evaluate_real_scorer.py").read_text().replace(
    "from app.main import rank_sizes  # noqa: E402", "def rank_sizes(*a, **k): return []")
_NS = {"__file__": str((HERE / "evaluate_real_scorer.py").resolve())}
exec(compile(ERS, "evaluate_real_scorer", "exec"), _NS)  # noqa: S102
complete_pairs, load = _NS["complete_pairs"], _NS["load"]

TARGETS = {"slim": (2, 1, 2), "regular": (8, 5, 8), "loose": (14, 12, 14)}
WEIGHTS = (0.4, 0.3, 0.3)


def scorer_pick(body, gl, pref, stretch):
    best = None
    for g in gl:
        s = sum(w * (g[k] - (body[k] + t - body[k] * stretch / 100.0)) ** 2
                for k, t, w in zip(("chest", "waist", "hip"), TARGETS[pref], WEIGHTS))
        if best is None or s < best[0]:
            best = (s, g["size"])
    return best[1]


# Drop the synthetic-only region_match feature from the supplied builder.
_orig_shared = fe.Prep.shared


def shared_no_region(self, F):
    cols, names = _orig_shared(self, F)
    keep = [i for i, n in enumerate(names) if n != "region_match"]
    return cols[:, keep], [names[i] for i in keep]


fe.Prep.shared = shared_no_region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    customers, garments, sizes, observations, splits = load(args.data_dir)
    truth = observations.groupby("pair_id")["pair_preferred_size"].first().astype(str)
    pairs = {p: complete_pairs(customers, garments, sizes, observations, splits, p)
             for p in ("train", "val", "test")}
    ids = {pid for part in pairs.values() for pid, *_ in part}

    main_set = fe.load_set(args.data_dir, True)
    R = fe.build_rows(main_set)
    R = R[R["pair_id"].isin(ids)].copy().sort_values("pair_id").reset_index(drop=True)
    partmap = main_set["splits_customers"].set_index("customer_id")["partition"]
    R["partition"] = R["customer_id"].map(partmap)

    F = fe.build_features(R)
    P = fe.Pairs(R)
    tr = (R["partition"] == "train").to_numpy()
    va = (R["partition"] == "val").to_numpy()
    te = (R["partition"] == "test").to_numpy()

    def sub(mask):
        r = R[mask].reset_index(drop=True)
        return r, fe.build_features(r), fe.Pairs(r)

    r_tr, F_tr, P_tr = sub(tr)
    r_va, F_va, P_va = sub(va)
    r_te, F_te, P_te = sub(te)

    hp = fe.select_hparams(r_tr, F_tr, P_tr, r_va, F_va, P_va)
    M = fe.Methods(r_tr, F_tr, P_tr, hp)

    # Learned model with the gate disabled: it must always name a size or NONE.
    M.lr.theta = 0.0
    out = M.predict(r_te, F_te, P_te)["logreg"]["chosen"]

    learned, scorer = {}, {}
    for i in range(P_te.n):
        pid = str(r_te.iloc[int(P_te.starts[i])]["pair_id"])
        gi = int(out[i])
        learned[pid] = "NONE" if gi < 0 else str(r_te.iloc[gi]["size_label"])
    for pair in pairs["test"]:
        pid, cid, gid, merged, c_row, g_row = pair
        body = {k: float(c_row[f"{k}_cm"]) for k in ("chest", "waist", "hip")}
        gl = [{"size": x.size_label, "chest": float(x.garment_chest_cm),
               "waist": float(x.garment_waist_cm), "hip": float(x.garment_hip_cm)}
              for x in merged.itertuples()]
        scorer[pid] = scorer_pick(body, gl, str(c_row.fit_preference),
                                  float(g_row.stretch_crosswise_pct))

    test_ids = [p[0] for p in pairs["test"]]
    y = np.array([truth[i] for i in test_ids])
    non = y != "NONE"

    def score(pred_map):
        p = np.array([pred_map[i] for i in test_ids])
        return {"total_top1": float((p == y).mean()),
                "conditional_top1": float((p[non] == y[non]).mean()) if non.any() else None,
                "abstention_rate": float((p == "NONE").mean()),
                "kappa": float(cohen_kappa_score(y, p))}

    result = {"data_dir": str(args.data_dir.resolve()), "n_test": int(len(y)),
              "n_none": int((y == "NONE").sum()), "hyperparameters": hp,
              "scorer": score(scorer), "learned_no_gate": score(learned)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
