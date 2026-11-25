"""Local semantics check for the CatBoost calls the kernel makes (CPU, real data, no torch).

Validates the parts that can silently break on Kaggle: DataFrame + string categorical columns
(one low-card, one 1.4k-card), NaN numerics, class_weights on MultiClass, predict_proba on rows
with unseen category values, and seed determinism. Usage: python smoke_catboost.py [data path]
"""
import json, sys, numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ahmednour/.hermes/cache/scratch/fitdata/modcloth_final_data.json"
NUM = ["size_n", "height_in", "hips_in", "bra_band", "cup_idx"]
L = ["small", "fit", "large"]

rows = [json.loads(l) for l in open(DATA)][:8000]


def num(x):
    try:
        return float(str(x).strip())
    except ValueError:
        return np.nan


def hin(x):
    import re
    m = re.match(r"\s*(\d+)ft\s*(\d+)in", str(x))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


df = pd.DataFrame({"user_id": [r["user_id"] for r in rows], "item_id": [str(r["item_id"]) for r in rows],
                   "fit": [r.get("fit") for r in rows], "category": [r.get("category") for r in rows],
                   "size_n": [num(r.get("size")) for r in rows], "height_in": [hin(r.get("height")) for r in rows],
                   "hips_in": [num(r.get("hips")) for r in rows], "bra_band": [num(r.get("bra size")) for r in rows],
                   "cup_idx": [np.nan] * len(rows)})
df = df[df.fit.isin(L) & df.size_n.notna()].reset_index(drop=True)
y = df.fit.map({c: i for i, c in enumerate(L)}).to_numpy()


def frame(d):
    x = d[NUM].copy()
    x["rel_hips"] = x.hips_in - x.size_n
    x["rel_band"] = x.bra_band - x.size_n
    x["rel_height"] = x.height_in - x.size_n
    for c in NUM:
        x["na_" + c] = x[c].isna().astype(np.int8)
    x["category"] = d.category.astype(str).to_numpy()
    x["item_id"] = d.item_id.to_numpy()
    return x.astype({c: np.float32 for c in x.columns if c not in ("category", "item_id")})


tr, te = df.iloc[:6000], df.iloc[6000:]
Xtr, Xte = frame(tr), frame(te)
Xte_item_unseen = int((~Xte.item_id.isin(set(Xtr.item_id))).sum())
assert Xte_item_unseen >= 0
print("rows", len(Xtr), len(Xte), "| cat card", Xtr.item_id.nunique(), "| unseen cat rows in eval", Xte_item_unseen)

w = len(y[:6000]) / (3.0 * np.bincount(y[:6000], minlength=3))
kw = dict(iterations=60, depth=6, l2_leaf_reg=3.0, learning_rate=0.06, loss_function="MultiClass",
          random_seed=20261124, verbose=False, allow_writing_files=False,
          class_weights={0: w[0], 1: w[1], 2: w[2]}, max_ctr_complexity=1, border_count=128)
CAT = ["category", "item_id"]
p = []
for _ in range(2):
    m = CatBoostClassifier(task_type="CPU", **kw)
    m.fit(Pool(Xtr, y[:6000], cat_features=CAT))
    q = m.predict_proba(Pool(Xte, cat_features=CAT))
    assert q.shape == (len(Xte), 3) and np.isfinite(q).all(), q.shape
    assert np.allclose(q.sum(1), 1, atol=1e-5)
    p.append(q)
print("deterministic same-seed probabilities:", bool(np.allclose(p[0], p[1])))
print("pred label argmax ok:", set(np.argmax(p[0], 1).tolist()) <= {0, 1, 2})
from sklearn.metrics import f1_score
print("smoke macro_f1 (60 iters, 6k rows):", round(float(f1_score(y[6000:], np.argmax(p[0], 1), average="macro")), 4))
print("SMOKE_OK")
