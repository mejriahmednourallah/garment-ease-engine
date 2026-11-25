"""Independently re-derive the reported metrics from the retrieved predictions.

Reads artifacts/val_predictions.csv + artifacts/metrics.json (downloaded from the private
Kaggle kernel), recomputes macro F1 / balanced accuracy / confusion / support in plain
stdlib, and fails loudly on any disagreement. Usage: python verify_run.py
"""
import json, sys
from collections import Counter, defaultdict

A = "artifacts"
rows = [l.rstrip("\n").split(",") for l in open(f"{A}/val_predictions.csv")]
head, body = rows[0], rows[1:]
iy, ip, iseen = head.index("y_true"), head.index("y_pred"), head.index("item_seen")
y = [r[iy] for r in body]
p = [r[ip] for r in body]
assert len(y) == len(p) and len(y) > 1000, (len(y), len(p))

def scores(y, p):
    labels = sorted(set(y) | set(p))
    cm = {t: Counter() for t in ["small", "fit", "large"]}
    for t, q in zip(y, p):
        cm[t][q] += 1
    f1, rec = {}, {}
    for c in labels:
        tp = sum(1 for t, q in zip(y, p) if t == c and q == c)
        fp = sum(1 for t, q in zip(y, p) if t != c and q == c)
        fn = sum(1 for t, q in zip(y, p) if t == c and q != c)
        f1[c] = 2 * tp / max(1, 2 * tp + fp + fn)
        support = tp + fn
        rec[c] = tp / max(1, support)
    return {
        "macro_f1": round(sum(f1.values()) / len(labels), 4),
        "balanced_accuracy": round(sum(rec.values()) / len(labels), 4),
        "accuracy": round(sum(a == b for a, b in zip(y, p)) / len(y), 4),
        "confusion_true_x_pred": [cm[t][q] for t in ["small", "fit", "large"] for q in ["small", "fit", "large"]],
        "support": dict(Counter(y)),
    }

got = scores(y, p)
meta = json.load(open(f"{A}/metrics.json"))
m = meta["model"]
print("file:", json.dumps(meta["model"], indent=1))
print("recomputed:", json.dumps(got, indent=1))
for k in ["macro_f1", "balanced_accuracy", "accuracy"]:
    assert abs(got[k] - m[k]) < 5e-4, (k, got[k], m[k])
flat = [x for row in m["confusion_true_x_pred"] for x in row]
assert flat == got["confusion_true_x_pred"], (m["confusion_true_x_pred"], got["confusion_true_x_pred"])
assert all(int(v) == got["support"][k] for k, v in m["support"].items()), (m["support"], got["support"])
assert "cuda" in json.dumps(meta).lower() or "tesla" in json.dumps(meta).lower()

seen = [i for i, r in enumerate(body) if r[iseen] == "True"]
print("rows", len(y), "| seen-item rows", len(seen), "| unseen-item rows", len(y) - len(seen))
print("seen-item macro F1", scores([y[i] for i in seen], [p[i] for i in seen])["macro_f1"])
print("VERIFIED: metrics.json matches predictions recomputed from scratch")
