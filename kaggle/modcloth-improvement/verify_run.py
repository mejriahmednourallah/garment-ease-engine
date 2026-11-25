"""Independently re-derive every reported final-test number from the retrieved predictions.

Stdlib only. Reads artifacts/test_predictions.csv, test_predictions_fitonly.csv, metrics.json,
leakage_audit.json and gpu_receipt.txt, recomputes macro F1 / balanced accuracy / accuracy /
per-class F1 / confusion / support plus the paired user-clustered delta CI, and fails loudly on
any disagreement. Usage: python verify_run.py
"""
import csv, json, random
from collections import Counter, defaultdict

A = "artifacts"
L = ["small", "fit", "large"]
rows = list(csv.DictReader(open(f"{A}/test_predictions.csv")))
meta = json.load(open(f"{A}/metrics.json"))
leak = json.load(open(f"{A}/leakage_audit.json"))
receipt = open(f"{A}/gpu_receipt.txt").read()
assert len(rows) > 5000, len(rows)


def scores(y, p):
    cm = defaultdict(Counter)
    for t, q in zip(y, p):
        cm[t][q] += 1
    f1, rec, sup = {}, {}, {}
    for c in L:
        tp = sum(1 for t, q in zip(y, p) if t == c and q == c)
        fp = sum(1 for t, q in zip(y, p) if t != c and q == c)
        fn = sum(1 for t, q in zip(y, p) if t == c and q != c)
        f1[c] = 2 * tp / max(1, 2 * tp + fp + fn)
        sup[c] = tp + fn
        rec[c] = tp / sup[c] if sup[c] else 0.0
    return {"macro_f1": round(sum(f1.values()) / 3, 4),
            "balanced_accuracy": round(sum(rec.values()) / 3, 4),
            "accuracy": round(sum(a == b for a, b in zip(y, p)) / len(y), 4),
            "per_class_f1": {c: round(v, 4) for c, v in f1.items()},
            "confusion_true_x_pred": [[cm[t][q] for q in L] for t in L],
            "support": dict(sup)}


y = [r["y_true"] for r in rows]
res = {"control_mlp_64_64": scores(y, [r["pred_control"] for r in rows]),
       "winner": scores(y, [r["pred_winner"] for r in rows])}
for k, v in res.items():
    m = meta["final"][k]
    for f in ("macro_f1", "balanced_accuracy", "accuracy"):
        assert abs(v[f] - m[f]) < 5e-4, (k, f, v[f], m[f])
    for c in L:
        assert abs(v["per_class_f1"][c] - m["per_class_f1"][c]) < 5e-4, (k, c)
    assert v["confusion_true_x_pred"] == m["confusion_true_x_pred"], k
    assert all(v["support"][c] == m["support"][c] for c in L), k
print("winner:", meta["final"]["winner"]["name"])
for k in ("control_mlp_64_64", "winner"):
    print(f"{k:22s} macro_f1 {res[k]['macro_f1']:.4f}  bal_acc {res[k]['balanced_accuracy']:.4f}  "
          f"acc {res[k]['accuracy']:.4f}  per_class {res[k]['per_class_f1']}")

obs = round(res["winner"]["macro_f1"] - res["control_mlp_64_64"]["macro_f1"], 4)
assert abs(obs - meta["delta_macro_f1"]) < 5e-4, (obs, meta["delta_macro_f1"])
lo, hi = meta["delta_ci95_user_clustered"]
assert lo <= obs <= hi, ("reported CI does not bracket the observed delta", lo, obs, hi)

# item slices
seen = [i for i, r in enumerate(rows) if r["item_seen"] == "True"]
sseen = scores([y[i] for i in seen], [rows[i]["pred_winner"] for i in seen])
assert abs(sseen["macro_f1"] - meta["final"]["item_slices"]["seen_items"]["winner"]["macro_f1"]) < 5e-4
print(f"seen-item rows {len(seen)} / {len(rows)} | winner macro_f1 {sseen['macro_f1']:.4f} "
      f"vs control {scores([y[i] for i in seen], [rows[i]['pred_control'] for i in seen])['macro_f1']:.4f}")

# independent paired user-clustered bootstrap (own RNG -> bounds need only agree approximately)
groups = defaultdict(list)
for i, r in enumerate(rows):
    groups[r["user_key"]].append(i)
gk = list(groups.values())
rng = random.Random(20261124)
pc, pw = [r["pred_control"] for r in rows], [r["pred_winner"] for r in rows]
d = []
for _ in range(2000):
    idx = [i for _ in range(len(gk)) for i in rng.choice(gk)]
    yy = [y[i] for i in idx]
    d.append(scores(yy, [pw[i] for i in idx])["macro_f1"] - scores(yy, [pc[i] for i in idx])["macro_f1"])
d.sort()
my = (round(d[49], 4), round(d[1949], 4))
print(f"delta {obs:+.4f} | kernel CI {lo}..{hi} | verifier CI {my[0]}..{my[1]} | P(delta>0) {sum(x > 0 for x in d)/len(d):.3f}")
assert abs(my[0] - lo) < 0.01 and abs(my[1] - hi) < 0.01, ("CI mismatch", my, (lo, hi))

# fit-only (matched protocol) sensitivity file
rows_a = list(csv.DictReader(open(f"{A}/test_predictions_fitonly.csv")))
assert [r["row_id"] for r in rows_a] == [r["row_id"] for r in rows], "row order differs between test files"
for k, col in (("control_mlp_64_64", "pred_control_fitonly"), ("winner", "pred_winner_fitonly")):
    got = scores(y, [r[col] for r in rows_a])["macro_f1"]
    assert abs(got - meta["final"]["matched_protocol_fit_only"][k]["macro_f1"]) < 5e-4, (k, got)

# split contract + receipt
assert leak["split_contract"]["dev_users"] == 9592 and leak["split_contract"]["dev_rows"] == 16729
assert leak["split_contract"]["pairwise_user_disjoint"] and leak["split_contract"]["final_users_removed_from_train"]
assert leak["split_contract"]["final_rows"] == len(rows), (leak["split_contract"]["final_rows"], len(rows))
assert "Tesla" in receipt or "P100" in receipt or "T4" in receipt, "no GPU named in the receipt"
assert '"matmul_check_sum"' in receipt
assert meta["selected"] in meta["dev_candidates"]
best = max(meta["dev_candidates"], key=lambda k: (meta["dev_candidates"][k]["macro_f1"],
                                                 meta["dev_candidates"][k]["balanced_accuracy"]))
assert best == meta["selected"], (best, meta["selected"])
print("dev ranking:", [(k, meta["dev_candidates"][k]["macro_f1"]) for k, _ in meta["dev_ranking"]])
print("VERIFIED: final metrics, delta, slices, dev selection and split contract recomputed from predictions")
