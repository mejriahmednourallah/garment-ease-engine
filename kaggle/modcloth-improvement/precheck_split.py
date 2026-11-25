"""Local pre-check of the deterministic split contract on the real file (no GPU, no training).

Reconstructs the previous run's seed-20261124 holdout, derives the final-test users from the
original training users, and asserts the properties the experiment depends on. Run before
pushing the kernel so a split mistake costs no Kaggle GPU version.
Usage: python precheck_split.py [/path/to/modcloth_final_data.json]
"""
import json, sys, numpy as np, pandas as pd

SEED = 20261124
FINAL_SEED = 20261125
LABELS = ["small", "fit", "large"]
DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ahmednour/.hermes/cache/scratch/fitdata/modcloth_final_data.json"

rows = [json.loads(l) for l in open(DATA)]
df = pd.DataFrame({"user_id": [r["user_id"] for r in rows],
                   "item_id": [r["item_id"] for r in rows],
                   "fit": [r.get("fit") for r in rows],
                   "size": [r.get("size") for r in rows]})
df = df[df.fit.isin(LABELS) & df["size"].notna()].reset_index(drop=True)

users = df.user_id.drop_duplicates().to_numpy()
perm = np.random.RandomState(SEED).permutation(len(users))
n_dev = int(round(0.2 * len(users)))
dev_users = set(users[perm[:n_dev]])
dev = df.user_id.isin(dev_users)
print("usable rows", len(df), "users", len(users), "items", df.item_id.nunique())
print("DEV users", len(dev_users), "rows", int(dev.sum()))
assert len(users) == 47958 and len(dev_users) == 9592 and dev.sum() == 16729, \
    "cannot reproduce previous holdout; split contract broken"

train_users = users[perm[n_dev:]]                      # original training users, same order
p2 = np.random.RandomState(FINAL_SEED).permutation(len(train_users))
n_final = int(round(0.15 * len(train_users)))
final_users = set(train_users[p2[:n_final]])
fin = df.user_id.isin(final_users)
fitm = ~dev & ~fin

assert not (dev_users & final_users)
assert fitm.sum() + fin.sum() + dev.sum() == len(df)
assert df.loc[fitm, "user_id"].nunique() == len(train_users) - len(final_users)
print("FIT users", df.loc[fitm, "user_id"].nunique(), "rows", int(fitm.sum()))
print("FINAL users", df.loc[fin, "user_id"].nunique(), "rows", int(fin.sum()))

fit_items = set(df.loc[fitm, "item_id"])
for name, m in (("dev", dev), ("final", fin)):
    seen = df.loc[m, "item_id"].isin(fit_items)
    print(f"{name}: seen-item rows {int(seen.sum())}/{int(m.sum())} "
          f"({100*seen.mean():.1f}%) | items {df.loc[m,'item_id'].nunique()} | "
          f"label counts {df.loc[m,'fit'].value_counts().to_dict()}")
# final users must not appear anywhere in the fit rows
assert not (set(df.loc[fitm, "user_id"]) & final_users)
print("SPLIT_OK")
