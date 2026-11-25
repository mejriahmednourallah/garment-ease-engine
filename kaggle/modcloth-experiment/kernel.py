# Garment Ease Engine -- ModCloth fit-feedback partial experiment (scoped).
# Data: UCSD/ModCloth via Kaggle dataset rmisra/clothing-fit-dataset-for-size-recommendation,
#       CC BY 4.0, Misra/Wan/McAuley RecSys 2018. Raw records are read from the read-only mount
#       and never copied into outputs; only per-row predictions of derived features are written.
# Predict 3-class fit (small / fit / large) from pre-purchase inputs only.
# Excluded: review text, quality (rating), length feedback, user_name, and any feedback-derived field.
import hashlib, json, os, re, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

import glob
DATA = next(iter(glob.glob("/kaggle/input/**/modcloth_final_data.json", recursive=True)), None)
if DATA is None:
    print("BLOCKER: modcloth_final_data.json not found under /kaggle/input; tree:", flush=True)
    for root, _, files in os.walk("/kaggle/input"):
        print(root, files[:5], flush=True)
    raise SystemExit(1)
print("data", DATA, flush=True)
OUT = "/kaggle/working"
SEED = 20261124
LABELS = ["small", "fit", "large"]
np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise SystemExit("BLOCKER: CUDA not available; refusing to fall back to CPU.")
dev = "cuda"
print("torch", torch.__version__, "| device", dev, "|", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0),
      "| mem_GB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), flush=True)

sha = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
print("data_sha256", sha, flush=True)

# ------------------------------ parse pre-purchase inputs ------------------------------
CUPS = {"aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "dd/e": 5, "ddd/f": 6,
        "dddd/g": 7, "ddddd/h": 8, "i": 9, "j": 10, "k": 11}

def height_in(x):
    m = re.match(r"\s*(\d+)ft\s*(\d+)in", str(x))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan

def num(x):
    try:
        return float(str(x).strip())
    except ValueError:
        return np.nan

rows = [json.loads(l) for l in open(DATA)]
df = pd.DataFrame({
    "user_id": [r["user_id"] for r in rows],
    "item_id": [r["item_id"] for r in rows],
    "fit": [r.get("fit") for r in rows],
    "category": [r.get("category") for r in rows],
    "size_n": [num(r.get("size")) for r in rows],
    "height_in": [height_in(r.get("height")) for r in rows],
    "hips_in": [num(r.get("hips")) for r in rows],
    "bra_band": [num(r.get("bra size")) for r in rows],
    "cup_idx": [CUPS.get(str(r.get("cup size", "")).strip().lower(), np.nan) for r in rows],
})
raw_rows, raw_users, raw_items = len(df), df.user_id.nunique(), df.item_id.nunique()
df = df[df.fit.isin(LABELS) & df.size_n.notna()].copy()
df["y"] = df.fit.map({c: i for i, c in enumerate(LABELS)})
df = df.reset_index(drop=True)
print(f"rows {raw_rows} -> usable {len(df)} | users {df.user_id.nunique()} | items {df.item_id.nunique()}",
      flush=True)

# ------------------------------ user-grouped holdout ------------------------------
users = df.user_id.drop_duplicates().to_numpy()
perm = np.random.RandomState(SEED).permutation(len(users))
n_val_u = int(round(0.2 * len(users)))
val_users = set(users[perm[:n_val_u]])
df["split"] = np.where(df.user_id.isin(val_users), "val", "train")
assert not (set(users[perm[:n_val_u]]) & set(users[perm[n_val_u:]])), "user leakage"
assert df.split.eq("train").any() and df.split.eq("val").any()
train_items = set(df.loc[df.split == "train", "item_id"])
df["item_seen"] = df.item_id.isin(train_items)

# ------------------------------ features ------------------------------
cats = [c for c in df.loc[df.split == "train", "category"].value_counts().index[:5]]
df["cat"] = df.category.where(df.category.isin(cats), "other")
cat_levels = cats + ["other"]

NUM = ["size_n", "height_in", "hips_in", "bra_band", "cup_idx"]
tr_mask = df.split.eq("train").to_numpy()
med = df.loc[tr_mask, NUM].median()
mu = df.loc[tr_mask, NUM].mean()
sd = df.loc[tr_mask, NUM].std().replace(0, 1.0)

def featurize(d):
    x = d[NUM].copy()
    miss = x.isna().astype(np.float32).to_numpy()
    x = x.fillna(med)
    z = ((x - mu) / sd).to_numpy(dtype=np.float32)
    oh = np.zeros((len(d), len(cat_levels)), dtype=np.float32)
    for i, c in enumerate(d.cat):
        oh[i, cat_levels.index(c)] = 1.0
    # relative size: body measurement minus purchased size, in standardized units
    rel = np.stack([z[:, NUM.index(b)] - z[:, NUM.index("size_n")]
                    for b in ("hips_in", "bra_band", "height_in")], 1)
    return np.hstack([z, rel, miss, oh]).astype(np.float32)

Xtr = featurize(df[tr_mask])
Xva = featurize(df[~tr_mask])
ytr = df.loc[tr_mask, "y"].to_numpy()
yva = df.loc[~tr_mask, "y"].to_numpy()
print("features", Xtr.shape[1], "| train rows", len(Xtr), "| val rows", len(Xva),
      "| val users", df.loc[~tr_mask, "user_id"].nunique(),
      "| val rows seen item", int(df.loc[~tr_mask, "item_seen"].sum()), flush=True)

def metrics(y, p):
    return {
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, p)), 4),
        "accuracy": round(float((y == p).mean()), 4),
        "per_class_f1": {LABELS[c]: round(float(f1_score(y, p, average=None, labels=[0, 1, 2],
                                                            zero_division=0)[c]), 4) for c in range(3)},
        "confusion_true_x_pred": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        "support": {LABELS[c]: int((y == c).sum()) for c in range(3)},
    }

# ------------------------------ baseline: majority class ------------------------------
maj = int(np.bincount(ytr, minlength=3).argmax())
base = metrics(yva, np.full_like(yva, maj))
print("BASELINE_MAJORITY", json.dumps({"predicted": LABELS[maj], **base}), flush=True)

# ------------------------------ trained model (GPU MLP) ------------------------------
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x):
        return self.net(x)

EPOCHS, BS, LR = 200, 1024, 1e-2
xt = torch.tensor(Xtr, device=dev)
yt = torch.tensor(ytr, device=dev)
xv = torch.tensor(Xva, device=dev)
model = MLP(Xtr.shape[1]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# inverse-frequency weights: without them the net collapses to the 69% majority class
w = len(ytr) / (3.0 * np.bincount(ytr, minlength=3))
lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
n = len(xt)
t0 = time.time()
curve = []
for ep in range(EPOCHS):
    model.train()
    order = torch.randperm(n, device=dev)          # train rows only; val never used for fitting/selection
    tot = 0.0
    for i in range(0, n, BS):
        idx = order[i:i + BS]
        opt.zero_grad()
        loss = lossf(model(xt[idx]), yt[idx])
        loss.backward()
        opt.step()
        tot += float(loss.detach()) * len(idx)
    curve.append({"epoch": ep + 1, "train_loss": round(tot / n, 4)})
    if (ep + 1) % 10 == 0:
        print("epoch", ep + 1, "train_loss", round(tot / n, 4), flush=True)
train_secs = round(time.time() - t0, 1)

model.eval()
with torch.no_grad():
    prob = torch.softmax(model(xv), 1).cpu().numpy()
    ptr = torch.softmax(model(xt), 1).argmax(1).cpu().numpy()
p = prob.argmax(1)
assert np.isfinite(prob).all() and len(p) == len(yva)
res = metrics(yva, p)
res.update({"model": "mlp_64_64", "epochs": EPOCHS, "batch_size": BS, "lr": LR,
            "class_weighted": True,
            "train_macro_f1": round(float(f1_score(ytr, ptr, average="macro", labels=[0, 1, 2],
                                                   zero_division=0)), 4),
            "params": int(sum(q.numel() for q in model.parameters())),
            "train_seconds": round(train_secs, 1), "device": torch.cuda.get_device_name(0)})
print("MODEL", json.dumps(res), flush=True)

seen = df.loc[~tr_mask, "item_seen"].to_numpy()
per_split = {
    "seen_items": {"rows": int(seen.sum()), "users": int(df.loc[~tr_mask].loc[seen, "user_id"].nunique()),
                   "model": metrics(yva[seen], p[seen]), "majority": metrics(yva[seen], np.full(int(seen.sum()), maj))},
    "unseen_items": {"rows": int((~seen).sum()), "users": int(df.loc[~tr_mask].loc[~seen, "user_id"].nunique()),
                     "model": metrics(yva[~seen], p[~seen]), "majority": metrics(yva[~seen], np.full(int((~seen).sum()), maj))},
}
print("ITEM_SPLIT", json.dumps(per_split), flush=True)

# ------------------------------ artefacts ------------------------------
val = df.loc[~tr_mask]
pd.DataFrame({
    "row_id": np.arange(len(yva)),
    "item_seen": seen,
    "y_true": [LABELS[v] for v in yva],
    "y_pred": [LABELS[v] for v in p],
    **{f"p_{c}": prob[:, i].round(6) for i, c in enumerate(LABELS)},
}).to_csv(f"{OUT}/val_predictions.csv", index=False)
pd.DataFrame(curve).to_csv(f"{OUT}/training_curve.csv", index=False)
json.dump({"majority": {"predicted": LABELS[maj], **base}, "model": res, "item_seen_split": per_split},
          open(f"{OUT}/metrics.json", "w"), indent=2)
json.dump({
    "experiment": "modcloth fit 3-class, pre-purchase inputs only, user-grouped holdout",
    "dataset": "rmisra/clothing-fit-dataset-for-size-recommendation :: modcloth_final_data.json",
    "dataset_origin": "UCSD ModCloth fit dataset; Misra, Wan & McAuley, RecSys 2018",
    "dataset_license": "CC BY 4.0",
    "data_sha256": sha,
    "raw_rows": raw_rows, "usable_rows": int(len(df)),
    "users": int(df.user_id.nunique()), "items": int(df.item_id.nunique()),
    "train_rows": int(tr_mask.sum()), "val_rows": int((~tr_mask).sum()),
    "train_users": int(df.loc[tr_mask, "user_id"].nunique()),
    "val_users": int(df.loc[~tr_mask, "user_id"].nunique()),
    "features": NUM + ["rel_" + b for b in ("hips_in", "bra_band", "height_in")]
                + ["missing_flags"] + cat_levels,
    "excluded_fields": ["quality (rating)", "length (fit feedback)", "review text", "user_name",
                        "fit label and any feedback-derived metadata"],
    "label_counts": df.fit.value_counts().to_dict(),
    "seed": SEED,
    "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
}, open(f"{OUT}/run_meta.json", "w"), indent=2)
print("DONE_MODCLOTH_FIT_V1", flush=True)
