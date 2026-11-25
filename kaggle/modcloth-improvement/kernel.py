# Garment Ease Engine -- ModCloth fit-feedback improvement experiment (Kaggle GPU, private).
# Data: UCSD/ModCloth via Kaggle dataset rmisra/clothing-fit-dataset-for-size-recommendation,
#       file modcloth_final_data.json (sha 9afbff64..., 82,790 rows). Kaggle mirror is labelled
#       CC BY 4.0; original-source reuse terms are NOT independently verified -- raw records stay
#       on the read-only mount and are never copied into outputs (only derived predictions and
#       salted user keys are written). Cite: Misra, Wan & McAuley, RecSys 2018.
# Predict 3-class fit (small / fit / large) from pre-purchase inputs only.
# Excluded on purpose: review text/summary, quality (rating), length (fit feedback), user_name,
#       waist/bust/shoe size (not part of the v1 feature contract), the fit label and anything
#       derived from it. item_id is a *pre-purchase* item attribute and is used only by candidates.
# Split contract (deterministic, verified locally against the previous run):
#       users in file order -> RandomState(20261124).permutation -> first 20% = DEV (the previous
#       run's holdout, 9,592 users / 16,729 rows, asserted). From the REMAINING (original train)
#       users -> RandomState(20261125) -> first 15% = FINAL TEST (~5,755 users / ~10,076 rows),
#       removed from training entirely. FIT = the rest. Final rows never touch fitting, feature
#       statistics, early stopping or model selection.
import hashlib, json, os, re, subprocess, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

DATA = next(iter(__import__("glob").glob("/kaggle/input/**/modcloth_final_data.json", recursive=True)), None)
if DATA is None:
    print("BLOCKER: modcloth_final_data.json not found under /kaggle/input", flush=True)
    raise SystemExit(1)
OUT = "/kaggle/working"
SEED, FINAL_SEED, DEV_FRAC, FINAL_FRAC = 20261124, 20261125, 0.2, 0.15
LABELS = ["small", "fit", "large"]
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def fp():
    import platform
    try:
        import catboost
        cb = catboost.__version__
    except Exception:                                   # noqa: BLE001
        cb = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
            "catboost": cb}

np.random.seed(SEED); torch.manual_seed(SEED)
if not torch.cuda.is_available():
    raise SystemExit("BLOCKER: CUDA not available; refusing to fall back to CPU for the torch models.")
DEV = "cuda"
log("gpu", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0),
    "| mem_GB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "|", fp())

sha = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
log("data_sha256", sha)

# ------------------------------ pre-purchase fields only ------------------------------
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
raw_keys = sorted({k for r in rows[:2000] for k in r})
df = pd.DataFrame({
    "user_id": [r["user_id"] for r in rows], "item_id": [str(r["item_id"]) for r in rows],
    "fit": [r.get("fit") for r in rows], "category": [r.get("category") for r in rows],
    "size_n": [num(r.get("size")) for r in rows], "height_in": [height_in(r.get("height")) for r in rows],
    "hips_in": [num(r.get("hips")) for r in rows], "bra_band": [num(r.get("bra size")) for r in rows],
    "cup_idx": [CUPS.get(str(r.get("cup size", "")).strip().lower(), np.nan) for r in rows],
})
raw_rows = len(df)
df = df[df.fit.isin(LABELS) & df.size_n.notna()].reset_index(drop=True)
df["y"] = df.fit.map({c: i for i, c in enumerate(LABELS)})
log(f"rows {raw_rows} -> usable {len(df)} | users {df.user_id.nunique()} | items {df.item_id.nunique()}")

# ------------------------------ split: dev (old holdout) / final / fit ------------------------------
users = df.user_id.drop_duplicates().to_numpy()
perm = np.random.RandomState(SEED).permutation(len(users))
n_dev = int(round(DEV_FRAC * len(users)))
dev_users = set(users[perm[:n_dev]])
df["split"] = np.where(df.user_id.isin(dev_users), "dev", "")
rest = users[perm[n_dev:]]                                  # original training users, original order
p2 = np.random.RandomState(FINAL_SEED).permutation(len(rest))
final_users = set(rest[p2[:int(round(FINAL_FRAC * len(rest)))]])
df.loc[df.split.eq("") & df.user_id.isin(final_users), "split"] = "final"
df.loc[df.split.eq(""), "split"] = "fit"

# hard contract: the dev slice must be exactly the previous run's holdout
if (len(dev_users), int(df.split.eq("dev").sum())) != (9592, 16729):
    print(f"BLOCKER: dev slice {(len(dev_users), int(df.split.eq('dev').sum()))} != previous holdout "
          "(9592 users, 16729 rows); refusing to claim a reproduced split.", flush=True)
    raise SystemExit(1)
assert not (dev_users & final_users) and not (final_users & set(df.loc[df.split == "fit", "user_id"]))
assert df.split.eq("fit").any() and df.split.eq("final").any() and df.split.eq("dev").any()
assert df.groupby("user_id").split.nunique().max() == 1, "a user spans two splits"
assert not set(df.loc[df.split == "final", "user_id"]) & set(df.loc[df.split != "final", "user_id"])
split_rows = df.split.value_counts().to_dict()
split_users = df.groupby("split").user_id.nunique().to_dict()
log("SPLIT rows", split_rows, "users", split_users)

fit_mask, dev_mask, fin_mask = (df.split.eq(s) for s in ("fit", "dev", "final"))
fit_items = set(df.loc[fit_mask, "item_id"])
df["item_seen"] = df.item_id.isin(fit_items)
for s, m in (("fit", fit_mask), ("dev", dev_mask), ("final", fin_mask)):
    log(f"item_overlap[{s}] seen {int(df.loc[m,'item_seen'].sum())}/{int(m.sum())} "
        f"({100*df.loc[m,'item_seen'].mean():.1f}%) | items {df.loc[m,'item_id'].nunique()}")

# ------------------------------ features (statistics from the fitting rows only) ------------------------------
cats = [c for c in df.loc[fit_mask, "category"].value_counts().index[:5]]
cat_levels = cats + ["other"]
df["cat"] = df.category.where(df.category.isin(cats), "other")
NUM = ["size_n", "height_in", "hips_in", "bra_band", "cup_idx"]

def stats_for(mask):
    return {"med": df.loc[mask, NUM].median(), "mu": df.loc[mask, NUM].mean(),
            "sd": df.loc[mask, NUM].std().replace(0, 1.0),
            "items": {v: i + 1 for i, v in enumerate(sorted(df.loc[mask, "item_id"].unique()))}}

def feat(d, st):
    """Standardized numerics + body-minus-size relations + missing flags + category one-hot + item index."""
    x = d[NUM].copy()
    miss = x.isna().astype(np.float32).to_numpy()
    z = (((x.fillna(st["med"]) - st["mu"]) / st["sd"])).to_numpy(dtype=np.float32)
    rel = np.stack([z[:, NUM.index(b)] - z[:, NUM.index("size_n")] for b in ("hips_in", "bra_band", "height_in")], 1)
    oh = np.zeros((len(d), len(cat_levels)), dtype=np.float32)
    for i, c in enumerate(d.cat):
        oh[i, cat_levels.index(c)] = 1.0
    item = np.array([st["items"].get(v, 0) for v in d.item_id], dtype=np.int64)   # 0 = unseen item
    return np.hstack([z, rel, miss, oh]).astype(np.float32), item

def catframe(d):
    """Raw numerics (NaN kept -- CatBoost handles missing) + string categoricals incl. item_id."""
    x = d[NUM].copy()
    x["rel_hips"] = x.hips_in - x.size_n
    x["rel_band"] = x.bra_band - x.size_n
    x["rel_height"] = x.height_in - x.size_n
    for c in NUM:
        x["na_" + c] = x[c].isna().astype(np.int8)
    x["category"] = d.cat.astype(str).to_numpy()
    x["item_id"] = d.item_id.to_numpy()
    return x.astype({c: np.float32 for c in x.columns if c not in ("category", "item_id")})

def metrics(y, p):
    return {"macro_f1": round(float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, p)), 4),
            "accuracy": round(float((y == p).mean()), 4),
            "per_class_f1": {LABELS[c]: round(float(f1_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)[c]), 4) for c in range(3)},
            "confusion_true_x_pred": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
            "support": {LABELS[c]: int((y == c).sum()) for c in range(3)}}

def cw(y):
    return len(y) / (3.0 * np.bincount(y, minlength=3))

# ------------------------------ torch models ------------------------------
class MLP(nn.Module):                       # exact control architecture from the previous run
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x, it):
        return self.net(x)

class MLPEmb(nn.Module):                    # candidate: categorical item embedding, regularized
    def __init__(self, d, n_items, emb=8):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, emb, padding_idx=0)
        self.net = nn.Sequential(nn.Linear(d + emb, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x, it):
        return self.net(torch.cat([x, self.emb(it)], 1))

def train_torch(ctor, X, item, y, epochs=200, bs=1024, lr=1e-2, wd=1e-4):
    torch.manual_seed(SEED)
    m = ctor().to(DEV)
    xt, it, yt = (torch.tensor(v, device=DEV) for v in (X, item, y))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cw(y), dtype=torch.float32, device=DEV))
    n, curve, t0 = len(xt), [], time.time()
    for ep in range(epochs):
        m.train(); order = torch.randperm(n, device=DEV); tot = 0.0
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            opt.zero_grad(); loss = lossf(m(xt[idx], it[idx]), yt[idx]); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        curve.append(round(tot / n, 4))
    secs = round(time.time() - t0, 1)
    return m, curve, secs

def pred_torch(m, X, item):
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(torch.tensor(X, device=DEV), torch.tensor(item, device=DEV)), 1).cpu().numpy()
    assert np.isfinite(p).all() and len(p) == len(X)
    return p

# ------------------------------ CatBoost candidate (item_id + category as categoricals) ------------------------------
try:
    from catboost import CatBoostClassifier, Pool
    HAVE_CB = True
except Exception as e:                                  # noqa: BLE001 - recorded, not hidden
    HAVE_CB, CB_IMPORT_ERROR = False, repr(e)
    log("WARN catboost import failed:", CB_IMPORT_ERROR)
CB_STATE = {"task_type": None, "note": None}

def fit_cat(Xd, yd, depth, l2, iters=500):
    w = cw(yd)
    kw = dict(iterations=iters, depth=depth, l2_leaf_reg=l2, learning_rate=0.06,
              loss_function="MultiClass", random_seed=SEED, verbose=False, allow_writing_files=False,
              class_weights={0: w[0], 1: w[1], 2: w[2]}, max_ctr_complexity=1, border_count=128)
    pool = Pool(Xd, yd, cat_features=CAT_COLS)
    for task in (["GPU"] if CB_STATE["task_type"] != "CPU" else ["CPU"]):
        try:
            m = CatBoostClassifier(task_type=task, devices="0", boosting_type="Plain", **kw) \
                if task == "GPU" else CatBoostClassifier(task_type="CPU", **kw)
            t0 = time.time(); m.fit(pool)
            CB_STATE.update(task_type=task, note="ok")
            return m, round(time.time() - t0, 1)
        except Exception as e:                          # noqa: BLE001 - recorded, not hidden
            CB_STATE.update(task_type="CPU", note=f"{task} fit failed: {e!r}")
            log(f"WARN catboost {task} fit failed, retrying CPU:", repr(e))
    raise RuntimeError(f"catboost could not fit on GPU or CPU: {CB_STATE['note']}")

CAT_COLS = ["category", "item_id"]

def cat_predict(m, Xd):
    return m.predict_proba(Pool(Xd, cat_features=CAT_COLS))

# ------------------------------ stage A: select on DEV (the previous run's holdout) ------------------------------
st_fit = stats_for(fit_mask)
Xf, If = feat(df[fit_mask], st_fit)
Xdv, Idv = feat(df[dev_mask], st_fit)
yf, ydv = df.loc[fit_mask, "y"].to_numpy(), df.loc[dev_mask, "y"].to_numpy()
CF, CDV, CTE = catframe(df[fit_mask]), catframe(df[dev_mask]), catframe(df[fin_mask])
log("features", Xf.shape[1], "| fit rows", len(Xf), "| dev rows", len(Xdv), "| final rows", int(fin_mask.sum()))

maj = int(np.bincount(yf, minlength=3).argmax())
base_dev = metrics(ydv, np.full_like(ydv, maj))
log("BASELINE_MAJORITY_DEV", json.dumps(base_dev))

dev_res, dev_prob = {}, {}

m, curve, secs = train_torch(lambda: MLP(Xf.shape[1]), Xf, If, yf)
pc = pred_torch(m, Xdv, Idv)
dev_prob["control_mlp_64_64"] = pc; dev_res["control_mlp_64_64"] = {
    **metrics(ydv, pc.argmax(1)), "family": "torch_mlp", "items_used": False,
    "epochs": 200, "lr": 1e-2, "weight_decay": 1e-4, "train_seconds": secs,
    "train_curve_tail": curve[-5:]}
log("DEV control_mlp_64_64", json.dumps(dev_res["control_mlp_64_64"]["macro_f1"]))

for wd, emb in ((1e-3, 8), (1e-2, 8)):
    name = f"mlp_emb{emb}_wd{wd:g}"
    m, curve, secs = train_torch(lambda: MLPEmb(Xf.shape[1], len(st_fit["items"]), emb), Xf, If, yf, wd=wd)
    p = pred_torch(m, Xdv, Idv)
    dev_prob[name] = p
    dev_res[name] = {**metrics(ydv, p.argmax(1)), "family": "torch_mlp_emb", "items_used": True,
                     "emb_dim": emb, "epochs": 200, "lr": 1e-2, "weight_decay": wd,
                     "train_seconds": secs, "train_curve_tail": curve[-5:]}
    log(f"DEV {name}", json.dumps({"macro_f1": dev_res[name]["macro_f1"], "bal_acc": dev_res[name]["balanced_accuracy"]}))

if HAVE_CB:
    for depth, l2 in ((6, 3.0), (6, 8.0), (8, 3.0), (8, 8.0)):
        name = f"cat_item_d{depth}_l2{l2:g}"
        try:
            mm, secs = fit_cat(CF, yf, depth, l2)
        except Exception as e:                          # noqa: BLE001 - degrade, never fake a result
            HAVE_CB = False
            log(f"WARN catboost candidates dropped at {name}:", repr(e))
            break
        p = cat_predict(mm, CDV)
        dev_prob[name] = p
        dev_res[name] = {**metrics(ydv, p.argmax(1)), "family": "catboost", "items_used": True,
                         "depth": depth, "l2_leaf_reg": l2, "iterations": 500, "lr": 0.06,
                         "task_type": CB_STATE["task_type"], "train_seconds": secs}
        log(f"DEV {name}", json.dumps({"macro_f1": dev_res[name]["macro_f1"],
                                       "bal_acc": dev_res[name]["balanced_accuracy"],
                                       "device": CB_STATE["task_type"]}))

# predeclared selection rule: dev macro F1, tie-break balanced accuracy
order = sorted(dev_res, key=lambda k: (-dev_res[k]["macro_f1"], -dev_res[k]["balanced_accuracy"], k))
winner = order[0]
log("SELECTED", winner, "| dev macro_f1", dev_res[winner]["macro_f1"], "| dev ranking",
    [(k, dev_res[k]["macro_f1"]) for k in order])

# ------------------------------ stage B: refit on fit+dev, score FINAL once ------------------------------
st_full = stats_for(~fin_mask)                       # encoder scope: fit+dev rows only
Xfu, Ifu = feat(df[~fin_mask], st_full)
Xte, Ite = feat(df[fin_mask], st_full)
Xte_a, Ite_a = feat(df[fin_mask], st_fit)        # matched-protocol frames for the fit-only models
yfu, yte = df.loc[~fin_mask, "y"].to_numpy(), df.loc[fin_mask, "y"].to_numpy()
CFU = catframe(df[~fin_mask])

def fit_predict(spec, refit):
    """spec: candidate name; refit=True -> fit on fit+dev for the final test."""
    X_tr, I_tr, y_tr, X_te, I_te = ((Xfu, Ifu, yfu, Xte, Ite) if refit else (Xf, If, yf, Xte_a, Ite_a))
    if spec == "control_mlp_64_64":
        m, _, secs = train_torch(lambda: MLP(X_tr.shape[1]), X_tr, I_tr, y_tr)
        return pred_torch(m, X_te, I_te), secs
    if spec.startswith("mlp_emb"):
        r = dev_res[spec]
        m, _, secs = train_torch(lambda: MLPEmb(X_tr.shape[1], len((st_full if refit else st_fit)["items"]),
                                                r["emb_dim"]), X_tr, I_tr, y_tr, wd=r["weight_decay"])
        return pred_torch(m, X_te, I_te), secs
    r = dev_res[spec]
    mm, secs = fit_cat(CFU if refit else CF, yfu if refit else yf, r["depth"], r["l2_leaf_reg"])
    return cat_predict(mm, CTE), secs

p_ctl, s_ctl = fit_predict("control_mlp_64_64", True)
p_win, s_win = fit_predict(winner, True)
# matched protocol: the stage-A models (fit on FIT only, dev never used) scored on FINAL too
p_ctl_a = pred_torch(train_torch(lambda: MLP(Xf.shape[1]), Xf, If, yf)[0], Xte_a, Ite_a)
p_win_a = fit_predict(winner, False)[0]

fin_res = {"majority": metrics(yte, np.full_like(yte, maj)),
           "control_mlp_64_64": {**metrics(yte, p_ctl.argmax(1)), "fit_rows": int((~fin_mask).sum()), "train_seconds": s_ctl},
           "winner": {"name": winner, **metrics(yte, p_win.argmax(1)), "fit_rows": int((~fin_mask).sum()), "train_seconds": s_win},
           "matched_protocol_fit_only": {"control_mlp_64_64": metrics(yte, p_ctl_a.argmax(1)),
                                         "winner": {**metrics(yte, p_win_a.argmax(1)), "name": winner}}}
seen_te = df.loc[fin_mask, "item_seen"].to_numpy()
fin_res["item_slices"] = {
    "seen_items": {"rows": int(seen_te.sum()), "control": metrics(yte[seen_te], p_ctl.argmax(1)[seen_te]),
                   "winner": metrics(yte[seen_te], p_win.argmax(1)[seen_te])},
    "unseen_items": {"rows": int((~seen_te).sum()), "control": metrics(yte[~seen_te], p_ctl.argmax(1)[~seen_te]),
                     "winner": metrics(yte[~seen_te], p_win.argmax(1)[~seen_te])}}

# paired user-clustered bootstrap of the macro-F1 delta on the final users (descriptive)
ukey = pd.Series([hashlib.sha256(("mc2026:" + u).encode()).hexdigest()[:12]
                  for u in df.loc[fin_mask, "user_id"]]).to_numpy()
groups = {k: np.where(ukey == k)[0] for k in np.unique(ukey)}
gk = np.array(list(groups.values()), dtype=object)
rng = np.random.RandomState(SEED)
deltas = []
for _ in range(2000):
    idx = np.concatenate([gk[i] for i in rng.randint(0, len(gk), len(gk))])
    a, b = p_win.argmax(1)[idx], p_ctl.argmax(1)[idx]; yy = yte[idx]
    deltas.append(f1_score(yy, a, average="macro", labels=[0, 1, 2], zero_division=0)
                  - f1_score(yy, b, average="macro", labels=[0, 1, 2], zero_division=0))
d = np.array(deltas)
obs = fin_res["winner"]["macro_f1"] - fin_res["control_mlp_64_64"]["macro_f1"]
ci = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
log("FINAL delta macro_f1 (winner-control)", round(obs, 4), "user-clustered 95% CI", ci,
    "| P(delta>0)", round(float((d > 0).mean()), 3))

# ------------------------------ GPU receipt ------------------------------
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:                              # noqa: BLE001
        return f"<failed {e!r}>"

smq = sh("nvidia-smi -q")
def field(key):
    for line in smq.splitlines():
        if key in line and ":" in line:
            return line.split(":", 1)[1].strip()
    return None
a = torch.randn(256, 256, device=DEV); b = torch.randn(256, 256, device=DEV)
gpu_receipt = {
    "kernel_ref": "ahmednourallah/garment-ease-modcloth-improvement-private",
    "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    "kaggle_url": os.environ.get("KAGGLE_URL_BASE"),
    "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "device_name": torch.cuda.get_device_name(0),
    "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    "total_memory_GB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    "torch_cuda": torch.version.cuda,
    "driver_version": field("Driver Version"), "cuda_driver_version": field("CUDA Version"),
    "gpu_product_name": field("Product Name"), "gpu_uuid": field("GPU UUID"),
    "matmul_check_sum": round(float((a @ b).sum().item()), 3),
    "catboost_task_type": CB_STATE["task_type"], "catboost_note": CB_STATE["note"],
    "nvidia_smi_q": smq, "env": fp(),
}
open(f"{OUT}/gpu_receipt.txt", "w").write(smq + "\n\n" + json.dumps(gpu_receipt, indent=2))

# ------------------------------ artefacts ------------------------------
def dframe(mask, probs, ys):
    idx = np.where(mask)[0]
    out = pd.DataFrame({"row_id": idx,
                        "user_key": [hashlib.sha256(("mc2026:" + u).encode()).hexdigest()[:12] for u in df.loc[mask, "user_id"]],
                        "item_seen": df.loc[mask, "item_seen"].to_numpy(), "y_true": [LABELS[v] for v in ys]})
    for nm, p in probs.items():
        out[f"pred_{nm}"] = [LABELS[v] for v in p.argmax(1)]
        for i, c in enumerate(LABELS):
            out[f"p_{nm}_{c}"] = p[:, i].round(6)
    return out

dframe(fin_mask, {"control": p_ctl, "winner": p_win}, yte).to_csv(f"{OUT}/test_predictions.csv", index=False)
dframe(fin_mask, {"control_fitonly": p_ctl_a, "winner_fitonly": p_win_a}, yte).to_csv(
    f"{OUT}/test_predictions_fitonly.csv", index=False)
dframe(dev_mask, dev_prob, ydv).to_csv(f"{OUT}/dev_predictions.csv", index=False)

json.dump({"baseline_majority_dev": base_dev, "dev_candidates": dev_res, "selected": winner,
           "selection_rule": "max dev macro_f1, tie-break balanced_accuracy",
           "dev_ranking": [(k, dev_res[k]["macro_f1"]) for k in order],
           "final": fin_res, "delta_macro_f1": round(obs, 4), "delta_ci95_user_clustered": ci,
           "delta_p_gt0": round(float((d > 0).mean()), 3)}, open(f"{OUT}/metrics.json", "w"), indent=2)

leakage = {
    "split_contract": {"dev_users": len(dev_users), "dev_rows": int(dev_mask.sum()),
                       "final_users": len(final_users), "final_rows": int(fin_mask.sum()),
                       "fit_users": split_users["fit"], "fit_rows": int(fit_mask.sum()),
                       "dev_reproduces_previous_holdout": True,
                       "pairwise_user_disjoint": True, "final_users_removed_from_train": True},
    "checks_passed": ["dev slice == previous run's seed-20261124 holdout (9,592 users / 16,729 rows)",
                      "dev, final and fit user sets pairwise disjoint; no user spans splits",
                      "final users removed from all fitting rows before training",
                      "feature statistics/medians/item vocabulary from fitting rows only "
                      "(stage A: fit; stage B: fit+dev)",
                      "final rows never used for fitting, early stopping, checkpointing or selection",
                      "item_id used as a pre-purchase categorical; item target statistics are fitted "
                      "inside CatBoost on train rows only (ordered target statistics)",
                      "no review text/summary, rating (quality), length feedback, user_name or user_id used"],
    "excluded_raw_keys_present_in_file": [k for k in raw_keys if k in
        ("quality", "length", "review_summary", "review_text", "user_name", "waist", "bust", "shoe size")],
    "all_raw_keys_seen": raw_keys,
    "item_overlap": {s: {"rows": int(df.loc[m].shape[0]), "seen_rows": int(df.loc[m, "item_seen"].sum())}
                     for s, m in (("dev", dev_mask), ("final", fin_mask))},
    "encoder_scope_note": "stage-A (dev) model uses fit-only statistics; stage-B (final) model uses fit+dev. "
                          "Stage-A probabilities are scored on final only as a matched-protocol sensitivity check.",
    "known_limits": ["items are ~99% shared across splits, so item-embedding gains cannot be read as cold-item gains",
                     "one deterministic split: intervals reflect test-user sampling only, not training-seed variance",
                     "no garment measurements in this dataset: these inputs are not physical ease"],
}
json.dump(leakage, open(f"{OUT}/leakage_audit.json", "w"), indent=2)
json.dump({
    "experiment": "modcloth fit 3-class improvement: control MLP vs item-aware models, dev-selected, one final test",
    "dataset": "rmisra/clothing-fit-dataset-for-size-recommendation :: modcloth_final_data.json",
    "dataset_origin": "UCSD ModCloth fit dataset; Misra, Wan & McAuley, RecSys 2018",
    "dataset_license": "Kaggle mirror labelled CC BY 4.0; original-source terms not independently verified "
                       "-- reused under the existing project scope, raw records not redistributed",
    "data_sha256": sha, "usable_rows": len(df), "raw_rows": raw_rows,
    "seed": SEED, "final_seed": FINAL_SEED, "dev_frac": DEV_FRAC, "final_frac": FINAL_FRAC,
    "features": NUM + ["rel_hips_in", "rel_bra_band", "rel_height_in", "missing_flags"] + cat_levels
                + ["item_id (candidates only)", "item_embedding (candidates only)"],
    "excluded_fields": ["review text/summary", "quality (rating)", "length (fit feedback)", "user_name", "user_id", "fit label"],
}, open(f"{OUT}/run_meta.json", "w"), indent=2)
open(f"{OUT}/run.log", "w").write("\n".join(LOG) + "\n")
log("DONE_MODCLOTH_IMPROVEMENT_V1")
