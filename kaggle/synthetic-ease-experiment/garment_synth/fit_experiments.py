#!/usr/bin/env python3
"""SYNTHETIC-data experiments for the garment-ease prototype.

Kept deliberately separate from any production recommendation endpoint: this file only reads the generated
CSVs and writes results/. Results measure CONSISTENCY WITH THE SIMULATOR, not real-world fit accuracy.

Methods compared
  majority           per-body-region majority label; per-garment-type modal preferred size (train rows only)
  fixed_size_M       always recommend size M (nearest available), never 'none acceptable'
  weighted_ease_prior   the existing starting model  penalty = sum_i w_i (actual_ease_i - desired_ease_i)^2
                        with a STAND-IN generic prior (see BASELINE_PRIOR; replace with the production config)
  weighted_ease_tuned   same formula, desired-ease offsets / tolerance / none-threshold tuned on TRAIN rows only
  logreg             interpretable logistic-regression models (regional labels, overall-acceptable, preferred size)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from generate_synthetic_garments import (DIMS, FABRICS, FIT_LABELS, GARMENT_TYPES, PREFS, REGION_NAMES, SCENARIOS,
                                         STYLES)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- baseline stand-in config (the real application config could not be inspected; see README) ----------------
BASELINE_PRIOR = {"desired_ease_cm": {"slim": [3, 3, 3], "regular": [6, 6, 6], "loose": [10, 10, 10]},  # chest,waist,hip
                  "weights": [1.0, 1.0, 1.0], "tolerance_cm": 3.0, "none_threshold_multiplier": 1.0}

# ---- columns a model may see at prediction time (everything else is label / latent / key / metadata) -----------
FEATURE_SOURCE_COLUMNS = ["sim_region", "chest_cm", "waist_cm", "hip_cm", "fit_preference", "garment_type", "garment_style",
                          "fabric_class", "stretch_crosswise_pct", "chart_target_region", "measured_dims", "size_rank",
                          "label_body_chest_cm", "label_body_waist_cm", "label_body_hip_cm", "garment_chest_cm",
                          "garment_waist_cm", "garment_hip_cm"]
LABEL_COLUMNS = ["fit_chest", "fit_waist", "fit_hip", "fit_overall_acceptable", "is_preferred_size", "pair_preferred_size",
                 "pair_none_acceptable"]
FORBIDDEN_FEATURE_SUBSTRINGS = ["preferred", "acceptable", "latent", "true_", "actual_", "sim_scenario", "stress_cohort",
                                "customer_id", "garment_id", "pair_id", "observation_id", "partition", "holdout"]


def is_forbidden_feature(col):
    """Labels, latents, IDs and split/cohort metadata must never reach a model. 'fit_preference' (stated) is allowed."""
    return (col.startswith("fit_") and col != "fit_preference") or any(s in col for s in FORBIDDEN_FEATURE_SUBSTRINGS)


# --------------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------------
def load_set(d: Path, with_splits=False):
    t = {n: pd.read_csv(d / f"{n}.csv") for n in
         ["customers", "garments", "garment_sizes", "observations", "latent_customers"]}
    if with_splits:
        t["splits_customers"] = pd.read_csv(d / "splits_customers.csv")
        t["splits_garments"] = pd.read_csv(d / "splits_garments.csv")
    return t


def build_rows(t):
    """One row per observation; features joined from customers/garments/sizes. Report-only metadata kept separately."""
    o = t["observations"]
    rows = o.merge(t["customers"].drop(columns=["synthetic_marker"]), on="customer_id", how="left")
    rows = rows.merge(t["garments"].drop(columns=["synthetic_marker", "stress_cohort"]), on="garment_id", how="left")
    rows = rows.merge(t["garment_sizes"].drop(columns=["synthetic_marker"]), on=["garment_id", "size_label"], how="left")
    # report-only metadata (NEVER features): scenario of the customer, taken from latent file for grouped reporting
    rows = rows.merge(t["latent_customers"][["customer_id", "sim_scenario"]], on="customer_id", how="left")
    assert len(rows) == len(o)
    return rows.reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------
# Pair structure helpers (rows must be contiguous by pair_id)
# --------------------------------------------------------------------------------------------------------------
class Pairs:
    def __init__(self, rows):
        codes, uniq = pd.factorize(rows["pair_id"], sort=False)
        assert np.all(np.diff(codes) >= 0), "rows must be grouped by pair"
        self.codes, self.n = codes, len(uniq)
        self.counts = np.bincount(codes, minlength=self.n)
        self.starts = np.cumsum(self.counts) - self.counts
        pref = rows["is_preferred_size"].to_numpy() == 1
        self.truth = np.full(self.n, -1)
        self.truth[codes[pref]] = np.flatnonzero(pref)
        self.cust = pd.factorize(rows["customer_id"].iloc[self.starts])[0]

    def argmax(self, score, mask=None):
        """Row index of max score among masked rows per pair; -1 if the pair has no masked rows."""
        mask = np.ones(len(score), bool) if mask is None else mask
        s = np.where(mask, score, -np.inf)
        pm = np.maximum.reduceat(s, self.starts)
        hit = mask & (s == np.repeat(pm, self.counts))
        idx = np.flatnonzero(hit)
        first = np.full(self.n, -1)
        pc = self.codes[idx]
        u, i0 = np.unique(pc, return_index=True)
        first[u] = idx[i0]
        return first


# --------------------------------------------------------------------------------------------------------------
# Features (prediction-time only) and train-only preprocessing
# --------------------------------------------------------------------------------------------------------------
def build_features(rows):
    f = rows[FEATURE_SOURCE_COLUMNS]  # slice: nothing else is reachable below
    assert not any(is_forbidden_feature(c) for c in f.columns)
    body = f[["chest_cm", "waist_cm", "hip_cm"]].to_numpy(float)
    gm = f[[f"garment_{d}_cm" for d in DIMS]].to_numpy(float)
    nom = f[[f"label_body_{d}_cm" for d in DIMS]].to_numpy(float)
    md = f["measured_dims"].fillna("").str.split("|")
    app = np.column_stack([md.apply(lambda x, d=d: d in x).to_numpy() for d in DIMS])
    onehot = lambda col, cats: np.column_stack([(f[col] == c).to_numpy(float) for c in cats])
    st = f["stretch_crosswise_pct"].to_numpy(float)
    return dict(ease=gm - body, nomoff=body - nom, app=app, size_rank=f["size_rank"].to_numpy(float),
                stretch=np.nan_to_num(st, nan=0.0) / 100.0, stretch_missing=np.isnan(st).astype(float),
                pref=onehot("fit_preference", PREFS), type=onehot("garment_type", GARMENT_TYPES),
                style=onehot("garment_style", STYLES), fabric=onehot("fabric_class", FABRICS),
                region_match=(f["sim_region"] == f["chart_target_region"]).to_numpy(float),
                pref_idx=f["fit_preference"].map({p: i for i, p in enumerate(PREFS)}).to_numpy())


class Prep:
    """Imputation medians are fitted on TRAINING rows only; scaling is fitted inside each model on training rows only."""

    def fit(self, F):
        ok = F["app"] & ~np.isnan(F["ease"])
        self.ease_med = np.array([np.median(F["ease"][ok[:, j], j]) for j in range(3)])
        ok2 = F["app"] & ~np.isnan(F["nomoff"])
        self.nom_med = np.array([np.median(F["nomoff"][ok2[:, j], j]) for j in range(3)])
        return self

    def dim_block(self, F, j):
        app = F["app"][:, j]
        e = np.where(app, F["ease"][:, j], 0.0)
        miss = app & np.isnan(e)
        e = np.where(miss, self.ease_med[j], e)
        n = np.where(app, F["nomoff"][:, j], 0.0)
        n = np.where(app & np.isnan(n), self.nom_med[j], n)
        slim, loose = F["pref"][:, 0], F["pref"][:, 2]
        cols = np.column_stack([e, e ** 2 / 10.0, e * slim, e * loose, e * F["stretch"], n, miss.astype(float),
                                e[:, None] * F["style"], e[:, None] * F["type"]])
        names = [f"{DIMS[j]}:{x}" for x in ["ease", "ease^2/10", "ease*slim", "ease*loose", "ease*stretch", "body-minus-label_body",
                                           "ease_missing"] + [f"ease*style={x}" for x in STYLES] + [f"ease*type={x}" for x in GARMENT_TYPES]]
        return cols, names

    def shared(self, F):
        cols = np.column_stack([F["pref"], F["type"], F["style"], F["fabric"], F["stretch"], F["stretch_missing"], F["region_match"]])
        names = [f"pref={p}" for p in PREFS] + [f"type={t}" for t in GARMENT_TYPES] + [f"style={s}" for s in STYLES] + \
                [f"fabric={x}" for x in FABRICS] + ["stretch", "stretch_missing", "region_match"]
        return cols, names

    def X_regional(self, F, j):
        a, an = self.dim_block(F, j)
        b, bn = self.shared(F)
        return np.hstack([a, b]), an + bn

    def X_row(self, F):
        blocks, names = [], []
        for j in range(3):
            a, an = self.dim_block(F, j)
            blocks.append(a), names.extend(an)
        b, bn = self.shared(F)
        app = F["app"].astype(float)
        return np.hstack(blocks + [app, b]), names + [f"applicable:{d}" for d in DIMS] + bn


def sub(F, m):
    return {k: v[m] for k, v in F.items()}


# --------------------------------------------------------------------------------------------------------------
# Learned interpretable models
# --------------------------------------------------------------------------------------------------------------
def lab_codes(rows):
    mp = {"tight": 0, "acceptable": 1, "loose": 2, "not_applicable": -1}
    return np.column_stack([rows[f"fit_{d}"].map(mp).to_numpy() for d in DIMS])


class ChoiceLogit:
    """Conditional (softmax) logit over the candidate sizes of a pair: linear, interpretable utility; L2-regularised."""

    def __init__(self, lam=1.0):
        self.lam = lam

    @staticmethod
    def _prep(X, pairs):
        has = pairs.truth >= 0
        rowmask = has[pairs.codes]
        sel = np.flatnonzero(has)
        counts = pairs.counts[sel]
        st = np.cumsum(counts) - counts
        ind = np.zeros(rowmask.sum())
        ind[st + (pairs.truth[sel] - pairs.starts[sel])] = 1.0
        return X[rowmask], st, counts, ind

    def _nll_grad(self, w, Xs, st, counts, ind):
        u = Xs @ w
        m = np.maximum.reduceat(u, st)
        e = np.exp(u - np.repeat(m, counts))
        S = np.add.reduceat(e, st)
        ll = u[ind == 1].sum() - (m + np.log(S)).sum()
        grad = Xs.T @ (e / np.repeat(S, counts) - ind)
        return -ll + 0.5 * self.lam * w @ w, grad + self.lam * w

    def fit(self, X, pairs):
        from scipy.optimize import minimize
        Xs, st, counts, ind = self._prep(X, pairs)
        res = minimize(self._nll_grad, np.zeros(X.shape[1]), args=(Xs, st, counts, ind), jac=True, method="L-BFGS-B",
                       options={"maxiter": 300})
        self.w = res.x
        return self

    def mean_nll(self, X, pairs):
        Xs, st, counts, ind = self._prep(X, pairs)
        u = Xs @ self.w
        m = np.maximum.reduceat(u, st)
        S = np.add.reduceat(np.exp(u - np.repeat(m, counts)), st)
        return float(-(u[ind == 1].sum() - (m + np.log(S)).sum()) / len(st))


class LogReg:
    def __init__(self, C=1.0, lam=1.0, theta=0.5):
        self.C, self.lam, self.theta = C, lam, theta

    def _fit_one(self, X, y):
        sc = StandardScaler().fit(X)  # fitted on training rows only
        m = LogisticRegression(C=self.C, max_iter=400).fit(sc.transform(X), y)
        return sc, m

    def fit(self, rows, F, pairs, shuffle_rng=None):
        self.prep = Prep().fit(F)
        y_lab = lab_codes(rows)
        y_all = rows["fit_overall_acceptable"].to_numpy()
        pairs_fit = pairs
        if shuffle_rng is not None:  # label-permutation sanity check
            y_lab, y_all = [shuffle_rng.permutation(a) for a in (y_lab, y_all)]
            pairs_fit = Pairs.__new__(Pairs)
            pairs_fit.__dict__.update(pairs.__dict__)
            pairs_fit.truth = pairs.starts + shuffle_rng.integers(0, pairs.counts)  # random "preferred" row per pair
            pairs_fit.truth[shuffle_rng.random(pairs.n) < 0.3] = -1
        self.reg = []
        for j in range(3):
            m = F["app"][:, j] & (y_lab[:, j] >= 0)
            X, self.reg_names = self.prep.X_regional(sub(F, m), j)
            self.reg.append(self._fit_one(X, y_lab[m, j]))
        X, self.row_names = self.prep.X_row(F)
        self.acc = self._fit_one(X, y_all)
        self.choice_sc = StandardScaler().fit(X)
        self.choice = ChoiceLogit(self.lam).fit(self.choice_sc.transform(X), pairs_fit)
        return self

    def predict(self, F):
        n = len(F["size_rank"])
        reg = np.full((n, 3), -1)
        for j in range(3):
            X, _ = self.prep.X_regional(F, j)
            sc, m = self.reg[j]
            reg[:, j] = np.where(F["app"][:, j], m.predict(sc.transform(X)), -1)
        X, _ = self.prep.X_row(F)
        p_acc = self.acc[1].predict_proba(self.acc[0].transform(X))[:, 1]
        u = self.choice_sc.transform(X) @ self.choice.w
        return dict(regional=reg, score=u, p_acc=p_acc)

    def coef_table(self, top=12):
        sc, m = self.acc
        c = pd.Series(m.coef_[0], index=self.row_names).sort_values(key=np.abs, ascending=False).head(top)
        u = pd.Series(self.choice.w, index=self.row_names).sort_values(key=np.abs, ascending=False).head(top)
        r = lambda x: {k: round(float(v), 3) for k, v in x.items()}
        return {"overall_acceptable_logreg_std_coef": r(c), "size_choice_logit_std_coef": r(u)}


def select_hparams(r_tr, F_tr, P_tr, r_va, F_va, P_va, Cgrid=(0.03, 0.3, 3.0), lam_grid=(0.3, 3.0, 30.0)):
    prep = Prep().fit(F_tr)
    Xtr, _ = prep.X_row(F_tr)
    Xva, _ = prep.X_row(F_va)
    sc = StandardScaler().fit(Xtr)
    losses = {C: log_loss(r_va["fit_overall_acceptable"],
                          LogisticRegression(C=C, max_iter=400).fit(sc.transform(Xtr), r_tr["fit_overall_acceptable"])
                          .predict_proba(sc.transform(Xva))[:, 1]) for C in Cgrid}
    C = min(losses, key=losses.get)
    nll = {lam: ChoiceLogit(lam).fit(sc.transform(Xtr), P_tr).mean_nll(sc.transform(Xva), P_va) for lam in lam_grid}
    lam = min(nll, key=nll.get)
    return {"C": float(C), "lam": float(lam), "val_logloss_by_C": {str(k): v for k, v in losses.items()},
            "val_choice_nll_by_lambda": {str(k): v for k, v in nll.items()}}


def select_theta(model, rows_va, F_va, P_va, grid=(0.3, 0.4, 0.5, 0.6, 0.7)):
    r = model.predict(F_va)
    best, bt = -1, 0.5
    for th in grid:
        a = float(np.mean(P_va.argmax(r["score"], r["p_acc"] >= th) == P_va.truth))
        if a > best:
            best, bt = a, th
    return bt


# --------------------------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------------------------
class WeightedEase:
    """penalty = sum_i w_i (actual_ease_i - desired_ease_i)^2 over dimensions with known measurements."""

    def __init__(self, cfg):
        self.des = np.array([cfg["desired_ease_cm"][p] for p in PREFS], float)  # (3 prefs, 3 dims)
        self.w = np.array(cfg["weights"], float)
        self.m = cfg["tolerance_cm"]
        self.tau = cfg["none_threshold_multiplier"]

    def config(self):
        return {"desired_ease_cm": {p: self.des[i].tolist() for i, p in enumerate(PREFS)}, "weights": self.w.tolist(),
                "tolerance_cm": self.m, "none_threshold_multiplier": self.tau}

    def _act(self, F):
        return F["app"] & ~np.isnan(F["ease"])

    def penalty(self, F):
        act = self._act(F)
        des = self.des[F["pref_idx"]]
        return (self.w * np.where(act, (np.nan_to_num(F["ease"]) - des) ** 2, 0.0)).sum(1)

    def predict(self, F, pairs, majority_reg):
        act = self._act(F)
        pen = self.penalty(F)
        thr = self.tau * (self.w * act * self.m ** 2).sum(1)
        score = -pen - 1e-6 * np.abs(F["size_rank"] - 3)
        chosen = pairs.argmax(score)
        none = (chosen >= 0) & (pen[np.maximum(chosen, 0)] > thr[np.maximum(chosen, 0)])
        chosen = np.where(none, -1, chosen)
        des = self.des[F["pref_idx"]]
        e = np.nan_to_num(F["ease"])
        reg = np.where(e < des - self.m, 0, np.where(e > des + self.m, 2, 1))
        reg = np.where(act, reg, np.where(F["app"], majority_reg[None, :], -1))
        return dict(regional=reg, score=score, chosen=chosen)

    def tune(self, F, pairs, rows):
        """Coordinate search on TRAIN rows only: desired-ease offsets, tolerance m, none-threshold multiplier."""
        maj = np.ones(3, int)

        def acc():
            p = self.predict(F, pairs, maj)["chosen"]
            return float(np.mean(p == pairs.truth))
        best = acc()
        grids = {"m": [2, 3, 4, 5, 6, 8], "tau": [0.5, 0.75, 1, 1.5, 2, 3]}
        for _ in range(3):
            for pi in range(3):
                for j in range(3):
                    cur = self.des[pi, j]
                    for dlt in [-4, -2, -1, 1, 2, 4]:
                        self.des[pi, j] = cur + dlt
                        a = acc()
                        if a > best + 1e-9:
                            best, cur = a, cur + dlt
                    self.des[pi, j] = cur
            for key, attr in [("m", "m"), ("tau", "tau")]:
                keep = getattr(self, attr)
                for v in grids[key]:
                    setattr(self, attr, v)
                    a = acc()
                    if a > best + 1e-9:
                        best, keep = a, v
                setattr(self, attr, keep)
        return best


def majority_models(rows, pairs):
    lab = lab_codes(rows)
    reg = np.array([np.bincount(lab[lab[:, j] >= 0, j], minlength=3).argmax() for j in range(3)])
    first = rows.iloc[pairs.starts]
    outcome = first[["garment_type", "pair_preferred_size"]]
    modal = outcome.groupby("garment_type")["pair_preferred_size"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    return reg, modal


def label_to_pos(rows, pairs, label_per_pair):
    """Map a size-label recommendation to the row index of the nearest available size in each pair."""
    order = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6, "3XL": 7}
    rank = rows["size_rank"].to_numpy()
    out = np.full(pairs.n, -1)
    for p in range(pairs.n):
        lab = label_per_pair[p]
        if lab == "NONE":
            continue
        s, c = pairs.starts[p], pairs.counts[p]
        out[p] = s + int(np.argmin(np.abs(rank[s:s + c] - order[lab])))
    return out


# --------------------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------------------
def regional_metrics(rows, reg_pred):
    true = lab_codes(rows)
    out = {}
    for j, d in enumerate(DIMS):
        m = true[:, j] >= 0
        y, p = true[m, j], reg_pred[m, j]
        rec = {FIT_LABELS[c]: float(np.mean(p[y == c] == c)) if np.any(y == c) else None for c in range(3)}
        out[d] = {"n": int(m.sum()), "accuracy": float(np.mean(y == p)), "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)),
                  "recall": rec}
    return out


def pair_metrics(rows, pairs, chosen, score=None, ks=(2, 3)):
    t, p = pairs.truth, chosen
    has = t >= 0
    out = {"n_pairs": int(pairs.n), "n_pairs_with_acceptable": int(has.sum()),
           "top1_incl_none": float(np.mean(p == t)),
           "top1_when_acceptable_exists": float(np.mean(p[has] == t[has])) if has.any() else None}
    tp, fp, fn = np.sum((t < 0) & (p < 0)), np.sum((t >= 0) & (p < 0)), np.sum((t < 0) & (p >= 0))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out["none_acceptable"] = {"support": int((t < 0).sum()), "precision": float(prec), "recall": float(rec),
                              "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0}
    rank = rows["size_rank"].to_numpy()
    ok1 = has & (p >= 0)
    out["within_one_size_when_acceptable"] = float(np.mean(ok1[has] & (np.abs(rank[np.maximum(p, 0)] - rank[np.maximum(t, 0)])[has] <= 1))) if has.any() else None
    if score is not None:
        for k in ks:
            hits = 0
            for q in np.flatnonzero(has):
                s, c = pairs.starts[q], pairs.counts[q]
                top = np.argsort(-score[s:s + c], kind="stable")[:k] + s
                hits += int(t[q] in top)
            out[f"top{k}_when_acceptable_exists"] = hits / has.sum() if has.any() else None
    else:
        for k in ks:
            out[f"top{k}_when_acceptable_exists"] = None
    return out


def bootstrap_ci(correct, cust, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    nc = cust.max() + 1
    num, den = np.bincount(cust, weights=correct, minlength=nc), np.bincount(cust, minlength=nc)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, nc, nc)
        vals.append(num[i].sum() / den[i].sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def overall_acc_metrics(rows, pred):
    y = rows["fit_overall_acceptable"].to_numpy()
    return {"f1_acceptable": float(f1_score(y, pred, zero_division=0)), "accuracy": float(np.mean(y == pred)),
            "positive_rate_true": float(y.mean())}


class Methods:
    """Everything fitted on training rows only."""

    def __init__(self, rows_tr, F_tr, pairs_tr, hp, baseline_cfg=BASELINE_PRIOR, shuffle=None):
        self.maj_reg, self.modal = majority_models(rows_tr, pairs_tr)
        self.prior = WeightedEase(json.loads(json.dumps(baseline_cfg)))
        self.tuned = WeightedEase(json.loads(json.dumps(baseline_cfg)))
        self.tuned_train_top1 = self.tuned.tune(F_tr, pairs_tr, rows_tr)
        self.lr = LogReg(hp["C"], hp["lam"], hp.get("theta", 0.5)).fit(
            rows_tr, F_tr, pairs_tr, shuffle_rng=None if shuffle is None else np.random.default_rng(shuffle))

    def predict(self, rows, F, pairs):
        out = {}
        n = len(rows)
        maj_reg = np.where(F["app"], self.maj_reg[None, :], -1)
        lab = [self.modal.get(t, "M") for t in rows["garment_type"].iloc[pairs.starts]]
        out["majority"] = dict(regional=maj_reg, chosen=label_to_pos(rows, pairs, lab), score=None,
                               overall=np.all((maj_reg == 1) | ~F["app"], axis=1).astype(int))
        out["fixed_size_M"] = dict(regional=None, chosen=label_to_pos(rows, pairs, ["M"] * pairs.n), score=None, overall=None)
        for name, m in [("weighted_ease_prior", self.prior), ("weighted_ease_tuned", self.tuned)]:
            r = m.predict(F, pairs, self.maj_reg)
            r["overall"] = np.all((r["regional"] == 1) | ~F["app"], axis=1).astype(int)
            out[name] = r
        r = self.lr.predict(F)
        elig = r["p_acc"] >= self.lr.theta
        r["chosen"] = pairs.argmax(r["score"], elig)
        r["overall"] = elig.astype(int)
        out["logreg"] = r
        return out


def evaluate(rows, preds, pairs, with_ci=True):
    res = {}
    for name, r in preds.items():
        d = {"pair": pair_metrics(rows, pairs, r["chosen"], r.get("score"))}
        if with_ci:
            d["pair"]["top1_incl_none_ci95_by_customer"] = bootstrap_ci((r["chosen"] == pairs.truth).astype(float), pairs.cust)
        if r["regional"] is not None:
            d["regional"] = regional_metrics(rows, r["regional"])
        if r["overall"] is not None:
            d["overall_acceptable_rows"] = overall_acc_metrics(rows, r["overall"])
        res[name] = d
    return res


def pool(parts):
    rows = pd.concat([p[0] for p in parts], ignore_index=True)
    preds = {}
    for name in parts[0][1]:
        preds[name] = {}
        offset = 0
        chunks = {k: [] for k in parts[0][1][name]}
        for rws, pr, pairs in parts:
            for k, v in pr[name].items():
                if v is None:
                    chunks[k].append(None)
                elif k == "chosen":
                    chunks[k].append(np.where(v >= 0, v + offset, -1))
                else:
                    chunks[k].append(v)
            offset += len(rws)
        for k, lst in chunks.items():
            preds[name][k] = None if lst[0] is None else np.concatenate(lst)
    return rows, preds


# --------------------------------------------------------------------------------------------------------------
def run(data_dir, out_dir, seed=0):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main, stress = load_set(data_dir / "main", True), load_set(data_dir / "stress")
    R = build_rows(main)
    Rs = build_rows(stress)
    Rs = Rs.merge(stress["customers"][["customer_id"]].assign(), on="customer_id")  # keep order
    part = main["splits_customers"].set_index("customer_id")["partition"]
    hold = main["splits_garments"].set_index("garment_id")
    R["partition"] = R["customer_id"].map(part)
    R["unseen_garment"] = R["garment_id"].map(hold["holdout_unseen_garment"]).astype(int)
    R["style"] = R["garment_style"]
    Rs["stress_cohort_cust"] = Rs["customer_id"].map(stress["customers"].set_index("customer_id")["stress_cohort"])
    Rs["stress_cohort_garm"] = Rs["garment_id"].map(stress["garments"].set_index("garment_id")["stress_cohort"])
    results = {"synthetic_notice": "SYNTHETIC data. Metrics show agreement with the simulator, not real-world accuracy."}

    def subset(mask):
        r = R[mask].reset_index(drop=True)
        return r, build_features(r), Pairs(r)

    tr, va, te = (R["partition"] == p for p in ("train", "val", "test"))
    # hyper-parameter C chosen on validation customers only
    r_tr, F_tr, P_tr = subset(tr)
    r_va, F_va, P_va = subset(va)
    hp = select_hparams(r_tr, F_tr, P_tr, r_va, F_va, P_va)
    # sanity: preprocessing statistics are train-only
    results["preprocessing"] = {"fit_on": "train partition rows only", "impute_median_ease_cm": Prep().fit(F_tr).ease_med.round(3).tolist()}

    # A. customer split
    M = Methods(r_tr, F_tr, P_tr, hp)
    hp["theta"] = M.lr.theta = select_theta(M.lr, r_va, F_va, P_va)  # acceptability gate chosen on validation customers
    results["config"] = {**hp, "baseline_prior_standin": BASELINE_PRIOR, "rows_train": int(tr.sum()),
                         "rows_val": int(va.sum()), "rows_test": int(te.sum())}
    results["tuned_baseline"] = {"config": M.tuned.config(), "train_pair_top1": M.tuned_train_top1}
    results["logreg_top_coefficients"] = M.lr.coef_table()
    r_te, F_te, P_te = subset(te)
    pred_te = M.predict(r_te, F_te, P_te)
    results["A_customer_split_test"] = evaluate(r_te, pred_te, P_te)
    # per-scenario pair top-1 on test customers
    per = {}
    scen = r_te["sim_scenario"].iloc[P_te.starts].to_numpy()
    for name, r in pred_te.items():
        per[name] = {s: float(np.mean((r["chosen"] == P_te.truth)[scen == s])) for s in SCENARIOS if (scen == s).any()}
    results["A_per_scenario_top1_incl_none"] = per

    # B. unseen garments (double hold-out: unseen customers AND unseen garments)
    r_b, F_b, P_b = subset(tr & (R["unseen_garment"] == 0))
    Mb = Methods(r_b, F_b, P_b, hp)
    r_bt, F_bt, P_bt = subset(te & (R["unseen_garment"] == 1))
    results["B_unseen_garment_test"] = evaluate(r_bt, Mb.predict(r_bt, F_bt, P_bt), P_bt)

    # C. leave-one-style-out (train customers x other styles -> test customers x held-out style), pooled
    parts = []
    for st in STYLES:
        r_c, F_c, P_c = subset(tr & (R["style"] != st))
        Mc = Methods(r_c, F_c, P_c, hp)
        r_ct, F_ct, P_ct = subset(te & (R["style"] == st))
        parts.append((r_ct, Mc.predict(r_ct, F_ct, P_ct), P_ct))
    rp, pp = pool(parts)
    Pp = Pairs(rp)
    results["C_unseen_style_pooled_leave_one_style_out"] = evaluate(rp, pp, Pp)

    # D. unseen scenario: train on S1+S2 customers, test on S3 customers (test partition)
    r_d, F_d, P_d = subset(tr & (R["sim_scenario"] != SCENARIOS[2]))
    Md = Methods(r_d, F_d, P_d, hp)
    r_dt, F_dt, P_dt = subset(te & (R["sim_scenario"] == SCENARIOS[2]))
    results["D_unseen_scenario_S3_test"] = evaluate(r_dt, Md.predict(r_dt, F_dt, P_dt), P_dt)

    # E. stress-test set (never used for fitting), models from A
    Fs = build_features(Rs)
    Ps = Pairs(Rs)
    pred_s = M.predict(Rs, Fs, Ps)
    results["E_stress_overall"] = evaluate(Rs, pred_s, Ps)
    coh = Rs["stress_cohort_cust"].iloc[Ps.starts].to_numpy()
    gcoh = Rs["stress_cohort_garm"].iloc[Ps.starts].to_numpy()
    by = {}
    for label, arr in [("customer_cohort", coh), ("garment_cohort", gcoh)]:
        by[label] = {}
        for c in np.unique(arr):
            m = arr == c
            by[label][c] = {name: {"n_pairs": int(m.sum()), "top1_incl_none": float(np.mean((r["chosen"] == Ps.truth)[m])),
                                   "none_rate_true": float(np.mean(Ps.truth[m] < 0))} for name, r in pred_s.items()
                            if name in ("weighted_ease_prior", "weighted_ease_tuned", "logreg", "majority")}
    results["E_stress_by_cohort"] = by

    # leakage sanity: label-permutation control (train labels shuffled -> should collapse toward trivial accuracy)
    Msh = Methods(r_tr, F_tr, P_tr, hp, shuffle=seed + 1)
    sh = Msh.predict(r_te, F_te, P_te)["logreg"]
    results["label_permutation_control"] = {"logreg_top1_incl_none": float(np.mean(sh["chosen"] == P_te.truth)),
                                            "note": "should be near the majority/trivial level, far below the real model"}
    results["feature_columns_used"] = FEATURE_SOURCE_COLUMNS
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (out_dir / "baseline_config_standin.json").write_text(json.dumps(BASELINE_PRIOR, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = run(a.data, a.out, a.seed)
    for name, v in r["A_customer_split_test"].items():
        print(f"{name:22s} top1(incl none)={v['pair']['top1_incl_none']:.3f}  none-F1={v['pair']['none_acceptable']['f1']:.3f}")
