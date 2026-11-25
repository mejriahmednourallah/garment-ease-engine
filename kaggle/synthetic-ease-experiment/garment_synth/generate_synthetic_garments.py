#!/usr/bin/env python3
"""SYNTHETIC garment-ease dataset generator.

*** EVERYTHING PRODUCED HERE IS SIMULATED (SYNTHETIC). ***
It contains no real people, no scraped records and no data from any paper.
Every range, distribution and rule below is an UNVALIDATED simulation
assumption chosen for software development and stress-testing. High model
accuracy on these labels shows consistency with this simulator only.

Usage:
    python generate_synthetic_garments.py --out data --seed 20261124

Outputs (per set: <out>/main and <out>/stress):
    customers.csv, garments.csv, garment_sizes.csv, observations.csv   (model-visible)
    latent_*.csv                                                       (simulator internals)
    splits_customers.csv, splits_garments.csv                          (main only)
plus <out>/{MANIFEST.json, dataset_stats.json, assumptions.json,
            DATA_DICTIONARY.md, README_SYNTHETIC.txt}
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GENERATOR_VERSION = "1.0.0"
MARKER = "SYNTHETIC"
DIMS = ("chest", "waist", "hip")
SIZE_ORDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "3XL"]
BASE_LABEL_IDX = SIZE_ORDER.index("M")
PREFS = ["slim", "regular", "loose"]
FIT_LABELS = ["tight", "acceptable", "loose"]
SCENARIOS = ["S1_asymmetric_threshold", "S2_stretch_coupling", "S3_style_relative", "S4_ood_shifted"]
KIND_ID = {"main": 1, "stress": 2}
COMP_ID = {"customers": 1, "garments": 2, "pairs": 3, "labels": 4, "splits": 5}

# --------------------------------------------------------------------------
# UNVALIDATED SIMULATION ASSUMPTIONS (all values in centimetres unless stated)
# --------------------------------------------------------------------------
REGION_NAMES = ["region_A", "region_B", "region_C", "region_D", "region_E"]  # abstract; not real populations
REGION_WEIGHTS = [0.25, 0.20, 0.20, 0.20, 0.15]
REGION_PARAMS = {  # median chest/waist/hip (cm), correlations, stated-preference probs, perception shift
    "region_A": dict(mean=(96, 82, 99), c_cw=0.88, c_ch=0.78, c_wh=0.83, pref=(0.22, 0.53, 0.25), shift=0.0, arch=(0.55, 0.15, 0.15, 0.15)),
    "region_B": dict(mean=(92, 78, 96), c_cw=0.86, c_ch=0.76, c_wh=0.82, pref=(0.30, 0.50, 0.20), shift=0.8, arch=(0.50, 0.10, 0.30, 0.10)),
    "region_C": dict(mean=(100, 88, 104), c_cw=0.92, c_ch=0.8, c_wh=0.88, pref=(0.15, 0.50, 0.35), shift=-0.6, arch=(0.50, 0.30, 0.10, 0.10)),
    "region_D": dict(mean=(90, 74, 94), c_cw=0.84, c_ch=0.74, c_wh=0.8, pref=(0.28, 0.52, 0.20), shift=0.5, arch=(0.55, 0.10, 0.20, 0.15)),
    "region_E": dict(mean=(98, 84, 100), c_cw=0.88, c_ch=0.78, c_wh=0.84, pref=(0.20, 0.50, 0.30), shift=-0.3, arch=(0.45, 0.20, 0.15, 0.20)),
}
SD_LOG = np.array([0.085, 0.115, 0.080])           # log-space SD of chest/waist/hip
ARCHETYPES = ["balanced", "waist_heavy", "hip_heavy", "chest_heavy"]  # statistical mixture components, not body "types"
ARCH_OFFSET_CM = np.array([[0, 0, 0], [1, 4, -1], [-1, -1, 4], [2, -2, -1]], dtype=float)
BODY_BOUNDS = {"chest": (60.0, 160.0), "waist": (50.0, 160.0), "hip": (66.0, 170.0)}  # absolute positivity/plausibility bounds
RATIO_MAIN = dict(waist_chest=(0.55, 1.20), hip_chest=(0.75, 1.35))                   # enforced in main set only
MEASUREMENT_SD = 1.0                                                                 # cm, tape-measure noise (main)

GARMENT_TYPES = ["tshirt", "shirt", "hoodie", "jacket", "trousers", "skirt", "dress"]
TYPE_WEIGHTS = [0.18, 0.14, 0.10, 0.10, 0.20, 0.08, 0.20]
NAN = np.nan
MEASURED_DIMS = {"tshirt": (1, 1, 0), "shirt": (1, 1, 0), "hoodie": (1, 1, 0), "jacket": (1, 1, 0),
                 "trousers": (0, 1, 1), "skirt": (0, 1, 1), "dress": (1, 1, 1)}
# garment-minus-nominal-body design ease for a "straight" style (cm), before style multiplier
DESIGN_EASE = {"tshirt": (10, 10, NAN), "shirt": (12, 12, NAN), "hoodie": (16, 12, NAN), "jacket": (16, 14, NAN),
               "trousers": (NAN, 2, 8), "skirt": (NAN, 2, 6), "dress": (6, 6, 8)}
# wearer's desired ease for a "regular" preference (cm); personal preference model, NOT a standard
DESIRED_BASE = np.array([(8, 8, NAN), (10, 10, NAN), (14, 12, NAN), (14, 12, NAN), (NAN, 2, 6), (NAN, 2, 6), (6, 6, 8)],
                        dtype=float)  # rows follow GARMENT_TYPES order
PREF_MULT = np.array([0.45, 1.0, 1.7])
PREF_ADD = np.array([-0.5, 0.0, 1.0])
STYLES = ["fitted", "straight", "relaxed", "oversized"]
STYLE_MULT = {"fitted": 0.65, "straight": 1.0, "relaxed": 1.25, "oversized": 1.6}
STYLE_WEIGHTS = {"tshirt": (0.2, 0.4, 0.25, 0.15), "shirt": (0.25, 0.45, 0.25, 0.05), "hoodie": (0.0, 0.25, 0.45, 0.30),
                 "jacket": (0.15, 0.45, 0.30, 0.10), "trousers": (0.25, 0.40, 0.30, 0.05), "skirt": (0.3, 0.5, 0.2, 0.0),
                 "dress": (0.3, 0.35, 0.25, 0.10)}
FABRICS = ["woven", "stretch_woven", "knit", "rib_knit"]
FABRIC_STRETCH_RANGE = {"woven": (0, 4), "stretch_woven": (3, 14), "knit": (15, 55), "rib_knit": (40, 100)}  # % crosswise
FABRIC_WEIGHTS = {"tshirt": (0, 0, 0.8, 0.2), "shirt": (0.7, 0.3, 0, 0), "hoodie": (0, 0, 1, 0), "jacket": (0.6, 0.2, 0.2, 0),
                  "trousers": (0.5, 0.5, 0, 0), "skirt": (0.4, 0.3, 0.3, 0), "dress": (0.3, 0.2, 0.4, 0.1)}
FABRIC_RIGIDITY = {"woven": 0.8, "stretch_woven": 0.5, "knit": 0.25, "rib_knit": 0.1}
STRETCH_CUT_REDUCTION = 0.08   # designers remove 0.08 cm of cut ease per 1% stretch per 100 cm of body (assumption)
GRADE_STEP_MEAN = np.array([5.0, 5.5, 5.0])   # cm per size step for chest/waist/hip
GARMENT_TOLERANCE_SD = 0.8     # cm, finished-garment manufacturing deviation from chart value
SCEN_STRETCH_MULT = np.array([0.4, 1.5, 1.0, 1.0])  # how much of the stretch is "usable" per scenario
S4_DESIRED_SHIFT = 2.5
LABEL_FLIP = {"main": 0.02, "S4": 0.04}


@dataclass
class GenConfig:
    seed: int = 20261124
    n_customers: int = 4000
    n_garments: int = 160
    pairs_per_customer: int = 6
    stress_customers: int = 600
    stress_garments: int = 40
    stress_pairs_per_customer: int = 4
    scenario_weights: tuple = (0.4, 0.3, 0.3)        # S1,S2,S3 in the main set; S4 is stress-only
    split_fractions: tuple = (0.70, 0.15, 0.15)
    unseen_garment_fraction: float = 0.20


PRESETS = {
    "small": dict(n_customers=400, n_garments=60, pairs_per_customer=5, stress_customers=120, stress_garments=24,
                  stress_pairs_per_customer=3),
}


def rng_for(seed, kind, comp):
    return np.random.default_rng(np.random.SeedSequence([seed, KIND_ID[kind], COMP_ID[comp]]))


# --------------------------------------------------------------------------
# Customers
# --------------------------------------------------------------------------
def sample_bodies(rng, region_idx, shift_log=0.0, sd_mult=1.0, corr_shrink=0.0, tail_frac=0.0):
    n = len(region_idx)
    out = np.empty((n, 3))
    arch = np.zeros(n, dtype=int)
    for r, name in enumerate(REGION_NAMES):
        m = np.where(region_idx == r)[0]
        if len(m) == 0:
            continue
        p = REGION_PARAMS[name]
        corr = np.array([[1, p["c_cw"], p["c_ch"]], [p["c_cw"], 1, p["c_wh"]], [p["c_ch"], p["c_wh"], 1]])
        corr = (1 - corr_shrink) * corr + corr_shrink * np.eye(3)
        sd = SD_LOG * sd_mult
        cov = np.outer(sd, sd) * corr
        z = rng.multivariate_normal(np.zeros(3), cov, size=len(m))
        if tail_frac > 0:
            heavy = rng.random(len(m)) < tail_frac
            z[heavy] *= 1.8
        a = rng.choice(4, size=len(m), p=p["arch"])
        out[m] = np.exp(np.log(p["mean"]) + shift_log + z) + ARCH_OFFSET_CM[a]
        arch[m] = a
    return out, arch


def clip_bodies(b, enforce_ratio=True):
    b = b.copy()
    for j, d in enumerate(DIMS):
        b[:, j] = np.clip(b[:, j], *BODY_BOUNDS[d])
    if enforce_ratio:
        b[:, 1] = np.clip(b[:, 1], RATIO_MAIN["waist_chest"][0] * b[:, 0], RATIO_MAIN["waist_chest"][1] * b[:, 0])
        b[:, 2] = np.clip(b[:, 2], RATIO_MAIN["hip_chest"][0] * b[:, 0], RATIO_MAIN["hip_chest"][1] * b[:, 0])
    return b


STRESS_CUSTOMER_COHORTS = [("ood_population", 0.15), ("unusual_preference", 0.10), ("extreme_size_small", 0.08),
                           ("extreme_size_large", 0.08), ("extreme_proportion", 0.08), ("missing_chest", 0.08),
                           ("missing_waist", 0.10), ("missing_hip", 0.08), ("missing_multiple", 0.10),
                           ("noisy_measurement", 0.08), ("all_body_missing", 0.02)]  # remainder = control
S4_COHORTS = {"ood_population", "unusual_preference", "extreme_size_small", "extreme_size_large"}


def gen_customers(rng, n, kind, cfg):
    region = rng.choice(5, size=n, p=REGION_WEIGHTS)
    cohort = np.array(["none"] * n, dtype=object)
    if kind == "stress":
        pos = 0
        order = rng.permutation(n)
        for name, frac in STRESS_CUSTOMER_COHORTS:
            k = int(frac * n)
            cohort[order[pos:pos + k]] = name
            pos += k
        cohort[order[pos:]] = "control"
    body = np.empty((n, 3))
    arch = np.zeros(n, dtype=int)
    for coh in np.unique(cohort):
        idx = np.where(cohort == coh)[0]
        if coh == "ood_population":
            b, a = sample_bodies(rng, region[idx], shift_log=0.06, sd_mult=1.35, corr_shrink=0.35, tail_frac=0.08)
            b = clip_bodies(b, enforce_ratio=False)
        elif coh == "extreme_size_small":
            c = rng.uniform(62, 72, len(idx))
            b = np.column_stack([c, c * rng.uniform(0.8, 1.0, len(idx)), c * rng.uniform(1.0, 1.15, len(idx))])
            b, a = clip_bodies(b, False), np.zeros(len(idx), int)
        elif coh == "extreme_size_large":
            c = rng.uniform(135, 158, len(idx))
            b = np.column_stack([c, c * rng.uniform(0.85, 1.15, len(idx)), c * rng.uniform(0.95, 1.15, len(idx))])
            b, a = clip_bodies(b, False), np.zeros(len(idx), int)
        elif coh == "extreme_proportion":
            c, _ = sample_bodies(rng, region[idx])
            c = c[:, 0]
            low = rng.random(len(idx)) < 0.5
            w = np.where(low, c * rng.uniform(0.56, 0.66, len(idx)), c * rng.uniform(1.10, 1.30, len(idx)))
            h = np.where(low, c * rng.uniform(1.05, 1.30, len(idx)), c * rng.uniform(0.85, 1.0, len(idx)))
            b, a = clip_bodies(np.column_stack([c, w, h]), False), np.zeros(len(idx), int)
        else:
            b, a = sample_bodies(rng, region[idx])
            b = clip_bodies(b, True)
        body[idx], arch[idx] = b, a
    # permute order so customer IDs carry no cohort/feature information
    perm = rng.permutation(n)
    region, cohort, body, arch = region[perm], cohort[perm], body[perm], arch[perm]

    # scenario per customer (labelling regime)
    scen = rng.choice(3, size=n, p=np.array(cfg.scenario_weights) / np.sum(cfg.scenario_weights))
    if kind == "stress":
        scen[np.isin(cohort, list(S4_COHORTS))] = 3

    # stated vs behavioural preference
    stated = np.array([rng.choice(3, p=REGION_PARAMS[REGION_NAMES[r]]["pref"]) for r in region])
    behav = stated.copy()
    drift = rng.random(n) < 0.12
    behav[drift] = np.clip(behav[drift] + rng.choice([-1, 1], size=drift.sum()), 0, 2)

    unusual = cohort == "unusual_preference"
    offset = rng.normal(0, np.where(unusual, 4.0, 1.5)[:, None], (n, 3))
    tol_base = np.clip(rng.normal(np.where(unusual, 2.8, 6.2), 1.3), 2.0, 10.0)
    dim_f = np.array([1.0, 1.0, 1.0])
    tol_t = tol_base[:, None] * dim_f * np.exp(rng.normal(0, 0.15, (n, 3)))
    tol_l = tol_t * np.exp(rng.normal(np.log(1.35), 0.25, (n, 1)))
    weights = rng.dirichlet([4, 4, 4], size=n) * 3
    lat = dict(region=region, cohort=cohort, true_body=body, arch=arch, scenario=scen, stated=stated, latent_pref=behav,
               pref_strength=np.exp(rng.normal(0, 0.15, n)), offset=offset, tol_t=tol_t, tol_l=tol_l, weights=weights,
               a_tight=np.exp(rng.normal(np.log(1.6), 0.3, n)), a_loose=np.exp(rng.normal(np.log(0.7), 0.3, n)),
               compress=np.clip(rng.normal(0.015, 0.01, n), 0, 0.04), noise_sd=rng.uniform(0.5, 1.3, n),
               bias=rng.normal(0, 0.5, (n, 3)), tau=rng.uniform(0.4, 1.4, n), style_pref=rng.normal(0.5, 0.4, n))

    # observed (reported) body measurements: noisy, rounded to 0.5 cm, then masked for stress cohorts
    sd = np.where(cohort == "noisy_measurement", 3.0, MEASUREMENT_SD)[:, None]
    obs = np.round((body + rng.normal(0, 1, (n, 3)) * sd) * 2) / 2
    for j, d in enumerate(DIMS):
        obs[:, j] = np.clip(obs[:, j], 30, None)
    for j, d in enumerate(DIMS):
        obs[cohort == f"missing_{d}", j] = np.nan
    multi = np.where(cohort == "missing_multiple")[0]
    for i in multi:
        keep = rng.integers(0, 3)
        for j in range(3):
            if j != keep:
                obs[i, j] = np.nan
    obs[cohort == "all_body_missing"] = np.nan

    prefix = "C" if kind == "main" else "S"
    ids = [f"{prefix}{i + 1:05d}" for i in range(n)]
    customers = pd.DataFrame({"synthetic_marker": MARKER, "customer_id": ids,
                              "sim_region": [REGION_NAMES[r] for r in region],
                              "chest_cm": obs[:, 0], "waist_cm": obs[:, 1], "hip_cm": obs[:, 2],
                              "fit_preference": [PREFS[p] for p in stated], "stress_cohort": cohort})
    latent = pd.DataFrame({"synthetic_marker": MARKER, "customer_id": ids,
                           "sim_scenario": [SCENARIOS[s] for s in scen], "body_archetype": [ARCHETYPES[a] for a in arch],
                           "true_chest_cm": body[:, 0], "true_waist_cm": body[:, 1], "true_hip_cm": body[:, 2],
                           "behavioural_preference": [PREFS[p] for p in behav], "preference_strength": lat["pref_strength"],
                           "style_preference": lat["style_pref"], "soft_tissue_compression": lat["compress"],
                           "perception_noise_sd_cm": lat["noise_sd"], "choice_temperature": lat["tau"],
                           "asym_tight_coef": lat["a_tight"], "asym_loose_coef": lat["a_loose"]})
    for j, d in enumerate(DIMS):
        latent[f"desired_offset_{d}_cm"] = offset[:, j]
        latent[f"tol_tight_{d}_cm"] = tol_t[:, j]
        latent[f"tol_loose_{d}_cm"] = tol_l[:, j]
        latent[f"importance_{d}"] = weights[:, j]
        latent[f"perception_bias_{d}_cm"] = lat["bias"][:, j]
    return customers, latent, lat


# --------------------------------------------------------------------------
# Garments and size charts
# --------------------------------------------------------------------------
STRESS_GARMENT_COHORTS = [("extreme_stretch", 0.15), ("missing_stretch", 0.10), ("sparse_chart", 0.15),
                          ("missing_garment_measure", 0.15), ("very_loose_cut", 0.10), ("very_fitted_cut", 0.10)]


def gen_garments(rng, n, kind):
    tidx = rng.choice(len(GARMENT_TYPES), size=n, p=TYPE_WEIGHTS)
    types = [GARMENT_TYPES[t] for t in tidx]
    styles = [STYLES[rng.choice(4, p=STYLE_WEIGHTS[t])] for t in types]
    fabrics = [FABRICS[rng.choice(4, p=FABRIC_WEIGHTS[t])] for t in types]
    stretch = np.array([rng.uniform(*FABRIC_STRETCH_RANGE[f]) for f in fabrics]).round(1)
    target = rng.choice(5, size=n, p=REGION_WEIGHTS)
    n_sizes = rng.choice([4, 5, 6, 7], size=n, p=[0.15, 0.35, 0.35, 0.15])
    cohort = np.array(["none"] * n, dtype=object)
    style_mult = np.array([STYLE_MULT[s] for s in styles])
    step_mult = np.ones(n)
    if kind == "stress":
        order = rng.permutation(n)
        pos = 0
        for name, frac in STRESS_GARMENT_COHORTS:
            k = max(1, int(frac * n))
            cohort[order[pos:pos + k]] = name
            pos += k
        cohort[order[pos:]] = "control"
        m = cohort == "extreme_stretch"
        stretch[m] = rng.uniform(100, 180, m.sum()).round(1)
        for i in np.where(m)[0]:
            fabrics[i] = "rib_knit"
        m = cohort == "sparse_chart"
        n_sizes[m], step_mult[m] = 3, 1.8
        style_mult[cohort == "very_loose_cut"] = 2.4
        style_mult[cohort == "very_fitted_cut"] = 0.1
    rigid = np.clip(np.array([FABRIC_RIGIDITY[f] for f in fabrics]) + rng.normal(0, 0.08, n), 0, 1)
    usable = rng.beta(2, 6, n)
    widening = rng.random(n) < 0.5
    prefix = "G" if kind == "main" else "X"
    gids = [f"{prefix}{i + 1:04d}" for i in range(n)]

    size_rows, lat_rows = [], []
    brand_off = rng.normal(0, 1.0, (n, 3))
    garment_off = rng.normal(0, 1.0, (n, 3))
    for g in range(n):
        t = types[g]
        md = np.array(MEASURED_DIMS[t], dtype=bool)
        ns = int(n_sizes[g])
        below = int(np.clip(np.round((ns - 1) / 2 + rng.normal(0, 0.6)), 0, ns - 1))  # sizes below 'M'; charts roughly centred on M
        start = int(np.clip(BASE_LABEL_IDX - below, 0, len(SIZE_ORDER) - ns))
        start = min(start, BASE_LABEL_IDX)  # 'M' always present
        start = max(start, BASE_LABEL_IDX - ns + 1)
        labels = SIZE_ORDER[start:start + ns]
        rel = np.arange(start, start + ns) - BASE_LABEL_IDX
        mean_reg = np.array(REGION_PARAMS[REGION_NAMES[target[g]]]["mean"], dtype=float)
        base_body = mean_reg + brand_off[g]
        step = np.maximum(3.0, GRADE_STEP_MEAN + rng.normal(0, 0.5, 3)) * step_mult[g]
        w = 0.08 if widening[g] else 0.0
        label_body = np.empty((ns, 3))
        for k, r in enumerate(rel):
            if r >= 0:
                cum = sum(step * (1 + w * (j - 1)) for j in range(1, r + 1)) if r > 0 else np.zeros(3)
            else:
                cum = -sum(np.maximum(3.0, step * (1 - 0.5 * w * (j - 1))) for j in range(1, -r + 1))
            label_body[k] = base_body + cum
        de = np.array(DESIGN_EASE[t], dtype=float) * style_mult[g] + garment_off[g]
        reduction = STRETCH_CUT_REDUCTION * stretch[g] * (label_body / 100.0)
        design = label_body + de - reduction
        lb_r = np.round(label_body * 2) / 2
        chart = np.round(design * 2) / 2
        actual = chart + rng.normal(0, GARMENT_TOLERANCE_SD, chart.shape)
        for arr in (lb_r, chart, actual):
            arr[:, ~md] = np.nan
        for k in range(ns):
            size_rows.append([gids[g], labels[k], int(rel[k] + BASE_LABEL_IDX), *lb_r[k], *chart[k]])
            lat_rows.append([gids[g], labels[k], *actual[k]])
    sizes = pd.DataFrame(size_rows, columns=["garment_id", "size_label", "size_rank"] +
                         [f"label_body_{d}_cm" for d in DIMS] + [f"garment_{d}_cm" for d in DIMS])
    sizes.insert(0, "synthetic_marker", MARKER)
    lat_sizes = pd.DataFrame(lat_rows, columns=["garment_id", "size_label"] + [f"actual_garment_{d}_cm" for d in DIMS])
    lat_sizes.insert(0, "synthetic_marker", MARKER)

    # observation masks (stress): missing garment measurements, missing stretch
    sz_idx = np.arange(len(sizes))
    gcoh = dict(zip(gids, cohort))
    miss_rows = sizes["garment_id"].map(gcoh).to_numpy() == "missing_garment_measure"
    mask_rng = rng.random((len(sizes), 3)) < 0.25
    for j, d in enumerate(DIMS):
        col = f"garment_{d}_cm"
        sizes.loc[miss_rows & mask_rng[:, j], col] = np.nan
    stretch_obs = stretch.copy()
    stretch_obs[cohort == "missing_stretch"] = np.nan

    garments = pd.DataFrame({"synthetic_marker": MARKER, "garment_id": gids, "garment_type": types, "garment_style": styles,
                             "fabric_class": fabrics, "stretch_crosswise_pct": stretch_obs, "stretch_direction": "crosswise",
                             "chart_target_region": [REGION_NAMES[t] for t in target], "n_sizes": n_sizes,
                             "measured_dims": ["|".join(d for d, f in zip(DIMS, MEASURED_DIMS[t]) if f) for t in types],
                             "stress_cohort": cohort})
    lat_g = pd.DataFrame({"synthetic_marker": MARKER, "garment_id": gids, "true_stretch_crosswise_pct": stretch,
                          "usable_stretch_fraction": usable, "fabric_rigidity": rigid, "design_style_multiplier": style_mult,
                          "grading_widening": widening.astype(int)})
    G = dict(type_idx=np.array([GARMENT_TYPES.index(t) for t in types]), stretch_true=stretch, usable=usable, rigid=rigid,
             style_mult=style_mult)
    return garments, sizes, lat_g, lat_sizes, G


# --------------------------------------------------------------------------
# Pairs, labels, preferred size
# --------------------------------------------------------------------------
def simulate_observations(rng_pairs, rng_lab, kind, cfg, C, G, customers, garments, sizes, lat_sizes, k_pairs):
    n_c, n_g = len(customers), len(garments)
    k = min(k_pairs, n_g)
    greg = np.array([REGION_NAMES.index(r) for r in garments["chart_target_region"]])
    pairs = []
    for c in range(n_c):
        w = np.where(greg == C["region"][c], 2.5, 1.0)
        g = rng_pairs.choice(n_g, size=k, replace=False, p=w / w.sum())
        pairs.append(np.column_stack([np.full(k, c), g]))
    pairs = np.vstack(pairs)
    P = len(pairs)
    gs_start = sizes.groupby("garment_id", sort=False).size().cumsum().shift(fill_value=0).reindex(garments["garment_id"]).to_numpy()
    gs_count = garments["n_sizes"].to_numpy()
    counts = gs_count[pairs[:, 1]]
    prep = np.repeat(np.arange(P), counts)
    starts = np.cumsum(counts) - counts
    offs = np.arange(counts.sum()) - np.repeat(starts, counts)
    sidx = gs_start[pairs[:, 1]][prep] + offs
    cidx, gidx = pairs[prep, 0], pairs[prep, 1]
    N = len(prep)

    body = C["true_body"][cidx]
    act = lat_sizes[[f"actual_garment_{d}_cm" for d in DIMS]].to_numpy()[sidx]
    app = ~np.isnan(act)
    act0 = np.where(app, act, 0.0)
    scen = C["scenario"][cidx]
    geo = np.where(app, act0 - body * (1 - C["compress"][cidx, None]), 0.0)
    smult = SCEN_STRETCH_MULT[scen] * G["usable"][gidx] * G["stretch_true"][gidx] / 100.0
    credit = np.where(app, np.minimum(smult[:, None] * act0, np.maximum(0, -geo)), 0.0)
    perceived = geo + credit

    tidx, lp = G["type_idx"][gidx], C["latent_pref"][cidx]
    des = DESIRED_BASE[tidx] * PREF_MULT[lp][:, None] + PREF_ADD[lp][:, None]
    des = des * C["pref_strength"][cidx][:, None] + C["offset"][cidx]
    des = np.where(app, des, 0.0)
    style_scale = np.where(scen == 2, 12.0, 5.0)  # everyone leans toward the garment's intended cut; S3 leans harder
    des += (C["style_pref"][cidx] * (G["style_mult"][gidx] - 1) * style_scale)[:, None]
    des += np.where((scen == 3)[:, None], S4_DESIRED_SHIFT, 0.0)
    des = np.where(app, des, 0.0)

    tt, tl = C["tol_t"][cidx].copy(), C["tol_l"][cidx].copy()
    rig = G["rigid"][gidx][:, None]
    tt = np.where((scen == 1)[:, None], tt * (1 - 0.5 * rig), tt)
    dpos = np.maximum(des, 0)
    tt = np.where((scen == 2)[:, None], 0.7 * tt + 0.15 * dpos, tt)
    tl = np.where((scen == 2)[:, None], 0.7 * tl + 0.25 * dpos, tl)
    tt = np.where((scen == 3)[:, None], tt * 1.5, tt)
    tl = np.where((scen == 3)[:, None], tl * 1.5, tl)

    sd = C["noise_sd"][cidx][:, None] * np.where((scen == 1)[:, None], 1.2, 1.0)
    eps = rng_lab.standard_normal((N, 3)) * sd
    heavy = rng_lab.standard_t(3, (N, 3)) / np.sqrt(3.0) * sd * 1.5
    eps = np.where((scen == 3)[:, None], heavy, eps)
    reg_shift = np.array([REGION_PARAMS[r]["shift"] for r in REGION_NAMES])[C["region"][cidx]] * (scen == 2)
    z = perceived + C["bias"][cidx] + eps + reg_shift[:, None]
    s2 = (scen == 1) & app[:, 1] & app[:, 2]
    z[:, 1] -= np.where(s2, 0.3 * np.maximum(0, des[:, 2] - z[:, 2]), 0.0)

    lab = np.where(z < des - tt, 0, np.where(z > des + tl, 2, 1))
    lab = np.where(app, lab, -1)
    fp = np.where(scen == 3, LABEL_FLIP["S4"], LABEL_FLIP["main"])
    flip = (rng_lab.random((N, 3)) < fp[:, None]) & app
    alt = np.where(rng_lab.random((N, 3)) < 0.5, 0, 2)
    lab = np.where(flip, np.where(lab == 1, alt, 1), lab)
    overall = np.all((lab == 1) | ~app, axis=1)

    tight = np.where(app, np.maximum(0, des - z), 0.0)
    loose = np.where(app, np.maximum(0, z - des), 0.0)
    w = C["weights"][cidx]
    at, al = C["a_tight"][cidx][:, None], C["a_loose"][cidx][:, None]
    D1 = (w * (at * tight ** 2 + al * loose)).sum(1)
    D2 = (w * (at * tight ** 2 + al * loose ** 2)).sum(1) + 0.5 * G["rigid"][gidx] * tight.sum(1)
    ph = lambda x: 9.0 * (np.sqrt(1 + (x / 3.0) ** 2) - 1)
    D3 = (w * (at * ph(tight) + al * ph(loose))).sum(1)
    D = np.select([scen == 0, scen == 1, scen == 2, scen == 3], [D1, D2, D3, D1])
    U = -D + C["tau"][cidx] * rng_lab.gumbel(size=N)

    Um = np.where(overall, U, -np.inf)
    pmax = np.maximum.reduceat(Um, starts)
    pmax_rep = np.repeat(pmax, counts)
    preferred = overall & (Um == pmax_rep)
    none_pair = np.isinf(pmax)
    pref_label = np.full(P, "NONE", dtype=object)
    sl = sizes["size_label"].to_numpy()
    pi = np.where(preferred)[0]
    pref_label[prep[pi]] = sl[sidx[pi]]

    lab_str = lambda a: np.where(a < 0, "not_applicable", np.array(FIT_LABELS)[np.clip(a, 0, 2)])
    oid = [f"O{i + 1:07d}" for i in range(N)]
    pid = np.array([f"P{i + 1:06d}" for i in range(P)])
    obs = pd.DataFrame({"synthetic_marker": MARKER, "observation_id": oid, "pair_id": pid[prep],
                        "customer_id": customers["customer_id"].to_numpy()[cidx],
                        "garment_id": garments["garment_id"].to_numpy()[gidx], "size_label": sl[sidx],
                        "fit_chest": lab_str(lab[:, 0]), "fit_waist": lab_str(lab[:, 1]), "fit_hip": lab_str(lab[:, 2]),
                        "fit_overall_acceptable": overall.astype(int), "is_preferred_size": preferred.astype(int),
                        "pair_preferred_size": pref_label[prep], "pair_none_acceptable": none_pair[prep].astype(int)})
    lat = pd.DataFrame({"synthetic_marker": MARKER, "observation_id": oid, "pair_id": pid[prep],
                        "sim_scenario": np.array(SCENARIOS)[scen], "discomfort": D, "utility": U})
    for j, d in enumerate(DIMS):
        lat[f"geometric_ease_{d}_cm"] = np.where(app[:, j], geo[:, j], np.nan)
        lat[f"stretch_credit_{d}_cm"] = np.where(app[:, j], credit[:, j], np.nan)
        lat[f"desired_ease_{d}_cm"] = np.where(app[:, j], des[:, j], np.nan)
        lat[f"perceived_ease_{d}_cm"] = np.where(app[:, j], z[:, j], np.nan)
        lat[f"label_flipped_{d}"] = (flip[:, j]).astype(int)
    return obs, lat


# --------------------------------------------------------------------------
# Bundle, splits, stats
# --------------------------------------------------------------------------
def generate_set(cfg, kind):
    n_c = cfg.n_customers if kind == "main" else cfg.stress_customers
    n_g = cfg.n_garments if kind == "main" else cfg.stress_garments
    k = cfg.pairs_per_customer if kind == "main" else cfg.stress_pairs_per_customer
    customers, lat_c, C = gen_customers(rng_for(cfg.seed, kind, "customers"), n_c, kind, cfg)
    garments, sizes, lat_g, lat_s, G = gen_garments(rng_for(cfg.seed, kind, "garments"), n_g, kind)
    obs, lat_o = simulate_observations(rng_for(cfg.seed, kind, "pairs"), rng_for(cfg.seed, kind, "labels"), kind, cfg,
                                       C, G, customers, garments, sizes, lat_s, k)
    out = {"customers": customers, "garments": garments, "garment_sizes": sizes, "observations": obs,
           "latent_customers": lat_c, "latent_garments": lat_g, "latent_garment_sizes": lat_s, "latent_observations": lat_o}
    if kind == "main":
        rng = rng_for(cfg.seed, kind, "splits")
        perm = rng.permutation(n_c)
        n_tr, n_va = int(cfg.split_fractions[0] * n_c), int(cfg.split_fractions[1] * n_c)
        part = np.empty(n_c, dtype=object)
        part[perm[:n_tr]], part[perm[n_tr:n_tr + n_va]], part[perm[n_tr + n_va:]] = "train", "val", "test"
        out["splits_customers"] = pd.DataFrame({"synthetic_marker": MARKER, "customer_id": customers["customer_id"],
                                                "partition": part})
        hold = np.zeros(n_g, dtype=int)
        hold[rng.permutation(n_g)[:int(cfg.unseen_garment_fraction * n_g)]] = 1
        out["splits_garments"] = pd.DataFrame({"synthetic_marker": MARKER, "garment_id": garments["garment_id"],
                                               "garment_style": garments["garment_style"], "holdout_unseen_garment": hold})
    return out


def dataset_stats(sets):
    stats = {}
    for kind, d in sets.items():
        o, lo, c, g, s = d["observations"], d["latent_observations"], d["customers"], d["garments"], d["garment_sizes"]
        pairs = o.drop_duplicates("pair_id")
        st = {"n_customers": len(c), "n_garments": len(g), "n_garment_size_rows": len(s), "n_pairs": len(pairs),
              "n_observations": len(o), "mean_sizes_per_pair": round(len(o) / len(pairs), 2)}
        for dim in DIMS:
            vc = o[f"fit_{dim}"].value_counts()
            st[f"label_counts_{dim}"] = {k: int(v) for k, v in vc.items()}
        st["pair_none_acceptable_rate"] = round(float(pairs["pair_none_acceptable"].mean()), 4)
        sc = lo.drop_duplicates("pair_id").set_index("pair_id")["sim_scenario"]
        pn = pairs.set_index("pair_id")["pair_none_acceptable"]
        st["none_rate_by_scenario"] = {k: round(float(v), 4) for k, v in pn.groupby(sc.reindex(pn.index)).mean().items()}
        st["pairs_by_scenario"] = {k: int(v) for k, v in sc.value_counts().items()}
        st["acceptable_sizes_per_pair_mean"] = round(float(o.groupby("pair_id")["fit_overall_acceptable"].sum().mean()), 3)
        st["preferred_size_label_counts"] = {k: int(v) for k, v in pairs["pair_preferred_size"].value_counts().items()}
        st["missing_rate_customers"] = {col: round(float(c[col].isna().mean()), 4) for col in ["chest_cm", "waist_cm", "hip_cm"]}
        st["missing_rate_garment_cells"] = {col: round(float(s[col].isna().mean()), 4) for col in
                                            [f"garment_{x}_cm" for x in DIMS]}
        st["customers_by_region"] = {k: int(v) for k, v in c["sim_region"].value_counts().items()}
        st["customers_by_stress_cohort"] = {k: int(v) for k, v in c["stress_cohort"].value_counts().items()}
        st["garments_by_type"] = {k: int(v) for k, v in g["garment_type"].value_counts().items()}
        st["garments_by_style"] = {k: int(v) for k, v in g["garment_style"].value_counts().items()}
        stats[kind] = st
    return stats


COLUMNS = {  # (column, dtype, role, unit, description)
    "customers": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("customer_id", "str", "key", "", "Random-order synthetic ID; C=main, S=stress. Carries no information. Never a model feature."),
        ("sim_region", "str", "feature", "", "Abstract simulated region A-E. NOT a real geography/population."),
        ("chest_cm", "float", "feature", "cm", "Reported chest circumference (true value + tape noise, 0.5 cm rounding). Blank = missing (stress set)."),
        ("waist_cm", "float", "feature", "cm", "Reported waist circumference; same conventions."),
        ("hip_cm", "float", "feature", "cm", "Reported hip circumference; same conventions."),
        ("fit_preference", "str", "feature", "", "Stated preference slim/regular/loose. May differ from behavioural preference (latent)."),
        ("stress_cohort", "str", "evaluation_metadata", "", "Stress-test cohort tag ('none' in main). Simulator metadata: NOT a model feature."),
    ],
    "garments": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("garment_id", "str", "key", "", "G=main, X=stress. Not a model feature."),
        ("garment_type", "str", "feature", "", "tshirt, shirt, hoodie, jacket, trousers, skirt, dress."),
        ("garment_style", "str", "feature", "", "Cut intent: fitted, straight, relaxed, oversized."),
        ("fabric_class", "str", "feature", "", "woven, stretch_woven, knit, rib_knit (simulation categories)."),
        ("stretch_crosswise_pct", "float", "feature", "%", "Simulated fabric extensibility: (stretched-relaxed)/relaxed*100, crosswise (circumferential) direction, under a notional fixed lab load, measured on fabric before construction. A material property, NOT on-body behaviour: ignores recovery/growth, seams, hems, load level, body shape and wearer tolerance. Blank = missing (stress)."),
        ("stretch_direction", "str", "feature", "", "Constant 'crosswise'. Lengthwise stretch is not modelled."),
        ("chart_target_region", "str", "feature", "", "Simulated region whose body distribution the size chart was drafted for."),
        ("n_sizes", "int", "feature", "", "Number of sizes in the chart."),
        ("measured_dims", "str", "feature", "", "Pipe-separated body regions the garment measures (e.g. 'chest|waist')."),
        ("stress_cohort", "str", "evaluation_metadata", "", "Stress-test garment cohort ('none' in main). NOT a model feature."),
    ],
    "garment_sizes": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("garment_id", "str", "key", "", "FK to garments."),
        ("size_label", "str", "key", "", "Ordered label (XXS..3XL subset, contiguous, always includes 'M')."),
        ("size_rank", "int", "feature", "", "Position in global label order (XXS=0..3XL=7); strictly increasing within a garment."),
        ("label_body_chest_cm", "float", "feature", "cm", "Retailer's nominal BODY measurement the size is labelled for. Not a garment measurement."),
        ("label_body_waist_cm", "float", "feature", "cm", "As above, waist."),
        ("label_body_hip_cm", "float", "feature", "cm", "As above, hip."),
        ("garment_chest_cm", "float", "feature", "cm", "Chart finished-garment full circumference, laid relaxed and unstretched (full circumference, not half-width). Blank = n/a for this garment type or missing (stress)."),
        ("garment_waist_cm", "float", "feature", "cm", "As above, waist (tops: hem circumference)."),
        ("garment_hip_cm", "float", "feature", "cm", "As above, hip."),
    ],
    "observations": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("observation_id", "str", "key", "", "One row per customer x garment x candidate size."),
        ("pair_id", "str", "key", "", "Customer x garment pair. Candidate sizes of a pair share this ID."),
        ("customer_id", "str", "key", "", "FK to customers (split unit)."),
        ("garment_id", "str", "key", "", "FK to garments."),
        ("size_label", "str", "key", "", "FK to garment_sizes."),
        ("fit_chest", "str", "label", "", "Simulated chest-region fit: tight/acceptable/loose/not_applicable."),
        ("fit_waist", "str", "label", "", "Simulated waist-region fit."),
        ("fit_hip", "str", "label", "", "Simulated hip-region fit."),
        ("fit_overall_acceptable", "int", "label", "", "1 iff every applicable region label is 'acceptable'."),
        ("is_preferred_size", "int", "label", "", "1 for the simulated preferred size of the pair (max latent utility among acceptable sizes)."),
        ("pair_preferred_size", "str", "label", "", "Preferred size label for the pair, or 'NONE' if no candidate is acceptable."),
        ("pair_none_acceptable", "int", "label", "", "1 iff the pair has no acceptable candidate size."),
    ],
    "splits_customers": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("customer_id", "str", "key", "", "FK to customers."),
        ("partition", "str", "split", "", "train/val/test; assigned by customer before any preprocessing."),
    ],
    "splits_garments": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("garment_id", "str", "key", "", "FK to garments."),
        ("garment_style", "str", "split", "", "Style, used for leave-one-style-out evaluation."),
        ("holdout_unseen_garment", "int", "split", "", "1 = garment reserved for the unseen-garment evaluation."),
    ],
    "latent_customers": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("customer_id", "str", "key", "", "FK to customers."),
        ("sim_scenario", "str", "latent", "", "Labelling regime S1..S4 for this customer."),
        ("body_archetype", "str", "latent", "", "Mixture component used to sample the body (statistical device only)."),
        ("true_chest_cm", "float", "latent", "cm", "True chest (before measurement noise/missingness)."),
        ("true_waist_cm", "float", "latent", "cm", "True waist."),
        ("true_hip_cm", "float", "latent", "cm", "True hip."),
        ("behavioural_preference", "str", "latent", "", "Preference actually driving simulated labels."),
        ("preference_strength", "float", "latent", "", "Multiplier on base desired ease."),
        ("style_preference", "float", "latent", "", "Used by Pull of desired ease toward the garment's intended cut (scale 5 cm in S1/S2/S4, 12 cm in S3)."),
        ("soft_tissue_compression", "float", "latent", "fraction", "Fractional compressibility of body circumference."),
        ("perception_noise_sd_cm", "float", "latent", "cm", "SD of per-observation perception noise."),
        ("choice_temperature", "float", "latent", "", "Scale of Gumbel noise in size choice."),
        ("asym_tight_coef", "float", "latent", "", "Discomfort coefficient on tightness."),
        ("asym_loose_coef", "float", "latent", "", "Discomfort coefficient on looseness."),
    ] + [(f"{p}_{d}{s}", "float", "latent", u, t) for d in DIMS for p, s, u, t in [
        ("desired_offset", "_cm", "cm", "Customer offset to desired ease."),
        ("tol_tight", "_cm", "cm", "Tolerance below desired ease before 'tight'."),
        ("tol_loose", "_cm", "cm", "Tolerance above desired ease before 'loose'."),
        ("importance", "", "", "Customer-specific region importance weight (mean 1)."),
        ("perception_bias", "_cm", "cm", "Systematic perception bias.")]],
    "latent_garments": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("garment_id", "str", "key", "", "FK to garments."),
        ("true_stretch_crosswise_pct", "float", "latent", "%", "Stretch before stress-set masking."),
        ("usable_stretch_fraction", "float", "latent", "fraction", "Share of stretch usable on a body; independent of stretch %."),
        ("fabric_rigidity", "float", "latent", "0-1", "Drives tightness intolerance in S1."),
        ("design_style_multiplier", "float", "latent", "", "Style cut multiplier (stress cohorts override)."),
        ("grading_widening", "int", "latent", "", "1 if grade steps widen toward larger sizes."),
    ],
    "latent_garment_sizes": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("garment_id", "str", "key", "", "FK to garments."),
        ("size_label", "str", "key", "", "FK to garment_sizes."),
    ] + [(f"actual_garment_{d}_cm", "float", "latent", "cm", "Actual finished garment (chart + manufacturing deviation, SD 0.8 cm).") for d in DIMS],
    "latent_observations": [
        ("synthetic_marker", "str", "marker", "", "Always 'SYNTHETIC'."),
        ("observation_id", "str", "key", "", "FK to observations."),
        ("pair_id", "str", "key", "", "FK to pair."),
        ("sim_scenario", "str", "latent", "", "Scenario of the customer."),
        ("discomfort", "float", "latent", "", "Scenario-specific discomfort used for preferred-size choice."),
        ("utility", "float", "latent", "", "-discomfort + Gumbel choice noise."),
    ] + [(f"{p}_{d}{s}", "float" if p != "label_flipped" else "int", "latent", u, t) for d in DIMS for p, s, u, t in [
        ("geometric_ease", "_cm", "cm", "Actual garment - compressed true body."),
        ("stretch_credit", "_cm", "cm", "Ease credited from usable stretch (only when geometric ease < 0)."),
        ("desired_ease", "_cm", "cm", "Customer-garment desired ease used for labels."),
        ("perceived_ease", "_cm", "cm", "Noisy perceived ease compared with thresholds."),
        ("label_flipped", "", "flag", "1 if the label was randomly moved to an adjacent class.")]],
}


def write_data_dictionary(path):
    lines = ["# DATA DICTIONARY — SYNTHETIC DATA", "",
             "All files are **SYNTHETIC**: simulated, not real observations. Units are centimetres. Roles: `feature` = may be "
             "used at prediction time; `label` = simulated outcome; `latent` = simulator internals (never features); "
             "`evaluation_metadata` / `split` / `key` / `marker` = not features.", ""]
    for f, cols in COLUMNS.items():
        lines += [f"## {f}.csv", "", "| column | type | role | unit | description |", "|---|---|---|---|---|"]
        lines += [f"| `{c}` | {t} | {r} | {u} | {d} |" for c, t, r, u, d in cols]
        lines.append("")
    Path(path).write_text("\n".join(lines))


def build_assumptions():
    A = lambda name, value, note="": dict(name=name, value=value, status="UNVALIDATED simulation assumption", note=note)
    return [
        A("body.regions", {k: dict(median_cm=v["mean"]) for k, v in REGION_PARAMS.items()}, "abstract regions, not real populations"),
        A("body.log_sd(chest,waist,hip)", SD_LOG.tolist(), "log-normal, correlated via region-specific correlation matrices"),
        A("body.correlations", {k: [v["c_cw"], v["c_ch"], v["c_wh"]] for k, v in REGION_PARAMS.items()}, "chest-waist, chest-hip, waist-hip"),
        A("body.archetype_offsets_cm", ARCH_OFFSET_CM.tolist(), "mixture components; no sex/gender/age modelled"),
        A("body.bounds_cm", BODY_BOUNDS), A("body.ratio_bounds_main", RATIO_MAIN),
        A("body.measurement_noise_sd_cm", MEASUREMENT_SD, "reported vs true; rounded to 0.5 cm"),
        A("garment.design_ease_cm", {k: [None if np.isnan(x) else x for x in v] for k, v in DESIGN_EASE.items()}, "straight style, before multiplier"),
        A("garment.style_multiplier", STYLE_MULT), A("garment.grade_step_mean_cm", GRADE_STEP_MEAN.tolist()),
        A("garment.manufacturing_tolerance_sd_cm", GARMENT_TOLERANCE_SD),
        A("fabric.stretch_range_pct", FABRIC_STRETCH_RANGE, "crosswise, notional fixed-load test"),
        A("fabric.cut_reduction_per_pct_stretch", STRETCH_CUT_REDUCTION, "cm per 1% stretch per 100 cm body; designer behaviour assumption"),
        A("fabric.rigidity", FABRIC_RIGIDITY), A("fabric.usable_stretch_fraction", "Beta(2,6), independent of stretch %", "scenario multipliers " + str(SCEN_STRETCH_MULT.tolist())),
        A("preference.desired_ease_regular_cm", DESIRED_BASE.tolist(), "rows follow GARMENT_TYPES order"),
        A("preference.multiplier_additive", dict(mult=PREF_MULT.tolist(), add=PREF_ADD.tolist())),
        A("preference.stated_vs_behavioural_mismatch", 0.12),
        A("labels.tolerance", "tight tol ~ N(6.2,1.3) clipped [2.0,10.0] (unusual_preference cohort: N(2.8,1.3)); loose tol = tight x lognormal(ln1.35,0.25)", "asymmetric"),
        A("labels.flip_prob", LABEL_FLIP), A("scenarios", SCENARIOS, "S4 appears only in the stress set"),
        A("choice.model", "argmax of -discomfort + Gumbel noise among acceptable sizes; scenario-specific discomfort shape"),
    ]


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def write_all(cfg, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = {k: generate_set(cfg, k) for k in ("main", "stress")}
    manifest = {"marker": MARKER, "generator_version": GENERATOR_VERSION, "config": asdict(cfg), "files": {}}
    for kind, d in sets.items():
        (out / kind).mkdir(exist_ok=True)
        for name, df in d.items():
            p = out / kind / f"{name}.csv"
            ff = "%.2f" if not name.startswith("latent") else "%.4f"
            df.to_csv(p, index=False, float_format=ff, lineterminator="\n")
            manifest["files"][f"{kind}/{name}.csv"] = dict(rows=len(df), sha256=sha256(p))
    stats = dataset_stats(sets)
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (out / "assumptions.json").write_text(json.dumps(build_assumptions(), indent=2, default=str))
    write_data_dictionary(out / "DATA_DICTIONARY.md")
    (out / "README_SYNTHETIC.txt").write_text(
        "SYNTHETIC DATA. Simulated for software development and stress-testing only. Not real customer observations, "
        "not evidence of fit accuracy, not derived from any paper or real person. latent_*.csv hold simulator internals "
        "and must never be used as model features.\n")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return sets, stats, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--preset", choices=list(PRESETS), help="small = quick test size")
    for f in ("seed", "n_customers", "n_garments", "pairs_per_customer", "stress_customers", "stress_garments",
              "stress_pairs_per_customer"):
        ap.add_argument("--" + f.replace("_", "-"), type=int, default=None)
    a = ap.parse_args(argv)
    kw = dict(PRESETS.get(a.preset, {}))
    kw.update({k: v for k, v in vars(a).items() if k not in ("out", "preset") and v is not None})
    cfg = GenConfig(**kw)
    sets, stats, _ = write_all(cfg, a.out)
    m = stats["main"]
    print(f"[{MARKER}] seed={cfg.seed} main: {m['n_customers']} customers, {m['n_garments']} garments, "
          f"{m['n_pairs']} pairs, {m['n_observations']} rows; none-acceptable={m['pair_none_acceptable_rate']:.1%}; "
          f"stress: {stats['stress']['n_observations']} rows -> {a.out}/")


if __name__ == "__main__":
    main()
