#!/usr/bin/env python3
"""Reproducible EDA for the synthetic garment-fit corpora.

Writes docs/eda/EDA.md. Numbers are computed from the CSVs, never typed by hand.
Run with the experiment venv:
  kaggle/synthetic-ease-experiment/.venv/bin/python .../eda.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "docs" / "eda" / "EDA.md"
SEEDS = ["data/main"] + [f"data_clean/seed_{s}/main" for s in (20270101, 20270102, 20270103)]
DIMS = ("chest", "waist", "hip")


def q(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "n/a"
    return f"{s.quantile(.05):.1f} / {s.median():.1f} / {s.quantile(.95):.1f}"


def pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def load(d: Path):
    splits = d / "splits_customers.csv"   # the stress corpus ships without partitions
    return (pd.read_csv(d / "customers.csv"), pd.read_csv(d / "garments.csv"),
            pd.read_csv(d / "garment_sizes.csv"), pd.read_csv(d / "observations.csv"),
            pd.read_csv(splits) if splits.exists() else None,
            pd.read_csv(d / "latent_customers.csv"))


def joined(d: Path):
    cust, gar, sizes, obs, splits, latent = load(d)
    drop = lambda f: f.drop(columns=[c for c in ("synthetic_marker",) if c in f.columns])
    cust, gar, sizes, splits, latent = map(drop, (cust, gar, sizes, splits, latent))
    o = obs.merge(cust, on="customer_id", validate="many_to_one")
    o = o.merge(gar, on="garment_id", suffixes=("", "_gar"), validate="many_to_one")
    o = o.merge(sizes, on=["garment_id", "size_label"], suffixes=("", "_sz"), validate="many_to_one")
    o = o.merge(splits, on="customer_id", validate="many_to_one")
    o = o.merge(latent[["customer_id", "sim_scenario"]], on="customer_id", validate="many_to_one")
    for k in DIMS:
        o[f"ease_{k}"] = o[f"garment_{k}_cm"] - o[f"{k}_cm"]
    return o


def inventory(d: Path) -> dict:
    cust, gar, sizes, obs, splits, _ = load(d)
    return {"customers": len(cust), "garments": len(gar), "size_rows": len(sizes),
            "pairs": obs.pair_id.nunique(), "observations": len(obs)}


def funnel(o: pd.DataFrame, partition: str) -> dict:
    t = o[o.partition == partition]
    complete_cols = [f"{k}_cm" for k in DIMS] + [f"garment_{k}_cm" for k in DIMS]
    ok = t[complete_cols].notna().all(axis=1) & t.stretch_crosswise_pct.between(0, 50)
    per_pair = t.assign(ok=ok).groupby(["pair_id", "garment_type"], as_index=False).agg(ok=("ok", "all"),
                                                                                        none=("pair_none_acceptable", "first"))
    return {"pairs": len(per_pair), "eligible": int(per_pair.ok.sum()),
            "eligible_types": sorted(t.loc[ok, "garment_type"].unique().tolist()),
            "eligible_none": int(per_pair.loc[per_pair.ok, "none"].sum())}


def main():
    lines = ["# Exploratory data analysis — synthetic garment-fit corpora", "",
             "All numbers below are computed by `kaggle/synthetic-ease-experiment/garment_synth/eda.py`",
             "from the CSV files it reads. Regenerate with:", "",
             "```bash",
             "kaggle/synthetic-ease-experiment/.venv/bin/python kaggle/synthetic-ease-experiment/garment_synth/eda.py",
             "```", "",
             "> **These data are simulated.** Every distribution, threshold and correlation is a design",
             "> assumption of the generator, not an observation of people. Nothing here estimates real",
             "> garment fit, and none of it comes from the 2023 or 2026 studies.", ""]

    # 1. inventory
    lines += ["## 1. Corpus inventory", "",
              "| corpus | customers | garments | size rows | customer×garment pairs | candidate observations |",
              "|---|---:|---:|---:|---:|---:|"]
    rows = []
    for s in SEEDS:
        d = HERE / s
        if not d.exists():
            continue
        inv = inventory(d)
        rows.append((s, inv))
        lines.append(f"| `{s}` | {inv['customers']:,} | {inv['garments']:,} | {inv['size_rows']:,} | "
                     f"{inv['pairs']:,} | {inv['observations']:,} |")
    stress = HERE / "data" / "stress"
    if stress.exists():
        inv = inventory(stress)
        lines.append(f"| `data/stress` | {inv['customers']:,} | {inv['garments']:,} | {inv['size_rows']:,} | "
                     f"{inv['pairs']:,} | {inv['observations']:,} |")
    lines += ["", f"Mean candidate sizes per pair range from "
                  f"{min(inv['observations'] / inv['pairs'] for _, inv in rows):.2f} to "
                  f"{max(inv['observations'] / inv['pairs'] for _, inv in rows):.2f}.", ""]

    # 2. schema / missingness
    d0 = HERE / "data/main"
    o0 = joined(d0)
    lines += ["## 2. Missingness", "",
              "Missing values are structural, not random noise: dimension-specific garment rows are emitted",
              "only for the regions a garment type declares as measured, which is why trousers and shirts",
              "carry no chest/hip garment values.", "",
              "| column | missing rate |", "|---|---:|"]
    for c in ["chest_cm", "waist_cm", "hip_cm", "garment_chest_cm", "garment_waist_cm",
              "garment_hip_cm", "stretch_crosswise_pct", "label_body_hip_cm"]:
        lines.append(f"| `{c}` | {pct(o0[c].isna().sum(), len(o0))} |")
    lines += ["", "### Missing garment measurements by garment type", "",
              "| garment type | rows | chest missing | waist missing | hip missing |",
              "|---|---:|---:|---:|---:|"]
    for t, g in o0.groupby("garment_type"):
        lines.append(f"| {t} | {len(g):,} | {pct(g.garment_chest_cm.isna().sum(), len(g))} | "
                     f"{pct(g.garment_waist_cm.isna().sum(), len(g))} | {pct(g.garment_hip_cm.isna().sum(), len(g))} |")
    lines.append("")

    # 3. eligibility funnel
    lines += ["## 3. Eligibility for EaseLab's input contract", "",
              "EaseLab requires chest, waist and hip for the customer and for every candidate size.",
              "Pairs failing that contract are excluded rather than imputed.", "",
              "| corpus | test pairs | eligible (all three regions present) | eligible share | of which `NONE` | garment types surviving |",
              "|---|---:|---:|---:|---:|---|"]
    for s in SEEDS:
        d = HERE / s
        if not d.exists():
            continue
        f = funnel(joined(d), "test")
        lines.append(f"| `{s}` | {f['pairs']:,} | {f['eligible']:,} | {pct(f['eligible'], f['pairs'])} | "
                     f"{f['eligible_none']:,} | {', '.join(f['eligible_types'])} |")
    lines += ["", "This funnel is the single most important caveat for every score in the README:",
              "the eligible cohort is a small, dress-only slice of each corpus.", ""]

    # 4. label balance
    lines += ["## 4. Regional fit-label distribution (corpus `data/main`, share of all candidate rows)", "",
              "| region | tight | acceptable | loose | not applicable |", "|---|---:|---:|---:|---:|"]
    for k in DIMS:
        col = o0[f"fit_{k}"]
        tot = len(col)
        lines.append(f"| {k} | {pct((col == 'tight').sum(), tot)} | {pct((col == 'acceptable').sum(), tot)} | "
                     f"{pct((col == 'loose').sum(), tot)} | {pct((col == 'not_applicable').sum(), tot)} |")
    pair = o0.groupby("pair_id")[["pair_none_acceptable", "pair_preferred_size"]].first()
    lines += ["", f"Pairs with **no acceptable size**: {pct(pair.pair_none_acceptable.sum(), len(pair))} "
                  f"({int(pair.pair_none_acceptable.sum()):,} of {len(pair):,}). This base rate is why total top-1",
              "alone cannot be read as a sizing result — always answering `NONE` scores the base rate.", ""]

    # 5. ease distributions and how labels relate to the app's fixed targets
    lines += ["## 5. Garment ease by regional label", "",
              "Ease is `garment − body`. The table shows the 5th / 50th / 95th percentile of ease for each",
              "label, per region, alongside the fixed target EaseLab assumes for `regular` fit.", "",
              "| region | tight | acceptable | loose | EaseLab `regular` target |", "|---|---|---|---|---:|"]
    app_target = {"chest": 8, "waist": 5, "hip": 8}
    for k in DIMS:
        g = o0[o0[f"fit_{k}"].isin(["tight", "acceptable", "loose"])]
        cells = [q(g.loc[g[f"fit_{k}"] == lab, f"ease_{k}"]) for lab in ("tight", "acceptable", "loose")]
        lines.append(f"| {k} | {cells[0]} | {cells[1]} | {cells[2]} | {app_target[k]} cm |")
    lines += ["", "Acceptable ease spans a wide range for every region; a single fixed target per region is",
              "a strong simplification of what the simulator treats as acceptable. This is a visible source",
              "of the gap between the fixed-target scorer and the learned model.", ""]

    # 6. ease by preference
    lines += ["### Median acceptable ease by stated preference", "",
              "| preference | chest | waist | hip |", "|---|---:|---:|---:|"]
    for p, g in o0.groupby("fit_preference"):
        cells = []
        for k in DIMS:
            sub = g.loc[g[f"fit_{k}"] == "acceptable", f"ease_{k}"]
            cells.append(f"{sub.median():.1f}" if len(sub) else "n/a")
        lines.append(f"| {p} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    # 7. none rate by scenario
    lines += ["## 6. No-acceptable-size rate by simulator scenario", "",
              "| scenario | pairs | none rate |", "|---|---:|---:|"]
    for scen, g in o0.groupby("sim_scenario"):
        p = g.groupby("pair_id").pair_none_acceptable.first()
        lines.append(f"| {scen} | {len(p):,} | {pct(p.sum(), len(p))} |")
    lines.append("")

    # 8. what the scorer sees vs what the label depends on
    lines += ["## 7. Signal available to each method", "",
              "The fixed-target scorer consumes only body chest/waist/hip, garment chest/waist/hip,",
              "`fit_preference` and one global stretch percentage. The corpora also carry garment type,",
              "style, fabric class, a retailer nominal-body chart, per-size `size_rank` and",
              "`measured_dims`. Those extra fields are prediction-time metadata, and the learned model uses",
              "them. The simulator generates labels from a latent rule that depends on garment style, stretch",
              "coupling and customer bias, so a model with more of those fields will fit it better — which is",
              "agreement with the generator, not evidence about real customers.", ""]

    # 9. how each corpus differs
    lines += ["## 8. Seed stability", "",
              "| seed | test pairs | eligible | eligible share | none rate in eligible |",
              "|---|---:|---:|---:|---:|"]
    for s in SEEDS:
        d = HERE / s
        if not d.exists():
            continue
        f = funnel(joined(d), "test")
        lines.append(f"| {s} | {f['pairs']:,} | {f['eligible']:,} | {pct(f['eligible'], f['pairs'])} | "
                     f"{pct(f['eligible_none'], f['eligible'])} |")
    lines += ["", "The eligible cohort and its `NONE` base rate both move with the seed, so a single-split",
              "comparison is fragile. The results reported in the README span three seeds for that reason.", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(lines)} lines)")


if __name__ == "__main__":
    main()
