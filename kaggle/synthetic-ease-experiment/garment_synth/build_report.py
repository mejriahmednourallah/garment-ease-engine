#!/usr/bin/env python3
"""Assemble REPORT.md from data/dataset_stats.json, data/assumptions.json and results/metrics.json (all SYNTHETIC)."""
import json
import sys
from pathlib import Path

D, R = Path(sys.argv[1] if len(sys.argv) > 1 else "data"), Path(sys.argv[2] if len(sys.argv) > 2 else "results")
st, man, res = (json.loads((D / "dataset_stats.json").read_text()), json.loads((D / "MANIFEST.json").read_text()),
                json.loads((R / "metrics.json").read_text()))
M = [("majority", "Majority"), ("fixed_size_M", "Fixed size M"), ("weighted_ease_prior", "Weighted-ease (stand-in prior)"),
     ("weighted_ease_tuned", "Weighted-ease (train-tuned)"), ("logreg", "Interpretable learned")]
f = lambda x: "–" if x is None else f"{x:.3f}"
L = ["# Synthetic garment-ease dataset: generator, assumptions and prototype results", "",
     "> **SYNTHETIC.** All data are simulated for software development and stress-testing. They are not customer observations, "
     "not evidence of fit accuracy, not a substitute for ethical human fitting studies, and not derived from the 2023 or 2026 papers "
     "or any real person. **Accuracy below measures agreement with this simulator only.**", "",
     f"Generator v{man['generator_version']}, seed {man['config']['seed']} (fixed; every file is byte-reproducible, hashes in `MANIFEST.json`).", "",
     "## 1. Generator", "",
     "`generate_synthetic_garments.py` writes a **main** set and a separately generated **stress** set. Model-visible files: "
     "`customers`, `garments`, `garment_sizes`, `observations` (keys + simulated labels). Simulator internals are in `latent_*.csv` and are never features. "
     "Body regions (chest/waist/hip) get labels `tight/acceptable/loose`; each customer×garment pair has all candidate sizes, an overall preferred size and a "
     "`NONE` outcome. (\"Regional\" labels are read as **body regions**; the five `sim_region` values are abstract populations.)", "",
     "Pipeline: correlated log-normal bodies per abstract region (+ archetype mixture) → noisy reported measurements → garments (type, style, fabric, "
     "crosswise stretch %) → graded charts (retailer *nominal body* columns kept separate from *finished-garment* columns) → latent perceived ease "
     "(true body, soft-tissue compression, partial stretch credit, manufacturing deviation, customer bias and noise) → asymmetric per-region thresholds → "
     "preferred size = max latent utility among acceptable sizes (Gumbel choice noise).", "",
     "| Scenario | Assumption differing from the others | Where used |", "|---|---|---|",
     "| S1 asymmetric threshold | rigid fabrics lower tight-tolerance; quadratic tight / linear loose discomfort | main |",
     "| S2 stretch coupling | more usable stretch, hip tightness pulls waist tighter, proportional tolerances (non-additive) | main |",
     "| S3 style-relative | strong pull toward garment's intended cut, pseudo-Huber discomfort, regional perception shifts | main |",
     "| S4 OOD shifted | +2.5 cm desired ease, wider tolerance, heavy-tailed noise, 4% label flips; shifted/heavy-tailed bodies | **stress only** |", "",
     "None of these is the baseline's `Σ w (ease − desired)²`; labels additionally pass through thresholds, noise, flips and a gated choice.", "",
     "## 2. Dataset sizes and class balance", "", "| | main | stress |", "|---|---|---|"]
for k, lab in [("n_customers", "customers"), ("n_garments", "garments"), ("n_garment_size_rows", "garment-size rows"),
               ("n_pairs", "customer×garment pairs"), ("n_observations", "observations (candidate sizes)")]:
    L.append(f"| {lab} | {st['main'][k]:,} | {st['stress'][k]:,} |")
L.append(f"| none acceptable (pairs) | {st['main']['pair_none_acceptable_rate']:.1%} | {st['stress']['pair_none_acceptable_rate']:.1%} |")
L += ["", "Region-level label balance (main, applicable rows):", "", "| body region | tight | acceptable | loose |", "|---|---|---|---|"]
for d in ("chest", "waist", "hip"):
    c = st["main"][f"label_counts_{d}"]; t = sum(v for k, v in c.items() if k != "not_applicable")
    L.append(f"| {d} | {c['tight']/t:.1%} | {c['acceptable']/t:.1%} | {c['loose']/t:.1%} |")
L += ["", f"None-acceptable rate by scenario (main): " + ", ".join(f"{k.split('_')[0]} {v:.1%}" for k, v in st["main"]["none_rate_by_scenario"].items()) + ". "
      "Stress cohorts: " + ", ".join(f"{k} ({v})" for k, v in st["stress"]["customers_by_stress_cohort"].items()) + ".", "",
      "## 3. Evaluation protocol", "",
      "Customer-level split (70/15/15) is fixed by the generator before any preprocessing; all rows of a customer stay together. Imputation medians, scalers, "
      "regularisation (chosen on validation customers), the none-acceptable gate and the tuned baseline's parameters use **training rows only**. "
      "Splits: **A** unseen customers; **B** unseen customers × unseen garments; **C** leave-one-style-out (unseen customers × unseen style, pooled); "
      "**D** train S1+S2 → test S3 (unseen scenario); **E** stress set (never fitted on), overall and by cohort. Customer ID, garment ID, latents, labels, "
      "scenario and cohort tags are excluded from features (enforced by tests). Pair top-1 counts `NONE` as a class; 95% CIs are customer-cluster bootstraps.", "",
      "Methods: majority / fixed size M; the existing weighted-ease formula with a **stand-in** generic prior "
      "(slim/regular/loose = 3/6/10 cm for all regions, unit weights, ±3 cm tolerance; the production config was not available, replace "
      "`BASELINE_PRIOR`); the same formula tuned on train rows; and an interpretable learned model (per-region multinomial logistic regression, overall-acceptable "
      "logistic regression, and a conditional-logit size-choice model with linear utility).", "", "## 4. Results", ""]
for blk, title in [("A_customer_split_test", "A. Unseen customers (test)"), ("B_unseen_garment_test", "B. Unseen customers and garments"),
                   ("C_unseen_style_pooled_leave_one_style_out", "C. Leave-one-style-out"), ("D_unseen_scenario_S3_test", "D. Unseen scenario S3"),
                   ("E_stress_overall", "E. Stress set (all cohorts)")]:
    L += [f"**{title}**", "", "| method | pairs | top-1 incl. NONE [95% CI] | top-1 when acceptable exists | top-2 | top-3 | within ±1 size | NONE P / R / F1 | region acc. chest / waist / hip |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, n in M:
        v = res[blk][k]; p = v["pair"]; ci = p["top1_incl_none_ci95_by_customer"]; z = p["none_acceptable"]
        reg = " / ".join(f"{v['regional'][d]['accuracy']:.2f}" for d in ("chest", "waist", "hip")) if "regional" in v else "–"
        L.append(f"| {n} | {p['n_pairs']} | {f(p['top1_incl_none'])} [{ci[0]:.2f}, {ci[1]:.2f}] | {f(p['top1_when_acceptable_exists'])} | "
                 f"{f(p['top2_when_acceptable_exists'])} | {f(p['top3_when_acceptable_exists'])} | {f(p['within_one_size_when_acceptable'])} | "
                 f"{z['precision']:.2f} / {z['recall']:.2f} / {z['f1']:.2f} | {reg} |")
    L.append("")
L += ["Macro-F1 and per-class recall for every body region are in `results/metrics.json`. Label-permutation control (shuffled training labels): "
      f"top-1 {res['label_permutation_control']['logreg_top1_incl_none']:.3f}, i.e. the trivial level.", "",
      "Stress cohorts, pair top-1 incl. NONE (learned vs tuned baseline vs majority):", "", "| cohort | pairs | true NONE | learned | tuned | majority |", "|---|---|---|---|---|---|"]
for grp in ("customer_cohort", "garment_cohort"):
    for c, v in res["E_stress_by_cohort"][grp].items():
        L.append(f"| {grp.split('_')[0]}: {c} | {v['logreg']['n_pairs']} | {v['logreg']['none_rate_true']:.2f} | {v['logreg']['top1_incl_none']:.2f} | "
                 f"{v['weighted_ease_tuned']['top1_incl_none']:.2f} | {v['majority']['top1_incl_none']:.2f} |")
A, C, E = res["A_customer_split_test"], res["C_unseen_style_pooled_leave_one_style_out"], res["E_stress_overall"]
t1 = lambda b, k: b[k]["pair"]["top1_incl_none"]
L += ["", "### What the numbers do and do not say", "",
      f"- In distribution (A, B, D) the learned model leads (A: {t1(A,'logreg'):.2f} vs tuned baseline {t1(A,'weighted_ease_tuned'):.2f} vs stand-in prior {t1(A,'weighted_ease_prior'):.2f}). "
      "The stand-in prior's weakness is mostly its generic ease table, which tuning on train rows repairs: the prior's gap is a statement about my stand-in, not your production config.",
      f"- **It does not extrapolate to unseen styles** (C: learned {t1(C,'logreg'):.2f} vs tuned baseline {t1(C,'weighted_ease_tuned'):.2f}). The style interaction terms have no signal for a style never seen in training, "
      "so the more flexible model degrades below the simpler tuned formula. This is the kind of failure an in-distribution benchmark would hide.",
      f"- On the stress set the learned model and tuned baseline are indistinguishable (E: {t1(E,'logreg'):.2f} vs {t1(E,'weighted_ease_tuned'):.2f}); the learned model is worse on extreme stretch, "
      "missing garment measurements and cases with several body measurements missing (it only ties the tuned baseline on unusual preferences). Rows with extreme size or proportion are mostly `NONE`, so the majority baseline looks competitive there; read NONE F1 as well.",
      "- Top-1 is modest everywhere (~0.5) because ~±1 size ambiguity, label noise, perception noise and choice noise are simulated. Top-3 (~0.96) and within-±1-size (~0.85) are much higher; these gaps are properties of the simulated noise, not of garments.",
      "- Model coefficients are readable (`results/metrics.json`, `logreg_top_coefficients`): the quadratic ease terms dominate, i.e. the learned model rediscovers 'distance from a preferred ease'. That is consistent with the simulator by construction.",
      "", "**High scores here demonstrate consistency with the simulator. They are not real-world accuracy and not evidence that the simulated assumptions match human preferences.**", "",
      "## 5. Limitations and choices with no real-evidence basis", "",
      "- Every range, distribution, correlation, tolerance, ease table, fabric class, stretch range and scenario rule is an **unvalidated assumption** (`assumptions.json`). No real fitting or anthropometric data were used or checked; the abstract regions are not real populations; no sex, gender, age, posture or ethnicity is modelled.",
      "- Body/garment values use centimetres; garment values are full circumferences laid relaxed and unstretched. Reported body values carry 1 cm noise.",
      "- **Stretch** is a crosswise, notional fixed-load percentage on fabric. Only a latent *usable fraction* (independent of the percentage) turns it into ease, and only when geometric ease is negative. Real behaviour also depends on load level, recovery, construction, seams, direction and wearer tolerance.",
      "- Designers' cut-ease reduction for stretch (0.08 cm per 1% per 100 cm), preference tables, style multipliers, size grading (3–7 sizes, ~5 cm steps) and asymmetry of tolerances are invented. Labels are drawn independently per body region apart from the stated couplings.",
      "- Calibration choice: tolerances were widened during development until ~30% of pairs have no acceptable size; this is an arbitrary target, not an observed rate.",
      "- The stand-in baseline is not your production model; I could not inspect the application's API or schema (none was available in this session), so compatibility is limited to column mapping (README).",
      "- Pair top-1 CIs account for customer clustering only, not generator-seed variability. One seed was run; rerun with other seeds before drawing conclusions about ranking of methods.",
      "- Simulated preferences are not human preferences; a fitting study with consenting participants is required for any claim about real fit."]
Path("REPORT.md").write_text("\n".join(L) + "\n")
print("REPORT.md written")
