# Synthetic garment-ease dataset: generator, assumptions and prototype results

> **SYNTHETIC.** All data are simulated for software development and stress-testing. They are not customer observations, not evidence of fit accuracy, not a substitute for ethical human fitting studies, and not derived from the 2023 or 2026 papers or any real person. **Accuracy below measures agreement with this simulator only.**

Generator v1.0.0, seed 20261124 (fixed; every file is byte-reproducible, hashes in `MANIFEST.json`).

## 1. Generator

`generate_synthetic_garments.py` writes a **main** set and a separately generated **stress** set. Model-visible files: `customers`, `garments`, `garment_sizes`, `observations` (keys + simulated labels). Simulator internals are in `latent_*.csv` and are never features. Body regions (chest/waist/hip) get labels `tight/acceptable/loose`; each customer×garment pair has all candidate sizes, an overall preferred size and a `NONE` outcome. ("Regional" labels are read as **body regions**; the five `sim_region` values are abstract populations.)

Pipeline: correlated log-normal bodies per abstract region (+ archetype mixture) → noisy reported measurements → garments (type, style, fabric, crosswise stretch %) → graded charts (retailer *nominal body* columns kept separate from *finished-garment* columns) → latent perceived ease (true body, soft-tissue compression, partial stretch credit, manufacturing deviation, customer bias and noise) → asymmetric per-region thresholds → preferred size = max latent utility among acceptable sizes (Gumbel choice noise).

| Scenario | Assumption differing from the others | Where used |
|---|---|---|
| S1 asymmetric threshold | rigid fabrics lower tight-tolerance; quadratic tight / linear loose discomfort | main |
| S2 stretch coupling | more usable stretch, hip tightness pulls waist tighter, proportional tolerances (non-additive) | main |
| S3 style-relative | strong pull toward garment's intended cut, pseudo-Huber discomfort, regional perception shifts | main |
| S4 OOD shifted | +2.5 cm desired ease, wider tolerance, heavy-tailed noise, 4% label flips; shifted/heavy-tailed bodies | **stress only** |

None of these is the baseline's `Σ w (ease − desired)²`; labels additionally pass through thresholds, noise, flips and a gated choice.

## 2. Dataset sizes and class balance

| | main | stress |
|---|---|---|
| customers | 4,000 | 600 |
| garments | 160 | 40 |
| garment-size rows | 885 | 214 |
| customer×garment pairs | 24,000 | 2,400 |
| observations (candidate sizes) | 132,677 | 12,829 |
| none acceptable (pairs) | 30.2% | 48.8% |

Region-level label balance (main, applicable rows):

| body region | tight | acceptable | loose |
|---|---|---|---|
| chest | 29.7% | 39.3% | 31.0% |
| waist | 31.2% | 35.4% | 33.4% |
| hip | 29.0% | 39.7% | 31.3% |

None-acceptable rate by scenario (main): S1 28.6%, S2 32.0%, S3 30.5%. Stress cohorts: ood_population (90), unusual_preference (60), missing_waist (60), missing_multiple (60), missing_hip (48), extreme_size_large (48), noisy_measurement (48), extreme_size_small (48), missing_chest (48), extreme_proportion (48), control (30), all_body_missing (12).

## 3. Evaluation protocol

Customer-level split (70/15/15) is fixed by the generator before any preprocessing; all rows of a customer stay together. Imputation medians, scalers, regularisation (chosen on validation customers), the none-acceptable gate and the tuned baseline's parameters use **training rows only**. Splits: **A** unseen customers; **B** unseen customers × unseen garments; **C** leave-one-style-out (unseen customers × unseen style, pooled); **D** train S1+S2 → test S3 (unseen scenario); **E** stress set (never fitted on), overall and by cohort. Customer ID, garment ID, latents, labels, scenario and cohort tags are excluded from features (enforced by tests). Pair top-1 counts `NONE` as a class; 95% CIs are customer-cluster bootstraps.

Methods: majority / fixed size M; the existing weighted-ease formula with a **stand-in** generic prior (slim/regular/loose = 3/6/10 cm for all regions, unit weights, ±3 cm tolerance; the production config was not available, replace `BASELINE_PRIOR`); the same formula tuned on train rows; and an interpretable learned model (per-region multinomial logistic regression, overall-acceptable logistic regression, and a conditional-logit size-choice model with linear utility).

## 4. Results

**A. Unseen customers (test)**

| method | pairs | top-1 incl. NONE [95% CI] | top-1 when acceptable exists | top-2 | top-3 | within ±1 size | NONE P / R / F1 | region acc. chest / waist / hip |
|---|---|---|---|---|---|---|---|---|
| Majority | 3600 | 0.298 [0.28, 0.32] | 0.000 | – | – | 0.000 | 0.30 / 1.00 / 0.46 | 0.39 / 0.36 / 0.41 |
| Fixed size M | 3600 | 0.144 [0.13, 0.16] | 0.205 | – | – | 0.629 | 0.00 / 0.00 / 0.00 | – |
| Weighted-ease (stand-in prior) | 3600 | 0.401 [0.38, 0.42] | 0.219 | 0.761 | 0.899 | 0.412 | 0.40 / 0.83 / 0.54 | 0.66 / 0.70 / 0.72 |
| Weighted-ease (train-tuned) | 3600 | 0.477 [0.46, 0.50] | 0.451 | 0.812 | 0.933 | 0.783 | 0.61 / 0.54 / 0.57 | 0.75 / 0.76 / 0.76 |
| Interpretable learned | 3600 | 0.519 [0.50, 0.54] | 0.522 | 0.874 | 0.964 | 0.846 | 0.65 / 0.51 / 0.57 | 0.80 / 0.82 / 0.81 |

**B. Unseen customers and garments**

| method | pairs | top-1 incl. NONE [95% CI] | top-1 when acceptable exists | top-2 | top-3 | within ±1 size | NONE P / R / F1 | region acc. chest / waist / hip |
|---|---|---|---|---|---|---|---|---|
| Majority | 722 | 0.325 [0.28, 0.36] | 0.000 | – | – | 0.000 | 0.33 / 1.00 / 0.49 | 0.40 / 0.36 / 0.40 |
| Fixed size M | 722 | 0.137 [0.11, 0.16] | 0.203 | – | – | 0.624 | 0.00 / 0.00 / 0.00 | – |
| Weighted-ease (stand-in prior) | 722 | 0.411 [0.38, 0.45] | 0.191 | 0.770 | 0.901 | 0.382 | 0.43 / 0.87 / 0.57 | 0.65 / 0.69 / 0.73 |
| Weighted-ease (train-tuned) | 722 | 0.456 [0.42, 0.49] | 0.444 | 0.823 | 0.934 | 0.809 | 0.67 / 0.48 / 0.56 | 0.74 / 0.74 / 0.78 |
| Interpretable learned | 722 | 0.517 [0.48, 0.56] | 0.532 | 0.875 | 0.965 | 0.836 | 0.63 / 0.49 / 0.55 | 0.80 / 0.82 / 0.82 |

**C. Leave-one-style-out**

| method | pairs | top-1 incl. NONE [95% CI] | top-1 when acceptable exists | top-2 | top-3 | within ±1 size | NONE P / R / F1 | region acc. chest / waist / hip |
|---|---|---|---|---|---|---|---|---|
| Majority | 3600 | 0.298 [0.28, 0.32] | 0.000 | – | – | 0.000 | 0.30 / 1.00 / 0.46 | 0.39 / 0.36 / 0.41 |
| Fixed size M | 3600 | 0.144 [0.13, 0.16] | 0.205 | – | – | 0.629 | 0.00 / 0.00 / 0.00 | – |
| Weighted-ease (stand-in prior) | 3600 | 0.401 [0.38, 0.42] | 0.219 | 0.761 | 0.899 | 0.412 | 0.40 / 0.83 / 0.54 | 0.66 / 0.70 / 0.72 |
| Weighted-ease (train-tuned) | 3600 | 0.456 [0.44, 0.47] | 0.402 | 0.791 | 0.919 | 0.735 | 0.58 / 0.58 / 0.58 | 0.75 / 0.74 / 0.77 |
| Interpretable learned | 3600 | 0.384 [0.36, 0.41] | 0.185 | 0.801 | 0.932 | 0.341 | 0.36 / 0.85 / 0.51 | 0.67 / 0.69 / 0.70 |

**D. Unseen scenario S3**

| method | pairs | top-1 incl. NONE [95% CI] | top-1 when acceptable exists | top-2 | top-3 | within ±1 size | NONE P / R / F1 | region acc. chest / waist / hip |
|---|---|---|---|---|---|---|---|---|
| Majority | 1080 | 0.316 [0.27, 0.36] | 0.000 | – | – | 0.000 | 0.32 / 1.00 / 0.48 | 0.39 / 0.34 / 0.38 |
| Fixed size M | 1080 | 0.141 [0.12, 0.17] | 0.206 | – | – | 0.622 | 0.00 / 0.00 / 0.00 | – |
| Weighted-ease (stand-in prior) | 1080 | 0.402 [0.36, 0.44] | 0.210 | 0.724 | 0.890 | 0.402 | 0.42 / 0.82 / 0.55 | 0.67 / 0.71 / 0.73 |
| Weighted-ease (train-tuned) | 1080 | 0.442 [0.40, 0.47] | 0.424 | 0.769 | 0.913 | 0.793 | 0.67 / 0.48 / 0.56 | 0.72 / 0.74 / 0.78 |
| Interpretable learned | 1080 | 0.482 [0.44, 0.52] | 0.480 | 0.840 | 0.947 | 0.836 | 0.67 / 0.49 / 0.56 | 0.79 / 0.81 / 0.81 |

**E. Stress set (all cohorts)**

| method | pairs | top-1 incl. NONE [95% CI] | top-1 when acceptable exists | top-2 | top-3 | within ±1 size | NONE P / R / F1 | region acc. chest / waist / hip |
|---|---|---|---|---|---|---|---|---|
| Majority | 2400 | 0.487 [0.46, 0.52] | 0.000 | – | – | 0.000 | 0.49 / 1.00 / 0.66 | 0.35 / 0.33 / 0.36 |
| Fixed size M | 2400 | 0.111 [0.10, 0.13] | 0.216 | – | – | 0.624 | 0.00 / 0.00 / 0.00 | – |
| Weighted-ease (stand-in prior) | 2400 | 0.492 [0.46, 0.51] | 0.181 | 0.727 | 0.872 | 0.383 | 0.60 / 0.82 / 0.69 | 0.64 / 0.64 / 0.64 |
| Weighted-ease (train-tuned) | 2400 | 0.506 [0.48, 0.53] | 0.341 | 0.744 | 0.889 | 0.634 | 0.73 / 0.68 / 0.70 | 0.70 / 0.68 / 0.68 |
| Interpretable learned | 2400 | 0.509 [0.48, 0.54] | 0.370 | 0.739 | 0.892 | 0.670 | 0.75 / 0.66 / 0.70 | 0.71 / 0.68 / 0.68 |

Macro-F1 and per-class recall for every body region are in `results/metrics.json`. Label-permutation control (shuffled training labels): top-1 0.298, i.e. the trivial level.

Stress cohorts, pair top-1 incl. NONE (learned vs tuned baseline vs majority):

| cohort | pairs | true NONE | learned | tuned | majority |
|---|---|---|---|---|---|
| customer: all_body_missing | 48 | 0.33 | 0.15 | 0.12 | 0.33 |
| customer: control | 120 | 0.30 | 0.48 | 0.47 | 0.30 |
| customer: extreme_proportion | 192 | 0.90 | 0.86 | 0.88 | 0.90 |
| customer: extreme_size_large | 192 | 0.96 | 0.96 | 0.96 | 0.96 |
| customer: extreme_size_small | 192 | 0.53 | 0.60 | 0.61 | 0.53 |
| customer: missing_chest | 192 | 0.39 | 0.45 | 0.44 | 0.39 |
| customer: missing_hip | 192 | 0.40 | 0.41 | 0.44 | 0.40 |
| customer: missing_multiple | 240 | 0.37 | 0.26 | 0.34 | 0.37 |
| customer: missing_waist | 240 | 0.34 | 0.42 | 0.41 | 0.34 |
| customer: noisy_measurement | 192 | 0.32 | 0.55 | 0.45 | 0.32 |
| customer: ood_population | 360 | 0.36 | 0.47 | 0.44 | 0.36 |
| customer: unusual_preference | 240 | 0.62 | 0.38 | 0.38 | 0.62 |
| garment: control | 620 | 0.42 | 0.51 | 0.46 | 0.42 |
| garment: extreme_stretch | 341 | 0.48 | 0.40 | 0.51 | 0.48 |
| garment: missing_garment_measure | 349 | 0.48 | 0.42 | 0.46 | 0.48 |
| garment: missing_stretch | 232 | 0.46 | 0.50 | 0.45 | 0.46 |
| garment: sparse_chart | 369 | 0.53 | 0.61 | 0.62 | 0.53 |
| garment: very_fitted_cut | 237 | 0.56 | 0.59 | 0.55 | 0.56 |
| garment: very_loose_cut | 252 | 0.57 | 0.56 | 0.52 | 0.57 |

### What the numbers do and do not say

- In distribution (A, B, D) the learned model leads (A: 0.52 vs tuned baseline 0.48 vs stand-in prior 0.40). The stand-in prior's weakness is mostly its generic ease table, which tuning on train rows repairs: the prior's gap is a statement about my stand-in, not your production config.
- **It does not extrapolate to unseen styles** (C: learned 0.38 vs tuned baseline 0.46). The style interaction terms have no signal for a style never seen in training, so the more flexible model degrades below the simpler tuned formula. This is the kind of failure an in-distribution benchmark would hide.
- On the stress set the learned model and tuned baseline are indistinguishable (E: 0.51 vs 0.51); the learned model is worse on extreme stretch, missing garment measurements and cases with several body measurements missing (it only ties the tuned baseline on unusual preferences). Rows with extreme size or proportion are mostly `NONE`, so the majority baseline looks competitive there; read NONE F1 as well.
- Top-1 is modest everywhere (~0.5) because ~±1 size ambiguity, label noise, perception noise and choice noise are simulated. Top-3 (~0.96) and within-±1-size (~0.85) are much higher; these gaps are properties of the simulated noise, not of garments.
- Model coefficients are readable (`results/metrics.json`, `logreg_top_coefficients`): the quadratic ease terms dominate, i.e. the learned model rediscovers 'distance from a preferred ease'. That is consistent with the simulator by construction.

**High scores here demonstrate consistency with the simulator. They are not real-world accuracy and not evidence that the simulated assumptions match human preferences.**

## 5. Limitations and choices with no real-evidence basis

- Every range, distribution, correlation, tolerance, ease table, fabric class, stretch range and scenario rule is an **unvalidated assumption** (`assumptions.json`). No real fitting or anthropometric data were used or checked; the abstract regions are not real populations; no sex, gender, age, posture or ethnicity is modelled.
- Body/garment values use centimetres; garment values are full circumferences laid relaxed and unstretched. Reported body values carry 1 cm noise.
- **Stretch** is a crosswise, notional fixed-load percentage on fabric. Only a latent *usable fraction* (independent of the percentage) turns it into ease, and only when geometric ease is negative. Real behaviour also depends on load level, recovery, construction, seams, direction and wearer tolerance.
- Designers' cut-ease reduction for stretch (0.08 cm per 1% per 100 cm), preference tables, style multipliers, size grading (3–7 sizes, ~5 cm steps) and asymmetry of tolerances are invented. Labels are drawn independently per body region apart from the stated couplings.
- Calibration choice: tolerances were widened during development until ~30% of pairs have no acceptable size; this is an arbitrary target, not an observed rate.
- The stand-in baseline is not your production model; I could not inspect the application's API or schema (none was available in this session), so compatibility is limited to column mapping (README).
- Pair top-1 CIs account for customer clustering only, not generator-seed variability. One seed was run; rerun with other seeds before drawing conclusions about ranking of methods.
- Simulated preferences are not human preferences; a fitting study with consenting participants is required for any claim about real fit.
