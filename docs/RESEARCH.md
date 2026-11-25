# What this project measured, on what data, and what it can and cannot claim

This document is the honest record of the evidence behind EaseLab. It separates four things that are
easy to conflate:

1. **Our deterministic scorer** — a three-region weighted ease penalty with hand-set targets.
2. **Our implementation of a published method** — Zhang et al.'s 2026 jacket framework, transcribed
   from the paper.
3. **Proxy experiments on real but mismatched data** — ModCloth customer fit feedback.
4. **Experiments on synthetic data** — a user-supplied simulator, used for software development.

Only (1) and (2) are about garment ease in the sense the project started with. (3) and (4) are not
evidence about real garment fit and are labelled as such everywhere they appear.

---

## 1. Starting point and the honest reason for the deviation

The original plan was `E_i = G_i − B_i` and `S = Σ w_i (E_i − E_i*)²`, with `E_i*` eventually
**learned from real fitting observations**. That requires, per participant: body circumferences,
finished-garment measurements of the sizes tried, and a regional fit judgement. Neither source study
releases that data:

| Study | Design | Data released | Licence |
|---|---|---|---|
| Dik et al., *Heliyon* 9(7):e17916 (2023) | 120 participants, Styku 3D scans, Optitex co-design; ANN predicts pattern/ease parameters from body + psychographic features; a GAN renders fitted avatars | **"Data will be made available on request."** Only a blank 15-item AIO questionnaire and an ethics letter are published as supplements | CC BY-NC-ND 4.0 |
| Zhang et al., *Textiles* 6(3):75 (2026) | 270 women, two jacket styles, full graded try-on; rule-based ease-deviation scoring with per-style ideal ease and six ease segments per dimension | **"The original contributions presented in this study are included in the article."** Full size charts and weights are in the paper; no participant-level rows, no code | CC BY 4.0 |

So the project improvised: targets, weights, tolerance bands and sample garment sizes are
**assumptions**, not measured preferences. That is stated in the README, in the app UI, and here.
Published aggregate tables are not participant data and were never treated as training data.

## 2. What is in this repository

| Component | Data | Status |
|---|---|---|
| `app/main.py` — three-region scorer, `/recommend`, `/feedback` | user-entered or synthetic | working prototype |
| `app/paper2026.py` — Zhang et al. equations, Tables 3–8 | transcribed from the paper | research implementation |
| `data/research/` — published jacket charts and aggregate results | paper tables | transcribed, provenance given |
| `kaggle/modcloth-experiment/`, `kaggle/modcloth-improvement/` | ModCloth fit feedback (CC BY 4.0 per the Kaggle mirror; original terms unverified) | real data, wrong target |
| `kaggle/synthetic-ease-experiment/` | simulated corpora, 4 seeds | development only |
| `docs/eda/EDA.md` | computed from the synthetic CSVs | reproducible |

## 3. Deviations from the two papers, explicitly

### Versus Dik et al. (2023)

| Aspect | Paper | This project |
|---|---|---|
| Model | ANN (reported 6 hidden layers) predicting pattern parameters from body + psychographics | no ANN; deterministic weighted ease |
| Psychographics | 15-item AIO survey, k-means into 5 groups | not collected |
| Fabric | handled via virtual co-design in Optitex | one global stretch percentage, heuristic |
| Output | predicted ease/pattern parameters + GAN visualisation | ranked sizes with per-region explanation |
| Evaluation | R² on 70 train / 50 test rows per garment | no reproduction attempted |

We did **not** recreate this model, and would not learn anything meaningful by training an ANN on
synthetic labels and reporting its fit to those labels.

### Versus Zhang et al. (2026)

| Aspect | Paper | This project |
|---|---|---|
| Score | `GE = G − B`; `FE = │GE − GI│`; `FS = W(segment)·FE`; `OFS = Σ FS` | implemented as specified |
| Inputs | bust, waist, shoulder width, height-based reference | same four |
| Ideal ease | Jacket A 6.0/13.0/0.0/0.0 cm; Jacket B 4.5/5.0/0.0/0.0 | same constants |
| Segments and weights | six intervals per dimension, per-style weights | transcribed verbatim |
| Fabric elasticity | explicitly excluded | excluded in this module, heuristic elsewhere |
| Accuracy | Top-1 77.8% / 73.0%; Top-2 97.8% / 95.2%; Top-3 99.6% / 98.9% | **not reproduced**; participant data unavailable |

Three printed inconsistencies were found and **documented rather than silently corrected**:

- **Waist target.** The paper states `GIwA = 13.0 cm`, but its worked example (Table 10) computes
  waist fit error against **6.0 cm** for all nine sizes. Our implementation uses the stated 13.0,
  ranks the reconstructed participant M, L, S as published, and produces different overall scores
  (2.99 for M against the printed 6.5).
- **Jacket B shoulder intervals.** Table 7 prints segment 2 as `−4 ≤ s < 2`, which overlaps segments
  3 and 4. The API refuses affected cases by default and requires an explicit interpretation rule;
  neither is validated.
- **Chart inconsistency.** Table 6 lists size L's fit-model bust as 85.5 cm, which contradicts the
  stated 4.5 cm ideal ease and the other rows.

Method reproduction, numerical-example verification, and participant-level accuracy validation are
three different claims. We can support the first two partially; we cannot support the third.

### 3.1 Where the 2023 data actually is, after searching

The participant-level dataset does not exist anywhere public. What does exist, and what we took:

| Resource | Status |
|---|---|
| Appendix 2 table: preferred ease for **64 pattern parameters × 5 psychographic groups × 4 garment types** | **Extracted** to `data/research/dik_2023_appendix2_pattern_ease_by_group.csv` by a script, and cross-checked against the paper's own claim (P2 largest for 60/64 parameters, P3 smallest for 50/64) |
| Appendix 2 "ease allowance preference charts" per size | **Inline images** in the published XML — no numeric values recoverable |
| Supplementary files | A blank 15-item AIO questionnaire and the ethics clearance letter. No responses |
| AHFE 2023 conference version, DOI 10.54941/ahfe1003635 | Retrieved; same study, no additional tables, no data deposit |
| Europe PMC / PMC JATS full text | Retrieved; source of the extraction above |
| THEi institutional repository copy | Returns HTTP 403 to non-browser clients; not retrieved |
| Zenodo / figshare / OSF / Mendeley / GitHub | No deposit found for this dataset |
| RGC project UGC/FDS25/H05/21 | Funding record only; no data output |

**The practical consequence.** The Appendix 2 table is the closest thing to real target-ease evidence
available without contacting the authors, and it changes what we can say: our invented constants are
not merely unsupported, they are *shape-wrong* — the published preferences vary by garment type and by
person cluster by roughly a factor of two, which a single global `slim/regular/loose` table cannot
express. But the values are per-panel pattern adjustments (0.03–14.31 cm), so converting them into
circumference ease is itself a modelling assumption, not a transcription. We therefore cite them and
do not silently substitute them.

Getting the real thing means an email to the corresponding author (THEi, Hong Kong) requesting the four
per-garment tables described in the paper's Section 3.5, under a data-sharing agreement consistent with
the HSESC HE2021-07 approval and the CC BY-NC-ND licence. That request has **not** been sent.

### 3.2 Can those published values be used now? A sensitivity test

`data/research/target_sensitivity.py` answers the only question that can be answered without inventing
a panel-to-circumference mapping: **how much does the choice of target change the recommendation?**

19 of the 64 parameters are girth-like (chest, waist, hip, sweep). Their spread across the five
psychographic groups splits into two clean regimes:

| Regime | Examples | Spread across P1–P5 | Share of a 5 cm size grade |
|---|---|---|---|
| **Panel-like** (per-panel adjustment) | blazer chest/waist, trousers hip | 0.26–0.82 cm (median 0.66) | 5–16% |
| **Circumference-like** | long-sleeved *Across Chest / Waist / Sweep* 3.72–5.97; short-sleeved *Across Chest / Waist* 4.08–7.16 | 2.05–3.08 cm | 41–62% |

Holding body and garment fixed and sweeping the target across each parameter's group range,
**5 of 19 parameters change the recommended size**, and all five are from the circumference-like shirts.
For short-sleeved shirts the swing is 3.08 cm — 0.62 of a size grade — enough to move the top pick from
L to XL.

Three consequences, in order of importance:

1. **Target error of ~3 cm is as damaging as anything in the scoring formula.** This is a measurable
   reason to spend the next effort on targets rather than on model class or extra features.
2. **The published evidence is usable as a bound, not as constants.** It tells us the plausible range
   for garment-type-specific targets. It cannot be pasted into `TARGET_EASE`, because the panel-like
   values would understate ease by roughly a factor of two to eight if treated as circumference.
3. **Garment type must enter the target.** A single `slim/regular/loose` table cannot represent a
   blazer's 0.39 cm chest spread and a shirt's 3.08 cm spread at once.

What this does **not** do is validate anything: the probe body and chart are fixed synthetic points, and
the sweep says nothing about which group a real customer belongs to.



## 4. Real-data proxy: ModCloth

ModCloth provides customer-reported fit (`small` / `fit` / `large`) with some body measurements —
but **no finished-garment measurements**, so ease is uncomputable. It is a fit-feedback proxy, not
an ease benchmark.

| Model | Macro F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| Majority class | 0.270 | 0.333 | 0.681 |
| MLP control, customer-grouped | 0.362 | 0.418 | 0.413 |
| Item-aware CatBoost, customer-grouped | **0.430** | **0.522** | 0.456 |

Item-aware modelling beat the control by **+0.068 macro F1** (customer-clustered bootstrap 95% CI
0.058–0.078). Two caveats matter more than the gain: the test users had appeared in an earlier
round's training set, and ~99% of test items were seen in training. So this is a warm-item result on
a reused split, and the model is *worse* than always answering "fit" on plain accuracy. It says
nothing about ease.

## 5. Synthetic data: what it is and what it did

A user-supplied generator simulates correlated bodies, graded garment charts, fabric stretch, latent
perceived ease, asymmetric regional thresholds, and a preferred size with choice noise. It ships a
main corpus and a stress corpus, plus four seeds in total here.

### 5.1 What the EDA establishes about the corpora

See `docs/eda/EDA.md` for the full tables. Key facts, all computed:

- **Missingness is structural, not random.** Dresses carry all three regions; shirts, jackets,
  hoodies and t-shirts carry no hip garment measurement; trousers and skirts carry no chest.
- **EaseLab's input contract covers a minority of pairs.** Of 3,600 test pairs per seed, between
  **532 and 806 (14.8%–22.4%)** have all three regions present for the customer and every candidate
  size — and in every seed **those pairs are dresses only**.
- **Acceptable ease is wide.** For chest, acceptable ease spans 5th–95th percentile roughly
  −4.5 to +25.5 cm, against EaseLab's single `regular` target of 8 cm. The fixed-target assumption is
  visibly lossy.
- **The `NONE` base rate is large**: 30.2% of pairs in the main corpus have no acceptable size, and
  33%–43% within the eligible cohorts. This is why total accuracy is a misleading headline metric.

### 5.2 What we measured

Eligible cohort only, identical pairs for every method:

| Corpus / seed | Existing scorer: total / conditional | Learned model, no gate: total / conditional | Always `NONE` (base rate) |
|---|---|---|---|
| `data/main` (test, n=603) | 18.4% / 30.1% | 53.2% (with gate) / 38.5% | 38.8% |
| seed 20270101 (n=806) | 19.6% / 34.1% | 37.8% / **65.9%** | 42.6% |
| seed 20270102 (n=532) | 20.1% / 30.1% | 40.4% / **60.6%** | 33.3% |
| seed 20270103 (n=672) | 15.6% / 24.8% | 40.2% / **63.8%** | 37.1% |

The learned model in the three fresh-seed rows **never abstains**, so its conditional accuracy is a
sizing result, not an artefact of refusing to answer. Cohen's κ rises from 0.078–0.127 to 0.309–0.326.

The **abstention gate alone does not hold up**. With its threshold chosen for balanced accuracy on
training/validation, it reached 43.7% / 35.5% / 34.4% total against always-`NONE` at
42.6% / 33.3% / 37.1%, while conditional accuracy fell to 19.2% / 16.1% / 12.5% and it abstained on
62%–68% of pairs. The earlier 53.2% headline came from a gate tuned on total accuracy on an
already-inspected split; that is a metric artefact, and it has been withdrawn.

## 6. What we can know

- The published 2026 equations, charts, segments and weights can be implemented and unit-checked;
  our implementation reproduces the paper's published top-three *ordering* for its only worked
  example while disagreeing on scores, for a documented reason.
- Given body and finished-garment measurements, a deterministic ease penalty is fully reproducible
  and auditable. Every number in the app is recomputable from its inputs.
- In the simulator, a fixed per-region target is a substantial simplification: acceptable ease varies
  with preference, garment type and style, and a model using those fields fits the simulator's labels
  far better than the three-region formula alone.
- In ModCloth, item identity adds measurable predictive signal for fit complaints, on a warm-item,
  reuse-split benchmark.
- The pipeline is correct end to end: 45 application tests pass, the API ranks and explains, feedback
  persists, and the research scripts are deterministic and hash-stamped.

## 7. What we cannot know, and will not claim

- **Real-world fit accuracy.** No participant-level fitting data from either paper, and no data of
  our own, exists in this repository.
- **That we reproduced either study.** The 2023 ANN/GAN is absent; the 2026 accuracy figures cannot
  be recomputed without the 270 participants' measurements and preferred sizes.
- **That the learned synthetic model transfers.** Its labels were generated by the same simulator it
  is fitted to, so a flexible model partly recovers the generator. High agreement there is expected
  and says nothing about people.
- **That the ModCloth result is an ease result.** ModCloth has no garment measurements; ease cannot
  be computed from it at all.
- **Fabric behaviour.** Stretch is one percentage applied to body circumference. Real stretch depends
  on direction, load, recovery, seams and construction; ASTM D2594-style measurement ranks fabrics
  but does not predict garment fit.
- **Generality.** Every eligible synthetic cohort here is dresses; the papers cover jackets, shirts,
  blazers and trousers; ModCloth is mostly dresses and tops. None of these populations is
  interchangeable.

## 8. Threats to validity, listed plainly

1. Only one worked example exists in the 2026 paper, and it cannot be regenerated from the paper's
   own tables.
2. The evaluated split for the first synthetic round had already been inspected before the follow-up,
   so it is not a clean holdout; the three fresh seeds are the honest test, and they are still
   simulator-only.
3. Eligible cohorts are 15%–22% of pairs and dress-only.
4. Metric choice dominates conclusions on the synthetic data: total accuracy rewards abstention
   because of the `NONE` base rate, so results are reported as conditional accuracy plus κ.
5. ModCloth test users appeared in an earlier round's training, and items are ~99% seen.
6. Neither the Kaggle mirror's CC BY 4.0 label nor the original ModCloth terms have been
   independently verified; raw records and row-level predictions are excluded from Git.
7. Three seeds check stability, not confidence; no seed-variance interval is claimed.

## 9. Reproducing

```bash
# application
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r dashboard/requirements.txt -r requirements-dev.txt
python -m pytest -q                       # 45 tests
uvicorn app.main:app --port 8000
streamlit run dashboard/streamlit_app.py

# EDA and synthetic studies (separate venv: needs scikit-learn)
kaggle/synthetic-ease-experiment/.venv/bin/python kaggle/synthetic-ease-experiment/garment_synth/eda.py
kaggle/synthetic-ease-experiment/.venv/bin/python kaggle/synthetic-ease-experiment/garment_synth/learned_only_study.py \
  --data-dir kaggle/synthetic-ease-experiment/garment_synth/data_clean/seed_20270101/main \
  --out /tmp/learned.json
kaggle/synthetic-ease-experiment/.venv/bin/python kaggle/synthetic-ease-experiment/garment_synth/ablation_seed_controlled.py \
  --data-dir kaggle/synthetic-ease-experiment/garment_synth/data_clean/seed_20270101/main \
  --out /tmp/gate.json
```

Input hashes and the git commit are written into every result JSON.

The generated corpora themselves (~130 MB) are not committed; they are rebuildable from the
generator. Each result JSON records the SHA-256 of every input CSV it used, so a regenerated corpus
can be checked against the recorded hashes before trusting a comparison:

```bash
cd kaggle/synthetic-ease-experiment/garment_synth
python generate_synthetic_garments.py --out data --seed 20261124
python generate_synthetic_garments.py --out data_clean/seed_20270101 --seed 20270101
```

## 10. The minimal study that would settle the open question

To validate the ease model rather than a simulator:

1. 100–150 participants per garment style, two styles, with informed consent and ethics approval.
2. Tape-measure body chest/waist/hip (plus shoulder and height if the 2026 method is to be tested).
3. Measure every size in the graded run as a finished garment, relaxed and unstretched.
4. Randomised try-on of all available sizes; record a three-level regional label and one preferred
   size per pair; record the fabric's measured stretch and recovery.
5. Pre-register the split by participant and by style; reserve one style entirely for testing.
6. Report conditional preferred-size accuracy, per-region F1, and κ — not Top-3 alone, which is easy
   on 7–9 near-equivalent sizes.

Until then, the correct summary of this repository is: **a working, auditable ease-ranking prototype
plus honest experiments showing where its fixed assumptions are likely to cost accuracy — not a
validated sizing system.**

## References

- Dik, N.Y.; Tsang, P.W.K.; Chan, A.P.; Lo, C.K.Y.; Chu, W.C. *A novel approach in predicting virtual
  garment fitting sizes with psychographic characteristics and 3D body measurements using artificial
  neural network and visualizing fitted bodies using generative adversarial network.* Heliyon 9(7),
  e17916, 2023. https://doi.org/10.1016/j.heliyon.2023.e17916
- Zhang, X.; Yang, J.; He, H.; Qu, H.; Luo, J. *An Interpretable Multi-Dimensional Fit Evaluation
  Framework for Online Apparel Size Recommendation.* Textiles 6(3), 75, 2026.
  https://doi.org/10.3390/textiles6030075
- Misra, R.; Wan, M.; McAuley, J. *Decomposing Fit Semantics for Product Size Recommendation in
  Metric Spaces.* RecSys 2018. (ModCloth / RentTheRunway fit datasets.)
