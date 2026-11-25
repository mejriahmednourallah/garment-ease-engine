# EaseLab — Garment Ease Engine

A local, explainable garment-size ranking prototype. Enter body and finished-garment measurements, a fit preference and an approximate fabric stretch; compare sizes and record real fit feedback for future research.

> **Research prototype, not a validated sizing system.** Targets, weights, fit bands, stretch adjustment and example sizes are illustrative assumptions—not learned from participants or copied from either study. No ML model or GAN is included in the size-ranking application; the separate exploratory ModCloth experiment below does train a classifier. Do not use results as safety-critical or reliable purchasing advice.

## Quick start

Python 3.10+ recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r dashboard/requirements.txt -r requirements-dev.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, from the same directory:

```bash
source .venv/bin/activate
streamlit run dashboard/streamlit_app.py
```

Open the Streamlit URL shown in the terminal. The API's interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`. Set `GARMENT_API_URL` if the API is elsewhere. Run `python -m pytest -q` for the tests.

## How it works

All measurements are circumferences in **cm**, not standard body-size labels. Supply finished garment measurements, not the wearer's measurements from a retailer's body size chart. For each size and region, actual ease is `garment - body`. The prototype approximates stretch as `body × stretch_percent / 100`; it compares `actual ease + stretch adjustment` with target ease, then sums squared differences with weights chest 0.4, waist 0.3, hip 0.3. Lower scores rank first. Scores are relative penalties, not probabilities. Stretch is a heuristic, not a textile mechanics model; real stretch depends on direction, force and construction.

The assumed target ease (cm) is:

| Preference | Chest | Waist | Hip |
|---|---:|---:|---:|
| Slim | 2 | 1 | 2 |
| Regular | 8 | 5 | 8 |
| Loose | 14 | 12 | 14 |

Regional labels indicate deviation from this adjusted target, not an independently validated assessment of comfort. Length, shoulders, shape, mobility, garment type, fabric direction and many other fit factors are not modeled.

## API example

```bash
curl -s http://127.0.0.1:8000/recommend -H 'Content-Type: application/json' -d '{
  "chest":96,"waist":80,"hip":102,
  "garments":[{"size":"S","chest":94,"waist":76,"hip":100},
              {"size":"M","chest":100,"waist":82,"hip":106},
              {"size":"L","chest":106,"waist":88,"hip":112}],
  "fit_preference":"regular","stretch_percent":5
}'
```

See `CONTRACT.md` or `/docs` for response details and `POST /feedback`. `data/sample_garments.csv` is a **made-up demo chart**, not study data; `data/sample_fits.csv` is an empty template, not real feedback. `data/research/` contains selected published aggregate results and the two 2026 jacket charts, with provenance and source discrepancies documented; it contains no participant-level dataset. The dashboard saves explicitly submitted feedback and its inputs in SQLite (`garment_ease.db` by default); set `GARMENT_DB_PATH` to change the file. No automatic training occurs.

### Privacy and deployment

Body measurements and feedback are sensitive personal data. The demo has no authentication, encryption at rest, consent management, retention policy, access controls, or user data deletion UI. It is intended for localhost use with test data only; do not expose it publicly or collect real participant data without adding those protections and establishing appropriate consent and data governance. Deleting the SQLite file removes stored feedback; protect backups too. The dashboard's configurable API URL sends the submitted measurements to that server—use only a trusted endpoint.

## Research context and data

**[`docs/RESEARCH.md`](docs/RESEARCH.md)** is the full account: the two source papers, what each did and released, exactly how this implementation deviates, every measured result, and what can and cannot be claimed. **[`docs/eda/EDA.md`](docs/eda/EDA.md)** is the reproducible exploratory analysis of the synthetic corpora.

- Dik et al. (2023), [ANN fit prediction and GAN visualization](https://doi.org/10.1016/j.heliyon.2023.e17916), studied 120 participants and four garment types (blazers, long- and short-sleeved shirts, trousers), using body measurements and psychographic characteristics. The paper reports aggregate results, but its participant-level data are **available on request**, not included here. This MVP does not reproduce its trained ANN, GAN or visualization.
- Zhang et al. (2026), [Interpretable Multi-Dimensional Fit Evaluation Framework](https://doi.org/10.3390/textiles6030075), evaluated 270 women in two jacket styles. It publishes garment charts and aggregate results, but we found no downloadable participant-level measurements and fit labels; its data statement directs further inquiries to the authors. It uses garment-specific ideal ease, segment-based weighting and differentiated penalties.

**How we adapted:** Because neither paper supplied an accessible participant-level dataset for this project, we improvised a transparent demonstration baseline with invented sample garment sizes, illustrative ease targets and weights, and the simpler squared-error formula. These assumptions are not the papers' fitted models, data, or validated results. The [research data notes](data/research/README.md) distinguish the published tables from missing observations. A separate ModCloth fit-feedback experiment can explore prediction from available customer and item information, but ModCloth lacks finished-garment measurements and cannot validate this ease engine.

### 2026 jacket framework — research implementation

A separate research mode implements Zhang et al.'s published 2026 jacket method (`app/paper2026.py`), without replacing the generic weighted-squared-error baseline. It uses the study's two jacket size charts, style-specific ideal ease, six regional ease segments and segment-specific weights. For each region it computes absolute error `abs(actual ease - ideal ease)`, multiplies by the selected segment weight, then sums regional scores; lower is better. Inputs are bust, waist, shoulder width and height in cm. This paper does not model fabric stretch; its height component uses the chart's height reference, **not** garment length.

Open the **Zhang 2026** page in the Streamlit sidebar, or call `POST /research/zhang2026/recommend`:

```bash
curl -s http://127.0.0.1:8000/research/zhang2026/recommend \
  -H 'Content-Type: application/json' \
  -d '{"jacket":"A","body":{"bust":84.4,"waist":73.5,"shoulder":39.3,"height":169.0}}'
```

**Reproducibility warning:** the paper's printed worked example disagrees with its stated waist target and some segment weights; the implemented equations/tables rank its reconstructed example M, L, S, but produce different overall scores. Jacket B's printed shoulder intervals overlap: by default the API refuses affected cases (HTTP 422); an explicit `b_shoulder_rule` chooses `printed_segment2` or `printed_segments3_4`, neither independently validated. See `app/paper2026.py` for the exact ambiguities and `tests/test_paper2026.py` for numerical checks. The published participant-level outcomes are unavailable, so this does **not** reproduce the paper's accuracy or validate recommendations.

### ModCloth fit-feedback experiment

A separate exploratory experiment trained a three-class fit classifier on ModCloth using a private Kaggle Tesla T4. On a user-grouped evaluation split, the class-weighted model achieved macro F1 **0.3600** and balanced accuracy **0.4217**, versus **0.2706** and **0.3333** for a majority-class baseline; its overall accuracy fell from **0.6834** to **0.4018**. These are not ease-engine results. The evaluation split was reused while developing variants, so these numbers are exploratory and need confirmation on untouched users. Most test items also appeared in training. The dataset lacks garment measurements; source rights require further verification. See [experiment details and artifacts](kaggle/modcloth-experiment/README.md).

A follow-up Kaggle GPU comparison added an item-aware CatBoost classifier and used a customer-grouped test set of 10,076 rows. Macro F1 increased from **0.3624** for the matched MLP control to **0.4303**, and balanced accuracy from **0.4184** to **0.5223**; accuracy increased from **0.4128** to **0.4555**, still below the majority baseline's **0.6808**. Nearly all test rows involved previously seen items. The test users were excluded from fitting in this follow-up but appeared in earlier training, so this is not fully independent of the whole development history. No garment measurements or ease labels were available. See the [follow-up methods and artifacts](kaggle/modcloth-improvement/README.md). The CatBoost model remains an experiment, separate from the ease-based size-ranking application.

### Synthetic garment-fit data

A user-provided synthetic dataset and generator were evaluated separately from real ModCloth feedback. Its measurements, charts, preferences and labels are **simulated**, not participant observations from either paper. Results measure agreement with that simulator, not real-world fit accuracy.

On the synthetic customer-held-out test split, the actual `app.main.rank_sizes` scorer could process **603 of 3,600** customer–garment pairs without imputing missing measurements; all eligible pairs were dresses. It matched the simulated preferred size on **111/603 (18.4%)**, versus **100/603 (16.6%)** for always choosing M. Both methods always choose a size and therefore count all **234** `NONE` (no acceptable size) cases as incorrect; always predicting `NONE`, the majority label in the eligible training pairs, scored **234/603 (38.8%)** overall and **0/369** where a size was preferred. On just the **369** test pairs with a preferred size, the scorer matched **111/369 (30.1%)** versus **100/369 (27.1%)** for M. These are scenario-specific simulator results, not real-world accuracy. Candidate coverage, lack of an abstention option and simulator assumptions limit interpretation.

A **separate experimental candidate** combines an interpretable fitted size-choice model with an acceptability gate; it is not the application's production scorer. On the same 603 synthetic dress pairs it matched **321/603 (53.2%)**, including **179/234** correct `NONE` cases and **142/369** correct preferred sizes. It incorrectly predicted `NONE` for **145/369** pairs that did have a simulated preferred size, and failed to predict `NONE` for **55/234** cases where none was acceptable. Gate and model settings were selected using the supplied training and validation partitions. The test split had already been inspected when the earlier baseline was reported, so this follow-up is exploratory, not an untouched end-to-end assessment. This candidate uses synthetic-specific fields such as simulated region/chart-region matching; it does not establish production transfer or real-world fit accuracy. This candidate mixes a learned model and a gate whose threshold was tuned on total accuracy, an objective that rewards abstaining; the ablation below shows the gate alone does not hold up. The experimental source is `kaggle/synthetic-ease-experiment/garment_synth/evaluate_candidate_easelab.py` (locally available, not integrated into the app).

A second round used **three freshly generated synthetic seeds** (20270101–03), never used for earlier tuning, to test whether the improvement is robust rather than an artefact of the already-inspected split.

**Learned size-choice model with no abstention** (so it cannot win by refusing to answer), same eligible cohort per seed:

| Seed | Existing scorer: conditional / total / κ | Learned, no gate: conditional / total / κ |
|---|---|---|
| 20270101 | 34.1% / 19.6% / 0.116 | **65.9%** / 37.8% / 0.309 |
| 20270102 | 30.1% / 20.1% / 0.127 | **60.6%** / 40.4% / 0.326 |
| 20270103 | 24.8% / 15.6% / 0.078 | **63.8%** / 40.2% / 0.322 |
| Mean | 29.7% / 18.4% / 0.107 | **63.4%** / 39.5% / 0.319 |

The gain holds on all three seeds and is a **sizing** gain, not an abstention trick: the model always names a size, and `NONE` pairs still count as wrong. Synthetic-only fields (`sim_region`, `region_match`) were removed from its features.

**The abstention gate alone is not a reliable improvement.** With its threshold chosen to maximise balanced accuracy on training/validation (not total accuracy), it scored 43.7%, 35.5% and 34.4% total against always-`NONE` at 42.6%, 33.3% and 37.1%, while conditional accuracy fell to 19.2%, 16.1% and 12.5% and it abstained on 62–68% of pairs. Thresholds tuned to total accuracy inflate scores precisely because 30–43% of pairs are `NONE`; total top-1 is therefore reported next to conditional accuracy and Cohen's κ.

**Why the learned model wins is also a caution.** Its labels come from the same simulator, so a flexible model can partly recover the generator's rule using fields the three-region ease score ignores — garment type, style, fabric, and the retailer's nominal body chart. That demonstrates the ease formula discards signal available in these synthetic features; it does **not** establish that real customers behave the same way. Results are simulator agreement only, cover complete-measurement pairs only (this simulator's eligible pairs are dresses), and are not a validation of the application or of either paper.

The supplied package's original learned-model report compared against its *stand-in* baseline, not the application. All evaluations here ran locally on CPU; no Kaggle GPU run for this dataset has been verified. Source archive SHA-256: `611d9ede782fddad1bc8b35a32daf00c96250884db17a54c04b2f87047a3cef9`. Aggregate result files are in `kaggle/synthetic-ease-experiment/garment_synth/results/`.

### Next steps before ML

Obtain permissioned, participant-level measurements, actual garment measurements and regional fit labels; record garment type, fabric properties and consent. Establish a documented schema and dataset rights, then split evaluation by participant (and, where relevant, garment or style) to avoid leakage. Compare interpretable baselines with learned target-ease models on untouched data and report subgroup uncertainty. A visualization model would need separate suitable 3D or image data and evaluation. The current feedback form alone is not enough to claim a trained or validated model. No expensive compute is needed for the deterministic MVP; Kaggle can be considered once appropriate data and a justified experiment exist.
