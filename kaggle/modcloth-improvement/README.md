# ModCloth fit-feedback — improvement experiment (Kaggle GPU, private)

Status: **complete and verified remotely**. Kernel
`ahmednourallah/garment-ease-modcloth-improvement-private` (private, version 1, Tesla T4,
finished 2026-11-24 19:33 UTC), code hash `9db70df2…` matches the pushed kernel (`CODE_MATCH`),
and every number below was recomputed from the downloaded predictions by `verify_run.py`.

## Question

Can 3-class fit (`small` / `fit` / `large`) be predicted from **pre-purchase inputs only** better
than the earlier MLP, when item identity is available as a categorical — and does the gain survive
on users that were never used for development? No garment measurements exist in this dataset, so
nothing here is physical garment ease.

## Split contract (deterministic, one development set, one untouched test set)

1. Users in file order → `RandomState(20261124).permutation` → first 20% = **DEV**. This is exactly
   the previous run's holdout (asserted: 9,592 users / 16,729 rows). It is the *only* set used for
   model/hyperparameter selection.
2. From the **remaining (original training) users** → `RandomState(20261125)` → first 15% =
   **FINAL TEST**, then **removed from all training rows** before any model is fitted in this run.
   These users were part of the earlier experiment's training data, so their labels could indirectly
   have influenced earlier choices carried into this experiment. This is a new user-disjoint test
   for this run, but not a fully independent test of the whole development history.
3. `FIT` = the rest.

| split | users | rows | role |
|---|---|---|---|
| fit | 32,611 | 55,985 | training (stage A); training + dev (stage B) |
| dev | 9,592 | 16,729 | selection only (previous run's holdout) |
| final | 5,755 | 10,076 | excluded from this run's fit and selection; used in earlier training |

Pairwise user-disjoint; no user spans splits; final rows absent from every fitting frame, from
feature statistics and from item vocabularies (`assert`s in `kernel.py`, audited in
`leakage_audit.json`). `item_seen` is defined against the fit-only item vocabulary.

## Models compared under the identical split

- **Control** — the previous run's MLP 64-64 (dropout 0.1, AdamW lr 1e-2, wd 1e-4, 200 epochs,
  batch 1024, inverse-frequency class weights), no item feature, unchanged.
- **Candidate A** — same MLP + regularized item embedding (dim 8, `padding_idx=0` for unseen items),
  wd 1e-3 / 1e-2.
- **Candidate B** — CatBoost `MultiClass` on GPU, `item_id` + `category` as categorical features
  (ordered target statistics fitted on train rows only), depth 6/8 × l2 3/8, 500 iterations,
  class weights. `task_type=GPU` confirmed in the receipt.

Selection rule, predeclared: **highest dev macro F1, tie-break balanced accuracy.**

### Stage A — dev (selection set)

| candidate | dev macro F1 | dev bal. acc |
|---|---|---|
| **cat_item_d8_l23** | **0.4362** | 0.5189 |
| cat_item_d8_l28 | 0.4352 | 0.5234 |
| mlp_emb8_wd0.01 | 0.4295 | 0.5283 |
| cat_item_d6_l23 | 0.4265 | 0.5263 |
| cat_item_d6_l28 | 0.4253 | 0.5266 |
| mlp_emb8_wd0.001 | 0.4244 | 0.5192 |
| control_mlp_64_64 | 0.3472 | 0.4256 |
| majority (`fit`) | 0.2706 | 0.3333 |

The control scores 0.3472 here versus 0.3600 in the previous run because 5,755 users that used to be
in training are now reserved for the final test — same code, less data, not a regression.
CatBoost fits took 7.5 s each on the T4; both MLPs took ~17–19 s for 200 epochs.

### Stage B — final test (10,076 rows / 5,755 users), scored once

| model | macro F1 | balanced acc | accuracy |
|---|---|---|---|
| majority (`fit`) | 0.2700 | 0.3333 | 0.6808 |
| control MLP 64-64 | 0.3624 | 0.4184 | 0.4128 |
| **winner `cat_item_d8_l23`** | **0.4303** | **0.5223** | 0.4555 |

**Effect: +0.0679 macro F1, +0.1039 balanced accuracy** for the item-aware CatBoost over the control.
Paired user-clustered bootstrap (2,000 resamples over the 5,755 final users):
95% CI **[0.0577, 0.0784]**, P(delta > 0) = 1.000. Independent stdlib recomputation gives
CI [0.0577, 0.0781] and the same point estimate.

Per-class F1: small 0.3702 (control 0.2527), fit 0.5242 (0.5233), large 0.3964 (0.3111) —
the gain sits in the minority classes; the majority `fit` class is essentially unchanged.
Winner confusion (true × pred, small/fit/large): `[[896, 384, 307], [2084, 2703, 2073], [273, 365, 991]]`;
support 1,587 / 6,860 / 1,629.

Matched-protocol check (both models fitted on **fit only**, dev never used, same 10,076 rows):
control 0.3392 vs winner 0.4305 macro F1. The improvement is therefore not an artefact of the
stage-B refit on dev, and no conclusion depends on which of the two variants is preferred.

Item slices (final): seen items 10,000 rows, control 0.3627 → winner 0.4312; unseen items 76 rows,
control 0.3163 → winner 0.2630 — too small to interpret, and it is the slice where an item feature
cannot help. **Say it plainly: the gain lives in items that already appear in training data.**

## Leakage and rights

- Never read: review text/summary, `quality` (rating), `length` (fit feedback), `user_name`,
  `user_id` (only hashed, salted, truncated to a cluster key), the `fit` label or anything derived
  from it. `waist`/`bust`/`shoe size` stay outside the v1 feature contract.
- `item_id` is a pre-purchase item attribute. Its target statistics are fitted inside CatBoost on
  training rows only; no label of a final-set user reaches the model.
- No early stopping or checkpoint selection anywhere — fixed iteration budgets, so the dev score is
  a plain selection set, not an early-stopping-contaminated one.
- Data: `rmisra/clothing-fit-dataset-for-size-recommendation` → `modcloth_final_data.json`,
  SHA-256 `9afbff64599f21f4755fa923acc2b32a166620e5e2ac72a97a1f4d876d92c3a4`, 82,790 rows.
  Origin: Misra, Wan & McAuley, RecSys 2018 (UCSD ModCloth). The Kaggle mirror is labelled
  CC BY 4.0; the original-source terms were **not** independently verified, so this stays an
  internal research use: raw records are never copied into outputs, only derived predictions and
  salted user keys.

## Limitations

- **Not garment ease.** No garment measurements exist here; the model never sees ease, only body
  inputs, purchased size, category and item identity.
- 99.2% of final rows use items that also appear in training, so this is a warm-item result.
  A cold-item claim is not available from this dataset split (76 rows).
- One deterministic split: the interval reflects final-user sampling only, not training-seed or
  split-seed variance.
- Seven dev configurations were compared; the winner's dev score carries mild selection optimism.
  The final users were excluded from this run's fit and selection, but appeared in earlier training;
  thus a completely independent evaluation of the entire development history is still needed.
- Accuracy falls below the majority-class baseline (0.456 vs 0.681) — the intended class-weighting
  trade-off for macro F1 / balanced accuracy; it is not a model that should be used for
  "will this customer be satisfied" accuracy claims.
- Better fit *classification* here means better ranking of size complaints per item class, not a
  calibrated probability of fit; probabilities are uncalibrated.

## Files

- `kernel.py` — exact code that ran remotely (hash-verified against the pulled kernel).
- `kernel-metadata.json` — private, GPU on, internet off, single dataset source.
- `precheck_split.py` — local rebuild of the split contract on the real file (asserts 9,592/16,729).
- `smoke_catboost.py` — local CPU check of the CatBoost call shapes/determinism (optional).
- `verify_run.py` — stdlib-only recomputation of every reported final number, the delta CI, the
  dev selection and the split assertions.
- `artifacts/` — `metrics.json`, `leakage_audit.json`, `run_meta.json`, `run.log`,
  `gpu_receipt.txt` (nvidia-smi, driver 580.178.04, CUDA 13.0, T4, matmul check),
  `test_predictions.csv`, `test_predictions_fitonly.csv`, `dev_predictions.csv`, kernel log.
  `run.log` is the full run log; the completion marker `DONE_MODCLOTH_IMPROVEMENT_V1` is printed
  to stdout after that file is written, so it lives in the retrieved Kaggle log, not in `run.log`.

Row-level predictions and the full GPU receipt are retained locally but excluded from Git. To independently verify, retrieve the private Kaggle outputs locally. Reproduce:

```bash
kaggle kernels push -p .                         # private, GPU, deterministic seeds
kaggle kernels output ahmednourallah/garment-ease-modcloth-improvement-private -p artifacts
python3 verify_run.py                            # fails loudly on any mismatch
```
