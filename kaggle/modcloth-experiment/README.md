# ModCloth fit-feedback — partial experiment (Kaggle GPU, private)

Status: **complete and verified**. Kaggle kernel:
`ahmednourallah/garment-ease-modcloth-fit-gpu-private` (private, version 5, Tesla T4).

## Question

Can 3-class fit (`small` / `fit` / `large`) be predicted from **pre-purchase inputs only**
on the public ModCloth fit dataset, using a **user-grouped held-out split**, and is that model
better than a majority-class baseline on macro F1 / balanced accuracy?

## Data and licence

- Source: `rmisra/clothing-fit-dataset-for-size-recommendation` (Kaggle mirror of the UCSD
  ModCloth fit dataset), file `modcloth_final_data.json`.
- Provenance: Misra, Wan & McAuley, *Decomposing Fit Semantics for Product Size Recommendation
  in Metric Spaces*, RecSys 2018.
- Licence: the original-source reuse terms have not been independently established; check both the UCSD source and the Kaggle mirror before redistributing or commercial use. The CC BY 4.0 label recorded in `artifacts/run_meta.json` is the run's metadata, not independently verified here.
- SHA-256 of the exact file used: `9afbff64599f21f4755fa923acc2b32a166620e5e2ac72a97a1f4d876d92c3a4`.
- Raw records are **not** stored in this repo and not published: the kernel reads the read-only
  Kaggle mount and writes only per-row derived predictions (no user ids, no feature values).

## Leakage exclusions

Inputs: purchased size, height, hips, bra band, cup (ordinal), 6 category indicators,
missing-value flags, and three "relative size" interactions (body measure − size, standardized).

Excluded on purpose: `quality` (rating), `length` (fit feedback), review/summary text,
`user_name`, the `fit` label and anything derived from it. Item identity is **not** a model
feature; item ids are only used to label the holdout (see below).

## Split

Users shuffled with seed 20261124; 20% of the 47,958 users (9,592 users, 16,729 rows) held out
entirely. Asserted disjoint user sets. Train: 38,366 users / 66,061 rows.
Item vocabulary overlap is reported, not enforced: **16,645 of 16,729 holdout rows (99.5%) belong
to items that also appear in training ("seen items"); only 84 rows (69 users) are unseen-item rows.**
Metrics are reported separately for both groups, so seen-item metrics are explicitly labelled.

## Results (holdout, n = 16,729 rows / 9,592 users)

| | macro F1 | balanced acc | accuracy |
|---|---|---|---|
| majority baseline (`fit` everywhere) | 0.2706 | 0.3333 | 0.6834 |
| MLP 64-64 (class-weighted, 200 epochs) | **0.3600** | **0.4217** | 0.4018 |

Model confusion (rows = true, cols = pred, order small/fit/large):
`[[934, 821, 920], [2727, 4394, 4311], [540, 688, 1394]]`; support 2,675 / 11,432 / 2,622.
Per-class F1: small 0.2717, fit 0.5070, large 0.3015.

Subgroups: seen items macro F1 0.3600 (n = 16,645) vs majority 0.2709; unseen items macro F1 0.3624
(n = 84, too small to interpret).

## Honest reading

- The model beats the majority baseline on macro F1 (+0.089) and balanced accuracy (+0.088) but
  **loses on accuracy** (0.402 vs 0.683) — the usual class-weighting trade-off; it no longer
  predicts only the 69% majority class.
- Train macro F1 is 0.4024 versus 0.3600 on the reported holdout; this gap alone does not establish whether data, optimization, or overfitting is the limiting factor.
- Earlier model versions were compared on the same user holdout while changing weighting and training settings. This holdout was therefore reused for model development: the reported score is **exploratory, not an unbiased final test**. An unweighted earlier run nearly collapsed to the majority class. Confirm the selected configuration on a separate untouched user-grouped test set.
- The "body minus size" inputs subtract standardized features; the dataset's size code is not a garment circumference. These features are not physical ease.
- Skipped by design: item embeddings / item fit priors (label-derived), garment measurement
  inference, ordinal-loss heads, per-category models. These are the obvious next levers, and any
  item-level variant must be scored on the unseen-item slice, which here is only 84 rows —
  so a clean cold-item claim is not available from this split.

## Files

- `kernel.py` — the exact code that ran remotely (`CODE_MATCH` confirmed against the pushed kernel).
- `kernel-metadata.json` — private, GPU on, internet off, single dataset source.
- `verify_run.py` — recomputes macro F1 / balanced accuracy / accuracy / confusion / support from
  the retrieved predictions with stdlib only and asserts they match `metrics.json`.
- `artifacts/` — retrieved outputs: `metrics.json`, `run_meta.json`, `training_curve.csv`,
  `val_predictions.csv` (16,729 rows: row_id, item_seen, y_true, y_pred, three probabilities),
  raw kernel log.

Row-level predictions are retained locally but excluded from Git; retrieve the private Kaggle outputs locally to verify. Reproduce: `kaggle kernels push -p .` then `kaggle kernels output ahmednourallah/garment-ease-modcloth-fit-gpu-private -p artifacts`.
