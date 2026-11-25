# Exploratory data analysis — synthetic garment-fit corpora

All numbers below are computed by `kaggle/synthetic-ease-experiment/garment_synth/eda.py`
from the CSV files it reads. Regenerate with:

```bash
kaggle/synthetic-ease-experiment/.venv/bin/python kaggle/synthetic-ease-experiment/garment_synth/eda.py
```

> **These data are simulated.** Every distribution, threshold and correlation is a design
> assumption of the generator, not an observation of people. Nothing here estimates real
> garment fit, and none of it comes from the 2023 or 2026 studies.

## 1. Corpus inventory

| corpus | customers | garments | size rows | customer×garment pairs | candidate observations |
|---|---:|---:|---:|---:|---:|
| `data/main` | 4,000 | 160 | 885 | 24,000 | 132,677 |
| `data_clean/seed_20270101/main` | 4,000 | 160 | 869 | 24,000 | 130,475 |
| `data_clean/seed_20270102/main` | 4,000 | 160 | 882 | 24,000 | 132,441 |
| `data_clean/seed_20270103/main` | 4,000 | 160 | 893 | 24,000 | 134,143 |
| `data/stress` | 600 | 40 | 214 | 2,400 | 12,829 |

Mean candidate sizes per pair range from 5.44 to 5.59.

## 2. Missingness

Missing values are structural, not random noise: dimension-specific garment rows are emitted
only for the regions a garment type declares as measured, which is why trousers and shirts
carry no chest/hip garment values.

| column | missing rate |
|---|---:|
| `chest_cm` | 0.0% |
| `waist_cm` | 0.0% |
| `hip_cm` | 0.0% |
| `garment_chest_cm` | 30.5% |
| `garment_waist_cm` | 0.0% |
| `garment_hip_cm` | 49.0% |
| `stretch_crosswise_pct` | 0.0% |
| `label_body_hip_cm` | 49.0% |

### Missing garment measurements by garment type

| garment type | rows | chest missing | waist missing | hip missing |
|---|---:|---:|---:|---:|
| dress | 27,258 | 0.0% | 0.0% | 0.0% |
| hoodie | 15,771 | 0.0% | 0.0% | 100.0% |
| jacket | 9,470 | 0.0% | 0.0% | 100.0% |
| shirt | 19,387 | 0.0% | 0.0% | 100.0% |
| skirt | 9,212 | 100.0% | 0.0% | 0.0% |
| trousers | 31,239 | 100.0% | 0.0% | 0.0% |
| tshirt | 20,340 | 0.0% | 0.0% | 100.0% |

## 3. Eligibility for EaseLab's input contract

EaseLab requires chest, waist and hip for the customer and for every candidate size.
Pairs failing that contract are excluded rather than imputed.

| corpus | test pairs | eligible (all three regions present) | eligible share | of which `NONE` | garment types surviving |
|---|---:|---:|---:|---:|---|
| `data/main` | 3,600 | 603 | 16.8% | 234 | dress |
| `data_clean/seed_20270101/main` | 3,600 | 806 | 22.4% | 343 | dress |
| `data_clean/seed_20270102/main` | 3,600 | 532 | 14.8% | 177 | dress |
| `data_clean/seed_20270103/main` | 3,600 | 672 | 18.7% | 249 | dress |

This funnel is the single most important caveat for every score in the README:
the eligible cohort is a small, dress-only slice of each corpus.

## 4. Regional fit-label balance (corpus `data/main`, applicable rows only)

| region | tight | acceptable | loose | not applicable |
|---|---:|---:|---:|---:|
| chest | 20.6% | 27.3% | 21.5% | 30.5% |
| waist | 31.2% | 35.4% | 33.4% | 0.0% |
| hip | 14.8% | 20.3% | 16.0% | 49.0% |

Pairs with **no acceptable size**: 30.2% (7,247 of 24,000). This base rate is why total top-1
alone cannot be read as a sizing result — always answering `NONE` scores the base rate.

## 5. Garment ease by regional label

Ease is `garment − body`. The table shows the 5th / 50th / 95th percentile of ease for each
label, per region, alongside the fixed target EaseLab assumes for `regular` fit.

| region | tight | acceptable | loose | EaseLab `regular` target |
|---|---|---|---|---:|
| chest | -20.5 / -3.5 / 13.0 | -4.5 / 9.5 / 25.5 | 10.0 / 23.5 / 40.5 | 8 cm |
| waist | -27.0 / -7.5 / 8.5 | -6.0 / 7.0 / 22.5 | 8.5 / 22.0 / 41.0 | 5 cm |
| hip | -20.5 / -5.5 / 5.5 | -4.5 / 7.0 / 19.0 | 8.5 / 20.0 / 35.0 | 8 cm |

Acceptable ease spans a wide range for every region; a single fixed target per region is
a strong simplification of what the simulator treats as acceptable. This is a visible source
of the gap between the fixed-target scorer and the learned model.

### Median acceptable ease by stated preference

| preference | chest | waist | hip |
|---|---:|---:|---:|
| loose | 15.0 | 11.5 | 11.5 |
| regular | 9.5 | 7.0 | 6.5 |
| slim | 4.5 | 3.0 | 3.0 |

## 6. No-acceptable-size rate by simulator scenario

| scenario | pairs | none rate |
|---|---:|---:|
| S1_asymmetric_threshold | 9,444 | 28.6% |
| S2_stretch_coupling | 6,918 | 32.0% |
| S3_style_relative | 7,638 | 30.5% |

## 7. Signal available to each method

The fixed-target scorer consumes only body chest/waist/hip, garment chest/waist/hip,
`fit_preference` and one global stretch percentage. The corpora also carry garment type,
style, fabric class, a retailer nominal-body chart, per-size `size_rank` and
`measured_dims`. Those extra fields are prediction-time metadata, and the learned model uses
them. The simulator generates labels from a latent rule that depends on garment style, stretch
coupling and customer bias, so a model with more of those fields will fit it better — which is
agreement with the generator, not evidence about real customers.

## 8. Seed stability

| seed | test pairs | eligible | eligible share | none rate in eligible |
|---|---:|---:|---:|---:|
| data/main | 3,600 | 603 | 16.8% | 38.8% |
| data_clean/seed_20270101/main | 3,600 | 806 | 22.4% | 42.6% |
| data_clean/seed_20270102/main | 3,600 | 532 | 14.8% | 33.3% |
| data_clean/seed_20270103/main | 3,600 | 672 | 18.7% | 37.1% |

The eligible cohort and its `NONE` base rate both move with the seed, so a single-split
comparison is fragile. The results reported in the README span three seeds for that reason.

