# SYNTHETIC garment-ease dataset and experiments

**Everything here is simulated.** It is for software development and stress-testing of an explainable ease-recommendation prototype. It is not real customer data, not evidence of fit accuracy, and not derived from any paper.

## Run

```bash
pip install -r requirements.txt
python generate_synthetic_garments.py --out data --seed 20261124          # default: 4000 customers, 160 garments, 6 pairs/customer
python generate_synthetic_garments.py --out data_small --preset small     # quick size
#   size flags: --n-customers --n-garments --pairs-per-customer --stress-customers --stress-garments --stress-pairs-per-customer
python fit_experiments.py --data data --out results                       # baselines + interpretable model, ~1 min
python build_report.py data results                                       # writes REPORT.md
python -m pytest tests -q                                                 # 42 validation tests (~10 s)
```

Same seed + sizes give byte-identical CSVs (see `data/MANIFEST.json`). Changing the stress-set size does not change the main set.

## Layout
- `data/main/` and `data/stress/`: `customers.csv`, `garments.csv`, `garment_sizes.csv`, `observations.csv` (model-visible + labels), `latent_*.csv` (simulator internals, never features); `splits_*.csv` in main only.
- `data/DATA_DICTIONARY.md`, `assumptions.json`, `dataset_stats.json`, `MANIFEST.json`, `README_SYNTHETIC.txt`.
- `REPORT.md`: generator, assumptions, sizes, class balance, results, limitations.

## Compatibility with the existing project
No application code, API or schema was available in this session, so none was inspected. Mapping to the starting model: `actual_ease_i = garment_{chest|waist|hip}_cm (garment_sizes) - {chest|waist|hip}_cm (customers)`; `desired_ease_i` and `weight_i` come from your config (`fit_preference`); the stand-in is `BASELINE_PRIOR` in `fit_experiments.py`. Replace it with the production config and confirm column names and the full-circumference convention against your schema before relying on any comparison. Generator and experiments import nothing from, and write nothing to, a production endpoint.
