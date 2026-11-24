# EaseLab — Garment Ease Engine

A local, explainable garment-size ranking prototype. Enter body and finished-garment measurements, a fit preference and an approximate fabric stretch; compare sizes and record real fit feedback for future research.

> **Research prototype, not a validated sizing system.** Targets, weights, fit bands, stretch adjustment and example sizes are illustrative assumptions—not learned from participants or copied from either study. No ML model or GAN is trained or included. Do not use results as safety-critical or reliable purchasing advice.

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

- Dik et al. (2023), [ANN fit prediction and GAN visualization](https://doi.org/10.1016/j.heliyon.2023.e17916), studied 120 participants and four garment types (blazers, long- and short-sleeved shirts, trousers), using body measurements and psychographic characteristics. This MVP does not reproduce its trained ANN, GAN or visualization. An open article does not automatically grant rights to use its data; verify terms before reuse.
- Zhang et al. (2026), [Interpretable Multi-Dimensional Fit Evaluation Framework](https://doi.org/10.3390/textiles6030075), evaluated 270 women in two jacket styles. It uses garment-specific ideal ease, segment-based weighting and differentiated penalties. This MVP is inspired by regional ease but uses the requested simpler squared-error formula, different inputs and illustrative constants; its results are not the paper's reported results. The paper's data statement directs further inquiries to its corresponding authors; no participant-level dataset is included here.

### Next steps before ML

Obtain permissioned, participant-level measurements, actual garment measurements and regional fit labels; record garment type, fabric properties and consent. Establish a documented schema and dataset rights, then split evaluation by participant (and, where relevant, garment or style) to avoid leakage. Compare interpretable baselines with learned target-ease models on untouched data and report subgroup uncertainty. A visualization model would need separate suitable 3D or image data and evaluation. The current feedback form alone is not enough to claim a trained or validated model. No expensive compute is needed for the deterministic MVP; Kaggle can be considered once appropriate data and a justified experiment exist.
