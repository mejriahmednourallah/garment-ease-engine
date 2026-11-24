# Garment Ease dashboard

Streamlit UI for the Garment Ease API (`POST /recommend`, `POST /feedback`). Runs the API
calls server-side, so no CORS setup is needed.

## Run

```bash
.venv/bin/pip install -r dashboard/requirements.txt
.venv/bin/python -m uvicorn app.main:app --port 8000   # API, separate terminal
.venv/bin/streamlit run dashboard/streamlit_app.py    # UI
```

Point the dashboard at another API with `GARMENT_API_URL=http://host:8000` or the
"API base URL" field in the sidebar. The sample chart comes from `data/sample_garments.csv`.

## Accessibility

- Every input has a visible, unique label and units in the label.
- Fit results are written out in words ("Tight — needs 3.5 cm more room"), never colour-only.
- Results use static HTML tables (`st.table`), which screen readers can read, rather than
  canvas-rendered grids.
- Errors are surfaced as alert text with a recovery hint; keyboard order matches visual order.

## Tests

```bash
.venv/bin/python -m pytest dashboard -q
```

Covers payload mapping, JSON POST headers/body, 422 error formatting and fit wording.
