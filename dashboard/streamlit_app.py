"""Streamlit dashboard for the Garment Ease Engine.

Talks to the FastAPI service over HTTP: POST /recommend and POST /feedback.
Run with: streamlit run dashboard/streamlit_app.py
"""

import csv
import os
import urllib.error
from pathlib import Path

import pandas as pd
import streamlit as st

from api import (
    FIT_OPTIONS,
    REGIONS,
    build_payload,
    describe_region,
    error_text,
    post_json,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CHART = ROOT / "data" / "sample_garments.csv"
MAX_GARMENTS = 50
DEFAULT_GARMENTS = (("S", 94, 76, 100), ("M", 100, 82, 106), ("L", 106, 88, 112))

st.set_page_config(page_title="Garment Ease — fit recommender", layout="wide")


def sample_rows():
    """Sample size chart from data/, with a tiny fallback if the file is missing."""
    try:
        with SAMPLE_CHART.open(newline="", encoding="utf-8") as handle:
            return [
                {"size": row["size"], "chest": float(row["chest"]),
                 "waist": float(row["waist"]), "hip": float(row["hip"])}
                for row in csv.DictReader(handle)
            ]
    except (OSError, KeyError, ValueError):
        return [{"size": size, "chest": chest, "waist": waist, "hip": hip}
                for size, chest, waist, hip in DEFAULT_GARMENTS]


def new_row(size="", chest=100.0, waist=82.0, hip=106.0):
    """One editable garment row with a stable id so remove/add keeps widget state straight."""
    st.session_state.row_seq += 1
    return {"id": st.session_state.row_seq, "size": size,
            "chest": chest, "waist": waist, "hip": hip}


def load_sample():
    st.session_state.garments = [new_row(**row) for row in sample_rows()]


if "row_seq" not in st.session_state:
    st.session_state.row_seq = 0
if "garments" not in st.session_state:
    load_sample()

api_url = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("GARMENT_API_URL", "http://localhost:8000"),
    help="Base URL of the Garment Ease API, e.g. http://localhost:8000. No trailing slash needed.",
)

st.title("Garment fit recommender")
st.warning("Research prototype, not validated fit advice. Use test data only; measurements are sent to the configured API.")
st.caption(
    "Enter your body measurements and a garment size chart. The API ranks sizes by how "
    "close each one is to the ease you prefer. All measurements are in centimetres (cm)."
)

st.header("Your measurements")
chest_col, waist_col, hip_col = st.columns(3)
chest = chest_col.number_input(
    "Chest (cm)", min_value=30.0, max_value=250.0, value=96.0, step=0.5,
    format="%.1f", help="Body circumference around the fullest part of the chest.",
)
waist = waist_col.number_input(
    "Waist (cm)", min_value=30.0, max_value=250.0, value=80.0, step=0.5,
    format="%.1f", help="Body circumference at the natural waist.",
)
hip = hip_col.number_input(
    "Hip (cm)", min_value=30.0, max_value=250.0, value=102.0, step=0.5,
    format="%.1f", help="Body circumference around the fullest part of the hips.",
)

st.header("Garment size chart")
st.caption("Finished garment circumferences in cm. Add up to 50 sizes.")

for index, garment in enumerate(st.session_state.garments):
    name = garment["size"].strip() or f"row {index + 1}"
    size_col, chest_col, waist_col, hip_col, remove_col = st.columns([1, 1, 1, 1, 1.3])
    garment["size"] = size_col.text_input(
        f"Size label — {name}", value=garment["size"], max_chars=32,
        key=f"size_{garment['id']}",
    )
    garment["chest"] = chest_col.number_input(
        f"Chest (cm) — {name}", min_value=30.0, max_value=300.0,
        value=float(garment["chest"]), step=0.5, format="%.1f",
        key=f"chest_{garment['id']}",
    )
    garment["waist"] = waist_col.number_input(
        f"Waist (cm) — {name}", min_value=30.0, max_value=300.0,
        value=float(garment["waist"]), step=0.5, format="%.1f",
        key=f"waist_{garment['id']}",
    )
    garment["hip"] = hip_col.number_input(
        f"Hip (cm) — {name}", min_value=30.0, max_value=300.0,
        value=float(garment["hip"]), step=0.5, format="%.1f",
        key=f"hip_{garment['id']}",
    )
    if remove_col.button(f"Remove {name}", key=f"remove_{garment['id']}",
                         help=f"Remove size {name}"):
        st.session_state.garments.remove(garment)
        st.rerun()

add_col, reset_col, _ = st.columns([1, 1, 3])
if add_col.button("Add a size", disabled=len(st.session_state.garments) >= MAX_GARMENTS):
    st.session_state.garments.append(new_row())
    st.rerun()
if reset_col.button("Reset to sample chart"):
    load_sample()
    st.rerun()

st.header("Fit preference")
pref_col, stretch_col = st.columns(2)
fit_preference = pref_col.selectbox(
    "How should it fit?", FIT_OPTIONS, index=FIT_OPTIONS.index("regular"),
    format_func=str.title,
    help="Slim sits close to the body; regular is everyday ease; loose is relaxed.",
)
stretch_percent = stretch_col.slider(
    "Fabric stretch (%)", min_value=0, max_value=50, value=0,
    help="Illustrative stretch adjustment as a percentage of body circumference. "
         "Use 0 if unknown; this is not a measured textile model.",
)

current_payload = build_payload(chest, waist, hip, st.session_state.garments,
                                fit_preference, stretch_percent)
if st.session_state.get("last") and st.session_state.last["request"] != current_payload:
    del st.session_state.last

if st.button("Recommend sizes", type="primary"):
    st.session_state.pop("last", None)
    missing = [str(index + 1) for index, garment in enumerate(st.session_state.garments)
               if not garment["size"].strip()]
    if not st.session_state.garments:
        st.error("Add at least one garment size before recommending.")
    elif missing:
        st.error(f"Size label missing for row {', '.join(missing)}. Every row needs a label.")
    else:
        payload = current_payload
        try:
            response = post_json(api_url, "/recommend", payload)
        except urllib.error.HTTPError as error:
            st.error(error_text(error))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            st.error(f"Could not reach the API at {api_url}. Check it is running and the "
                     f"URL is correct. ({error})")
        else:
            st.session_state.last = {"request": payload, "response": response}

last = st.session_state.get("last")
if last:
    response = last["response"]
    ranked = response["ranked"]
    st.header("Recommended sizes")
    st.caption(
        f"Fit preference: {response['fit_preference'].title()} · "
        f"stretch {response['stretch_percent']:g}% · lower score is a closer overall fit."
    )

    if ranked:
        best = ranked[0]
        st.success(f"Best match: size {best['size']} (score {best['score']:.2f}).")

        summary = []
        for rank, item in enumerate(ranked, start=1):
            row = {"Rank": rank, "Size": item["size"], "Score": f"{item['score']:.2f}"}
            row.update({region.title(): describe_region(item["regions"][region])
                        for region in REGIONS})
            summary.append(row)
        st.table(pd.DataFrame(summary))

        for item in ranked:
            with st.expander(f"Size {item['size']} — score {item['score']:.2f}"):
                detail = [
                    {
                        "Region": region.title(),
                        "Body (cm)": f"{values['body_cm']:g}",
                        "Garment (cm)": f"{values['garment_cm']:g}",
                        "Target ease (cm)": f"{values['target_ease_cm']:g}",
                        "Actual ease (cm)": f"{values['actual_ease_cm']:g}",
                        "Stretch credit (cm)": f"{values['stretch_credit_cm']:g}",
                        "Gap vs target (cm)": f"{values['delta_cm']:g}",
                        "Result": describe_region(values),
                    }
                    for region, values in item["regions"].items()
                ]
                st.table(pd.DataFrame(detail))

        st.header("Feedback")
        st.caption("Record how a size actually fit. This is saved against the recommendation above.")
        with st.form("feedback_form"):
            size = st.selectbox("Size you tried", [item["size"] for item in ranked], index=None)
            actual_fit = {}
            fit_columns = st.columns(3)
            for column, region in zip(fit_columns, REGIONS):
                actual_fit[region] = column.radio(
                    f"{region.title()} fit", ["fits", "tight", "loose"], horizontal=True,
                    index=None, format_func=str.title, key=f"actual_{region}",
                )
            if st.form_submit_button("Save feedback"):
                if size is None or any(value is None for value in actual_fit.values()):
                    st.error("Choose the size you tried and a fit rating for every region.")
                else:
                    feedback = {"request": last["request"], "size": size, "actual_fit": actual_fit}
                    try:
                        saved = post_json(api_url, "/feedback", feedback)
                    except urllib.error.HTTPError as error:
                        st.error(error_text(error))
                    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
                        st.error(f"Could not reach the API at {api_url}. ({error})")
                    else:
                        st.success(f"Feedback saved (#{saved['id']}).")
