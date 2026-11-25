"""Research-only interpretation of Zhang et al.'s published jacket framework."""
import os
import urllib.error

import pandas as pd
import streamlit as st

from dashboard.api import error_text, post_json

st.set_page_config(page_title="2026 jacket research", layout="wide")
st.title("2026 jacket research framework")
st.warning("Experimental interpretation of one paper, not validated fit advice. It uses only the two jacket charts in the study and does not model fabric stretch. Its worked example and Jacket B shoulder intervals contain inconsistencies.")
st.caption("Zhang et al., Textiles (2026), DOI 10.3390/textiles6030075. Inputs are body measurements in cm. The height calculation uses a size-chart reference, not garment length.")
api_url = st.sidebar.text_input("API base URL", value=os.environ.get("GARMENT_API_URL", "http://localhost:8000"))
jacket = st.selectbox("Published jacket style", ["A", "B"], format_func=lambda x: "A — regular" if x == "A" else "B — tight")
cols = st.columns(4)
bust = cols[0].number_input("Bust (cm)", min_value=1.0, max_value=300.0, value=84.4)
waist = cols[1].number_input("Waist (cm)", min_value=1.0, max_value=300.0, value=73.5)
shoulder = cols[2].number_input("Shoulder width (cm)", min_value=1.0, max_value=150.0, value=39.3)
height = cols[3].number_input("Height (cm)", min_value=1.0, max_value=300.0, value=169.0)
rule = "error"
if jacket == "B":
    st.info("The paper's shoulder segment 2 overlaps segments 3 and 4. The default refuses to score affected sizes. Alternate interpretations are unvalidated.")
    rule = st.selectbox("Shoulder overlap interpretation", ["error", "printed_segment2", "printed_segments3_4"],
                        format_func=lambda x: {"error": "Do not guess (default)", "printed_segment2": "Use segment 2", "printed_segments3_4": "Use segment 3 or 4"}[x])
if st.button("Rank published jacket sizes", type="primary"):
    payload = {"jacket": jacket, "body": {"bust": bust, "waist": waist, "shoulder": shoulder, "height": height}, "b_shoulder_rule": rule}
    try:
        result = post_json(api_url, "/research/zhang2026/recommend", payload)
    except urllib.error.HTTPError as error:
        st.error(error_text(error))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        st.error(f"Could not reach the API: {error}")
    else:
        st.caption(result["warning"])
        st.table(pd.DataFrame([{"Rank": i, "Size": r["size"], "Overall fit score": round(r["ofs"], 2)} for i, r in enumerate(result["ranked"], 1)]))
        for item in result["ranked"]:
            with st.expander(f"Size {item['size']} — score {item['ofs']:.2f}"):
                st.table(pd.DataFrame([{"Region": region, "Body (cm)": v["body_cm"], "Garment / height reference (cm)": v["garment_cm"], "Actual ease (cm)": v["garment_ease_cm"], "Ideal ease (cm)": v["ideal_ease_cm"], "Error (cm)": v["fit_error_cm"], "Segment": v["segment"], "Weight": v["segment_weight"], "Section score": v["sectional_score"]} for region, v in item["regions"].items()]))
