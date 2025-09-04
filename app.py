


import streamlit as st
import pandas as pd
from transformers import pipeline

# Force all text to be black
st.markdown(
    """
    <style>
    body, p, div, span, label, h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load OpenMed NER model (for disease extraction)
@st.cache_resource
def load_model():
    return pipeline("token-classification",
                    model="OpenMed/OpenMed-NER-OncologyDetect-SnowMed-568M",
                    aggregation_strategy="simple")

ner = load_model()

# Simple mapping of lab tests -> possible conditions
diagnosis_map = {
    "hemoglobin": "Anemia",
    "platelets": "Thrombocytopenia",
    "glucose": "Diabetes",
    "cholesterol": "Hyperlipidemia"
}

st.title("🩺 Medical Diagnostics Assistant")
st.write("Upload lab test results and extract possible conditions.")

uploaded_file = st.file_uploader("Upload CSV with test results", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    conditions = []
    for col in df.columns:
        if col.lower() in diagnosis_map:
            conditions.append(diagnosis_map[col.lower()])

    if conditions:
        st.subheader("🔎 Possible Conditions")
        st.write(", ".join(set(conditions)))
    else:
        st.info("No matching conditions found.")
