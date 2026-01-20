# ===============================
# Fraud / Anomaly Detection Dashboard
# Streamlit app
# ===============================

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Add project root to Python path
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# -------------------------------
# Import feature pipeline
# -------------------------------
from src.features import build_features

# -------------------------------
# Load trained model and scaler
# -------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models/isolation_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")

# Verify model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler file not found! Make sure they exist in the models/ folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------
# Streamlit Layout
# -------------------------------
st.title("Fraud / Anomaly Detection Dashboard")
st.markdown("""
Upload a CSV of transactions to analyze. The system will build features, predict anomaly scores, and flag suspicious transactions.
""")

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Upload transaction CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Drop columns that should not be used for model
    # -------------------------------
    for col in ["Class", "anomaly_score", "anomaly_flag"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Build features
    # -------------------------------
    X, _, feature_df = build_features(df, scaler=scaler)

    # -------------------------------
    # Ensure feature count matches model
    # -------------------------------
    st.write("Number of features passed to model:", X.shape[1])
    if X.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch! Model expects {model.n_features_in_} features, but input has {X.shape[1]}.")
        st.stop()

    # -------------------------------
    # Predict anomalies
    # -------------------------------
    scores = model.decision_function(X)
    flags = model.predict(X)

    feature_df["anomaly_score"] = scores
    feature_df["anomaly_flag"] = (flags == -1).astype(int)

    # -------------------------------
    # Show processed data
    # -------------------------------
    st.subheader("Processed Data with Anomaly Scores")
    st.dataframe(feature_df.head())

    # -------------------------------
    # Anomaly summary
    # -------------------------------
    st.subheader("Anomaly Summary")
    st.write(feature_df["anomaly_flag"].value_counts())

    # -------------------------------
    # Optional download
    # -------------------------------
    csv = feature_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Data with Anomalies",
        data=csv,
        file_name='evaluated_transactions.csv',
        mime='text/csv',
    )
