from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from src.features import build_features
import os

# -------------------------------
# Load trained model and scaler
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../models/isolation_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Fraud / Anomaly Detection API")

# -------------------------------
# Input data schema
# -------------------------------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to DataFrame
    df = pd.DataFrame([transaction.dict()])

    # Build features
    X, _, _ = build_features(df, scaler=scaler)

    # Get anomaly score
    score = model.decision_function(X)[0]
    flag = model.predict(X)[0]  # -1 = anomaly, 1 = normal

    return {
        "anomaly_score": float(score),
        "anomaly_flag": int(flag == -1)
    }
