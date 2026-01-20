import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame, scaler: StandardScaler = None):
    """
    Build engineered features for fraud / anomaly detection.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data WITHOUT the fraud label.
        Expected columns include at least:
        - Time
        - Amount

    scaler : StandardScaler, optional
        If provided, this scaler will be used to transform the Amount column.
        If None, a new scaler will be fitted.

    Returns
    -------
    X : np.ndarray
        Final feature matrix to be used by ML models.

    scaler : StandardScaler
        Fitted scaler object (needed later for inference).

    feature_df : pd.DataFrame
        DataFrame containing engineered features (useful for debugging/monitoring).
    """

    # Work on a copy to avoid modifying raw input data
    feature_df = df.copy()

    # -----------------------------
    # Amount scaling
    # -----------------------------
    if scaler is None:
        scaler = StandardScaler()
        feature_df["Amount_scaled"] = scaler.fit_transform(
            feature_df[["Amount"]]
        )
    else:
        feature_df["Amount_scaled"] = scaler.transform(
            feature_df[["Amount"]]
        )

    # -----------------------------
    # Transaction velocity feature
    # -----------------------------
    feature_df["transaction_count"] = 1
    feature_df["tx_per_min"] = (
        feature_df["transaction_count"]
        .rolling(window=60)
        .sum()
    )

    # -----------------------------
    # Rolling amount statistics
    # -----------------------------
    feature_df["rolling_mean_amount"] = (
        feature_df["Amount"]
        .rolling(window=60)
        .mean()
    )

    feature_df["rolling_std_amount"] = (
        feature_df["Amount"]
        .rolling(window=60)
        .std()
    )

    # -----------------------------
    # Time-based feature
    # -----------------------------
    feature_df["hour"] = (feature_df["Time"] // 3600) % 24

    # -----------------------------
    # Handle missing values from rolling operations
    # -----------------------------
    feature_df = feature_df.fillna(0)

    # -----------------------------
    # Final feature matrix
    # -----------------------------
    X = feature_df.values

    return X, scaler, feature_df
