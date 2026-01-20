#  Real-Time Fraud & Anomaly Detection System (Unsupervised ML)

An **end-to-end, production-style machine learning system** that detects suspicious credit card transactions **without labeled fraud data**, simulating real-world industry constraints.

This project complements supervised churn prediction by covering the **unsupervised anomaly detection** side of applied machine learning.

---

##  Why This Project Matters

In real financial systems:

- Fraud labels are **rare, delayed, or unreliable**
- Models must detect **abnormal behavior**, not known fraud patterns
- Feature engineering matters more than algorithms
- Deployment and monitoring matter as much as model accuracy

This project demonstrates **industry-ready ML system design**, not just notebook experimentation.

---

##  Problem Statement

- Detect anomalous / suspicious transactions in real time
- No fraud labels available during training
- Balance business cost of false positives vs false negatives
- Serve predictions through an API and dashboard

---


---

##  Dataset

- **Source:** Kaggle Credit Card Fraud Dataset  
- **Transactions:** 284,807  
- **Features:** PCA-transformed (`V1`–`V28`), `Time`, `Amount`
- **Fraud Rate:** ~0.17%

 Fraud labels (`Class`) are used **only for post-training sanity checks**, never during training.

---

## Feature Engineering (Core Strength)

Custom behavioral and temporal features engineered:

- Transaction velocity
- Rolling mean and standard deviation of amounts
- Time-based features (hour of day)
- Transaction frequency per minute
- Scaled transaction amount

 Initial rows may contain zeros due to rolling window initialization — this is **expected and correct behavior**.

---

## Models Used

### Isolation Forest (Primary Model)

- Handles high-dimensional data efficiently
- Scales well for large datasets
- Ideal for unsupervised anomaly detection

(Optional comparison with Local Outlier Factor for justification.)

---

## Evaluation Without Labels

Because fraud labels are unavailable in real-time systems:

- Anomaly score distribution analysis
- Percentile-based thresholding
- Manual inspection of flagged transactions
- Post-hoc sanity check using true labels

### Example Sanity Check

| Anomaly Flag | Non-Fraud | Fraud |
|-------------|-----------|-------|
| Normal      | 281,722   | 236   |
| Anomalous   | 2,593     | 256   |

 Strong enrichment of fraud among detected anomalies.

---

## Deployment

###  FastAPI (Real-Time Inference)

Start the API server:

```bash
uvicorn src.api:app --reload
```

### Streamlit Dashboard

Launch the dashboard:
```bash
streamlit run dashboards/app.py
```
### Environment Setup

```bash
python -m venv fraud_env
source fraud_env/bin/activate   
fraud_env\Scripts\activate
pip install -r requirements.txt
```

