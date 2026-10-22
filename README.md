# Customer Churn Prediction System

Predicting which telecom customers are likely to cancel their subscription
("churn"), so the business can target retention offers at the customers most
at risk — before they leave.

Built with **Python, scikit-learn and SMOTE** (imbalanced-learn).

---

## Problem

Acquiring a new customer costs far more than keeping an existing one, yet
most churn happens silently: by the time a customer cancels, it is too late
to win them back. This project builds a machine-learning model that scores
each customer with a churn probability, using their demographics, contract,
subscribed services, billing behaviour and support-call history.

The hard part is the **class imbalance**: only ~19% of customers churn, so a
naive model that always predicts "will stay" is 81% accurate — and completely
useless. This project handles the imbalance explicitly with SMOTE and judges
models on recall, F1 and ROC-AUC, not accuracy alone.

## Dataset

`generate_data.py` creates a realistic synthetic telecom dataset
(`data/churn.csv`, **2,000 customers, 22 columns**) modelled on the classic
Telco Customer Churn schema:

- **Demographics** — gender, senior citizen, partner, dependents
- **Account** — tenure (months), contract type, paperless billing, payment method
- **Services** — phone, multiple lines, internet (DSL / fibre / none), online
  security, backup, device protection, tech support, streaming TV/movies
- **Billing & behaviour** — monthly charges, total charges, support calls
- **Target** — `churn` (1 = left, 0 = stayed); overall churn rate **18.9%**

The label is generated from a latent probability model with realistic drivers
(month-to-month contracts, high charges, low tenure, fibre internet and 3+
support calls raise churn risk; long contracts, tech support and online
security reduce it), so the patterns the model learns are the ones a real
retention team would recognise.

## Approach / Pipeline (`train.py`)

1. **EDA** — shape, missing values, churn rate overall and by contract type,
   churned-vs-stayed averages, feature correlations with churn.
2. **Preprocessing** — `ColumnTransformer`: one-hot encoding for the 15
   categorical features, standard scaling for the 5 numeric features
   (46 encoded features in total).
3. **Split** — stratified 80/20 train/test split (test set never touched
   during training or resampling).
4. **SMOTE** — oversamples the minority class in the *training* split only
   (1,600 → 2,596 rows, balanced 50/50).
5. **Models** — Logistic Regression, Random Forest (300 trees) and Gradient
   Boosting, all trained on the SMOTE-balanced data.
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC and confusion
   matrix on the held-out test set.
7. **Model selection & saving** — the best model by ROC-AUC is saved as a
   single inference pipeline (preprocessor + model) in `model.pkl`, so it can
   score raw customer rows directly with `joblib.load(...)`.

## Results (held-out test set, 400 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.735 | 0.392 | **0.747** | **0.514** | **0.801** |
| Gradient Boosting | **0.810** | **0.492** | 0.387 | 0.433 | 0.788 |
| Random Forest | 0.803 | 0.467 | 0.373 | 0.415 | 0.760 |

**Best model: Logistic Regression (ROC-AUC 0.801).** The tree ensembles are
more accurate overall, but they achieve it by missing most churners
(recall ≈ 0.38). Logistic Regression catches **75% of all churners**
(recall 0.747) at the cost of more false alarms — the right trade-off for a
retention campaign, where contacting a happy customer is cheap but losing a
churner is expensive. Its confusion matrix: TN 238, FP 87, FN 19, TP 56.

**Strongest churn signals** (largest standardised coefficients):
contract type (two-year contracts strongly protective, month-to-month
risky), tenure (longer = safer), monthly charges, and having online security
or tech support (both protective) — matching the EDA, where month-to-month
customers churn at 25.0% vs only 4.8% on two-year contracts.

## How to Run

```bash
pip install -r requirements.txt

python3 generate_data.py   # writes data/churn.csv
python3 train.py           # EDA, SMOTE, trains 3 models, saves model.pkl
```

`train.py` regenerates the dataset automatically if `data/churn.csv` is
missing. To score new customers with the saved model:

```python
import joblib
model = joblib.load("model.pkl")
churn_probability = model.predict_proba(new_customers_df)[:, 1]
```

## Project Structure

```
customer-churn-prediction/
├── generate_data.py     # synthetic dataset generator (seeded, reproducible)
├── train.py             # EDA + preprocessing + SMOTE + training + evaluation
├── data/
│   └── churn.csv        # 2,000-row dataset
├── model.pkl            # saved best-model inference pipeline
├── requirements.txt
└── README.md
```

## Future Improvements

- **Threshold tuning** — pick the decision threshold from a cost model
  (cost of a retention offer vs. cost of a lost customer) instead of 0.5.
- **Calibration** — Platt/isotonic calibration so predicted probabilities can
  be used directly for campaign budgeting.
- **More models** — XGBoost/LightGBM and a small neural network comparison.
- **Real data** — validate on the public IBM Telco Churn dataset.
- **API deployment** — wrap `model.pkl` in a FastAPI scoring endpoint.
- **Explainability** — SHAP values per customer, so retention agents know
  *why* a customer is flagged, not just that they are.

## Tech Stack

Python 3 · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · joblib
