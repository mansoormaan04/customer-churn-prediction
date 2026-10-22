"""Generate a realistic synthetic telecom customer churn dataset.

The dataset mimics the well-known Telco Customer Churn schema: demographics,
account information, subscribed services and billing details, plus a binary
``churn`` label (1 = customer left, 0 = customer stayed).

The label is drawn from a latent churn-probability model, so the usual
real-world patterns hold (month-to-month contracts, high monthly charges,
low tenure, fibre internet and frequent support calls all raise the churn
risk). This gives the models in ``train.py`` a genuine signal to learn
instead of pure noise.

Output: data/churn.csv  (~2,000 rows, deterministic with seed 42)

Usage:
    python3 generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2000
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "churn.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Demographics
    # ------------------------------------------------------------------
    customer_id = [f"CUST-{i:05d}" for i in range(1, n_rows + 1)]
    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.504, 0.496])
    senior_citizen = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------
    # Tenure in months: mixture of new customers and long-term customers.
    tenure = np.where(
        rng.random(n_rows) < 0.35,
        rng.integers(1, 13, size=n_rows),          # many short-tenure users
        rng.integers(13, 73, size=n_rows),         # the rest spread to 6 years
    )
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows,
        p=[0.55, 0.21, 0.24],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_rows, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_rows,
        p=[0.34, 0.19, 0.22, 0.25],
    )

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------
    phone_service = rng.choice(["Yes", "No"], size=n_rows, p=[0.91, 0.09])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_rows, p=[0.42, 0.58]),
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_rows, p=[0.34, 0.44, 0.22]
    )

    def addon(has_internet: np.ndarray, p_yes: float) -> np.ndarray:
        """Add-on service column: 'No internet service' when offline."""
        return np.where(
            ~has_internet,
            "No internet service",
            rng.choice(["Yes", "No"], size=n_rows, p=[p_yes, 1 - p_yes]),
        )

    has_net = internet_service != "No"
    online_security = addon(has_net, 0.28)
    online_backup = addon(has_net, 0.34)
    device_protection = addon(has_net, 0.34)
    tech_support = addon(has_net, 0.29)
    streaming_tv = addon(has_net, 0.38)
    streaming_movies = addon(has_net, 0.39)

    # Support calls: dissatisfied / problem-prone customers call more.
    support_calls = rng.poisson(lam=1.1, size=n_rows)
    support_calls = np.clip(support_calls, 0, 9)

    # ------------------------------------------------------------------
    # Billing — monthly charges depend on the subscribed services.
    # ------------------------------------------------------------------
    base = np.select(
        [internet_service == "DSL", internet_service == "Fiber optic"],
        [45.0, 70.0],
        default=20.0,
    )
    addon_count = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
        + (multiple_lines == "Yes").astype(int)
    )
    monthly_charges = base + addon_count * rng.uniform(4.0, 9.0, size=n_rows)
    monthly_charges = np.clip(
        monthly_charges + rng.normal(0, 4.0, size=n_rows), 18.0, 120.0
    ).round(2)
    total_charges = (monthly_charges * tenure * rng.uniform(0.92, 1.05, size=n_rows)).round(2)

    # ------------------------------------------------------------------
    # Churn label from a latent probability model (realistic drivers).
    # ------------------------------------------------------------------
    logit = (
        -1.55
        + 1.05 * (contract == "Month-to-month")
        + 0.35 * (contract == "One year")
        - 0.85 * (contract == "Two year")
        + 0.028 * (monthly_charges - 65.0)
        - 0.035 * tenure
        + 0.42 * (internet_service == "Fiber optic")
        + 0.45 * (payment_method == "Electronic check")
        + 0.38 * (support_calls >= 3)
        + 0.22 * (support_calls == 2)
        - 0.55 * (tech_support == "Yes")
        - 0.45 * (online_security == "Yes")
        + 0.30 * (senior_citizen == 1)
        - 0.25 * (partner == "Yes")
        + 0.18 * (paperless_billing == "Yes")
    )
    churn_prob = sigmoid(logit)
    churn = (rng.random(n_rows) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
            "churn": churn,
        }
    )
    return df


def main() -> None:
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    rate = df["churn"].mean()
    print(f"Wrote {OUTPUT_PATH}  ({len(df)} rows, {df.shape[1]} columns)")
    print(f"Churn rate: {rate:.1%}  (churned={int(df['churn'].sum())}, "
          f"stayed={int((1 - df['churn']).sum())})")


if __name__ == "__main__":
    main()
