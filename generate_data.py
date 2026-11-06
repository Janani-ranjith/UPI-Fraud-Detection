"""Generates a synthetic UPI transaction dataset (data/upi_transactions.csv).
Replace it with a real dataset that has the same columns + 'is_fraud' to train on real data."""
import os
import numpy as np
import pandas as pd

def generate(path, n=20000, seed=7):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "amount": np.round(r.lognormal(6.2, 1.1, n), 2),
        "hour": r.integers(0, 24, n),
        "day_of_week": r.integers(0, 7, n),
        "txn_type": r.choice(["P2P", "Merchant", "Bill Payment", "Recharge"], n, p=[.40, .35, .15, .10]),
        "device": r.choice(["Android", "iOS", "Web"], n, p=[.70, .22, .08]),
        "is_new_payee": r.binomial(1, .20, n),
        "txns_last_24h": r.poisson(2.5, n),
        "avg_amount_30d": np.round(r.lognormal(6.0, .7, n), 2),
        "account_age_days": r.integers(3, 2500, n),
        "location_mismatch": r.binomial(1, .08, n),
        "failed_attempts": r.poisson(.25, n),
    })
    ratio = df.amount / df.avg_amount_30d.clip(lower=1)
    night = df.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    z = (-5.6 + 2.2 * (ratio > 4) + 0.5 * np.log1p(df.amount / 1000) + 1.1 * night + 1.6 * df.is_new_payee
         + 2.0 * df.location_mismatch + 0.8 * df.failed_attempts + 0.18 * np.maximum(df.txns_last_24h - 5, 0)
         + 1.0 * (df.account_age_days < 30) + 0.7 * (df.txn_type == "P2P") + 0.5 * (df.device == "Web")
         + r.normal(0, .3, n))
    df["is_fraud"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    d = generate(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "upi_transactions.csv"))
    print(f"Created data/upi_transactions.csv  rows={len(d)}  fraud rate={d.is_fraud.mean():.2%}")
