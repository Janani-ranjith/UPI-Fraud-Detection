CAT = ["txn_type", "device"]
NUM = ["amount", "hour", "day_of_week", "is_new_payee", "txns_last_24h",
       "avg_amount_30d", "account_age_days", "location_mismatch", "failed_attempts", "amount_ratio"]

def add_features(df):
    df = df.copy()
    df["amount_ratio"] = df["amount"] / df["avg_amount_30d"].clip(lower=1)
    return df
