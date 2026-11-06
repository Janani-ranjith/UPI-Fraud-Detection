"""Trains 3 Machine Learning models + 1 Deep Neural Network and saves them to models/."""
import json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import CAT, NUM, add_features
from generate_data import generate

BASE = os.path.dirname(os.path.abspath(__file__))
DATA, MODELS = os.path.join(BASE, "data", "upi_transactions.csv"), os.path.join(BASE, "models")

def build(clf):
    prep = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("prep", prep), ("clf", clf)])

def train():
    if not os.path.exists(DATA):
        generate(DATA)
    df = add_features(pd.read_csv(DATA))
    tr, te = train_test_split(df, test_size=0.2, stratify=df.is_fraud, random_state=42)
    fraud, legit = tr[tr.is_fraud == 1], tr[tr.is_fraud == 0]           # oversample fraud (class imbalance)
    extra = fraud.sample(max(int(0.5 * len(legit)) - len(fraud), 0), replace=True, random_state=1)
    tr = pd.concat([tr, extra]).sample(frac=1, random_state=1)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=1),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=1),
        "Deep Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32, 16), early_stopping=True,
                                             max_iter=300, random_state=1),
    }
    os.makedirs(MODELS, exist_ok=True)
    out = {"models": {}, "dataset": {"rows": len(df), "fraud_rate": round(df.is_fraud.mean() * 100, 2)}}
    for name, clf in models.items():
        pipe = build(clf).fit(tr[CAT + NUM], tr.is_fraud)
        p = pipe.predict_proba(te[CAT + NUM])[:, 1]
        pred = (p >= 0.5).astype(int)
        out["models"][name] = {
            "accuracy": round(accuracy_score(te.is_fraud, pred), 4),
            "precision": round(precision_score(te.is_fraud, pred, zero_division=0), 4),
            "recall": round(recall_score(te.is_fraud, pred), 4),
            "f1": round(f1_score(te.is_fraud, pred), 4),
            "roc_auc": round(roc_auc_score(te.is_fraud, p), 4),
            "confusion": confusion_matrix(te.is_fraud, pred).tolist(),
        }
        joblib.dump(pipe, os.path.join(MODELS, name.replace(" ", "_") + ".joblib"))
        print(f"{name:22s} F1={out['models'][name]['f1']}  AUC={out['models'][name]['roc_auc']}")
        if name == "Random Forest":
            names = [n.split("__")[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
            imp = sorted(zip(names, pipe.named_steps["clf"].feature_importances_), key=lambda x: -x[1])[:8]
            out["importances"] = [{"feature": f, "value": round(float(v), 4)} for f, v in imp]
    json.dump(out, open(os.path.join(MODELS, "metrics.json"), "w"), indent=2)
    print("Models saved in models/")

if __name__ == "__main__":
    train()
