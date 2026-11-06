import json, os
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from common import CAT, NUM, add_features
from train import MODELS, train

if not os.path.exists(os.path.join(MODELS, "metrics.json")):
    print("No trained models found - training now (takes ~1 minute)...")
    train()

NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Deep Neural Network"]
pipes = {n: joblib.load(os.path.join(MODELS, n.replace(" ", "_") + ".joblib")) for n in NAMES}
metrics = json.load(open(os.path.join(MODELS, "metrics.json")))
app = Flask(__name__)

def reasons(x):
    r = []
    if x["amount_ratio"] > 4: r.append(f"Amount is {x['amount_ratio']:.1f}x higher than the 30-day average")
    if x["hour"] <= 5: r.append("Transaction made late at night (12am-5am)")
    if x["is_new_payee"]: r.append("Payment to a new, unknown payee")
    if x["location_mismatch"]: r.append("Location differs from the usual location")
    if x["failed_attempts"] >= 2: r.append(f"{int(x['failed_attempts'])} failed PIN/login attempts")
    if x["txns_last_24h"] >= 8: r.append("Unusually many transactions in the last 24 hours")
    if x["account_age_days"] < 30: r.append("Very new account")
    return r

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)

@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json(force=True)
    try:
        row = {k: float(d[k]) for k in NUM if k != "amount_ratio"}
        row.update({k: str(d[k]) for k in CAT})
    except (KeyError, ValueError):
        return jsonify(error="Please fill all fields with valid values"), 400
    x = add_features(pd.DataFrame([row]))
    probs = {n: float(p.predict_proba(x[CAT + NUM])[0, 1]) for n, p in pipes.items()}
    risk = sum(probs.values()) / len(probs)
    verdict = "HIGH RISK - Block" if risk >= 0.7 else "SUSPICIOUS - Review" if risk >= 0.4 else "SAFE"
    return jsonify(risk=round(risk * 100, 1), verdict=verdict,
                   models={k: round(v * 100, 1) for k, v in probs.items()}, reasons=reasons(x.iloc[0]))

if __name__ == "__main__":
    app.run(debug=False, port=5000)
