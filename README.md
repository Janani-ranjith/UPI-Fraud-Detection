# 🛡️ UPI Fraud Detection using Machine Learning & Deep Learning

A web application that detects fraudulent UPI transactions in real time. Enter the details of a transaction and four models (three Machine Learning models and one Deep Neural Network) estimate the fraud risk, explain the risk factors, and show how each model performs.

Built with **Flask**, **scikit-learn**, **pandas** and **Chart.js**.

## 📌 Overview

UPI has made digital payments fast and simple, but it has also created new opportunities for fraud. This project trains several models on transaction behaviour (amount, time of day, payee history, device, location, failed attempts and more) and uses them to classify a transaction as **Safe**, **Suspicious** or **High Risk**.

## ✨ Features

- **Real-time fraud check:** enter a transaction and get an instant risk score
- **Four models compared side by side:**
  - Logistic Regression (ML)
  - Random Forest (ML)
  - Gradient Boosting (ML)
  - Deep Neural Network, 3 hidden layers: 64-32-16 (DL)
- **Ensemble verdict:** average risk of all models mapped to Safe / Suspicious / High Risk
- **Explainable results:** shows the risk factors behind a flagged transaction
- **Model comparison dashboard:** accuracy, precision, recall, F1, ROC-AUC and confusion matrix
- **Top fraud indicators:** feature-importance chart from the Random Forest
- **Auto-training:** models are trained automatically on the first run
- **Responsive design** with light and dark mode

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Deep Learning | Multi-layer neural network (`MLPClassifier`) |
| Data processing | pandas, NumPy |
| Model storage | joblib |
| Frontend | HTML, CSS, JavaScript, Chart.js |

## 📁 Project Structure

```
upi-fraud-detection/
├── app.py              # Flask server and prediction API
├── train.py            # Trains and evaluates the 4 models
├── generate_data.py    # Generates synthetic UPI transaction data
├── common.py           # Shared feature definitions
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web interface
├── data/               # Created on first run (upi_transactions.csv)
└── models/             # Created on first run (trained models + metrics)
```

## 🧠 How It Works

1. **Data:** a synthetic dataset of 20,000 UPI transactions (about 6.5% fraud) is generated. You can replace it with a real dataset.
2. **Features:** amount, hour, day of week, transaction type, device, new payee, transactions in the last 24h, 30-day average amount, account age, location mismatch, failed attempts, and the amount-to-average ratio.
3. **Preprocessing:** numeric features are scaled and categorical features are one-hot encoded.
4. **Imbalance handling:** fraud cases are oversampled in the training set.
5. **Training:** four models are trained and evaluated on a held-out 20% test set.
6. **Prediction:** the Flask API returns each model's fraud probability and the average risk score.

| Risk score | Verdict |
|---|---|
| below 40% | ✅ Safe |
| 40% to 70% | ⚠️ Suspicious, review |
| 70% and above | 🚨 High risk, block |

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or newer
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```
   - Windows: `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python app.py
   ```
   On the first run, the dataset is generated and all models are trained (about 1 minute).

5. **Open** http://127.0.0.1:5000 in your browser.

### Optional commands

```bash
python generate_data.py   # create the dataset only
python train.py           # retrain the models only
```

## 🧪 Try It

Use the **Safe example** and **Fraud example** buttons on the page to fill the form automatically, then click **Check Fraud Risk**.

## 📂 Using Your Own Dataset

Save your data as `data/upi_transactions.csv` with these columns and run `python train.py`:

`amount, hour, day_of_week, txn_type, device, is_new_payee, txns_last_24h, avg_amount_30d, account_age_days, location_mismatch, failed_attempts, is_fraud`

(`is_fraud` is 1 for fraud and 0 for genuine.)

## 🔌 API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/api/predict` | POST | Returns fraud risk, verdict, per-model scores and risk factors |
| `/api/metrics` | GET | Returns evaluation metrics for all models |

## ⚠️ Note

The included dataset is **synthetic** and meant for learning and demonstration. Metrics on it do not reflect real-world performance. For production use, train on real, labelled transaction data and validate thoroughly.

## 🧯 Troubleshooting

- **`python` not found:** try `python3`.
- **Port 5000 in use:** change `port=5000` at the bottom of `app.py`.
- **Charts not showing:** Chart.js loads from a CDN, so an internet connection is needed.

## 🔮 Future Improvements

- Use TensorFlow/Keras or PyTorch for deeper neural networks
- Train on real-world datasets
- Add SHAP explanations
- Add a database to store checked transactions
- Support CSV batch checking
- Add user authentication and an admin dashboard
