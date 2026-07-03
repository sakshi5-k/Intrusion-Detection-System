"""
predict_live.py
----------------
Simulates a live IDS feed: streams test-set rows one at a time (standing in
for flows that would normally arrive from Wireshark/Zeek/CICFlowMeter in
real time), scores each with the trained Random Forest model, and prints an
alert whenever traffic is classified as malicious.

In a real deployment, replace `stream_rows()` with a feed that converts
live captured packets into the same flow-feature vector (e.g. via
CICFlowMeter or a Zeek log tailer) and call `score_row()` on each one.
"""

import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


def load_artifacts():
    model = joblib.load(MODELS_DIR / "rf_model.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    X_test = np.load(MODELS_DIR / "X_test.npy")
    y_test = np.load(MODELS_DIR / "y_test.npy")
    return model, le, X_test, y_test


def stream_rows(X, n=40, delay=0.05):
    """Yield a random sample of rows one at a time, as if arriving live."""
    idx = np.random.default_rng(7).choice(len(X), size=min(n, len(X)), replace=False)
    for i in idx:
        yield i, X[i]
        time.sleep(delay)


def score_row(model, le, row):
    proba = model.predict_proba(row.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.classes_[pred_idx]
    confidence = float(proba[pred_idx])
    return pred_label, confidence


def main():
    model, le, X_test, y_test = load_artifacts()
    print("Starting simulated live traffic monitor (Ctrl+C to stop)...\n")

    alerts = 0
    total = 0
    for i, row in stream_rows(X_test):
        total += 1
        pred_label, confidence = score_row(model, le, row)
        true_label = le.classes_[y_test[i]]

        if pred_label != "normal":
            alerts += 1
            print(
                f"[ALERT] flow#{i:5d}  predicted={pred_label:8s} "
                f"confidence={confidence:.2f}  (ground truth: {true_label})"
            )
        else:
            print(f"[ ok  ] flow#{i:5d}  predicted=normal    confidence={confidence:.2f}")

    print(f"\nMonitored {total} flows — raised {alerts} alert(s).")


if __name__ == "__main__":
    main()
