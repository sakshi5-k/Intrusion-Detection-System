"""
train_rf.py
-----------
Trains a Random Forest classifier on the preprocessed traffic data and
saves the model + a feature-importance report.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


def main():
    X_train = np.load(MODELS_DIR / "X_train.npy")
    X_test = np.load(MODELS_DIR / "X_test.npy")
    y_train = np.load(MODELS_DIR / "y_train.npy")
    y_test = np.load(MODELS_DIR / "y_test.npy")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    feature_columns = json.load(open(MODELS_DIR / "feature_columns.json"))

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=== Random Forest — Test Set Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, MODELS_DIR / "rf_model.joblib")

    # Feature importance report
    importances = sorted(
        zip(feature_columns, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop 10 most important features:")
    for name, score in importances[:10]:
        print(f"  {name:30s} {score:.4f}")

    with open(MODELS_DIR / "rf_feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR / 'rf_model.joblib'}")


if __name__ == "__main__":
    main()
