"""
preprocess.py
-------------
Loads the raw traffic CSV, encodes categoricals, scales numeric features,
splits into train/test sets, and persists the fitted scaler/encoder + label
classes so train/eval/predict scripts can reuse them consistently.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "traffic.csv"
MODELS_DIR = ROOT / "models"

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABEL_COL = "label"


def load_and_prepare():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_dataset.py` first "
            "(or drop in a real labeled traffic CSV with the same schema)."
        )
    df = pd.read_csv(DATA_PATH)

    # One-hot encode categorical protocol/service/flag fields
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)

    # Encode the multi-class label
    le = LabelEncoder()
    y = le.fit_transform(df_encoded[LABEL_COL])
    X = df_encoded.drop(columns=[LABEL_COL])

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled, X_test_scaled, y_train, y_test,
        scaler, le, feature_columns,
    )


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, scaler, le, feature_columns = load_and_prepare()

    np.save(MODELS_DIR / "X_train.npy", X_train)
    np.save(MODELS_DIR / "X_test.npy", X_test)
    np.save(MODELS_DIR / "y_train.npy", y_train)
    np.save(MODELS_DIR / "y_test.npy", y_test)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    print("Preprocessing complete.")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}")
    print(f"  Classes:     {list(le.classes_)}")


if __name__ == "__main__":
    main()
