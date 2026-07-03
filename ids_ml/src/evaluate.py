"""
evaluate.py
-----------
Loads both trained models and prints side-by-side evaluation metrics:
accuracy, per-class precision/recall/F1, confusion matrix, and a binary
(attack vs normal) ROC-AUC score. Also saves confusion matrix plots.
"""

from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, accuracy_score
)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


def plot_confusion(cm, labels, title, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def binary_auc(y_test, prob_attack, le):
    # Treat everything that isn't "normal" as the positive (attack) class
    normal_idx = list(le.classes_).index("normal")
    y_binary = (y_test != normal_idx).astype(int)
    return roc_auc_score(y_binary, prob_attack)


def main():
    X_test = np.load(MODELS_DIR / "X_test.npy")
    y_test = np.load(MODELS_DIR / "y_test.npy")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    labels = list(le.classes_)
    normal_idx = labels.index("normal")

    # --- Random Forest ---
    rf = joblib.load(MODELS_DIR / "rf_model.joblib")
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    rf_attack_prob = 1 - rf_proba[:, normal_idx]

    print("\n========== RANDOM FOREST ==========")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(classification_report(y_test, rf_pred, target_names=labels))
    print(f"Binary (attack vs normal) ROC-AUC: {binary_auc(y_test, rf_attack_prob, le):.4f}")
    cm_rf = confusion_matrix(y_test, rf_pred)
    plot_confusion(cm_rf, labels, "Random Forest Confusion Matrix", MODELS_DIR / "rf_confusion_matrix.png")

    # --- DNN (optional, only if trained) ---
    dnn_path = MODELS_DIR / "dnn_model.keras"
    if dnn_path.exists():
        import tensorflow as tf
        dnn = tf.keras.models.load_model(dnn_path)
        dnn_proba = dnn.predict(X_test, verbose=0)
        dnn_pred = np.argmax(dnn_proba, axis=1)
        dnn_attack_prob = 1 - dnn_proba[:, normal_idx]

        print("\n========== DEEP NEURAL NETWORK ==========")
        print(f"Accuracy: {accuracy_score(y_test, dnn_pred):.4f}")
        print(classification_report(y_test, dnn_pred, target_names=labels))
        print(f"Binary (attack vs normal) ROC-AUC: {binary_auc(y_test, dnn_attack_prob, le):.4f}")
        cm_dnn = confusion_matrix(y_test, dnn_pred)
        plot_confusion(cm_dnn, labels, "DNN Confusion Matrix", MODELS_DIR / "dnn_confusion_matrix.png")
    else:
        print("\n(DNN model not found — run train_dnn.py to include it in evaluation.)")

    print(f"\nConfusion matrix plots saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
