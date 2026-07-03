"""
train_dnn.py
------------
Trains a feedforward Deep Neural Network (TensorFlow/Keras) on the same
preprocessed data, as a comparison point to the Random Forest baseline.
"""

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models, callbacks

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


def build_model(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X_train = np.load(MODELS_DIR / "X_train.npy")
    X_test = np.load(MODELS_DIR / "X_test.npy")
    y_train = np.load(MODELS_DIR / "y_train.npy")
    y_test = np.load(MODELS_DIR / "y_test.npy")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")

    n_classes = len(le.classes_)
    model = build_model(X_train.shape[1], n_classes)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n=== DNN — Test accuracy: {acc:.4f}, loss: {loss:.4f} ===")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model.save(MODELS_DIR / "dnn_model.keras")
    print(f"Saved model to {MODELS_DIR / 'dnn_model.keras'}")


if __name__ == "__main__":
    main()
