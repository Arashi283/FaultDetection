"""
Machine Fault Model Training — corrected deployment version

Uses:
    features_X.npy
    labels_y.npy

Trains:
    1) Random Forest baseline
    2) Small Keras neural network
Then:
    - saves RF confusion matrix
    - saves NN training curve
    - saves the StandardScaler parameters needed by live prediction
    - converts the NN to TensorFlow Lite

Classes:
    0 = Normal
    1 = Mild Fault
    2 = Severe Fault
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FEATURES_PATH = "features_X.npy"
LABELS_PATH = "labels_y.npy"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CLASS_NAMES = {
    0: "Normal",
    1: "Mild Fault",
    2: "Severe Fault",
}


def load_data():
    X = np.load(FEATURES_PATH).astype(np.float32)
    y = np.load(LABELS_PATH).astype(np.int64)

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")
    print(f"[INFO] Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # IMPORTANT: live prediction must use the same mean/std.
    np.savez(
        "feature_scaler.npz",
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )
    print("[SAVED] feature_scaler.npz")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n===== Random Forest baseline =====")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[RESULT] RF accuracy: {acc * 100:.2f}%")
    print(classification_report(
        y_test,
        y_pred,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
        digits=4,
    ))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Random Forest — Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xticks(range(3), ["Normal", "Mild", "Severe"])
    plt.yticks(range(3), ["Normal", "Mild", "Severe"])
    plt.tight_layout()
    plt.savefig("rf_confusion_matrix.png", dpi=120)
    plt.close()

    print("[SAVED] rf_confusion_matrix.png")
    return rf


def train_neural_network(X_train, y_train, X_test, y_test):
    print("\n===== Neural Network =====")

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        verbose=1,
    )

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Neural Network Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nn_training_curve.png", dpi=120)
    plt.close()

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] NN test accuracy: {test_acc * 100:.2f}%")
    print("[SAVED] nn_training_curve.png")

    return model


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open("fault_detection_model.tflite", "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"[SAVED] fault_detection_model.tflite ({size_kb:.2f} KB)")

    return tflite_model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    train_random_forest(X_train, y_train, X_test, y_test)

    nn = train_neural_network(
        X_train, y_train,
        X_test, y_test
    )

    convert_to_tflite(nn)

    print("\n[DONE] Training and TFLite conversion complete.")
