import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow import keras


# =========================================================
# CONFIGURATION
# =========================================================

FEATURES_FILE = "features_X_single_vibration.npy"
LABELS_FILE = "labels_y_single_vibration.npy"

SCALER_FILE = "feature_scaler_single_vibration.npz"
MODEL_FILE = "fault_detection_model_single_vibration.tflite"

RANDOM_STATE = 42


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

X = np.load(FEATURES_FILE)
y = np.load(LABELS_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)

print()
print("Class counts:")
print("Normal:", np.sum(y == 0))
print("Mild Fault:", np.sum(y == 1))
print("Severe Fault:", np.sum(y == 2))


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# Save scaler parameters for live prediction
np.savez(
    SCALER_FILE,
    mean=scaler.mean_,
    scale=scaler.scale_
)

print()
print("[SAVED]", SCALER_FILE)


# =========================================================
# RANDOM FOREST
# =========================================================

print()
print("========================================")
print("RANDOM FOREST")
print("========================================")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

rf_model.fit(
    X_train_scaled,
    y_train
)

rf_predictions = rf_model.predict(
    X_test_scaled
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print()
print(
    f"Random Forest test accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=[
            "Normal",
            "Mild Fault",
            "Severe Fault"
        ]
    )
)


# =========================================================
# RANDOM FOREST CONFUSION MATRIX
# =========================================================

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

print("Random Forest Confusion Matrix:")
print(rf_cm)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=[
        "Normal",
        "Mild Fault",
        "Severe Fault"
    ]
).plot(
    ax=ax
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "rf_confusion_matrix_single_vibration.png"
)

plt.close()

print(
    "[SAVED] "
    "rf_confusion_matrix_single_vibration.png"
)


# =========================================================
# NEURAL NETWORK
# =========================================================

print()
print("========================================")
print("NEURAL NETWORK")
print("========================================")

# Input = 35 features
model = keras.Sequential([
    keras.layers.Input(
        shape=(35,)
    ),

    keras.layers.Dense(
        32,
        activation="relu"
    ),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dense(
        3,
        activation="softmax"
    )
])


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================================================
# TRAIN
# =========================================================

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=16,
    verbose=1
)


# =========================================================
# TEST NEURAL NETWORK
# =========================================================

test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print()
print(
    f"[RESULT] NN test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# NN PREDICTIONS
# =========================================================

nn_probabilities = model.predict(
    X_test_scaled,
    verbose=0
)

nn_predictions = np.argmax(
    nn_probabilities,
    axis=1
)

print()
print("Neural Network Classification Report:")

print(
    classification_report(
        y_test,
        nn_predictions,
        target_names=[
            "Normal",
            "Mild Fault",
            "Severe Fault"
        ]
    )
)


# =========================================================
# NN CONFUSION MATRIX
# =========================================================

nn_cm = confusion_matrix(
    y_test,
    nn_predictions
)

print()
print("Neural Network Confusion Matrix:")
print(nn_cm)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

ConfusionMatrixDisplay(
    confusion_matrix=nn_cm,
    display_labels=[
        "Normal",
        "Mild Fault",
        "Severe Fault"
    ]
).plot(
    ax=ax
)

plt.title(
    "Neural Network Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "nn_confusion_matrix_single_vibration.png"
)

plt.close()

print(
    "[SAVED] "
    "nn_confusion_matrix_single_vibration.png"
)


# =========================================================
# TRAINING CURVE
# =========================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Neural Network Training Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "nn_training_curve_single_vibration.png"
)

plt.close()

print(
    "[SAVED] "
    "nn_training_curve_single_vibration.png"
)


# =========================================================
# TFLITE CONVERSION
# =========================================================

print()
print("========================================")
print("TFLITE CONVERSION")
print("========================================")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

tflite_model = converter.convert()

with open(
    MODEL_FILE,
    "wb"
) as f:

    f.write(
        tflite_model
    )

print(
    "[SAVED]",
    MODEL_FILE
)

print(
    "TFLite model size:",
    len(tflite_model),
    "bytes"
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print()
print("========================================")
print("[DONE] Training complete.")
print("========================================")