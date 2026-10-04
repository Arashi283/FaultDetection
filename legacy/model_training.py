"""
Model Training Script — Sound-based Machine Fault Detection
------------------------------------------------------------------
Ye script feature_extraction.py se save kiye hue features (features_X.npy,
labels_y.npy) load karta hai, ek classifier train karta hai, aur usko
TensorFlow Lite (TFLite) format mein convert karta hai taaki Raspberry Pi
Pico pe deploy ho sake.

2 model options diye hain:
1. Random Forest (sklearn)  -> simple, fast, achi baseline accuracy
2. Small Neural Network (TensorFlow/Keras) -> TFLite conversion ke liye zaroori
   (Random Forest ko TFLite mein direct convert nahi kar sakte, isliye final
   deployment ke liye Neural Network hi use karna hoga)

Usage:
    python model_training.py
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


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
FEATURES_PATH = "features_X.npy"
LABELS_PATH = "labels_y.npy"
RANDOM_STATE = 42
TEST_SIZE = 0.2   # 80% train, 20% test


# ---------------------------------------------------------
# 1. Data load + split
# ---------------------------------------------------------
def load_data():
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    print(f"[INFO] Loaded: X shape = {X.shape}, y shape = {y.shape}")
    print(f"[INFO] Classes: {np.unique(y)} (0=normal, 1=abnormal — ya apne multi-class labels)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Normalize features (zaroori hai NN ke liye, RF ke liye optional but harm nahi karta)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------------------------------------------------
# 2. Random Forest — baseline model (quick accuracy check ke liye)
# ---------------------------------------------------------
def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n===== Random Forest (Baseline) =====")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Random Forest Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Random Forest — Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("rf_confusion_matrix.png", dpi=120)
    print("[SAVED] rf_confusion_matrix.png")

    return rf


# ---------------------------------------------------------
# 3. Small Neural Network — TFLite conversion ke liye (final deployment model)
# ---------------------------------------------------------
def train_neural_network(X_train, y_train, X_test, y_test, num_classes):
    print("\n===== Neural Network (for TFLite / Pico deployment) =====")

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

    print(model.summary())

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        verbose=1,
    )

    # Training curve plot
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Neural Network Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nn_training_curve.png", dpi=120)
    print("[SAVED] nn_training_curve.png")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Neural Network Test Accuracy: {test_acc*100:.2f}%")

    return model


# ---------------------------------------------------------
# 4. Neural Network ko TFLite mein convert karna (Pico deployment ke liye)
# ---------------------------------------------------------
def convert_to_tflite(model, output_path="fault_detection_model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimize karo taaki model chota ho jaye (microcontroller ke liye zaroori)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\n[SAVED] {output_path} ({size_kb:.2f} KB)")

    if size_kb > 200:
        print("[WARNING] Model 200KB se bada hai — Pico ke 264KB RAM mein tight fit hoga.")
        print("          Layers chote karo (Dense units kam karo) agar issue aaye.")
    else:
        print("[OK] Model size Pico ke liye theek hai.")

    return tflite_model


# ---------------------------------------------------------
# 5. Prediction + Confidence Score dikhana (demo ke liye zaroori)
# ---------------------------------------------------------
CLASS_NAMES = {0: "Normal", 1: "Mild Fault", 2: "Severe Fault"}


def predict_with_confidence(model, X_sample):
    """
    Ek sample (ya batch) pe prediction karta hai aur confidence % bhi dikhata hai.
    Ye wahi logic hai jo baad mein Pico pe bhi use hoga (serial output ke liye).

    X_sample: shape (1, num_features) — ek single sample ka feature vector
    """
    probabilities = model.predict(X_sample, verbose=0)[0]  # softmax output, sab classes ki probability
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class]) * 100

    label = CLASS_NAMES.get(predicted_class, f"Class {predicted_class}")
    print(f"\n[PREDICTION] {label}  (Confidence: {confidence:.1f}%)")

    # Sab classes ki probability bhi dikhao (demo ke liye achha lagta hai)
    print("  Breakdown:")
    for class_id, class_name in CLASS_NAMES.items():
        if class_id < len(probabilities):
            print(f"    {class_name}: {probabilities[class_id]*100:.1f}%")

    return predicted_class, confidence


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_data()
    num_classes = len(np.unique(y_train))

    # Step 1: Random Forest se quick baseline check karo
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)

    # Step 2: Neural Network train karo (ye wala hi Pico pe jayega)
    nn_model = train_neural_network(X_train, y_train, X_test, y_test, num_classes)

    # Step 3: TFLite mein convert karo
    convert_to_tflite(nn_model)

    # Step 4: Demo — ek test sample pe confidence score ke saath prediction dikhao
    print("\n===== Demo: Confidence Score Example =====")
    sample = X_test[0:1]  # test set se ek sample utha lo demo ke liye
    predict_with_confidence(nn_model, sample)
    print(f"[ACTUAL LABEL] {CLASS_NAMES.get(int(y_test[0]), y_test[0])}")

    print("\n[DONE] Ab 'fault_detection_model.tflite' file ready hai Pico pe deploy karne ke liye.")