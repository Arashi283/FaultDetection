"""
Laptop real-time prediction — corrected version.

Requires:
    fault_detection_model.tflite
    feature_scaler.npz

The scaler MUST be the one produced by model_training_fixed.py.
"""

import serial
import numpy as np
import librosa
import time
import tensorflow as tf

SERIAL_PORT = "COM6"       # <-- change if Device Manager shows another port
BAUD_RATE = 115200

MODEL_PATH = "fault_detection_model.tflite"
SCALER_PATH = "feature_scaler.npz"

AUDIO_SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

CLASS_NAMES = {
    0: "Normal",
    1: "Mild Fault",
    2: "Severe Fault",
}


def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def load_scaler(path):
    d = np.load(path)
    return d["mean"].astype(np.float32), d["scale"].astype(np.float32)


def parse_pico_line(line):
    try:
        vib_part, aud_part = line.strip().split("|")
        vib_values = vib_part.replace("VIB:", "")
        aud_values = aud_part.replace("AUD:", "")

        vibration = np.array(
            [float(v) for v in vib_values.split(",") if v],
            dtype=np.float32,
        )
        audio = np.array(
            [float(a) for a in aud_values.split(",") if a],
            dtype=np.float32,
        )

        return vibration, audio
    except Exception:
        return None, None


def extract_features_live(vibration, audio):
    # ---- SOUND: 30 features ----
    if len(audio) > 0:
        fft_result = np.fft.fft(audio)
        fft_magnitude = np.abs(fft_result)[:len(fft_result) // 2]

        fft_features = [
            np.mean(fft_magnitude),
            np.std(fft_magnitude),
            np.max(fft_magnitude),
            np.sum(fft_magnitude ** 2),
        ]

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=AUDIO_SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=min(N_FFT, len(audio)),
            hop_length=HOP_LENGTH,
        )

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        sound_features = np.concatenate(
            [fft_features, mfcc_mean, mfcc_std]
        )
    else:
        sound_features = np.zeros(30, dtype=np.float32)

    # ---- VIBRATION: 30 features ----
    vibration_features = []

    if len(vibration) == 0:
        vibration = np.zeros(1, dtype=np.float32)

    # The current trained feature format contains six vibration
    # channels. The live hardware has one piezo channel, so the
    # existing project design repeats that channel six times.
    for _ in range(6):
        rms = np.sqrt(np.mean(vibration ** 2))
        peak = np.max(np.abs(vibration))
        std = np.std(vibration)

        fft_v = np.abs(np.fft.fft(vibration))[:len(vibration) // 2]

        if len(fft_v) == 0:
            fft_mean = 0.0
            fft_energy = 0.0
        else:
            fft_mean = np.mean(fft_v)
            fft_energy = np.sum(fft_v ** 2)

        vibration_features.extend([
            rms, peak, std, fft_mean, fft_energy
        ])

    return np.concatenate([
        sound_features,
        np.array(vibration_features, dtype=np.float32),
    ]).astype(np.float32)


def predict_tflite(interpreter, feature_vector):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array([feature_vector], dtype=np.float32)

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data,
    )
    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    predicted_class = int(np.argmax(output))
    confidence = float(output[predicted_class]) * 100.0

    return predicted_class, confidence, output


if __name__ == "__main__":
    print("[INFO] Loading model...")
    interpreter = load_tflite_model(MODEL_PATH)
    print("[OK] Model loaded")

    print("[INFO] Loading feature scaler...")
    scaler_mean, scaler_scale = load_scaler(SCALER_PATH)
    print("[OK] Scaler loaded")

    print(f"[INFO] Connecting to {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)

    print("[OK] Connected.")
    print("[INFO] Live predictions started.\n")

    try:
        while True:
            raw_line = (
                ser.readline()
                .decode("utf-8", errors="ignore")
                .strip()
            )

            if not raw_line or raw_line == "READY":
                continue

            vibration, audio = parse_pico_line(raw_line)

            if vibration is None or len(vibration) == 0:
                continue

            feats = extract_features_live(vibration, audio)

            # CRITICAL: use exactly the same scaling used during training.
            feats_scaled = (feats - scaler_mean) / scaler_scale

            pred_class, confidence, probs = predict_tflite(
                interpreter,
                feats_scaled,
            )

            label = CLASS_NAMES.get(
                pred_class,
                f"Class {pred_class}"
            )

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"Prediction: {label} "
                f"(Confidence: {confidence:.1f}%)"
            )

            print(
                f"    Normal: {probs[0] * 100:.1f}% | "
                f"Mild: {probs[1] * 100:.1f}% | "
                f"Severe: {probs[2] * 100:.1f}%"
            )

    except KeyboardInterrupt:
        print("\n[STOPPED]")

    finally:
        ser.close()
