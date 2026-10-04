"""
Laptop Real-Time Prediction Script
------------------------------------------------------------------
Ye script laptop pe chalta hai (isi 'pico_env' environment mein jisme
TensorFlow install kiya tha). Pico se USB serial ke through data leta
hai, features nikalta hai (training ke waise hi), aur TFLite model se
real-time prediction karta hai — confidence score ke saath.

Usage:
    python laptop_realtime_predict.py

Pehle apna Pico ka COM port set karo niche (SERIAL_PORT variable).
Device Manager (Windows) mein "Ports (COM & LPT)" ke andar dekh sakte ho
kaunsa COM number hai jab Pico connect ho.
"""

import serial
import numpy as np
import librosa
import time
import tensorflow as tf

# ---------------------------------------------------------
# CONFIG — apne hisab se change karo
# ---------------------------------------------------------
SERIAL_PORT = "COM5"        # <-- Apna Pico ka COM port yahan daalo
BAUD_RATE = 115200
MODEL_PATH = "fault_detection_model.tflite"

AUDIO_SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

CLASS_NAMES = {0: "Normal", 1: "Mild Fault", 2: "Severe Fault"}


# ---------------------------------------------------------
# 1. TFLite model load karo
# ---------------------------------------------------------
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(interpreter, feature_vector):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array([feature_vector], dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_class = int(np.argmax(output))
    confidence = float(output[predicted_class]) * 100
    return predicted_class, confidence, output


# ---------------------------------------------------------
# 2. Pico se aayi ek line ko parse karke features nikalna
# ---------------------------------------------------------
def parse_pico_line(line):
    """
    Line format: VIB:<comma-separated>|AUD:<comma-separated>
    Isse vibration aur audio arrays nikalta hai.
    """
    try:
        vib_part, aud_part = line.strip().split("|")
        vib_values = vib_part.replace("VIB:", "")
        aud_values = aud_part.replace("AUD:", "")

        vibration = np.array([float(v) for v in vib_values.split(",") if v], dtype=np.float32)
        audio = np.array([float(a) for a in aud_values.split(",") if a], dtype=np.float32)
        return vibration, audio
    except Exception:
        return None, None


def extract_features_live(vibration, audio):
    """
    Training ke time jaisa hi feature vector banata hai — 30 sound features
    + 30 vibration features = 60 total. Yahan sirf EK vibration channel
    (piezo) hai training ke 6 channels ki jagah, isliye usi ek channel
    ko 6 baar duplicate/repeat karke shape match karayenge (approx fix).
    """
    # ---- SOUND FEATURES ----
    if len(audio) > 0:
        fft_result = np.fft.fft(audio)
        fft_magnitude = np.abs(fft_result)[: len(fft_result) // 2]
        fft_features = [
            np.mean(fft_magnitude), np.std(fft_magnitude),
            np.max(fft_magnitude), np.sum(fft_magnitude ** 2),
        ]
        mfcc = librosa.feature.mfcc(
            y=audio, sr=AUDIO_SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=min(N_FFT, len(audio)), hop_length=HOP_LENGTH
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        sound_features = np.concatenate([fft_features, mfcc_mean, mfcc_std])
    else:
        sound_features = np.zeros(30)   # mic na ho toh zero-fill

    # ---- VIBRATION FEATURES (piezo, ek channel ko 6 baar repeat) ----
    vibration_features = []
    for _ in range(6):   # training mein 6 accelerometer channels the
        rms = np.sqrt(np.mean(vibration ** 2))
        peak = np.max(np.abs(vibration))
        std = np.std(vibration)
        fft_v = np.abs(np.fft.fft(vibration))[: len(vibration) // 2]
        fft_mean = np.mean(fft_v)
        fft_energy = np.sum(fft_v ** 2)
        vibration_features.extend([rms, peak, std, fft_mean, fft_energy])

    feature_vector = np.concatenate([sound_features, np.array(vibration_features)])
    return feature_vector.astype(np.float32)


# ---------------------------------------------------------
# MAIN — Serial se live data lo, predict karo
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"[INFO] {MODEL_PATH} load ho raha hai...")
    interpreter = load_tflite_model(MODEL_PATH)
    print("[OK] Model loaded")

    print(f"[INFO] {SERIAL_PORT} se connect ho raha hai...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("[OK] Connected. Live predictions shuru ho rahi hain...\n")

    try:
        while True:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not raw_line or raw_line == "READY":
                continue

            vibration, audio = parse_pico_line(raw_line)
            if vibration is None or len(vibration) == 0:
                continue

            feats = extract_features_live(vibration, audio)
            pred_class, confidence, probs = predict_tflite(interpreter, feats)
            label = CLASS_NAMES.get(pred_class, f"Class {pred_class}")

            print(f"[{time.strftime('%H:%M:%S')}] Prediction: {label}  (Confidence: {confidence:.1f}%)")
            print(f"    Normal: {probs[0]*100:.1f}% | Mild: {probs[1]*100:.1f}% | Severe: {probs[2]*100:.1f}%")

    except KeyboardInterrupt:
        print("\n[STOPPED] User ne band kiya.")
    finally:
        ser.close()