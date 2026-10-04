"""
Feature Extraction Script — MaFaulDa Dataset (CSV format)
------------------------------------------------------------------
MaFaulDa dataset .wav files nahi, .csv files deta hai. Har CSV file mein
8 columns hote hain (50kHz sampling rate, 5 second recording):

    Column 1        -> Tachometer (rotation speed) — hum ise skip karenge
    Columns 2 to 4  -> Underhang bearing accelerometer (vibration) — piezo jaisa
    Columns 5 to 7  -> Overhang bearing accelerometer (vibration) — piezo jaisa
    Column 8        -> Microphone (sound) — MEMS mic jaisa

Ye script har CSV file se:
    1. Microphone column (sound) se FFT + MFCC nikalta hai
    2. Accelerometer columns (vibration) se time-domain + FFT stats nikalta hai
Aur dono ko combine karke ek feature vector banata hai (sensor fusion).

Expected folder structure:
    dataset/
        normal/          <- normal/normal/*.csv
        mild_fault/       <- imbalance/6g, 10g, 15g/*.csv
        severe_fault/     <- imbalance/20g, 25g, 30g, 35g/*.csv

Usage:
    python feature_extraction.py
"""

import os
import numpy as np
import pandas as pd
import librosa

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DATA_DIR = "archive"
SAMPLE_RATE = 50000     # MaFaulDa 50kHz pe recorded hai
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

MIC_COLUMN = 7           # Column 8 (0-indexed = 7) -> microphone
VIBRATION_COLUMNS = [1, 2, 3, 4, 5, 6]  # Columns 2-7 (0-indexed 1 to 6) -> accelerometers


# ---------------------------------------------------------
# 1. Ek CSV file se features nikalna (sound + vibration dono)
# ---------------------------------------------------------
def extract_features_from_csv(file_path):
    """
    MaFaulDa CSV file load karke uske mic (sound) aur accelerometer
    (vibration) columns se combined feature vector banata hai.
    """
    data = pd.read_csv(file_path, header=None)

    # ---- SOUND FEATURES (microphone column) ----
    mic_signal = data.iloc[:, MIC_COLUMN].values.astype(np.float32)

    # FFT summary stats
    fft_result = np.fft.fft(mic_signal)
    fft_magnitude = np.abs(fft_result)[: len(fft_result) // 2]
    fft_features = [
        np.mean(fft_magnitude),
        np.std(fft_magnitude),
        np.max(fft_magnitude),
        np.sum(fft_magnitude ** 2),
    ]

    # MFCC (librosa float32 signal expect karta hai)
    mfcc = librosa.feature.mfcc(
        y=mic_signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    sound_features = np.concatenate([fft_features, mfcc_mean, mfcc_std])

    # ---- VIBRATION FEATURES (accelerometer columns) ----
    vibration_features = []
    for col in VIBRATION_COLUMNS:
        signal = data.iloc[:, col].values.astype(np.float32)

        # Time-domain stats
        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))
        std = np.std(signal)

        # Frequency-domain stats (chota FFT summary)
        fft_v = np.abs(np.fft.fft(signal))[: len(signal) // 2]
        fft_mean = np.mean(fft_v)
        fft_energy = np.sum(fft_v ** 2)

        vibration_features.extend([rms, peak, std, fft_mean, fft_energy])

    vibration_features = np.array(vibration_features)

    # ---- COMBINE (Sensor Fusion) ----
    feature_vector = np.concatenate([sound_features, vibration_features])
    return feature_vector


# ---------------------------------------------------------
# 2. Poore dataset folder pe features nikalna
# ---------------------------------------------------------
def build_dataset(data_dir):
    """
    dataset/normal/*.csv       -> label 0 (Normal)
    dataset/mild_fault/*.csv    -> label 1 (Mild Fault)
    dataset/severe_fault/*.csv  -> label 2 (Severe Fault)
    """
    label_map = {"normal": 0, "mild_fault": 1, "severe_fault": 2}
    X = []
    y = []

    for label_name, label_id in label_map.items():
        folder = os.path.join(data_dir, label_name)
        if not os.path.exists(folder):
            print(f"[SKIP] Folder nahi mila: {folder}")
            continue

        files = []
        for root, _, filenames in os.walk(folder):
            for fn in filenames:
                if fn.endswith(".csv"):
                    files.append(os.path.join(root, fn))
        print(f"[INFO] {label_name}: {len(files)} files mile")

        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                feats = extract_features_from_csv(fpath)
                X.append(feats)
                y.append(label_id)
            except Exception as e:
                print(f"[ERROR] {fname} process nahi hui: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"\n[DONE] Total samples: {X.shape[0]}, Feature vector size: {X.shape[1] if len(X) else 0}")
    return X, y


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    X, y = build_dataset(DATA_DIR)

    if len(X) > 0:
        np.save("features_X.npy", X)
        np.save("labels_y.npy", y)
        print("[SAVED] features_X.npy aur labels_y.npy")
    else:
        print("\n[NOTE] Koi data nahi mila. DATA_DIR path check karo, aur")
        print("       folder structure 'dataset/normal/', 'dataset/mild_fault/', 'dataset/severe_fault/' jaisa rakho.")
        print("       (Files .csv format mein honi chahiye, MaFaulDa se)")
