"""
Feature Extraction Script — Sound-based Machine Fault Detection
------------------------------------------------------------------
Ye script kisi bhi .wav audio file se do tarah ke features nikalta hai:
1. FFT (Fast Fourier Transform)  -> frequency domain representation
2. MFCC (Mel-Frequency Cepstral Coefficients) -> standard audio ML feature

Usage:
    python feature_extraction.py

Isko apne dataset folder pe chalane ke liye niche 'DATA_DIR' path change karo.
Folder structure expect karta hai (severity levels ke saath):
    dataset/
        normal/         <- normal machine sound .wav files
        mild_fault/      <- halka fault (mild imbalance/misalignment) .wav files
        severe_fault/    <- zyada severe fault .wav files

(Agar sirf 2 classes chahiye normal/abnormal, to niche LABEL_MAP change kar do)
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIG — apne hisab se change karo
# ---------------------------------------------------------
DATA_DIR = "dataset"          # yaha 'normal' aur 'abnormal' subfolders honge
SAMPLE_RATE = 16000            # MIMII dataset 16kHz pe recorded hai
N_MFCC = 13                    # kitne MFCC coefficients nikalne hain (standard = 13)
N_FFT = 2048                   # FFT window size
HOP_LENGTH = 512                # frame ke beech ka gap


# ---------------------------------------------------------
# 1. Ek single audio file se features nikalna
# ---------------------------------------------------------
def extract_features(file_path):
    """
    Ek audio file load karke uske FFT aur MFCC features return karta hai.
    """
    # Audio load karo (mono mein convert ho jayega automatically)
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # ---- FFT FEATURE ----
    # Poore signal ka FFT nikal ke uski magnitude (strength) leta hai
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)[: len(fft_result) // 2]  # sirf positive frequencies

    # FFT se summary statistics (chota feature vector banane ke liye)
    fft_features = {
        "fft_mean": np.mean(fft_magnitude),
        "fft_std": np.std(fft_magnitude),
        "fft_max": np.max(fft_magnitude),
        "fft_energy": np.sum(fft_magnitude ** 2),
    }

    # ---- MFCC FEATURE ----
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # Har coefficient ka time ke saath average aur variation le lete hain
    mfcc_mean = np.mean(mfcc, axis=1)   # shape: (N_MFCC,)
    mfcc_std = np.std(mfcc, axis=1)     # shape: (N_MFCC,)

    # Sab kuch ek single feature vector mein jodo (model ko yahi denge)
    feature_vector = np.concatenate([
        list(fft_features.values()),
        mfcc_mean,
        mfcc_std,
    ])

    return feature_vector, mfcc, fft_magnitude, signal, sr


# ---------------------------------------------------------
# 2. Poore dataset folder pe features nikalna (normal + abnormal)
# ---------------------------------------------------------
def build_dataset(data_dir):
    """
    dataset/normal/*.wav        -> label 0 (Normal)
    dataset/mild_fault/*.wav     -> label 1 (Mild Fault)
    dataset/severe_fault/*.wav   -> label 2 (Severe Fault)

    (Agar sirf normal/abnormal chahiye, isse replace kar do:
     label_map = {"normal": 0, "abnormal": 1})
    """
    label_map = {"normal": 0, "mild_fault": 1, "severe_fault": 2}
    X = []
    y = []

    for label_name, label_id in label_map.items():
        folder = os.path.join(data_dir, label_name)
        if not os.path.exists(folder):
            print(f"[SKIP] Folder nahi mila: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.endswith((".wav", ".mp3"))]
        print(f"[INFO] {label_name}: {len(files)} files mile")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                feats, _, _, _, _ = extract_features(fpath)
                X.append(feats)
                y.append(label_id)
            except Exception as e:
                print(f"[ERROR] {fname} process nahi hui: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"\n[DONE] Total samples: {X.shape[0]}, Feature vector size: {X.shape[1] if len(X) else 0}")
    return X, y


# ---------------------------------------------------------
# 3. Visualization — ek sample ka FFT + MFCC plot karna (samajhne ke liye)
# ---------------------------------------------------------
def visualize_single_file(file_path, save_path="sample_visualization.png"):
    feats, mfcc, fft_magnitude, signal, sr = extract_features(file_path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Raw waveform
    axes[0].plot(signal)
    axes[0].set_title("Raw Audio Waveform")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")

    # FFT spectrum
    freqs = np.fft.fftfreq(len(signal), 1 / sr)[: len(signal) // 2]
    axes[1].plot(freqs, fft_magnitude)
    axes[1].set_title("FFT — Frequency Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlim(0, sr / 2)

    # MFCC heatmap
    img = librosa.display.specshow(mfcc, x_axis="time", sr=sr, hop_length=HOP_LENGTH, ax=axes[2])
    axes[2].set_title("MFCC")
    fig.colorbar(img, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"[SAVED] Visualization: {save_path}")


# ---------------------------------------------------------
# MAIN — yaha se chalao
# ---------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Poore dataset pe features nikalo
    X, y = build_dataset(DATA_DIR)

    if len(X) > 0:
        # Features ko file mein save kar do (baad mein model training ke liye direct load karenge)
        np.save("features_X.npy", X)
        np.save("labels_y.npy", y)
        print("[SAVED] features_X.npy aur labels_y.npy")

        # Step 2 (optional): Ek sample file ka visualization dekhna
        # visualize_single_file("dataset/normal/sample1.wav")
    else:
        print("\n[NOTE] Koi data nahi mila. DATA_DIR path check karo, aur")
        print("       folder structure 'dataset/normal/', 'dataset/mild_fault/', 'dataset/severe_fault/' jaisa rakho.")