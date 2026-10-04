import os
import numpy as np
import pandas as pd
import librosa


# =========================================================
# CONFIGURATION
# =========================================================

# Dataset is inside the current project:
# D:\EmbededProject\archive
DATA_DIR = "archive"

# MaFaulDa recordings use 50 kHz
SAMPLE_RATE = 50000

N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# CSV columns are 0-indexed
# Column 8 = microphone
MIC_COLUMN = 7

# Use ONE vibration/accelerometer channel
# Column 2 = first accelerometer channel
VIBRATION_COLUMN = 1


# =========================================================
# FEATURE EXTRACTION FROM ONE CSV
# =========================================================

def extract_features_from_csv(file_path):

    # Read CSV
    data = pd.read_csv(
        file_path,
        header=None
    )

    # -----------------------------------------------------
    # AUDIO FEATURES
    # -----------------------------------------------------

    mic_signal = data.iloc[:, MIC_COLUMN].values.astype(
        np.float32
    )

    # FFT
    fft_result = np.fft.fft(mic_signal)

    fft_magnitude = np.abs(
        fft_result
    )[:len(fft_result) // 2]

    fft_features = np.array([
        np.mean(fft_magnitude),
        np.std(fft_magnitude),
        np.max(fft_magnitude),
        np.sum(fft_magnitude ** 2)
    ])

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=mic_signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    # 4 FFT + 13 MFCC mean + 13 MFCC std
    # = 30 audio features
    sound_features = np.concatenate([
        fft_features,
        mfcc_mean,
        mfcc_std
    ])


    # -----------------------------------------------------
    # ONE VIBRATION CHANNEL
    # -----------------------------------------------------

    vibration = data.iloc[:, VIBRATION_COLUMN].values.astype(
        np.float32
    )

    # Time-domain features
    rms = np.sqrt(
        np.mean(vibration ** 2)
    )

    peak = np.max(
        np.abs(vibration)
    )

    std = np.std(
        vibration
    )

    # Frequency-domain features
    fft_v = np.abs(
        np.fft.fft(vibration)
    )[:len(vibration) // 2]

    fft_mean = np.mean(
        fft_v
    )

    fft_energy = np.sum(
        fft_v ** 2
    )

    # 5 vibration features
    vibration_features = np.array([
        rms,
        peak,
        std,
        fft_mean,
        fft_energy
    ])


    # -----------------------------------------------------
    # SENSOR FUSION
    # -----------------------------------------------------

    # 30 audio + 5 vibration = 35 features
    feature_vector = np.concatenate([
        sound_features,
        vibration_features
    ])

    return feature_vector


# =========================================================
# PROCESS ONE FOLDER
# =========================================================

def process_folder(
    folder_path,
    label_name,
    label_id
):

    X = []
    y = []

    if not os.path.exists(folder_path):

        print(
            "[ERROR] Folder not found:",
            folder_path
        )

        return X, y

    # Find CSV files
    files = []

    for root, _, filenames in os.walk(
        folder_path
    ):

        for filename in filenames:

            if filename.lower().endswith(".csv"):

                files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    print()
    print(
        f"[INFO] {label_name}: "
        f"{len(files)} CSV files found"
    )

    # Process every CSV
    for index, file_path in enumerate(
        files,
        start=1
    ):

        try:

            features = extract_features_from_csv(
                file_path
            )

            X.append(
                features
            )

            y.append(
                label_id
            )

            print(
                f"[OK] {index}/{len(files)} "
                f"{label_name}: "
                f"{os.path.basename(file_path)}"
            )

        except Exception as e:

            print(
                "[ERROR]",
                file_path,
                e
            )

    return X, y


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    X = []
    y = []


    # =====================================================
    # CLASS 0 — NORMAL
    #
    # archive/
    #   normal/
    #       normal/
    #           *.csv
    # =====================================================

    normal_folder = os.path.join(
        DATA_DIR,
        "normal",
        "normal"
    )

    X_part, y_part = process_folder(
        normal_folder,
        "NORMAL",
        0
    )

    X.extend(
        X_part
    )

    y.extend(
        y_part
    )


    # =====================================================
    # CLASS 1 — MILD FAULT
    #
    # 6g
    # 10g
    # 15g
    # =====================================================

    mild_levels = [
        "6g",
        "10g",
        "15g"
    ]

    for level in mild_levels:

        folder = os.path.join(
            DATA_DIR,
            "imbalance",
            "imbalance",
            level
        )

        X_part, y_part = process_folder(
            folder,
            f"MILD FAULT ({level})",
            1
        )

        X.extend(
            X_part
        )

        y.extend(
            y_part
        )


    # =====================================================
    # CLASS 2 — SEVERE FAULT
    #
    # 20g
    # 25g
    # 30g
    # 35g
    # =====================================================

    severe_levels = [
        "20g",
        "25g",
        "30g",
        "35g"
    ]

    for level in severe_levels:

        folder = os.path.join(
            DATA_DIR,
            "imbalance",
            "imbalance",
            level
        )

        X_part, y_part = process_folder(
            folder,
            f"SEVERE FAULT ({level})",
            2
        )

        X.extend(
            X_part
        )

        y.extend(
            y_part
        )


    # =====================================================
    # CONVERT TO NUMPY ARRAYS
    # =====================================================

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print()
    print("========================================")
    print("DATASET COMPLETE")
    print("========================================")

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )


    if len(y) > 0:

        print()
        print("Class counts:")

        print(
            "Normal:",
            np.sum(y == 0)
        )

        print(
            "Mild Fault:",
            np.sum(y == 1)
        )

        print(
            "Severe Fault:",
            np.sum(y == 2)
        )


        # =================================================
        # SAVE FEATURES
        # =================================================

        np.save(
            "features_X_single_vibration.npy",
            X
        )

        np.save(
            "labels_y_single_vibration.npy",
            y
        )

        print()
        print(
            "[SAVED] "
            "features_X_single_vibration.npy"
        )

        print(
            "[SAVED] "
            "labels_y_single_vibration.npy"
        )


    else:

        print()
        print(
            "[ERROR] No data was processed."
        )

        print(
            "Check the archive folder structure."
        )