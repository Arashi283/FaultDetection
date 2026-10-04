# Sound Fault Detection

A machine fault detection system using audio and vibration signals.

The project extracts signal features from machine sensor data, trains
neural-network-based classification models, converts the trained models
to TensorFlow Lite, and performs real-time prediction using a Raspberry
Pi Pico and laptop.

## Fault Classes

The system classifies machine conditions into three classes:

- Normal
- Mild Fault
- Severe Fault

## Project Overview

The project contains two different machine-fault-detection pipelines.

### 1. Sensor-Fusion Pipeline

The sensor-fusion pipeline combines audio data with vibration data.

It uses:

- Microphone/audio features
- Six vibration channels

Total input features:

**60 features**

The trained TensorFlow Lite model is:

`models/fault_detection_model.tflite`

### 2. Single-Vibration Pipeline

The single-vibration pipeline uses audio data together with one vibration
channel.

It uses:

- Microphone/audio features
- One vibration channel

Total input features:

**35 features**

The trained TensorFlow Lite model is:

`models/fault_detection_model_single_vibration.tflite`

The two models use different feature dimensions and their scalers are
not interchangeable.

---

# Feature Extraction

## Audio Features

The audio processing pipeline extracts features from the microphone
signal using FFT-based features and MFCC features.

The feature extraction configuration includes:

- Sampling rate: 16 kHz
- 13 MFCC coefficients
- FFT size: 2048
- Hop length: 512

## Vibration Features

The vibration pipelines extract numerical features from the vibration
signals.

The sensor-fusion pipeline processes six vibration channels, while the
single-vibration pipeline processes one vibration channel.

---

# Hardware

The real-time system uses a Raspberry Pi Pico together with:

- Piezo vibration sensor
- INMP441 I2S microphone
- USB serial connection to a laptop

## Raspberry Pi Pico

The Pico collects vibration and audio signals and sends them to the
laptop through USB serial communication.

### Piezo Sensor

The vibration sensor is connected to:

`GP26`

### INMP441

The microphone connections are:

| INMP441 | Raspberry Pi Pico |
|---|---|
| SCK | GP16 |
| WS | GP17 |
| SD | GP19 |

Audio sampling rate:

`16 kHz`

---

# Repository Structure

```text
sound-fault-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── feature_extraction_csv.py
│   ├── feature_extraction_single_vibration.py
│   ├── model_training_fixed.py
│   ├── model_training_single_vibration.py
│   ├── laptop_realtime_predict_fixed.py
│   └── pico_code.py
│
├── models/
│   ├── fault_detection_model.tflite
│   └── fault_detection_model_single_vibration.tflite
│
├── data/
│   └── processed/
│       ├── features_X.npy
│       ├── labels_y.npy
│       ├── features_X_single_vibration.npy
│       └── labels_y_single_vibration.npy
│
├── scalers/
│   ├── feature_scaler.npz
│   └── feature_scaler_single_vibration.npz
│
└── legacy/
    ├── feature_extraction.py
    ├── model_training.py
    └── laptop_realtime_predict.py
