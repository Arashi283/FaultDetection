"""
Pico Code — Sensor Data Streaming (MicroPython)
------------------------------------------------------------------
Ye code Raspberry Pi Pico pe chalega (Thonny IDE se upload karo).

Kya karta hai:
1. Piezo sensor (GP26, ADC) se vibration data continuously sample karta hai
2. MEMS mic (INMP441, I2S — GP2/GP3/GP4) se audio data sample karta hai
3. Dono ko ek chhote "window" (chunk) mein collect karke USB serial se
   laptop ko bhejta hai, comma-separated values ke roop mein

Laptop side (laptop_realtime_predict.py) is data ko receive karke
features nikalega aur model se prediction karega.

IMPORTANT: Ye code hardware aane ke baad test/calibrate karna hoga —
I2S pin numbers aur sample rates apne exact wiring ke hisab se
adjust karne ki zarurat pad sakti hai.
"""

from machine import Pin, ADC, I2S
import time
import sys

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
PIEZO_PIN = 26          # GP26 (ADC0) — piezo vibration sensor
VIBRATION_SAMPLES = 500  # kitne vibration samples ek window mein lene hain

# I2S mic config (INMP441)
I2S_ID = 0
SCK_PIN = 16  # clock (physical pin 21)
WS_PIN = 17   # word select (physical pin 22)
SD_PIN = 19   # data (physical pin 24)
# NOTE: L/R pin (physical pin 23, GP18) should be wired directly to
# GND (left channel) or 3V3 (right channel) — NOT to a GPIO. It does
# not need to be referenced in code.
AUDIO_SAMPLE_RATE = 16000
AUDIO_BITS = 16
AUDIO_BUFFER_SIZE = 2000  # bytes per read

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
piezo = ADC(Pin(PIEZO_PIN))

try:
    audio_in = I2S(
        I2S_ID,
        sck=Pin(SCK_PIN),
        ws=Pin(WS_PIN),
        sd=Pin(SD_PIN),
        mode=I2S.RX,
        bits=AUDIO_BITS,
        format=I2S.MONO,
        rate=AUDIO_SAMPLE_RATE,
        ibuf=4000,
    )
    MIC_AVAILABLE = True
except Exception as e:
    # Agar I2S setup fail ho (wiring issue ya pin mismatch), vibration-only
    # mode mein chal jao taaki poora system down na ho
    print("MIC INIT FAILED, running vibration-only mode:", e)
    MIC_AVAILABLE = False


def read_vibration_window():
    """Piezo sensor se ek window ka vibration data collect karta hai."""
    samples = []
    for _ in range(VIBRATION_SAMPLES):
        samples.append(piezo.read_u16())
        time.sleep_us(200)   # ~5kHz sampling rate (adjust as needed)
    return samples


def read_audio_window():
    """MEMS mic se ek window ka audio data collect karta hai."""
    if not MIC_AVAILABLE:
        return []
    buf = bytearray(AUDIO_BUFFER_SIZE)
    num_bytes = audio_in.readinto(buf)
    # 16-bit samples -> int list mein convert karo
    samples = []
    for i in range(0, num_bytes - 1, 2):
        val = buf[i] | (buf[i + 1] << 8)
        if val >= 32768:
            val -= 65536
        samples.append(val)
    return samples


def send_window():
    """Ek window ka data (vibration + audio) serial se bhejta hai."""
    vib = read_vibration_window()
    aud = read_audio_window()

    # Format: VIB:<comma-separated vibration>|AUD:<comma-separated audio>
    vib_str = ",".join(str(v) for v in vib)
    aud_str = ",".join(str(a) for a in aud)
    line = "VIB:{}|AUD:{}".format(vib_str, aud_str)
    print(line)   # USB serial pe print hi transmission hai


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
print("READY")   # laptop ko signal ki Pico shuru ho gaya
time.sleep(1)

while True:
    send_window()
    time.sleep(0.5)   # har 0.5 second mein ek naya reading bhejo