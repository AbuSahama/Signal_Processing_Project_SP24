import os
import numpy as np
from scipy.io.wavfile import write

def save_audio(filename, signal, sample_rate):

    os.makedirs("generated/audio", exist_ok=True)

    signal = signal / np.max(np.abs(signal))
    signal = np.int16(signal * 32767)

    filepath = os.path.join("generated/audio", filename)

    write(filepath, sample_rate, signal)

    print(f"Audio saved successfully: {filepath}")