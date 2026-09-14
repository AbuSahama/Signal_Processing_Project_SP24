"""
audio.py

Save/load signals as WAV files, and play/record audio via the system's
default audio device.
"""

from __future__ import annotations

import os

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray
from scipy.io.wavfile import read, write

FloatArray = NDArray[np.float64]

DEFAULT_AUDIO_DIR = "generated/audio"


def save_audio(
    filename: str,
    signal_data: FloatArray,
    sample_rate: float,
    output_dir: str = DEFAULT_AUDIO_DIR,
) -> str:
    """Normalize `signal_data` to int16 range and save it as a WAV file.

    Returns the path the file was saved to.
    """
    if len(signal_data) == 0:
        raise ValueError("signal_data must not be empty")

    max_value = np.max(np.abs(signal_data))
    normalized = signal_data / max_value if max_value > 0 else signal_data
    int16_data = np.int16(normalized * 32767)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    write(filepath, sample_rate, int16_data)

    print(f"Audio saved successfully: {filepath}")
    return filepath


def load_audio(filename: str) -> tuple[int, FloatArray]:
    """Load a WAV file and return (sample_rate, signal) with the signal
    normalized to [-1, 1] as float32."""
    sample_rate, signal_data = read(filename)
    signal_data = signal_data.astype(np.float32)

    peak = np.max(np.abs(signal_data))
    if peak > 0:
        signal_data = signal_data / peak

    return sample_rate, signal_data


def play_audio(signal_data: FloatArray, sample_rate: float) -> None:
    """Play `signal_data` through the default output device.

    Prints a friendly message instead of raising if no audio device is
    available (e.g. running headless/in CI).
    """
    try:
        sd.play(signal_data, sample_rate)
        print("Audio playback started.")
    except sd.PortAudioError as exc:
        print(f"Could not start playback (no audio device available?): {exc}")


def stop_audio() -> None:
    """Stop any in-progress playback or recording."""
    sd.stop()
    print("Audio playback stopped.")


def record_audio(duration: float, sample_rate: float) -> FloatArray:
    """Record `duration` seconds of mono audio from the default input device."""
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    print("Recording started...")
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except sd.PortAudioError as exc:
        raise RuntimeError(
            f"Could not record audio (no input device available?): {exc}"
        ) from exc

    print("Recording finished.")
    return recording.flatten()


if __name__ == "__main__":
    print("Testing audio.py...\n")

    sample_rate = 44100
    duration = 1
    frequency = 440
    amplitude = 0.5

    time = np.arange(0, duration, 1 / sample_rate)
    test_signal = amplitude * np.sin(2 * np.pi * frequency * time)

    print("Test signal created.")
    print(f"Sample rate : {sample_rate} Hz")
    print(f"Duration    : {duration} second")
    print(f"Frequency   : {frequency} Hz")
    print(f"Samples     : {len(test_signal)}\n")

    filepath = save_audio("test_audio.wav", test_signal, sample_rate)
    assert os.path.exists(filepath), "expected saved WAV file to exist"

    loaded_rate, loaded_signal = load_audio(filepath)
    assert loaded_rate == sample_rate, "loaded sample rate mismatch"
    assert len(loaded_signal) == len(test_signal), "loaded signal length mismatch"
    assert np.max(np.abs(loaded_signal)) <= 1.0 + 1e-6, "loaded signal not normalized"
    print("Save/load round-trip verified.\n")

    try:
        save_audio("empty.wav", np.array([]), sample_rate)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty signal")

    print("Testing playback (skipped if no audio device is present)...")
    play_audio(test_signal, sample_rate)
    sd.wait()
    print("Playback finished.\n")

    print("audio.py is working correctly!")