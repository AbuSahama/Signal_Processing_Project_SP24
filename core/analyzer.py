from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy import signal

FloatArray = NDArray[np.float64]

__all__ = [
    "calculate_fft",
    "calculate_fft_two_sided",
    "plot_fft",
    "calculate_stft",
    "plot_stft",
]

def validate_signal(signal_data: FloatArray) -> None:
    if len(signal_data) <= 0:
        raise ValueError("Signal must not be empty")


def validate_sample_rate(sample_rate: float) -> None:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")


def calculate_fft(
    signal_data: FloatArray, sample_rate: float
) -> tuple[FloatArray, FloatArray]:
    signal_data = np.asarray(signal_data, dtype=np.float64)
    validate_signal(signal_data)
    validate_sample_rate(sample_rate)

    n = len(signal_data)

    fft_result = np.fft.fft(signal_data)
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)

    amplitude = np.abs(fft_result) / n

    positive_frequencies = frequencies[: n // 2]
    positive_amplitude = amplitude[: n // 2].copy()

    positive_amplitude[1:] *= 2

    return positive_frequencies, positive_amplitude

def calculate_fft_two_sided(
    signal_data: FloatArray, sample_rate: float
) -> tuple[FloatArray, FloatArray]:
    
    signal_data = np.asarray(signal_data, dtype=np.float64)
    validate_signal(signal_data)
    validate_sample_rate(sample_rate)

    n = len(signal_data)
    fft_result = np.fft.fft(signal_data)
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)
    amplitude = np.abs(fft_result) / n

    return np.fft.fftshift(frequencies), np.fft.fftshift(amplitude)

def plot_fft(signal_data: FloatArray, sample_rate: float, db_scale: bool = False) -> None:
    frequencies, magnitude = calculate_fft(signal_data, sample_rate)

    y = magnitude
    ylabel = "Magnitude"
    if db_scale:
        floor = 1e-12  # avoid log(0)
        y = 20 * np.log10(np.maximum(magnitude, floor))
        ylabel = "Magnitude (dB)"

    plt.figure()
    plt.plot(frequencies, y, color="g")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    plt.title("FFT Spectrum")
    plt.grid(True)
    plt.show()


def calculate_stft(
    signal_data: FloatArray,
    sample_rate: float,
    window: str = "hann",
    nperseg: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    
    signal_data = np.asarray(signal_data, dtype=np.float64)
    validate_signal(signal_data)
    validate_sample_rate(sample_rate)

    if nperseg is not None and nperseg > len(signal_data):
        raise ValueError(
            f"nperseg ({nperseg}) cannot exceed the signal length ({len(signal_data)})"
        )

    frequencies, times, stft_result = signal.stft(
        signal_data, fs=sample_rate, window=window, nperseg=nperseg
    )
    magnitude = np.abs(stft_result)
    return frequencies, times, magnitude

def plot_stft(
    signal_data: FloatArray,
    sample_rate: float,
    window: str = "hann",
    nperseg: int | None = None,
) -> None:
    """Compute and display the STFT spectrogram in a new figure."""
    frequencies, times, magnitude = calculate_stft(
        signal_data, sample_rate, window=window, nperseg=nperseg
    )

    plt.figure()
    plt.pcolormesh(times, frequencies, magnitude, shading="gouraud")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Spectrogram")
    plt.colorbar(label="Magnitude")
    plt.show()

