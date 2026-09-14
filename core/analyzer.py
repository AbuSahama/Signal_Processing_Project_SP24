"""
analyzer.py

FFT and STFT analysis of a generated signal, plus matching plot helpers.
Calculation and plotting are kept separate so calculate_fft/calculate_stft
can be reused by a GUI (which needs raw arrays) without pulling in
matplotlib's pyplot state machine.
"""

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
    """Compute the single-sided FFT amplitude/magnitude spectrum of `signal_data`.

    Returns (frequencies, magnitude), both containing only the
    non-negative-frequency half of the spectrum. `magnitude` is scaled so
    that a pure sine/cosine component of amplitude A shows up as a peak
    of height ~A (DC is not doubled; all other bins are, to account for
    the discarded negative-frequency half).
    """
    signal_data = np.asarray(signal_data, dtype=np.float64)
    validate_signal(signal_data)
    validate_sample_rate(sample_rate)

    n = len(signal_data)

    fft_result = np.fft.fft(signal_data)
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)

    # Two-sided amplitude spectrum, normalized by n.
    amplitude = np.abs(fft_result) / n

    positive_frequencies = frequencies[: n // 2]
    positive_amplitude = amplitude[: n // 2].copy()

    # Fold the negative-frequency energy back in (everything except DC).
    positive_amplitude[1:] *= 2

    return positive_frequencies, positive_amplitude


def calculate_fft_two_sided(
    signal_data: FloatArray, sample_rate: float
) -> tuple[FloatArray, FloatArray]:
    """Compute the full (two-sided) FFT amplitude spectrum, frequency-shifted
    so 0 Hz sits in the middle of the returned arrays.

    Unlike `calculate_fft`, this keeps the negative-frequency half instead
    of folding it into the positive side. Most real-signal use cases don't
    need this (the single-sided spectrum is a full, non-redundant summary),
    but it's useful when a signal's symmetry around 0 Hz is itself the thing
    you want to see plotted — e.g. a sinc pulse, whose textbook Fourier
    transform is a rectangle centered on DC. `calculate_fft`'s single-sided
    output only shows the right half of that rectangle, which can look like
    it "starts" at 0 Hz rather than being centered on it.

    Returns (frequencies, magnitude), both length n, ordered from
    -sample_rate/2 to +sample_rate/2 (approximately; exact bounds depend on
    whether n is even or odd).
    """
    signal_data = np.asarray(signal_data, dtype=np.float64)
    validate_signal(signal_data)
    validate_sample_rate(sample_rate)

    n = len(signal_data)
    fft_result = np.fft.fft(signal_data)
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)
    amplitude = np.abs(fft_result) / n

    return np.fft.fftshift(frequencies), np.fft.fftshift(amplitude)


def plot_fft(signal_data: FloatArray, sample_rate: float, db_scale: bool = False) -> None:
    """Compute and display the FFT magnitude spectrum in a new figure.

    `db_scale=True` plots 20*log10(magnitude) instead of linear magnitude,
    which is often easier to read when peaks span a wide dynamic range.
    """
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
    """Compute the STFT magnitude spectrogram of `signal_data`.

    `window` and `nperseg` are exposed (rather than hardcoded) so a GUI
    can offer time/frequency-resolution trade-offs without touching this
    function's internals.

    Returns (frequencies, times, magnitude).
    """
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


if __name__ == "__main__":
    print("Testing analyzer.py...\n")

    sample_rate = 1000
    duration = 1
    frequency = 50
    amplitude = 1

    t = np.arange(0, duration, 1 / sample_rate)
    test_signal = amplitude * np.sin(2 * np.pi * frequency * t)

    print("Test signal created.")
    print(f"Frequency   : {frequency} Hz")
    print(f"Sample rate : {sample_rate} Hz")
    print(f"Samples     : {len(test_signal)}\n")

    frequencies, magnitude = calculate_fft(test_signal, sample_rate)
    assert len(frequencies) == len(magnitude), "FFT frequency/magnitude length mismatch"

    peak_index = np.argmax(magnitude)
    peak_freq = frequencies[peak_index]
    peak_mag = magnitude[peak_index]
    assert abs(peak_freq - frequency) <= 1, (
        f"expected FFT peak near {frequency} Hz, got {peak_freq} Hz"
    )
    assert abs(peak_mag - amplitude) <= 0.05, (
        f"expected FFT peak magnitude near {amplitude}, got {peak_mag}"
    )
    print("FFT calculation successful!")
    print(f"Number of FFT frequency points : {len(frequencies)}")
    print(f"Detected peak frequency        : {peak_freq:.1f} Hz")
    print(f"Detected peak magnitude        : {peak_mag:.3f}\n")

    stft_frequencies, stft_times, stft_magnitude = calculate_stft(
        test_signal, sample_rate
    )
    assert stft_magnitude.shape == (len(stft_frequencies), len(stft_times)), (
        "STFT magnitude shape mismatch"
    )
    print("STFT calculation successful!")
    print(f"Number of STFT frequency points : {len(stft_frequencies)}")
    print(f"Number of STFT time points      : {len(stft_times)}\n")

    try:
        calculate_fft(np.array([]), sample_rate)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty signal")

    try:
        calculate_fft(test_signal, sample_rate=-100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive sample_rate")

    try:
        calculate_stft(test_signal, sample_rate, nperseg=len(test_signal) + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for nperseg exceeding signal length")

    freqs_2s, mag_2s = calculate_fft_two_sided(test_signal, sample_rate)
    assert len(freqs_2s) == len(test_signal), "two-sided FFT length mismatch"
    assert freqs_2s[0] < 0 < freqs_2s[-1], "two-sided FFT should span negative to positive frequencies"
    mirror_freq = np.argmin(np.abs(freqs_2s - frequency))
    mirror_neg = np.argmin(np.abs(freqs_2s + frequency))
    assert abs(mag_2s[mirror_freq] - mag_2s[mirror_neg]) < 1e-6, (
        "two-sided spectrum should be symmetric about 0 Hz for a real signal"
    )
    print("Two-sided FFT calculation successful (symmetric about 0 Hz, as expected).\n")

    print("All checks passed. analyzer.py is working correctly!")