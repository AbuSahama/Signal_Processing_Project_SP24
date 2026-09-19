"""
filters.py

Digital Low-Pass, High-Pass, and Band-Pass filters for generated signals.

Filtering calculations are kept separate from plotting so that the
filter functions can be reused by the CLI or GUI.
"""


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy import signal

FloatArray = NDArray[np.float64]


def validate_signal(signal_data: FloatArray) -> None:
    """Validate that signal is not empty."""
    if len(signal_data) <= 0:
        raise ValueError("Signal must not be empty")


def validate_order(order: int) -> None:
    """Validate the filter order."""
    if not isinstance(order, int) or order <= 0:
        raise ValueError("Filter order must be a positive integer")


def validate_filt_length(signal_data: FloatArray, order: int, n_sections: int = 1) -> None:
    """Ensure the signal is long enough for filtfilt's default padding.

    filtfilt pads the signal by roughly 3x the filter's effective length.
    For band-pass filters, `a`/`b` arrays are twice the length of `order`
    (n_sections=2), so we pass that in explicitly.
    """
    effective_len = order * n_sections * 2 + 1  # length of a/b returned by butter
    min_len = 3 * (effective_len - 1)
    if len(signal_data) <= min_len:
        raise ValueError(
            f"Signal is too short for filtfilt with order={order}. "
            f"Need more than {min_len} samples, got {len(signal_data)}. "
            f"Use a shorter order or a longer signal."
        )


def validate_cutoff(cutoff: float, sample_rate: float) -> None:
    """Validate a single cutoff frequency."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    nyquist = sample_rate / 2

    if cutoff <= 0:
        raise ValueError("cutoff frequency must be positive")

    if cutoff >= nyquist:
        raise ValueError(f"Cutoff frequency must be less than the Nyquist frequency {nyquist}Hz")


def validate_band(low_cutoff: float, high_cutoff: float, sample_rate: float) -> None:
    """Validate band-pass filter cutoff frequencies."""

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")

    nyquist = sample_rate / 2

    if low_cutoff <= 0:
        raise ValueError("Low cutoff frequency must be positive")

    if high_cutoff <= 0:
        raise ValueError("High cutoff frequency must be positive")

    if high_cutoff <= low_cutoff:
        raise ValueError(
            "High cutoff frequency must be greater than low cutoff frequency"
        )

    if high_cutoff >= nyquist:
        raise ValueError(
            f"High cutoff frequency must be less than "
            f"Nyquist frequency ({nyquist} Hz)"
        )


def low_pass_filter(
    signal_data: FloatArray, sample_rate: float, cutoff_frequency: float, order: int = 5
) -> FloatArray:
    """Apply a Butterworth low-pass filter.
    Frequencies below the cutoff are mostly preserved while frequencies above the cutoff are attenuated."""

    signal_data = np.asarray(signal_data, dtype=np.float64)
    validate_signal(signal_data)
    validate_order(order)
    validate_cutoff(cutoff_frequency, sample_rate)
    validate_filt_length(signal_data, order, n_sections=1)

    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_frequency / nyquist

    # b, a = signal.butter(order, normalized_cutoff, btype="low")

    # filtered_signal = signal.filtfilt(b, a, signal_data)
    sos = signal.butter(order, normalized_cutoff, btype="low", output="sos")
    filtered_signal = signal.sosfiltfilt(sos, signal_data)

    return filtered_signal


def high_pass_filter(
    signal_data: FloatArray, sample_rate: float, cutoff_frequency: float, order: int = 5
) -> FloatArray:
    """Apply a Butterworth high-pass filter.
    Frequencies below the cutoff are attenuated while frequencies above the cutoff are mostly preserved."""

    signal_data = np.asarray(signal_data, dtype=np.float64)
    validate_signal(signal_data)
    validate_order(order)
    validate_cutoff(cutoff_frequency, sample_rate)
    validate_filt_length(signal_data, order, n_sections=1)

    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_frequency / nyquist

    # b, a = signal.butter(order, normalized_cutoff, btype="high")

    # filtered_signal = signal.filtfilt(b, a, signal_data)
    sos = signal.butter(order, normalized_cutoff, btype="high", output="sos")
    filtered_signal = signal.sosfiltfilt(sos, signal_data)

    return filtered_signal


def band_pass_filter(
    signal_data: FloatArray,
    sample_rate: float,
    low_cutoff: float,
    high_cutoff: float,
    order: int = 5,
) -> FloatArray:
    """Apply a Butterworth band-pass filter. Frequencies between low_cutoff and high_cutoff are mostly preserved
    while frequencies outside this range are attenuated."""

    signal_data = np.asarray(signal_data, dtype=np.float64)
    validate_signal(signal_data)
    validate_order(order)
    validate_band(low_cutoff, high_cutoff, sample_rate)
    validate_filt_length(signal_data, order, n_sections=2)

    nyquist = sample_rate / 2

    normalized_cutoffs = [low_cutoff / nyquist, high_cutoff / nyquist]

    # b, a = signal.butter(order, normalized_cutoffs, btype="band")

    # filtered_signal = signal.filtfilt(b, a, signal_data)
    sos = signal.butter(order, normalized_cutoffs, btype="band", output="sos")
    filtered_signal = signal.sosfiltfilt(sos, signal_data)

    return filtered_signal


def plot_filter_comparison(
    original_signal: FloatArray, filtered_signal: FloatArray, sample_rate: float, title: str
) -> None:
    """Plot original and filtered signals in the time domain."""

    time = np.arange(len(original_signal)) / sample_rate

    plt.figure()

    plt.plot(time, original_signal, label="Original")
    plt.plot(time, filtered_signal, label="Filtered Signal")

    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()