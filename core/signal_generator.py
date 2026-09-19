"""
signal_generator.py

Utilities for generating common test/reference waveforms (sine, cosine,
square, triangle, sinc, chirp, noise) as NumPy arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal

__all__ = [
    "create_time_vector",
    "create_sinc_time_vector",
    "generate_sine",
    "generate_cosine"
    "generate_square",
    "generate_triangle",
    "generate_sinc",
    "generate_chirp",
    "generate_noise",
    "normalize",
]

FloatArray = NDArray[np.float64]


def validate_duration_and_rate(duration: float, sample_rate: float) -> None:
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")


def create_time_vector(duration: float, sample_rate: float) -> FloatArray:
    """Create a time vector from 0 to `duration` (exclusive) at `sample_rate` Hz."""
    validate_duration_and_rate(duration, sample_rate)
    return np.arange(0, duration, 1 / sample_rate)


def create_sinc_time_vector(duration: float, sample_rate: float) -> FloatArray:
    """Create a symmetric time vector centered on 0, from -duration/2 to duration/2.

    Useful for sinc pulses, which are conventionally centered at t=0.
    """
    validate_duration_and_rate(duration, sample_rate)
    return np.arange(-duration / 2, duration / 2, 1 / sample_rate)


def generate_sine(
    t: FloatArray, frequency: float, amplitude: float, phase: float = 0
) -> FloatArray:
    """Generate a sine wave. `phase` is in degrees."""
    phase_rad = np.deg2rad(phase)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)

def generate_cosine(
    t: FloatArray, frequency: float, amplitude: float, phase: float = 0
) -> FloatArray:
    """Generate a cosine wave. `phase` is in degrees."""
    phase_rad = np.deg2rad(phase)
    return amplitude * np.cos(2 * np.pi * frequency * t + phase_rad)

def generate_square(
    t: FloatArray, frequency: float, amplitude: float, duty_cycle: float = 50
) -> FloatArray:
    """Generate a square wave. `duty_cycle` is a percentage in (0, 100)."""
    if not 0 < duty_cycle < 100:
        raise ValueError(f"duty_cycle must be in (0, 100), got {duty_cycle}")
    return amplitude * signal.square(2 * np.pi * frequency * t, duty=duty_cycle / 100)


def generate_triangle(t: FloatArray, frequency: float, amplitude: float) -> FloatArray:
    """Generate a symmetric triangle wave."""
    return amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)


def generate_sinc(t: FloatArray, frequency: float, amplitude: float) -> FloatArray:
    """Generate a sinc pulse. Pair with `create_sinc_time_vector` for a
    symmetric, zero-centered result."""
    return amplitude * np.sinc(frequency * t)


def generate_chirp(
    t: FloatArray,
    start_frequency: float,
    end_frequency: float,
    duration: float,
    amplitude: float,
    method: str = "linear",
) -> FloatArray:
    """Generate a frequency sweep (chirp) from `start_frequency` to
    `end_frequency` over `duration` seconds."""
    return amplitude * signal.chirp(
        t, f0=start_frequency, t1=duration, f1=end_frequency, method=method
    )


def generate_noise(
    t: FloatArray, amplitude: float = 1.0, seed: int | None = None
) -> FloatArray:
    """Generate Gaussian white noise, scaled by `amplitude` (used as the
    standard deviation). Pass `seed` for reproducible output."""
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(len(t))


def normalize(sig: FloatArray, target_amplitude: float = 1.0) -> FloatArray:
    """Rescale `sig` so its peak absolute value equals `target_amplitude`."""
    peak = np.max(np.abs(sig))
    if peak == 0:
        return sig
    return sig * (target_amplitude / peak)


if __name__ == "__main__":
    print("Testing signal_generator.py...\n")

    duration = 1
    sample_rate = 10000
    frequency = 20
    amplitude = 1
    
    expected_samples = int(duration * sample_rate)

    t = create_time_vector(duration, sample_rate)
    t_sinc = create_sinc_time_vector(duration, sample_rate)

    signals = {
        "sine": generate_sine(t, frequency, amplitude),
        "cosine": generate_sine(t, frequency, amplitude),
        "square": generate_square(t, frequency, amplitude),
        "triangle": generate_triangle(t, frequency, amplitude),
        "sinc": generate_sinc(t_sinc, frequency, amplitude),
        "chirp": generate_chirp(t, 5, 50, duration, amplitude),
        "noise": generate_noise(t, amplitude, seed=42),
    }

    # Real sanity checks, not just length printouts.
    assert len(t) == expected_samples, "time vector length mismatch"
    for name, sig in signals.items():
        assert len(sig) == expected_samples, f"{name} length mismatch"
        assert np.all(np.isfinite(sig)), f"{name} contains non-finite values"

    assert np.isclose(np.max(generate_sine(t, frequency, 1)), 1, atol=1e-2), \
        "sine amplitude out of range"
    assert np.max(np.abs(generate_sine(t, frequency, 2))) <= 2 + 1e-9, \
        "sine amplitude exceeds requested value"

    try:
        create_time_vector(-1, sample_rate)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative duration")

    print(f"Number of samples : {len(t)}")
    print(f"Duration          : {duration} seconds")
    print(f"Sample rate       : {sample_rate} Hz")
    print(f"Frequency         : {frequency} Hz")
    print(f"Amplitude         : {amplitude}\n")

    print("Generated signals:")
    for name, sig in signals.items():
        print(f"{name.capitalize():9}: {len(sig)} samples")

    print("\nAll checks passed. signal_generator.py is working correctly!")