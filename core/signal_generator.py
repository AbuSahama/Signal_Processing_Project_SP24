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
    validate_duration_and_rate(duration, sample_rate)
    return np.arange(0, duration, 1 / sample_rate)


def create_sinc_time_vector(duration: float, sample_rate: float) -> FloatArray:
    validate_duration_and_rate(duration, sample_rate)
    return np.arange(-duration / 2, duration / 2, 1 / sample_rate)


def generate_sine(
    t: FloatArray, frequency: float, amplitude: float, phase: float = 0
) -> FloatArray:
    phase_rad = np.deg2rad(phase)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)

def generate_cosine(
    t: FloatArray, frequency: float, amplitude: float, phase: float = 0
) -> FloatArray:
    phase_rad = np.deg2rad(phase)
    return amplitude * np.cos(2 * np.pi * frequency * t + phase_rad)

def generate_square(
    t: FloatArray, frequency: float, amplitude: float, duty_cycle: float = 50
) -> FloatArray:
    if not 0 < duty_cycle < 100:
        raise ValueError(f"duty_cycle must be in (0, 100), got {duty_cycle}")
    return amplitude * signal.square(2 * np.pi * frequency * t, duty=duty_cycle / 100)


def generate_triangle(t: FloatArray, frequency: float, amplitude: float) -> FloatArray:
    return amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)


def generate_sinc(t: FloatArray, frequency: float, amplitude: float) -> FloatArray:
    return amplitude * np.sinc(frequency * t)


def generate_chirp(
    t: FloatArray,
    start_frequency: float,
    end_frequency: float,
    duration: float,
    amplitude: float,
    method: str = "linear",
) -> FloatArray:
    return amplitude * signal.chirp(
        t, f0=start_frequency, t1=duration, f1=end_frequency, method=method
    )


def generate_noise(
    t: FloatArray, amplitude: float = 1.0, seed: int | None = None
) -> FloatArray:
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(len(t))


def normalize(sig: FloatArray, target_amplitude: float = 1.0) -> FloatArray:
    """Rescale `sig` so its peak absolute value equals `target_amplitude`."""
    peak = np.max(np.abs(sig))
    if peak == 0:
        return sig
    return sig * (target_amplitude / peak)
