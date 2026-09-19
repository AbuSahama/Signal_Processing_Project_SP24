"""
signal_specs.py

Single source of truth for "what does each waveform need, and how do I
build it". Both main.py (CLI) and gui.py (GUI) import SIGNAL_SPECS
and generate() instead of duplicating a per-waveform if/elif chain.

To add a new waveform: write a *_build function, add a SignalSpec entry
to SIGNAL_SPECS, and both front ends pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

# import signal_generator as sg
from . import signal_generator as sg

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParamSpec:
    """Describes one numeric input a waveform needs (frequency, phase, etc.)."""

    key: str
    label: str
    minimum: float | None = None
    maximum: float | None = None
    default: float = 0.0


@dataclass(frozen=True)
class SignalSpec:
    """Describes one waveform type: its extra inputs, builder, and title."""

    name: str
    params: Sequence[ParamSpec]
    build: Callable[[FloatArray, dict], FloatArray]
    title: Callable[[dict], str]
    uses_sinc_time: bool = False


def _sine_build(t: FloatArray, p: dict) -> FloatArray:
    return sg.generate_sine(t, p["frequency"], p["amplitude"], p["phase"])

def _cosine_build(t: FloatArray, p: dict) -> FloatArray:
    return sg.generate_cosine(t, p["frequency"], p["amplitude"], p["phase"])


def _square_build(t: FloatArray, p: dict) -> FloatArray:
    return sg.generate_square(t, p["frequency"], p["amplitude"], p["duty_cycle"])


def _triangle_build(t: FloatArray, p: dict) -> FloatArray:
    return sg.generate_triangle(t, p["frequency"], p["amplitude"])


def _sinc_build(t: FloatArray, p: dict) -> FloatArray:
    return sg.generate_sinc(t, p["frequency"], p["amplitude"])


def _chirp_build(t: FloatArray, p: dict) -> FloatArray:
    return sg.generate_chirp(
        t, p["start_frequency"], p["end_frequency"], p["duration"], p["amplitude"]
    )


# def _noise_build(t: FloatArray, p: dict) -> FloatArray:
#     return sg.generate_noise(t, p["amplitude"], seed=p.get("seed"))


SIGNAL_SPECS: dict[str, SignalSpec] = {
    "Sine": SignalSpec(
        name="Sine",
        params=[
            ParamSpec("frequency", "Frequency (Hz)", minimum=0),
            ParamSpec("phase", "Phase Shift (degrees)"),
        ],
        build=_sine_build,
        title=lambda p: (
            f"Sine Wave | f={p['frequency']} Hz | "
            f"A={p['amplitude']} | Phase={p['phase']}\u00b0"
        ),
    ),

    "Cosine": SignalSpec(
            name="Cosine",
            params=[
                ParamSpec("frequency", "Frequency (Hz)", minimum=0),
                ParamSpec("phase", "Phase Shift (degrees)"),
            ],
            build=_cosine_build,
            title=lambda p: (
                f"Cosine Wave | f={p['frequency']} Hz | "
                f"A={p['amplitude']} | Phase={p['phase']}\u00b0"
            ),
        ),
    
    "Square": SignalSpec(
        name="Square",
        params=[
            ParamSpec("frequency", "Frequency (Hz)", minimum=0),
            ParamSpec(
                "duty_cycle", "Duty Cycle (%)", minimum=0, maximum=100, default=50
            ),
        ],
        build=_square_build,
        title=lambda p: (
            f"Square Wave | f={p['frequency']} Hz | "
            f"A={p['amplitude']} | Duty={p['duty_cycle']}%"
        ),
    ),
    "Triangle": SignalSpec(
        name="Triangle",
        params=[ParamSpec("frequency", "Frequency (Hz)", minimum=0)],
        build=_triangle_build,
        title=lambda p: f"Triangle Wave | f={p['frequency']} Hz | A={p['amplitude']}",
    ),
    "Sinc": SignalSpec(
        name="Sinc",
        params=[ParamSpec("frequency", "Frequency Scaling Factor", minimum=0)],
        build=_sinc_build,
        title=lambda p: f"Sinc Signal | f={p['frequency']} | A={p['amplitude']}",
        uses_sinc_time=True,
    ),
    "Chirp": SignalSpec(
        name="Chirp",
        params=[
            ParamSpec("start_frequency", "Initial Frequency (Hz)", minimum=0),
            ParamSpec("end_frequency", "Final Frequency (Hz)", minimum=0),
        ],
        build=_chirp_build,
        title=lambda p: (
            f"Chirp Signal | f0={p['start_frequency']} Hz | "
            f"f1={p['end_frequency']} Hz | T={p['duration']}s | A={p['amplitude']}"
        ),
    ),
    
}


def generate(
    signal_name: str, common: dict, extra: dict
) -> tuple[FloatArray, FloatArray, str]:
    """Build a signal by name.

    `common` must contain amplitude, duration, sample_rate.
    `extra` must contain the signal-specific params listed in
    SIGNAL_SPECS[signal_name].params.

    Returns (t, x, title).
    """
    if signal_name not in SIGNAL_SPECS:
        raise KeyError(f"Unknown signal type: {signal_name!r}")

    spec = SIGNAL_SPECS[signal_name]
    params = {**common, **extra}

    if spec.uses_sinc_time:
        t = sg.create_sinc_time_vector(common["duration"], common["sample_rate"])
    else:
        t = sg.create_time_vector(common["duration"], common["sample_rate"])

    x = spec.build(t, params)
    title = spec.title(params)
    return t, x, title


if __name__ == "__main__":
    print("Testing signal_specs.py...\n")

    common = {"amplitude": 1.0, "duration": 1.0, "sample_rate": 1000}

    extras = {
        "Sine": {"frequency": 5, "phase": 0},
        "Cosine": {"frequency": 5, "phase": 0},
        "Square": {"frequency": 5, "duty_cycle": 50},
        "Triangle": {"frequency": 5},
        "Sinc": {"frequency": 5},
        "Chirp": {"start_frequency": 5, "end_frequency": 50},
        "Noise": {"seed": 42},
    }

    for name in SIGNAL_SPECS:
        t, x, title = generate(name, common, extras[name])
        assert len(t) == len(x), f"{name}: t/x length mismatch"
        print(f"{name:9}: {len(x)} samples | {title}")

    print("\nAll signal_specs checks passed!")