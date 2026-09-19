"""
main.py

Command-line front end for the signal generator/analyzer.

Supports:
- Generating signals
- Loading WAV files
- Recording microphone audio
- Time-domain plotting
- Low-pass, high-pass, and band-pass filtering
- FFT analysis
- STFT analysis
- Audio playback
- Saving signals as WAV files
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from analyzer import plot_fft, plot_stft

from filters import (
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    plot_filter_comparison,
)

from audio import (
    load_audio,
    play_audio,
    record_audio,
    save_audio,
    stop_audio,
)

from signal_specs import SIGNAL_SPECS, generate

from utils import (
    get_choice,
    get_float_input,
    get_int_input,
    validate_filename,
)


SIGNAL_NAMES = list(SIGNAL_SPECS.keys())


def choose_signal() -> str:
    """Prompt the user to pick a waveform by number."""

    print()
    print("Choose a signal:")

    for i, name in enumerate(SIGNAL_NAMES, start=1):
        print(f"{i}. {name}")

    print()

    choices = [str(i) for i in range(1, len(SIGNAL_NAMES) + 1)]

    picked = get_choice(
        "Enter your choice: ",
        choices,
    )

    return SIGNAL_NAMES[int(picked) - 1]


def generate_selected_signal(signal_name: str):
    """Collect signal parameters and generate the selected signal.

    Returns:
        t, signal_data, sample_rate, title
    """

    amplitude = get_float_input(
        "Enter amplitude: ",
        minimum=0,
    )

    duration = get_float_input(
        "Enter duration (seconds): ",
        minimum=0.001,
    )

    sample_rate = get_int_input(
        "Enter sampling rate (Hz): ",
        minimum=1,
    )

    common = {
        "amplitude": amplitude,
        "duration": duration,
        "sample_rate": sample_rate,
    }

    spec = SIGNAL_SPECS[signal_name]

    extra = {}

    for param in spec.params:

        extra[param.key] = get_float_input(
            f"Enter {param.label.lower()}: ",
            minimum=param.minimum,
            maximum=param.maximum,
        )

    t, signal_data, title = generate(
        signal_name,
        common,
        extra,
    )

    return t, signal_data, sample_rate, title


def load_wav_signal():
    """Load a WAV file and create its corresponding time vector.

    Returns:
        t, signal_data, sample_rate, title
    """

    filename = input(
        "Enter WAV filename: "
    ).strip()

    try:
        sample_rate, signal_data = load_audio(
            filename
        )

    except (OSError, ValueError) as exc:

        print(
            f"Could not load WAV file: {exc}"
        )

        return None

    t = np.arange(
        len(signal_data)
    ) / sample_rate

    title = f"Loaded Audio | {filename}"

    return (
        t,
        signal_data,
        sample_rate,
        title,
    )


def record_microphone_signal():
    """Record mono audio from the default microphone.

    Returns:
        t, signal_data, sample_rate, title
    """

    duration = get_float_input(
        "Enter recording duration (seconds): ",
        minimum=0.001,
    )

    sample_rate = get_int_input(
        "Enter sampling rate (Hz): ",
        minimum=1,
    )

    try:

        signal_data = record_audio(
            duration,
            sample_rate,
        )

    except RuntimeError as exc:

        print(exc)

        return None

    t = np.arange(
        len(signal_data)
    ) / sample_rate

    title = (
        f"Recorded Audio | "
        f"Duration={duration}s | "
        f"Sample Rate={sample_rate} Hz"
    )

    return (
        t,
        signal_data,
        sample_rate,
        title,
    )


def plot_signal(
    t,
    signal_data,
    title: str,
) -> None:
    """Plot a signal in the time domain."""

    plt.figure()

    plt.plot(
        t,
        signal_data,
    )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.show()

def filter_signal(signal_data,    sample_rate: int):
    """Ask the user to select a filter and return the filtered signal."""

    print()
    print("Filter Options:")
    print("1. Low-Pass Filter")
    print("2. High-Pass Filter")
    print("3. Band-Pass Filter")
    print("4. Skip Filter")

    filter_choice = get_choice(
        "Enter your choice: ",
        ["1", "2", "3", "4"],
    )

    # --------------------------------
    # No filter
    # --------------------------------

    if filter_choice == "4":

        print("No filter applied.")

        return signal_data

    nyquist = sample_rate / 2

    # --------------------------------
    # Low-pass filter
    # --------------------------------

    if filter_choice == "1":

        cutoff_freq = get_float_input(
            f"Enter cutoff frequency (Hz) "
            f"[0 < f < {nyquist}]: ",
            minimum=0,
            maximum=nyquist,
        )

        if cutoff_freq >= nyquist:
            print(
                "Cutoff frequency must be less than "
                f"Nyquist frequency ({nyquist} Hz)."
            )

            return signal_data

        filtered_signal = low_pass_filter(
            signal_data,
            sample_rate,
            cutoff_freq,
        )

        print(
            f"Low-pass filter applied "
            f"(cutoff = {cutoff_freq} Hz)."
        )

        plot_filter_comparison(
            signal_data,
            filtered_signal,
            sample_rate,
            (
                f"Low-Pass Filter | "
                f"Cutoff = {cutoff_freq} Hz"
            ),
        )

        return filtered_signal

    # --------------------------------
    # High-pass filter
    # --------------------------------

    if filter_choice == "2":

        cutoff_freq = get_float_input(
            f"Enter cutoff frequency (Hz) "
            f"[0 < f < {nyquist}]: ",
            minimum=0,
            maximum=nyquist,
        )

        if cutoff_freq >= nyquist:
            print(
                "Cutoff frequency must be less than "
                f"Nyquist frequency ({nyquist} Hz)."
            )

            return signal_data

        filtered_signal = high_pass_filter(
            signal_data,
            sample_rate,
            cutoff_freq,
        )

        print(
            f"High-pass filter applied "
            f"(cutoff = {cutoff_freq} Hz)."
        )

        plot_filter_comparison(
            signal_data,
            filtered_signal,
            sample_rate,
            (
                f"High-Pass Filter | "
                f"Cutoff = {cutoff_freq} Hz"
            ),
        )

        return filtered_signal

    # --------------------------------
    # Band-pass filter
    # --------------------------------

    if filter_choice == "3":

        print()
        print(
            f"Nyquist frequency = {nyquist:.2f} Hz"
        )

        low_cutoff = get_float_input(
            f"Enter low cutoff frequency (Hz) "
            f"[0 < f < {nyquist}]: ",
            minimum=0,
            maximum=nyquist,
        )

        high_cutoff = get_float_input(
            f"Enter high cutoff frequency (Hz) "
            f"[{low_cutoff} < f < {nyquist}]: ",
            minimum=0,
            maximum=nyquist,
        )

        # Validate low cutoff
        if low_cutoff <= 0:

            print(
                "Low cutoff frequency must be greater "
                "than 0 Hz."
            )

            return signal_data

        # Validate high cutoff
        if high_cutoff >= nyquist:

            print(
                "High cutoff frequency must be less than "
                f"Nyquist frequency ({nyquist} Hz)."
            )

            return signal_data

        # Validate cutoff ordering
        if high_cutoff <= low_cutoff:

            print(
                "High cutoff frequency must be greater "
                "than low cutoff frequency."
            )

            return signal_data

        filtered_signal = band_pass_filter(
            signal_data,
            sample_rate,
            low_cutoff,
            high_cutoff,
        )

        print()
        print("Band-pass filter applied.")
        print(
            f"Lower cutoff : {low_cutoff:.2f} Hz"
        )
        print(
            f"Upper cutoff : {high_cutoff:.2f} Hz"
        )

        plot_filter_comparison(
            signal_data,
            filtered_signal,
            sample_rate,
            (
                f"Band-Pass Filter | "
                f"Low = {low_cutoff} Hz | "
                f"High = {high_cutoff} Hz"
            ),
        )

        return filtered_signal

    return signal_data

def analyze_signal(
    signal_data,
    sample_rate: int,
    filtered_signal=None,
) -> None:
    """Display the analysis menu and analyze the selected signal."""

    # Decide which signal to analyze
    if filtered_signal is not None:
        analyze_filtered = get_choice(
            "Do you want to analyze the filtered signal? (y/n): ",
            ["y", "n"],
        )

        if analyze_filtered == "y":
            signal_to_analyze = filtered_signal
        else:
            signal_to_analyze = signal_data

    else:
        signal_to_analyze = signal_data

    print()
    print("Analyzer Options:")
    print("1. FFT")
    print("2. STFT")
    print("3. Skip analyzer")

    analysis_choice = get_choice(
        "Enter your choice: ",
        ["1", "2", "3"],
    )

    if analysis_choice == "1":
        plot_fft(
            signal_to_analyze,
            sample_rate,
        )

    elif analysis_choice == "2":
        plot_stft(
            signal_to_analyze,
            sample_rate,
        )



def save_signal(
    signal_data,
    sample_rate: int,
) -> None:
    """Ask whether the user wants to save the current signal."""

    save = input(
        "Do you want to save the signal? (y/n): "
    ).strip().lower()

    if save != "y":
        return

    filename = input(
        "Enter filename: "
    )

    filename = validate_filename(
        filename
    )

    if filename is not None:

        save_audio(
            filename,
            signal_data,
            sample_rate,
        )


def play_signal(
    signal_data,
    sample_rate: int,
) -> None:
    """Ask whether the user wants to play the current signal."""

    play = input(
        "Do you want to play the signal? (y/n): "
    ).strip().lower()

    if play != "y":
        return

    play_audio(
        signal_data,
        sample_rate,
    )

    input(
        "Press Enter to stop playback..."
    )

    stop_audio()


def process_signal(
    t,
    signal_data,
    sample_rate: int,
    title: str,
) -> None:
    """Process a generated, loaded, or recorded signal."""

    print()
    print("Signal ready!")
    print(
        f"Number of samples : "
        f"{len(signal_data)}"
    )
    print(
        f"Sampling rate     : "
        f"{sample_rate} Hz"
    )
    print()

    # --------------------------------
    # 1. Original signal
    # --------------------------------

    plot_signal(
        t,
        signal_data,
        title,
    )

    # --------------------------------
    # 2. Filtering
    # --------------------------------

    filtered_signal = filter_signal(
        signal_data,
        sample_rate,
    )

    # --------------------------------
    # 3. FFT / STFT
    # --------------------------------

    analyze_signal(
        filtered_signal,
        sample_rate,
    )

    # --------------------------------
    # 4. Save
    # --------------------------------

    save_signal(
        filtered_signal,
        sample_rate,
    )

    # --------------------------------
    # 5. Playback
    # --------------------------------

    play_signal(
        filtered_signal,
        sample_rate,
    )


def main() -> None:
    """Run the command-line application."""

    print("=" * 50)
    print("Signal Generator and Analyzer")
    print("=" * 50)

    print()
    print("Select signal source:")
    print("1. Generate Signal")
    print("2. Load WAV File")
    print("3. Record Audio")
    print("4. Exit")
    print()

    source_choice = get_choice(
        "Enter your choice: ",
        ["1", "2", "3", "4"],
    )

    if source_choice == "4":

        print("Program finished.")

        return

    result = None

    if source_choice == "1":

        signal_name = choose_signal()

        result = generate_selected_signal(
            signal_name
        )

    elif source_choice == "2":

        result = load_wav_signal()

    elif source_choice == "3":

        result = record_microphone_signal()

    if result is None:

        print(
            "\nNo signal was processed."
        )

        return

    (
        t,
        signal_data,
        sample_rate,
        title,
    ) = result

    process_signal(
        t,
        signal_data,
        sample_rate,
        title,
    )

    print()
    print("Program finished.")


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nInterrupted. Exiting."
        )

        sys.exit(0)