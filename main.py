import numpy as np
import matplotlib.pyplot as plt
from audio import save_audio
import sys


from signal_generator import (
    generate_sine,
    generate_square,
    generate_triangular,
    generate_sinc,
    generate_chirp,
    generate_cosine
)

choice = input(
    "Choose a signal:\n"
    "1. Sine\n"
    "2. Cosine\n"
    "3. Square\n"
    "4. Triangular\n"
    "5. Sinc\n"
    "6. Chirp\n"
    "Enter your choice: "
).lower()

''' Currently, the program directly converts user input using float() and int().
    If the user enters an invalid value (e.g., "abc" instead of a number),
    Python raises a ValueError and the application crashes.
    Before the final testing and submission we should  wrap all user inputs inside try/except
    blocks (or create reusable helper functions) to catch invalid inputs,
    display a meaningful error message, and ask the user to enter the value
    again instead of terminating the program.'''

amp = float(input("Amplitude: "))
duration = float(input("Duration (s): "))
sample_rate = int(input("Sampling Rate (Hz): "))

# Universal Time Vector
t = np.arange(0, duration, 1 / sample_rate)

if choice in ["1", "sine"]:
    freq = float(input("Frequency (Hz): "))
    phase_ = float(input("Phase Shift (degrees): "))
    x = generate_sine(t, freq, amp, phase_)
    title = f"Sine Wave | f={freq} Hz | A={amp} | P={phase_}°"

elif choice in ["2", "cosine"]:
    freq = float(input("Frequency (Hz): "))
    phase_ = float(input("Phase Shift (degrees): "))
    x = generate_cosine(t, freq, amp, phase_)
    title = f"Cosine Wave | f={freq} Hz | A={amp} | P={phase_}°"

elif choice in ["3", "square"]:
    freq = float(input("Frequency (Hz): "))
    duty_cycle = float(input("Duty Cycle (%): "))
    x = generate_square(t, freq, amp, duty_cycle)
    title = f"Square Wave | f={freq} Hz | A={amp} | Duty={duty_cycle}%"

elif choice in ["4", "triangular"]:
    freq = float(input("Frequency (Hz): "))
    x = generate_triangular(t, freq, amp)
    title = f"Triangle Wave | f={freq} Hz | A={amp}"

elif choice in ["5", "sinc"]:
    t = np.arange(-duration/2, duration/2, 1/sample_rate)
    freq = float(input("Frequency (Hz): "))
    x = generate_sinc(t, freq, amp)
    title = f"Sinc Wave | f={freq} Hz | A={amp}"

elif choice in ["6", "chirp"]:
    freq = float(input("Initial Frequency (Hz): "))
    f1 = float(input("Final Frequency (Hz): "))
    t1=duration
    x = generate_chirp(t, freq, t1, f1, amp)
    title = (
    f"Chirp Wave | f₀={freq} Hz | "
    f"f₁={f1} Hz | T={t1}s | A={amp}"
    )


else:
    print("Invalid choice!")
    sys.exit()


              

plt.plot(t,x)
plt.title(title)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

save = input("Save as WAV? (y/n): ").strip().lower()

if save == "y":
    filename = input("Enter filename: ").strip()

    if not filename.endswith(".wav"):
        filename += ".wav"

    save_audio(filename, x, sample_rate)

else:
     print("Audio not saved.")