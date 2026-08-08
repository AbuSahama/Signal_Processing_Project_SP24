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
    signal_name = "Sine Wave"
    freq = float(input("Frequency (Hz): "))
    phase_ = float(input("Phase Shift (degrees): "))
    x = generate_sine(t, freq, amp, phase_)
    title = f"{signal_name} | f={freq} Hz | A={amp} | Phase={phase_}°"

elif choice in ["2", "cosine"]:
    signal_name = "Cosine Wave"
    freq = float(input("Frequency (Hz): "))
    phase_ = float(input("Phase Shift (degrees): "))
    x = generate_cosine(t, freq, amp, phase_)
    title = f"{signal_name} | f={freq} Hz | A={amp} | Phase={phase_}°"
    
    
elif choice in ["3", "square"]:
    signal_name = "Square Wave"
    freq = float(input("Frequency (Hz): "))
    duty_cycle = float(input("Duty Cycle (%): "))
    x = generate_square(t, freq, amp, duty_cycle)
    title = f"{signal_name} | f={freq} Hz | A={amp} | Duty={duty_cycle}%"
    
elif choice in ["4", "triangular"]:
    signal_name = "Triangular Wave"
    freq = float(input("Frequency (Hz): "))
    x = generate_triangular(t, freq, amp)
    title = f"{signal_name} | f={freq} Hz | A={amp}"
        
    

elif choice in ["5", "sinc"]:
    signal_name = "Sinc Wave"
    t = np.arange(-duration/2, duration/2, 1/sample_rate)
    freq = float(input("Frequency (Hz): "))
    x = generate_sinc(t, freq, amp)
    title = f"{signal_name} | f={freq} Hz | A={amp}"
        
    
elif choice in ["6", "chirp"]:
    signal_name = "Chirp Wave"
    freq = float(input("Initial Frequency (Hz): "))
    f1 = float(input("Final Frequency (Hz): "))
    t1=duration
    x = generate_chirp(t, freq, t1, f1, amp)
    title = f"{signal_name} | f0={freq} Hz | f1={f1} Hz | A={amp}"
    
else:
    print("Invalid choice!")
    sys.exit()


plt.plot(t,x, color="#054D21")
plt.title(title, fontsize=15,
                family="Arial",
                color="#670255")

plt.xlabel("Time (s)", fontsize=10,
                        family="Arial",
                        fontweight="bold",
                        color="#47A18C")
plt.ylabel("Amplitude", fontsize=10,
                        family="Arial",
                        fontweight="bold",
                      color="#438C7A")
plt.tick_params(axis="both",
                colors="#1D2523")
plt.grid(axis="y",
         linewidth=2,
         color="lightgray",
         linestyle="dashed")
plt.show()

save = input("Save as WAV? (y/n): ").strip().lower()

if save == "y":
    filename = signal_name.lower().replace(" ", "_")

    if not filename.endswith(".wav"):
        filename += ".wav"

    save_audio(filename, x, sample_rate)

else:
     print("Audio not saved.")