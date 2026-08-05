import numpy as py
import matplotlib.pyplot as plt
from scipy import signal

from signal_generator import generate_sine, generate_square


choice = input(
    "Choose a signal:\n"
    "1. Sine\n"
    "2. Square\n"
    "Enter your choice: "
)

freq = float(input("Frequency (Hz): "))
amp = float(input("Amplitude: "))
duration = float(input("Duration (s): "))
sample_rate = int(input("Sampling Rate (Hz): "))

if choice == "1":
    t, x = generate_sine(freq, amp, duration, sample_rate)

elif choice == "2":
    duty_cycle = float(input("Duty Cycle (%): "))
    t, x = generate_square(freq, amp, duration, sample_rate, duty_cycle)
else:
    exit()

# titles = {
#     "1": "Sine Wave",
#     "2": "Square Wave"
# }

titles = {
    "1": f"Sine Wave | f={freq} Hz | A={amp}",
    "2": f"Square Wave | f={freq} Hz | A={amp} | Duty={duty_cycle}%"
}





# for testing 
# if __name__=="__main__":
#     t, x = generate_sine(freq=float(input("Enter the frequency: ")),
#                          amp=float(input("Enter the amplitude of the sine wave: ")),
#                          duration=float(input("Enter the duration (in senconds): ")),
#                          sample_rate=int(input("Enter the sampling rate (in Hz): "))
#                          )

# if __name__=="__main__":
#     t, x = generate_square(freq=float(input("Enter the frequency: ")),
#                          amp=float(input("Enter the amplitude of the sine wave: ")),
#                          duration=float(input("Enter the duration (in senconds): ")),
#                          sample_rate=int(input("Enter the sampling rate (in Hz): ")),
#                          duty_cycle=float(input("Enter the duty cycle in percentage: "))
#                          )
    

plt.plot(t,x)
#plt.plot("Sine Wave")
plt.title(titles[choice])

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()