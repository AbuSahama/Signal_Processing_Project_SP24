import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Function to generate Sine signal 
def generate_sine(freq, amp, duration, sample_rate):
    t=np.arange(0, duration, 1/sample_rate)
    x=amp*np.sin(2*np.pi*freq*t)
    return t, x

# Function to generate Square signal 
def generate_square(freq, amp, duration, sample_rate, duty_cycle):
    t=np.arange(0, duration, 1/sample_rate)
    x=amp*signal.square(2*np.pi*freq*t, duty=duty_cycle/100)
    return t, x

