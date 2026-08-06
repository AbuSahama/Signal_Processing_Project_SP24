import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Function to generate Sine signal 
# def generate_sine(freq, amp, duration, sample_rate):
#     t=np.arange(0, duration, 1/sample_rate)
#     x=amp*np.sin(2*np.pi*freq*t)
#     return t, x
def generate_sine(t, freq, amp, phase_=0):
    phase_ = np.deg2rad(phase_)
    x = amp * np.sin(2 * np.pi * freq * t + phase_)
    return x

def generate_cosine(t, freq, amp, phase_=0):
    phase_ = np.deg2rad(phase_)
    x = amp * np.cos(2 * np.pi * freq * t + phase_)
    return x


# Function to generate Square signal 
def generate_square(t, freq, amp, duty_cycle):
    x=amp*signal.square(2*np.pi*freq*t, duty=duty_cycle/100)
    return x

def generate_triangular(t, freq, amp):
    x=amp*signal.sawtooth(2*np.pi*freq*t, width=0.5)
    return x

def generate_sinc(t, freq, amp):
    x=amp*np.sinc(t * freq)
    return x

def generate_chirp(t, freq, t1, f1, amp):
    x=amp*signal.chirp(t, freq, t1, f1, method='linear')
    return x

    

