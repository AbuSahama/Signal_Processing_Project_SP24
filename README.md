# Signal Generator and Analyzer

**Signal Processing Project (SP24)**

A Python-based application for generating, visualizing, playing, recording, and analyzing audio signals.


## Objective

This project aims to build an interactive application for generating and analyzing audio signals using Python.


## Features

### Signal Generator

- Sine Wave
- Square Wave
- Triangle Wave
- Chirp Signal
- Sinc Signal

### Signal Controls

- Adjustable Frequency
- Adjustable Amplitude
- Adjustable Sampling Rate
- Adjustable Duration
- Duty Cycle (Square Wave)

### Audio

- Play Signal
- Stop Playback
- Save WAV File
- Load WAV File
- Record Audio

### Signal Analysis

- Time Domain Waveform
- FFT Spectrum
- STFT Spectrogram


## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- PyQt6
- SoundDevice
- SoundFile
- Git
- GitHub
  

# Project Setup

## 1. Clone the Repository

```bash
git clone https://github.com/AbuSahama/Signal_Processing_Project_SP24.git
```


## 2. Navigate to the Project Directory

```bash
cd Signal_Processing_Project_SP24
```


## 3. Create a Virtual Environment

A virtual environment creates an isolated Python environment for this project so that project-specific libraries do not affect other Python projects.

```bash
python3 -m venv .venv
```


## 4. Activate the Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows 

```bash
.venv\Scripts\activate
```


## 5. Install Required Libraries

Install all required dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```



# Running the Project

Run the application using:

```bash
python main.py
```


# Project Structure

```
Signal_Processing_Project_SP24/
│
├── main.py                  # Entry point of the application
├── gui.py                   # Graphical User Interface
├── signal_generator.py      # Signal generation algorithms
├── analyzer.py              # FFT and STFT analysis
├── audio.py                 # Audio playback, recording, WAV handling
├── utils.py                 # Helper functions
│
├── generated/               # Generated WAV files and exported graphs
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Files ignored by Git
```



## Project Status


Current Progress:

- [x] Repository Created
- [x] Project Structure Designed
- [ ] Signal Generator
- [ ] Audio Playback
- [ ] WAV Handling
- [ ] FFT Analysis
- [ ] STFT Analysis
- [ ] GUI Development
- [ ] Testing
