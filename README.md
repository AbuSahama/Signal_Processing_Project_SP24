**Signal Processing Project (SP24)**

A Python-based application for generating, visualizing, playing, recording, and analyzing audio signals — built as an interactive web app with Streamlit.


## Objective

This project aims to develop an interactive Python-based application for generating, visualizing, and analyzing audio signals. It provides tools for signal generation, FFT-based frequency-domain analysis, STFT-based time-frequency analysis, and digital filtering, allowing users to observe and compare signals in both the time and frequency domains.

## Features

### Signal Generator

- Sine Wave
- Cosine Wave
- Square Wave
- Triangle Wave
- Sinc Signal
- Chirp Signal

### Signal Controls

- Adjustable Frequency
- Adjustable Amplitude
- Adjustable Sampling Rate
- Adjustable Duration
- Phase Shift (Sine / Cosine)
- Duty Cycle (Square Wave)

### Audio

- Play Signal (in-browser)
- Upload an existing `.wav` file to analyze instead of generating one
- Record audio from the browser microphone
- Export original and filtered signals as `.wav` or `.csv`

### Filtering

- Butterworth Low-Pass, High-Pass, and Band-Pass filters
- Adjustable cutoff frequency/frequencies and filter order

### Signal Analysis

- Time Domain Waveform (original vs. filtered)
- FFT Spectrum (single-sided, with optional dB scale; two-sided view for Sinc)
- STFT Spectrogram (side-by-side original/filtered comparison when a filter is active)


## Technologies

- Python
- NumPy
- SciPy
- Streamlit
- Plotly
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


## 6. Run the App

Streamlit apps are launched with the `streamlit` command, not `python`, since Streamlit runs its own local web server and opens the app in your browser.

```bash
streamlit run app.py
```

This starts a local server (by default at `http://localhost:8501`) and opens the app automatically.


## Project Structure

```text
Signal_Processing_Project_SP24/
│
├── app.py                    # Entry point of the application (Streamlit UI)
├── core/
│   ├── __init__.py
│   ├── signal_generator.py    # Signal generation algorithms
│   ├── signal_specs.py        # Waveform parameter definitions
│   ├── analyzer.py            # FFT and STFT analysis
│   ├── filters.py             # Butterworth low/high/band-pass filters
│   └── audio.py               # System-audio playback, recording, WAV I/O    
│
├── generated/
│   └── audio/                  # Generated/saved WAV files
├── assets/
│   └── logo.png                # Aplication logo for browser tab
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignored files
```
