**Signal Processing Project (SP24)**

A application for generating, visualizing, playing, recording, and analyzing audio signals — built as an interactive web app with Streamlit.

## Objective

This project aims to develop an interactive application for generating, visualizing, and analyzing audio signals. It provides tools for signal generation, FFT-based frequency-domain analysis, STFT-based time-frequency analysis, noise addition, and digital filtering. The application enables users to observe the effects of noise and filtering, compare signals in both the time and frequency domains, listen to the original and filtered audio signals, and export the processed signals in WAV or CSV format.

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

- Play signals in-browser
- Upload an existing `.wav` file to analyze instead of generating one
- Record audio from the browser microphone (via Streamlit's built-in `st.audio_input` — no system audio drivers required, works on any cloud host)
- Export original and filtered signals as `.wav` or `.csv`

### Filtering

- Butterworth Low-Pass, High-Pass, and Band-Pass filters (zero-phase, via SOS/`sosfiltfilt` for numerical stability at higher orders)
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
- Matplotlib *(used mainly for visualization in the CLI application through main.py)*
- Git / GitHub


## Project Structure

```text
Signal_Processing_Project_SP24/
│
├── app.py                      # Entry point of the application (Streamlit UI)
├── core/
│   ├── __init__.py
│   ├── signal_generator.py     # Signal generation algorithms
│   ├── signal_specs.py         # Waveform parameter definitions
│   ├── analyzer.py             # FFT and STFT analysis
│   ├── filters.py              # Butterworth low/high/band-pass filters
│   └── audio.py                # Optional local system-audio helpers (not used by the web app)
│
├── assets/
│   └── logo.png                 # Application logo / browser tab favicon
│
├── requirements.txt             # Python dependencies for the deployed app
├── README.md                    # Project documentation
└── .gitignore                   # Git ignored files
```

`generated/` (saved local WAV files from `core/audio.py`) is created on demand and git-ignored — it isn't part of the repo.

---

## Running Locally


### Check Python Version
Make sure Python 3.12 is installed and being used before setting up the project.

Check your Python version:

python3 --version

It should show:

Python 3.12.x

If python3 points to another version, use the Python 3.12 executable explicitly when creating the virtual environment.

### 1. Clone the Repository

```bash
git clone https://github.com/AbuSahama/Signal_Processing_Project_SP24.git
cd Signal_Processing_Project_SP24
```

### 2. Create a Virtual Environment

A virtual environment keeps this project's dependencies isolated from other Python projects on your machine.

```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the App

Streamlit apps are launched with the `streamlit` command, not `python`, since Streamlit runs its own local web server.

```bash
streamlit run app.py
```

This starts a local server (by default at `http://localhost:8501`) and opens the app automatically.
