# Signal Generator and Analyzer

**Signal Processing Project (SP24)**

A Python-based application for generating and analyzing audio signals.


## Objective

The objective of this project is to develop an interactive application that generates and analyzes different audio signals. The application supports waveform generation, audio playback, recording, WAV file handling, FFT analysis, and STFT spectrogram visualization.


## Features

### Signal Generator

- Generate Sine Wave
- Generate Square Wave
- Generate Triangle Wave
- Generate Chirp Signal
- Generate Sinc Signal

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


## Technologies Used

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

---

## 2. Navigate to the Project Directory

```bash
cd Signal_Processing_Project_SP24
```

---

## 3. Create a Virtual Environment

A virtual environment creates an isolated Python environment for this project so that project-specific libraries do not affect other Python projects.

```bash
python3 -m venv .venv
```

---

## 4. Activate the Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows 

```bash
.venv\Scripts\activate
```

---

## 5. Install Required Libraries

Install all required dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## 6. Verify Installation (Optional)

```bash
python --version
pip list
```

---

# Running the Project

Run the application using:

```bash
python main.py
```

---

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

---

# File Description

### main.py

Starts the application and initializes all modules.

---

### gui.py

Contains the graphical user interface.

Responsibilities:

- User input
- Buttons
- Navigation
- Display graphs
- Connect all modules

---

### signal_generator.py

Responsible for generating different waveforms.

Functions include:

- Sine Wave
- Square Wave
- Triangle Wave
- Chirp Signal
- Sinc Signal

---

### analyzer.py

Performs signal analysis.

Includes:

- Fast Fourier Transform (FFT)
- Short-Time Fourier Transform (STFT)

---

### audio.py

Handles all audio-related operations.

Includes:

- Audio Playback
- Stop Playback
- Audio Recording
- Save WAV
- Load WAV

---

### utils.py

Contains reusable helper functions used throughout the project.

Examples:

- Signal validation
- Normalization
- Utility calculations

---

### generated/

Stores all generated output files such as:

- WAV files
- FFT images
- STFT images

---

### requirements.txt

Lists all Python packages required to run the project.

Install them using:

```bash
pip install -r requirements.txt
```

---

### README.md

Provides project documentation, setup instructions, usage guide, and team information.

---

### .gitignore

Prevents unnecessary files from being uploaded to GitHub.

Examples:

- `.venv/`
- `__pycache__/`
- `.DS_Store`
- `*.pyc`

## Project Status

🚧 Under Development

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
