"""
app.py

SignalLab (signal generator and analyzer) — a Streamlit front end for the signal-processing toolkit
(signal_generator.py / signal_specs.py / filters.py / analyzer.py).

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
from scipy import signal as sp_signal

from core.analyzer import (
    calculate_fft,
    calculate_fft_two_sided,
    calculate_stft,
)
from core.signal_generator import generate_noise

from core.filters import (
    band_pass_filter,
    high_pass_filter,
    low_pass_filter,
)

from core.signal_specs import SIGNAL_SPECS, generate

# --------------------------------------------------------------------------
# Page setup + theme
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Signal Generator and Analyzer",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Original dark palette.
ACCENT = "#7C9BFF"
ACCENT_SOFT = "rgba(124, 155, 255, 0.15)"
COMPARE = "#FE0000"          # "filtered" trace
ORIGINAL_TRACE = "#0EFF01"   # "original" trace 
BG = "#0E1117"
CARD_BG = "#161B22"
BORDER = "#262C36"
TEXT = "#F2F4F8"
SUBTEXT = "#9AA4B2"


st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {BG}; }}

        h1, h2, h3 {{
            color: {TEXT} !important;
            font-weight: 800 !important;
        }}

        /* Header block */
        .app-header {{
            text-align: center;
            padding: 0.5rem 0 1.5rem 0;
        }}
        .app-header .app-title {{
            font-size: 2.5rem;
            font-weight: 800;
            color: {TEXT};
            letter-spacing: -0.02em;
        }}
        .app-header .app-subtitle {{
            color: {SUBTEXT};
            font-size: 1.05rem;
            margin-top: 0.4rem;
        }}
        .app-header .app-badge {{
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            background-color: {ACCENT_SOFT};
            color: {ACCENT};
            font-size: 0.85rem;
            font-weight: 600;
        }}

        /* Stat tiles */
        .plain-stat-label {{
            color: {SUBTEXT};
            font-size: 0.9rem;
            margin-bottom: 0.15rem;
        }}
        .plain-stat-value {{
            color: {TEXT};
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.2;
        }}

        .comparison-label {{
            color: {SUBTEXT};
            font-size: 0.85rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: -0.6rem;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {CARD_BG};
            border-right: 1px solid {BORDER};
        }}
        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.1rem;
        }}
        .sidebar-logo .logo-mark {{
            font-size: 1.35rem;
            color: {ACCENT};
        }}
        .sidebar-logo .logo-text {{
            font-size: 1.2rem;
            font-weight: 800;
            color: {TEXT};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {SUBTEXT};
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            color: {ACCENT} !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {ACCENT} !important;
        }}

        /* Card-style containers (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 14px !important;
            border-color: {BORDER} !important;
            background-color: {CARD_BG};
        }}
        div[data-testid="stExpander"] {{
            border: 1px solid {BORDER};
            border-radius: 14px;
            background-color: {CARD_BG};
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 10px;
            border: 1px solid {BORDER};
            background-color: {CARD_BG};
            color: {TEXT};
            font-weight: 600;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: {ACCENT};
            color: {ACCENT};
        }}

        /* Uploader / audio-input dropzone */
        section[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploaderDropzone"] {{
            border: 1.5px dashed {BORDER} !important;
            border-radius: 14px !important;
        }}

        hr {{
            border-color: {BORDER};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", y=1.08, x=0),
)

# Always show the modebar (camera/zoom/pan/reset/fullscreen), not just on hover.
PLOT_CONFIG = dict(displayModeBar=True, displaylogo=False)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def to_wav_bytes(sig: np.ndarray, sample_rate: float) -> bytes:
    """Normalize to int16 and encode as an in-memory WAV file."""
    peak = np.max(np.abs(sig))
    normalized = sig / peak if peak > 0 else sig
    int16_data = np.int16(normalized * 32767)
    buf = io.BytesIO()
    wav_write(buf, int(sample_rate), int16_data)
    return buf.getvalue()


def to_csv_bytes(t: np.ndarray, sig: np.ndarray) -> bytes:
    buf = io.StringIO()
    buf.write("time_s,amplitude\n")
    for ti, xi in zip(t, sig):
        buf.write(f"{ti:.8f},{xi:.8f}\n")
    return buf.getvalue().encode("utf-8")


def read_uploaded_wav(uploaded_file) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Read an uploaded WAV file into (t, x, sample_rate, title).

    Stereo files are downmixed to mono (averaged) since the rest of the
    toolkit works on 1D signals. Amplitude is peak-normalized to [-1, 1],
    matching audio.load_audio's convention, but reimplemented here so this
    file doesn't need to import audio.py (which imports sounddevice at
    module level — a dependency this app doesn't otherwise need, and one
    that can fail to import at all on a server with no audio hardware).
    """
    sample_rate, raw = wav_read(uploaded_file)
    raw = np.asarray(raw)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    raw = raw.astype(np.float64)

    peak = np.max(np.abs(raw))
    x = raw / peak if peak > 0 else raw

    t = np.arange(len(x)) / sample_rate
    duration = len(x) / sample_rate
    title = f"Uploaded: {uploaded_file.name} | {sample_rate:,.0f} Hz | {duration:.3f}s"
    return t, x, float(sample_rate), title


FILTER_LABELS = {
    "None": None,
    "Low-Pass": "low",
    "High-Pass": "high",
    "Band-Pass": "band",
}


def apply_filter(sig: np.ndarray, sample_rate: float, kind: str, params: dict) -> np.ndarray:
    if kind == "low":
        return low_pass_filter(sig, sample_rate, params["cutoff"], params["order"])
    if kind == "high":
        return high_pass_filter(sig, sample_rate, params["cutoff"], params["order"])
    if kind == "band":
        return band_pass_filter(
            sig, sample_rate, params["low_cutoff"], params["high_cutoff"], params["order"]
        )
    raise ValueError(f"Unknown filter kind: {kind}")


# --------------------------------------------------------------------------
# Sidebar — signal + filter controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <span class="logo-mark">〰️</span>
            <span class="logo-text">SignalLab</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Waveform generator · filters · spectral analysis")
    st.write("")

    with st.container(border=True):
        st.markdown("**Signal Source**")
        source = st.radio(
            "Source", ["Generate", "Upload audio (.wav)", "Record (microphone)"], horizontal=True
        )

        uploaded_t = uploaded_x = uploaded_title = None
        uploaded_sample_rate = 10000.0  # fallback so the Filter section below has a nyquist to work with

        if source == "Generate":
            signal_name = st.selectbox("Waveform type", list(SIGNAL_SPECS.keys()))
            spec = SIGNAL_SPECS[signal_name]

            col1, col2 = st.columns(2)
            with col1:
                amplitude = st.number_input("Amplitude", value=1.0, min_value=0.0, step=0.1)
                sample_rate = st.number_input(
                    "Sample rate (Hz)", value=10000.0, min_value=1.0, step=100.0
                )
            with col2:
                duration = st.number_input("Duration (s)", value=1.0, min_value=0.001, step=0.1)
                

            extra_params: dict = {}
            if spec.params:
                st.markdown("**Waveform parameters**")
                for p in spec.params:
                    extra_params[p.key] = st.number_input(
                        p.label,
                        value=float(p.default),
                        min_value=float(p.minimum) if p.minimum is not None else None,
                        max_value=float(p.maximum) if p.maximum is not None else None,
                        key=f"param_{p.key}",
                    )
            
            if signal_name == "Chirp":
                extra_params["duration"] = duration
        else:
            signal_name = None

            if source == "Upload audio (.wav)":
                audio_file = st.file_uploader("Choose Audio File", type=["wav"])
                empty_hint = "Drag & drop or click to select a .wav file to analyze."
            else:  # Record (microphone)
                # Browser-side recording via st.audio_input — no sounddevice/system
                # audio device involved, so this works the same locally and when
                # deployed (unlike audio.py's record_audio, which needs a real
                # input device on the machine running the server).
                audio_file = st.audio_input("Record audio")
                empty_hint = "Record a clip to analyze it instead of generating one."

            if audio_file is not None:
                try:
                    uploaded_t, uploaded_x, uploaded_sample_rate, uploaded_title = read_uploaded_wav(
                        audio_file
                    )
                except Exception as exc:
                    st.error(f"Could not read this audio: {exc}")
            else:
                st.caption(empty_hint)
            sample_rate = uploaded_sample_rate

    st.markdown("**Add noise**")
    add_noise = st.checkbox("Overlay noise on the generated signal")
    noise_amplitude = 0.0
    noise_seed = 42
    if add_noise:
      nc1, nc2 = st.columns(2)
      with nc1:
          noise_amplitude = st.number_input("Noise amplitude", value=0.1, min_value=0.0, step=0.05)
      with nc2:
          noise_seed = st.number_input("Noise seed", value=42, step=1)

    st.write("")

    with st.container(border=True):
        st.markdown("**🎛️ Filter**")
        filter_choice = st.selectbox("Type", list(FILTER_LABELS.keys()))
        filter_kind = FILTER_LABELS[filter_choice]
        filter_params: dict = {}

        nyquist = sample_rate / 2
        if filter_kind in ("low", "high"):
            # filter_params["cutoff"] = st.slider(
            #     "Cutoff frequency (Hz)", 1.0, float(max(nyquist - 1, 1.0)), min(50.0, nyquist / 2)

            # )
            cutoff_min = 1.0
            cutoff_max = float(max(nyquist - 1, 1.0))
            cutoff_default = min(50.0, nyquist / 2)

            st.session_state.setdefault("cutoff_val", cutoff_default)

            def _sync_cutoff_from_slider():
                st.session_state.cutoff_val = st.session_state.cutoff_slider

            def _sync_cutoff_from_number():
                st.session_state.cutoff_val = st.session_state.cutoff_number

            # cc1, cc2 = st.columns([3, 2])
            # with cc1:
            st.slider(
                "Cutoff frequency (Hz)", cutoff_min, cutoff_max,
                value=st.session_state.cutoff_val, step=1.0, key="cutoff_slider",
                on_change=_sync_cutoff_from_slider,
            )
            # with cc2:
            #     st.number_input(
            #         "Exact (Hz)", min_value=cutoff_min, max_value=cutoff_max,
            #         value=st.session_state.cutoff_val, step=1.0, key="cutoff_number",
            #         on_change=_sync_cutoff_from_number,
            #     )

            filter_params["cutoff"] = st.session_state.cutoff_val
            filter_params["order"] = st.slider("Filter order", 1, 10, 5)


        elif filter_kind == "band":
            lo, hi = st.slider(
                "Passband (Hz)",
                1.0,
                float(max(nyquist - 1, 2.0)),
                (min(20.0, nyquist / 4), min(100.0, nyquist / 2)),
            )
            filter_params["low_cutoff"] = lo
            filter_params["high_cutoff"] = hi
            filter_params["order"] = st.slider("Filter order", 1, 10, 5)
        else:
            st.caption("Select a filter type to reveal its controls.")

    st.write("")

    with st.container(border=True):
        st.markdown("**Display Options**")
        db_scale = st.checkbox("Show FFT magnitude in dB", value=False)


# --------------------------------------------------------------------------
# Generate (or load) + (optionally) filter the signal
# --------------------------------------------------------------------------

error = None
t = x = title = None
x_filtered = None

if source == "Generate":
    try:
        common = {"amplitude": amplitude, "duration": duration, "sample_rate": sample_rate}
        t, x, title = generate(signal_name, common, extra_params)
    except ValueError as exc:
        error = str(exc)
else:
    if uploaded_x is None:
        error = "Upload or record an audio clip in the sidebar, or switch the source back to Generate."
    else:
        t, x, sample_rate, title = uploaded_t, uploaded_x, uploaded_sample_rate, uploaded_title

# if add_noise and noise_amplitude > 0:
#     x = x + generate_noise(t, noise_amplitude, seed=int(noise_seed))
#     title += f" + Noise (A={noise_amplitude})"
# if error is None and add_noise and noise_amplitude>0:
#     x=x+generate_noise(t, noise_amplitude, seed=int(noise_seed)) 
#     title += f" + Noise (A={noise_amplitude})"

if error is None and x is not None and add_noise and noise_amplitude > 0:
    x = x + generate_noise(
        t,
        noise_amplitude,
        seed=int(noise_seed)
    )
    title += f" + Noise (A={noise_amplitude})"

if error is None and filter_kind is not None:
    try:
        x_filtered = apply_filter(x, sample_rate, filter_kind, filter_params)
    except ValueError as exc:
        st.warning(f"Filter not applied: {exc}")
        x_filtered = None
#st.image("assets/logo.png", use_container_width=True)
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">Signal Generator and Analyzer</div>
        <div class="app-subtitle">{title if title else "Configure a signal in the sidebar to get started"}</div>
        <div class="app-badge">〰️ Waveform · Filter · Spectral analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if error:
    st.error(error)
    st.stop()

active_signal = x_filtered if x_filtered is not None else x


def plain_stat(col, label: str, value: str) -> None:
    col.markdown(
        f'<div class="plain-stat-label">{label}</div>'
        f'<div class="plain-stat-value">{value}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Stat row, in its own card
# --------------------------------------------------------------------------

with st.container(border=True):
    s1, s2, s3, s4 = st.columns(4)
    plain_stat(s1, "Samples", f"{len(active_signal):,}")
    plain_stat(s2, "Sample rate", f"{sample_rate:,.0f} Hz")
    plain_stat(s3, "Peak amplitude", f"{np.max(np.abs(active_signal)):.3f}")
    plain_stat(s4, "RMS amplitude", f"{np.sqrt(np.mean(active_signal ** 2)):.3f}")

st.write("")

st.markdown("""
<style>
div[data-baseweb="tab-list"] {
    display: flex;
    width: 100%;
}

button[data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Time Domain", "Frequency Domain (FFT)", "Spectrogram (STFT)", "Audio & Export"])

# --- Time Domain ----------------------------------------------------------
with tabs[0]:
    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=x, name="Original", line=dict(color=ORIGINAL_TRACE, width=1.4)))
        if x_filtered is not None:
            fig.add_trace(
                go.Scatter(x=t, y=x_filtered, name="Filtered", line=dict(color=COMPARE, width=1.4))
            )
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Time (s)", yaxis_title="Amplitude", height=440)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

# --- Frequency Domain -------------------------------------------------------
with tabs[1]:
    with st.container(border=True):
        if source == "Generate" and signal_name == "Sinc":
            st.caption(
                "Sinc's Fourier transform is a rectangle centered on 0 Hz. The usual "
                "single-sided view only shows the right half of that rectangle, which "
                "can look like it starts at 0 Hz rather than being centered on it — so "
                "this tab shows the full two-sided spectrum instead, just for Sinc."
            )
            freqs, mag = calculate_fft_two_sided(x, sample_rate)
            y = mag
            ylabel = "Magnitude"
            if db_scale:
                y = 20 * np.log10(np.maximum(mag, 1e-12))
                ylabel = "Magnitude (dB)"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs, y=y, name="Original", line=dict(color=ORIGINAL_TRACE, width=1.2)))

            if x_filtered is not None:
                freqs_f, mag_f = calculate_fft_two_sided(x_filtered, sample_rate)
                yf = mag_f
                if db_scale:
                    yf = 20 * np.log10(np.maximum(mag_f, 1e-12))
                fig.add_trace(
                    go.Scatter(x=freqs_f, y=yf, name="Filtered", line=dict(color=COMPARE, width=1.2))
                )

            fig.add_vline(x=0, line_dash="dot", line_color=SUBTEXT)
            fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Frequency (Hz)", yaxis_title=ylabel, height=440)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        else:
            freqs, mag = calculate_fft(x, sample_rate)
            y = mag
            ylabel = "Magnitude"
            if db_scale:
                y = 20 * np.log10(np.maximum(mag, 1e-12))
                ylabel = "Magnitude (dB)"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs, y=y, name="Original", line=dict(color=ORIGINAL_TRACE, width=1.2)))

            if x_filtered is not None:
                freqs_f, mag_f = calculate_fft(x_filtered, sample_rate)
                yf = mag_f
                if db_scale:
                    yf = 20 * np.log10(np.maximum(mag_f, 1e-12))
                fig.add_trace(
                    go.Scatter(x=freqs_f, y=yf, name="Filtered", line=dict(color=COMPARE, width=1.2))
                )

            fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Frequency (Hz)", yaxis_title=ylabel, height=440)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

# --- Spectrogram ------------------------------------------------------------
with tabs[2]:
    with st.container(border=True):
        if len(x) < 32:
            st.info("Signal is too short for a meaningful spectrogram — increase duration or sample rate.")
        elif x_filtered is None:
            f_stft, t_stft, mag_stft = calculate_stft(x, sample_rate)
            fig = go.Figure(
                data=go.Heatmap(
                    z=mag_stft, x=t_stft, y=f_stft, colorscale="Viridis", colorbar=dict(title="Magnitude")
                )
            )
            fig.update_layout(
                **PLOTLY_LAYOUT, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", height=440
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        else:
            # Side-by-side original vs. filtered, on a shared color scale so the
            # comparison is apples-to-apples rather than each panel auto-scaling
            # to its own max.
            f_o, t_o, mag_o = calculate_stft(x, sample_rate)
            f_f, t_f, mag_f = calculate_stft(x_filtered, sample_rate)
            shared_max = max(mag_o.max(), mag_f.max())

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Original", "Filtered"),
                shared_yaxes=True,
                horizontal_spacing=0.06,
            )
            fig.add_trace(
                go.Heatmap(
                    z=mag_o, x=t_o, y=f_o, colorscale="Viridis", zmin=0, zmax=shared_max, showscale=False
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=mag_f,
                    x=t_f,
                    y=f_f,
                    colorscale="Viridis",
                    zmin=0,
                    zmax=shared_max,
                    colorbar=dict(title="Magnitude"),
                ),
                row=1,
                col=2,
            )
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
            fig.update_layout(**PLOTLY_LAYOUT, height=440, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            st.caption("Both panels share the same color scale, so darker/brighter areas are directly comparable.")




# --- Audio & Export -----------------------------------------------------------
with tabs[3]:
    ac1, ac2 = st.columns(2)
    with ac1:
        with st.container(border=True):
            st.markdown("**Original**")
            st.audio(to_wav_bytes(x, sample_rate), format="audio/wav")
            st.download_button(
                "Download original .wav", to_wav_bytes(x, sample_rate), file_name="original.wav"
            )
            st.download_button(
                "Download original .csv", to_csv_bytes(t, x), file_name="original.csv"
            )
    with ac2:
        with st.container(border=True):
            st.markdown("**Filtered**")
            if x_filtered is not None:
                st.audio(to_wav_bytes(x_filtered, sample_rate), format="audio/wav")
                st.download_button(
                    "Download filtered .wav",
                    to_wav_bytes(x_filtered, sample_rate),
                    file_name="filtered.wav",
                )
                st.download_button(
                    "Download filtered .csv", to_csv_bytes(t, x_filtered), file_name="filtered.csv"
                )
            else:
                st.caption("Enable a filter in the sidebar to hear/export the filtered signal.")

    st.caption(
        "Playback uses your browser's audio, not the system audio device — this works the same "
        "locally and when deployed (e.g. Streamlit Cloud)."
    )