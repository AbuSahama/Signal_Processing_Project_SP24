from __future__ import annotations
import io
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write

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

st.set_page_config(
    page_title="Signal Generator and Analyzer",
    page_icon="/Users/abusahama/Signal_Processing_Project_SP24/assests/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Theme constants — single source of truth for every color used in the CSS


ACCENT = "#7C9BFF"
ACCENT_SOFT = "rgba(124, 155, 255, 0.15)"
COMPARE = "#FE0000"          # "filtered" trace color
ORIGINAL_TRACE = "#0EFF01"   # "original" trace color
BG = "#0E1117"
CARD_BG = "#161B22"
BORDER = "#262C36"
TEXT = "#F2F4F8"
SUBTEXT = "#9AA4B2"



st.markdown(
    f"""
    <style>
        html {{ scroll-behavior: smooth; }}  /* smooth-scrolls the #dashboard anchor jump */
        .stApp {{ background-color: {BG}; }}

        h1, h2, h3 {{
            color: {TEXT} !important;
            font-weight: 800 !important;
        }}

        /* Header block (dashboard title above the tabs) */
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

        /* Stat tiles (Samples / Sample rate / Peak / RMS) */
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

        /* Tabs (Time Domain / FFT / STFT / Export) */
        .stTabs [data-baseweb="tab-list"] {{
            display: flex;
            width: 100%;
            gap: 1.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {SUBTEXT};
            font-weight: 600;
        }}
        .stTabs button[data-baseweb="tab"] {{
            flex: 1;
            justify-content: center;
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

        /* ---------------- Landing / welcome hero ---------------- */
        .landing-wrap {{
            max-width: 900px;
            margin: 2rem 0 0 0;
            padding: 0;
        }}
        .landing-title {{
            font-size: clamp(2.0rem, 4.0vw, 3.0rem);
            font-weight: 800;
            color: {TEXT};
            letter-spacing: -0.03em;
            line-height: 1.05;
            margin-top: 0.4rem;
        }}
        .landing-subtitle {{
            color: {SUBTEXT};
            font-size: 1.2rem;
            margin-top: 1.4rem;
            line-height: 1.6;
        }}
        .landing-secondary-link {{
            color: {TEXT};
            font-size: 1.05rem;
            text-decoration: underline;
            text-decoration-color: {BORDER};
            text-underline-offset: 4px;
        }}
        .landing-cta-btn {{
            display: inline-block;
            background-color: {ACCENT};
            color: {BG} !important;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 0.7rem 1.8rem;
            border-radius: 10px;
            text-decoration: none !important;
            transition: filter 0.15s ease;
        }}
        .landing-cta-btn:hover {{
            filter: brightness(1.08);
        }}

        /* Pseudocode / code-editor style mockup card under the hero */
        .landing-mockup {{
            margin-top: 3rem;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid {BORDER};
            background: linear-gradient(135deg, {ACCENT_SOFT}, transparent 60%), {CARD_BG};
        }}
        .landing-mockup-window {{
            margin: 1.8rem;
            border-radius: 10px;
            background-color: #0B0E14;
            border: 1px solid {BORDER};
            overflow: hidden;
        }}
        .landing-mockup-titlebar {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.6rem 0.8rem;
            border-bottom: 1px solid {BORDER};
        }}
        .landing-dot {{
            width: 11px;
            height: 11px;
            border-radius: 50%;
        }}
        .landing-mockup-filename {{
            margin-left: 0.5rem;
            color: {SUBTEXT};
            font-size: 0.8rem;
            font-family: monospace;
        }}
        .landing-mockup-code {{
            padding: 1.1rem 1.3rem 1.4rem 1.3rem;
            font-family: "SFMono-Regular", Consolas, monospace;
            font-size: 0.92rem;
            line-height: 1.85;
            color: {TEXT};
        }}
        .landing-mockup-code .ln {{
            color: {SUBTEXT};
            display: inline-block;
            width: 1.4rem;
        }}
        .landing-mockup-code .kw {{ color: {ACCENT}; }}
        .landing-mockup-code .fn {{ color: {ORIGINAL_TRACE}; }}
        .landing-mockup-code .str {{ color: #FFC86B; }}

        /* Top bar (GitHub link, top-of-page anchor target) */
        .topbar-github {{
            display: flex;
            justify-content: flex-end;
        }}
        .topbar-github a {{
            display: inline-flex;
            align-items: center;
            color: {SUBTEXT};
            transition: color 0.15s ease;
        }}
        .topbar-github a:hover {{
            color: {TEXT};
        }}

        div[data-testid="stButton"] button[kind="primary"] {{
            background-color: {ACCENT} !important;
            border-color: {ACCENT} !important;
            color: {BG} !important;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 0.7rem 0;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{
            filter: brightness(1.08);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


GITHUB_URL = "https://github.com/AbuSahama/Signal_Processing_Project_SP24"


def render_topbar() -> None:
    """Top bar shown once at the very top of the page: an invisible
    `#top` anchor (used by the dashboard's "Back to top" link) on the
    left, and the GitHub repo link on the right."""
    left, right = st.columns([8, 1])
    with left:
        st.markdown('<a id="top"></a>', unsafe_allow_html=True)
    with right:
        st.markdown(
            f"""
            <div class="topbar-github">
                <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer" title="View on GitHub">
                    <svg height="22" width="22" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                        -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07
                        -1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82
                        .64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                        .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                        0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )



render_topbar()
st.markdown(
    f"""
    <div class="landing-wrap">
       <div class="landing-title"><span style="color:{ACCENT};">ES Signals Project</span><br>Signal Generator and Analyzer</div>
        <div class="landing-subtitle">
            Team Members: Abu Sahama (24f3100239),  Rajiv Ratan (24f1100039),  Sonali (24f2100339)
        </div>
        <div style="margin-top: 1.8rem; display: flex; align-items: center; gap: 1.6rem; flex-wrap: wrap;">
            <a href="#dashboard" class="landing-cta-btn">Get started</a>
            <span class="landing-secondary-link">Signals · FFT · Spectrogram (STFT) · Filters</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pseudocode mockup card: a fake "code editor" summarizing the app's logic,

st.markdown(
    f"""
    <div class="landing-wrap">
        <div class="landing-mockup">
            <div class="landing-mockup-window">
                <div class="landing-mockup-titlebar">
                    <span class="landing-mockup-filename">Project's Basic Pseudo Code</span>
                </div>
                <div class="landing-mockup-code">
                    <div>
                        <span class="ln">1</span>
                        <span class="kw">BEGIN</span>
                    </div>
                    <div>
                        <span class="ln">2</span>
                        <span class="kw">SELECT</span> signal source
                    </div>
                    <div>
                        <span class="ln">3</span>
                        <span class="fn">GENERATE</span> /
                        <span class="fn">UPLOAD</span> /
                        <span class="fn">RECORD</span> signal
                    </div>
                    <div>
                        <span class="ln">4</span>
                        <span class="fn">ADD</span> optional noise
                    </div>
                    <div>
                        <span class="ln">5</span>
                        <span class="kw">ANALYZE</span>
                        <span style="color:{SUBTEXT};">
                        </span>
                    </div>
                    <div>
                        <span class="ln">6</span>
                        &nbsp;&nbsp;Time Domain →
                        <span class="fn">FFT</span> →
                        <span class="fn">STFT</span>
                    </div>
                    <div>
                        <span class="ln">7</span>
                        <span class="kw">IF</span> filtering is enabled
                        <span class="kw">THEN</span>
                    </div>
                    <div>
                        <span class="ln">8</span>
                        &nbsp;&nbsp;<span class="fn">Butterworth</span>
                        <span class="str">LPF / HPF / BPF</span>
                    </div>
                    <div>
                        <span class="ln">9</span>
                        &nbsp;&nbsp;<span class="fn">SOS</span> zero-phase filtering
                    </div>
                    <div>
                        <span class="ln">10</span>
                        <span class="fn">COMPARE</span>
                        original vs filtered
                    </div>
                    <div>
                        <span class="ln">11</span>
                        <span class="fn">PLAY</span> /
                        <span class="fn">EXPORT</span>
                        <span class="str">WAV / CSV</span>
                    </div>
                    <div>
                        <span class="ln">12</span>
                        <span class="kw">END</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <a id="dashboard"></a>
    """,
    unsafe_allow_html=True,
)



# Shared Plotly layout/config for every chart in the dashboard below.

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", y=1.08, x=0),
)

PLOT_CONFIG = dict(displayModeBar=True, displaylogo=False)


def to_wav_bytes(sig: np.ndarray, sample_rate: float) -> bytes:
    """Normalize a signal to int16 range and encode it as an in-memory WAV file."""
    peak = np.max(np.abs(sig))
    normalized = sig / peak if peak > 0 else sig
    int16_data = np.int16(normalized * 32767)
    buf = io.BytesIO()
    wav_write(buf, int(sample_rate), int16_data)
    return buf.getvalue()


def to_csv_bytes(t: np.ndarray, sig: np.ndarray) -> bytes:
    """Encode a (time, amplitude) pair as a downloadable CSV file."""
    buf = io.StringIO()
    buf.write("time_s,amplitude\n")
    for ti, xi in zip(t, sig):
        buf.write(f"{ti:.8f},{xi:.8f}\n")
    return buf.getvalue().encode("utf-8")


def read_uploaded_wav(uploaded_file) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Read an uploaded .wav file, downmix to mono, and normalize to [-1, 1]."""
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
    """Dispatch to the matching Butterworth filter in core.filters."""
    if kind == "low":
        return low_pass_filter(sig, sample_rate, params["cutoff"], params["order"])
    if kind == "high":
        return high_pass_filter(sig, sample_rate, params["cutoff"], params["order"])
    if kind == "band":
        return band_pass_filter(
            sig, sample_rate, params["low_cutoff"], params["high_cutoff"], params["order"]
        )
    raise ValueError(f"Unknown filter kind: {kind}")



# Sidebar — signal source, noise, filter and display controls.

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
        uploaded_sample_rate = 44100.0

        if source == "Generate":
            signal_name = st.selectbox("Waveform type", list(SIGNAL_SPECS.keys()))
            spec = SIGNAL_SPECS[signal_name]

            col1, col2 = st.columns(2)
            with col1:
                amplitude = st.number_input("Amplitude", value=1.0, min_value=0.0, step=0.1)
                sample_rate = st.number_input(
                    "Sample rate (Hz)", value=44100.0, min_value=1.0, step=100.0
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
            else:
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
            cutoff_min = 1.0
            cutoff_max = float(max(nyquist - 1, 1.0))
            cutoff_default = min(50.0, nyquist / 2)

            
            st.session_state.setdefault("cutoff_val", cutoff_default)

            def _sync_cutoff_from_slider():
                st.session_state.cutoff_val = st.session_state.cutoff_slider

            st.slider(
                "Cutoff frequency (Hz)", cutoff_min, cutoff_max,
                value=st.session_state.cutoff_val, step=1.0, key="cutoff_slider",
                on_change=_sync_cutoff_from_slider,
            )

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



# Build the active signal: generate / load it, optionally add noise, then
# optionally run it through the selected filter.

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

if error is None and x is not None and add_noise and noise_amplitude > 0:
    x = x + generate_noise(t, noise_amplitude, seed=int(noise_seed))
    title += f" + Noise (A={noise_amplitude})"

if error is None and filter_kind is not None:
    try:
        x_filtered = apply_filter(x, sample_rate, filter_kind, filter_params)
    except ValueError as exc:
        st.warning(f"Filter not applied: {exc}")
        x_filtered = None


# Dashboard header (sits right below the #dashboard anchor the hero links to)

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">Generate and Analyze</div>
        <div class="app-subtitle">{title if title else "Configure a signal in the sidebar to get started"}</div>
        <div class="app-badge">〰️ Waveform · Filter · Spectral analysis</div>
        <div style="margin-top: 0.6rem;">
            <a href="#top" style="color:{SUBTEXT}; font-size: 0.85rem; text-decoration: none;">&uarr; Back to top</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if error:
    st.error(error)
    st.stop()

active_signal = x_filtered if x_filtered is not None else x


def plain_stat(col, label: str, value: str) -> None:
    """Render a single label/value stat tile inside the given column."""
    col.markdown(
        f'<div class="plain-stat-label">{label}</div>'
        f'<div class="plain-stat-value">{value}</div>',
        unsafe_allow_html=True,
    )


with st.container(border=True):
    s1, s2, s3, s4 = st.columns(4)
    plain_stat(s1, "Samples", f"{len(active_signal):,}")
    plain_stat(s2, "Sample rate", f"{sample_rate:,.0f} Hz")
    plain_stat(s3, "Peak amplitude", f"{np.max(np.abs(active_signal)):.3f}")
    plain_stat(s4, "RMS amplitude", f"{np.sqrt(np.mean(active_signal ** 2)):.3f}")

st.write("")

tabs = st.tabs(["Time Domain", "Frequency Domain (FFT)", "Spectrogram (STFT)", "Audio & Export"])

#  Time Domain 
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

#  Frequency Domain (FFT) 
with tabs[1]:
    with st.container(border=True):

        if source == "Generate" and signal_name == "Sinc":
            
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

            fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Frequency (Hz)", yaxis_title=ylabel, height=440,)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

#  Spectrogram (STFT) 
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
            

#  Audio & Export
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