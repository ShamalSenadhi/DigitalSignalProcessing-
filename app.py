import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import soundfile as sf
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="🎸 Guitar Tuner Dashboard",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-attractive interface
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
    }
    
    h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00d4ff, #6bcf7f, #ffd93d, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3.5em;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px #00d4ff); }
        to { filter: drop-shadow(0 0 20px #6bcf7f); }
    }
    
    .subtitle {
        color: #6bcf7f;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 2em;
        text-shadow: 0 0 10px rgba(107, 207, 127, 0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1f3a;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #2d3561 0%, #1a1f3a 100%);
        border-radius: 8px;
        color: #00d4ff;
        font-weight: bold;
        padding: 10px 20px;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #6bcf7f 100%);
        color: #0a0e27;
        border: 2px solid #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        text-align: center;
        margin: 10px 0;
    }
    
    .string-button {
        background: linear-gradient(135deg, #2d3561 0%, #1a1f3a 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid;
        margin: 5px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .string-button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px;
    }
    
    .status-intune {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        color: #0a0e27;
        padding: 20px;
        border-radius: 15px;
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.6);
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .status-sharp {
        background: linear-gradient(135deg, #ff4444 0%, #ff6b6b 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 68, 68, 0.6);
    }
    
    .status-flat {
        background: linear-gradient(135deg, #4444ff 0%, #6b6bff 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 30px rgba(68, 68, 255, 0.6);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #e94560 0%, #8b2d61 100%);
        color: white;
        font-size: 1.3em;
        font-weight: bold;
        border-radius: 15px;
        padding: 15px 40px;
        border: 3px solid #f472b6;
        width: 100%;
        box-shadow: 0 0 25px rgba(233, 69, 96, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 35px rgba(233, 69, 96, 0.8);
        border-color: #ff6b9d;
    }
    
    .info-box {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00d4ff;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 100%);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .visualization-section {
        background: linear-gradient(135deg, #0f1729 0%, #1a1f3a 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #6bcf7f;
        box-shadow: 0 0 25px rgba(107, 207, 127, 0.3);
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None
if 'filtered_audio' not in st.session_state:
    st.session_state.filtered_audio = None
if 'dominant_freq' not in st.session_state:
    st.session_state.dominant_freq = None
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None
if 'selected_string' not in st.session_state:
    st.session_state.selected_string = 'E4 (1st - High E)'

# Guitar string data
STRING_FREQUENCIES = {
    'E2 (6th - Low E)': 82.41,
    'A2 (5th)': 110.00,
    'D3 (4th)': 146.83,
    'G3 (3rd)': 196.00,
    'B3 (2nd)': 246.94,
    'E4 (1st - High E)': 329.63
}

STRING_COLORS = {
    'E2 (6th - Low E)': '#ff6b6b',
    'A2 (5th)': '#ffd93d',
    'D3 (4th)': '#6bcf7f',
    'G3 (3rd)': '#4ecdc4',
    'B3 (2nd)': '#a78bfa',
    'E4 (1st - High E)': '#f472b6'
}

TUNE_TOLERANCE = 2.0

# Functions
def process_audio(audio_data, sample_rate, target_freq):
    """Process audio and return results"""
    # Apply filters
    nyquist = sample_rate / 2
    Q = 30
    w0 = 50 / nyquist
    b_notch, a_notch = signal.iirnotch(w0, Q, sample_rate)
    
    cutoff = 500 / nyquist
    b_low, a_low = signal.butter(4, cutoff, btype='low', analog=False)
    
    temp = signal.filtfilt(b_notch, a_notch, audio_data)
    filtered_audio = signal.filtfilt(b_low, a_low, temp)
    
    # Find dominant frequency
    N = len(filtered_audio)
    yf = fft(filtered_audio)
    xf = fftfreq(N, 1/sample_rate)
    
    pos_mask = xf > 0
    xf = xf[pos_mask]
    yf = np.abs(yf[pos_mask])
    
    mask = (xf >= 70) & (xf <= 400)
    freq_range = xf[mask]
    mag_range = yf[mask]
    
    if len(mag_range) == 0:
        return None, None, None, None, None, None
    
    peak_idx = np.argmax(mag_range)
    dominant_freq = freq_range[peak_idx]
    
    # Compute FFTs for visualization
    freq_before, mag_before = xf, yf
    
    N_filtered = len(filtered_audio)
    yf_filtered = fft(filtered_audio)
    xf_filtered = fftfreq(N_filtered, 1/sample_rate)
    pos_mask_filtered = xf_filtered > 0
    freq_after = xf_filtered[pos_mask_filtered]
    mag_after = np.abs(yf_filtered[pos_mask_filtered])
    
    return filtered_audio, dominant_freq, freq_before, mag_before, freq_after, mag_after

def determine_tuning_status(detected_freq, target_freq):
    """Determine tuning status"""
    diff = detected_freq - target_freq
    
    if abs(diff) <= TUNE_TOLERANCE:
        return "IN TUNE ✓", "status-intune", diff, "🎉 Perfect! Your string is tuned correctly!"
    elif diff > 0:
        return "SHARP ↑", "status-sharp", diff, "⬇️ String is too high - Loosen the string (turn tuning peg counter-clockwise)"
    else:
        return "FLAT ↓", "status-flat", diff, "⬆️ String is too low - Tighten the string (turn tuning peg clockwise)"

# Header
st.markdown("<h1>🎸 GUITAR TUNER DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Professional FFT Analysis & Digital Signal Processing</p>", unsafe_allow_html=True)

# Main Layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🎻 SELECT GUITAR STRING")
    
    # String selection buttons
    cols = st.columns(2)
    for idx, (string_name, freq) in enumerate(STRING_FREQUENCIES.items()):
        color = STRING_COLORS[string_name]
        with cols[idx % 2]:
            if st.button(
                f"{string_name}\n{freq} Hz",
                key=f"string_{idx}",
                use_container_width=True
            ):
                st.session_state.selected_string = string_name
            
            # Highlight selected
            if st.session_state.selected_string == string_name:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color} 0%, {color}99 100%); 
                     padding: 5px; border-radius: 8px; text-align: center; 
                     border: 3px solid {color}; box-shadow: 0 0 20px {color}88;'>
                    <b style='color: white;'>SELECTED ✓</b>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📁 AUDIO OPERATIONS")
    
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Supports WAV, MP3, FLAC, OGG, M4A formats"
    )
    
    if uploaded_file:
        try:
            audio_bytes = uploaded_file.read()
            audio_data, sample_rate = librosa.load(BytesIO(audio_bytes), sr=None, mono=True)
            
            max_samples = int(5 * sample_rate)
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            st.session_state.current_filename = uploaded_file.name
            
            st.success(f"✅ Audio loaded: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Error loading audio: {str(e)}")

with col_right:
    st.markdown("### 📊 CURRENT STATUS")
    
    if st.session_state.audio_data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: #00d4ff; margin: 0;'>📄 File</h4>
                <p style='color: white; font-size: 0.9em; margin: 5px 0;'>{st.session_state.current_filename[:20]}...</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: #6bcf7f; margin: 0;'>🔊 Sample Rate</h4>
                <p style='color: white; font-size: 1.2em; margin: 5px 0;'>{st.session_state.sample_rate} Hz</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            duration = len(st.session_state.audio_data) / st.session_state.sample_rate
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: #ffd93d; margin: 0;'>⏱️ Duration</h4>
                <p style='color: white; font-size: 1.2em; margin: 5px 0;'>{duration:.2f} s</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No audio file loaded yet")
    
    target_freq = STRING_FREQUENCIES[st.session_state.selected_string]
    st.markdown(f"""
    <div class='metric-card'>
        <h4 style='color: #f472b6; margin: 0;'>🎯 TARGET FREQUENCY</h4>
        <p style='color: white; font-size: 2em; font-weight: bold; margin: 10px 0;'>{target_freq:.2f} Hz</p>
        <p style='color: #888; font-size: 0.9em;'>{st.session_state.selected_string}</p>
    </div>
    """, unsafe_allow_html=True)

# Main Analysis Button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("⚡ ANALYZE & TUNE ⚡", use_container_width=True):
    if st.session_state.audio_data is None:
        st.error("❌ Please upload an audio file first!")
    else:
        with st.spinner("🔄 Processing audio..."):
            target_freq = STRING_FREQUENCIES[st.session_state.selected_string]
            
            filtered_audio, dominant_freq, freq_before, mag_before, freq_after, mag_after = process_audio(
                st.session_state.audio_data,
                st.session_state.sample_rate,
                target_freq
            )
            
            if dominant_freq is None:
                st.error("❌ Could not detect frequency. Please check the audio file.")
            else:
                st.session_state.filtered_audio = filtered_audio
                st.session_state.dominant_freq = dominant_freq
                
                status, status_class, diff, instruction = determine_tuning_status(dominant_freq, target_freq)
                
                st.markdown("---")
                st.markdown("## 🎯 TUNING RESULT")
                
                # Result display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card' style='border-color: #00d4ff;'>
                        <h3 style='color: white; margin: 0;'>Detected</h3>
                        <h1 style='color: #00d4ff; margin: 10px 0;'>{dominant_freq:.2f}</h1>
                        <p style='color: #888;'>Hz</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card' style='border-color: #6bcf7f;'>
                        <h3 style='color: white; margin: 0;'>Target</h3>
                        <h1 style='color: #6bcf7f; margin: 10px 0;'>{target_freq:.2f}</h1>
                        <p style='color: #888;'>Hz</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    diff_color = '#00ff88' if abs(diff) <= TUNE_TOLERANCE else ('#ff4444' if diff > 0 else '#4444ff')
                    st.markdown(f"""
                    <div class='metric-card' style='border-color: {diff_color};'>
                        <h3 style='color: white; margin: 0;'>Difference</h3>
                        <h1 style='color: {diff_color}; margin: 10px 0;'>{diff:+.2f}</h1>
                        <p style='color: #888;'>Hz</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Status display
                st.markdown(f"<div class='{status_class}'>{status}</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='info-box'>
                    <h4 style='color: #00d4ff;'>📝 Instructions:</h4>
                    <p style='color: white; font-size: 1.1em;'>{instruction}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations in tabs
                st.markdown("---")
                st.markdown("## 📊 DETAILED ANALYSIS")
                
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🌊 Waveform",
                    "📊 Spectrum",
                    "📈 FFT Before",
                    "📉 FFT After",
                    "🔧 Filters",
                    "🎯 Tuning Meter"
                ])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0f1729')
                    ax.set_facecolor('#0f1729')
                    
                    time = np.linspace(0, len(st.session_state.audio_data)/st.session_state.sample_rate, 
                                     len(st.session_state.audio_data))
                    samples = min(10000, len(st.session_state.audio_data))
                    
                    color = STRING_COLORS[st.session_state.selected_string]
                    ax.plot(time[:samples], st.session_state.audio_data[:samples], 
                           color=color, linewidth=0.8, alpha=0.8)
                    ax.fill_between(time[:samples], st.session_state.audio_data[:samples], 
                                   alpha=0.3, color=color)
                    
                    ax.set_title('Time Domain Signal', color='#00d4ff', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Time (s)', color='white', fontsize=11)
                    ax.set_ylabel('Amplitude', color='white', fontsize=11)
                    ax.grid(True, alpha=0.2, color='#2d3561')
                    ax.tick_params(colors='white')
                    
                    st.pyplot(fig)
                    plt.close()
                
                with tab2:
                    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0f1729')
                    ax.set_facecolor('#0f1729')
                    
                    ax.plot(freq_before, mag_before, color='#6bcf7f', linewidth=1, alpha=0.8)
                    ax.fill_between(freq_before, mag_before, alpha=0.3, color='#6bcf7f')
                    ax.set_xlim([0, 1000])
                    
                    ax.set_title('Frequency Spectrum', color='#6bcf7f', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Frequency (Hz)', color='white', fontsize=11)
                    ax.set_ylabel('Magnitude', color='white', fontsize=11)
                    ax.grid(True, alpha=0.2, color='#2d3561')
                    ax.tick_params(colors='white')
                    
                    st.pyplot(fig)
                    plt.close()
                
                with tab3:
                    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0f1729')
                    ax.set_facecolor('#0f1729')
                    
                    ax.plot(freq_before, mag_before, color='#ff6b6b', linewidth=1, alpha=0.8)
                    ax.fill_between(freq_before, mag_before, alpha=0.2, color='#ff6b6b')
                    ax.set_xlim([0, 600])
                    
                    ax.set_title('Unfiltered Spectrum', color='#ff6b6b', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Frequency (Hz)', color='white', fontsize=11)
                    ax.set_ylabel('Magnitude', color='white', fontsize=11)
                    ax.grid(True, alpha=0.2, color='#2d3561')
                    ax.tick_params(colors='white')
                    
                    st.pyplot(fig)
                    plt.close()
                
                with tab4:
                    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0f1729')
                    ax.set_facecolor('#0f1729')
                    
                    ax.plot(freq_after, mag_after, color='#6bcf7f', linewidth=1, alpha=0.8)
                    ax.fill_between(freq_after, mag_after, alpha=0.2, color='#6bcf7f')
                    ax.set_xlim([0, 600])
                    
                    ax.set_title('Filtered Spectrum', color='#6bcf7f', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Frequency (Hz)', color='white', fontsize=11)
                    ax.set_ylabel('Magnitude', color='white', fontsize=11)
                    ax.grid(True, alpha=0.2, color='#2d3561')
                    ax.tick_params(colors='white')
                    
                    st.pyplot(fig)
                    plt.close()
                
                with tab5:
                    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0f1729')
                    ax.set_facecolor('#0f1729')
                    
                    nyquist = st.session_state.sample_rate / 2
                    Q = 30
                    w0 = 50 / nyquist
                    b_notch, a_notch = signal.iirnotch(w0, Q, st.session_state.sample_rate)
                    
                    cutoff = 500 / nyquist
                    b_low, a_low = signal.butter(4, cutoff, btype='low', analog=False)
                    
                    w_notch, h_notch = signal.freqz(b_notch, a_notch, worN=2000, fs=st.session_state.sample_rate)
                    w_low, h_low = signal.freqz(b_low, a_low, worN=2000, fs=st.session_state.sample_rate)
                    
                    ax.plot(w_notch, 20*np.log10(abs(h_notch)), 
                           color='#ffd93d', label='Notch Filter (50Hz)', linewidth=2)
                    ax.plot(w_low, 20*np.log10(abs(h_low)), 
                           color='#6bcf7f', label='Lowpass Filter (500Hz)', linewidth=2)
                    ax.set_xlim([0, 600])
                    
                    ax.set_title('Filter Frequency Response', color='#ffd93d', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Frequency (Hz)', color='white', fontsize=11)
                    ax.set_ylabel('Gain (dB)', color='white', fontsize=11)
                    ax.legend(facecolor='#0f1729', edgecolor='#2d3561', labelcolor='white')
                    ax.grid(True, alpha=0.2, color='#2d3561')
                    ax.tick_params(colors='white')
                    
                    st.pyplot(fig)
                    plt.close()
                
                with tab6:
                    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0f1729')
                    ax.set_facecolor('#0f1729')
                    ax.axis('off')
                    
                    # Tuning meter
                    meter_width = 0.8
                    meter_height = 0.12
                    meter_x = 0.1
                    meter_y = 0.5
                    
                    # Background
                    ax.add_patch(plt.Rectangle((meter_x, meter_y), meter_width, meter_height,
                                               facecolor='#1a1f3a', edgecolor='#2d3561', linewidth=3))
                    
                    # Zones
                    ax.add_patch(plt.Rectangle((meter_x, meter_y), meter_width*0.35, meter_height,
                                               facecolor='#4444ff', alpha=0.3))
                    ax.add_patch(plt.Rectangle((meter_x + meter_width*0.35, meter_y), 
                                               meter_width*0.3, meter_height,
                                               facecolor='#00ff88', alpha=0.3))
                    ax.add_patch(plt.Rectangle((meter_x + meter_width*0.65, meter_y), 
                                               meter_width*0.35, meter_height,
                                               facecolor='#ff4444', alpha=0.3))
                    
                    # Needle
                    max_diff = 10
                    needle_pos = np.clip(diff / max_diff, -1, 1)
                    needle_x = meter_x + meter_width/2 + (needle_pos * meter_width/2 * 0.9)
                    
                    needle_color = '#00ff88' if abs(diff) <= TUNE_TOLERANCE else ('#ff4444' if diff > 0 else '#4444ff')
                    ax.plot([needle_x, needle_x], [meter_y, meter_y + meter_height], 
                           color=needle_color, linewidth=6, alpha=0.9)
                    ax.plot([needle_x], [meter_y + meter_height + 0.03], 
                           marker='v', markersize=25, color=needle_color)
                    
                    # Center line
                    center_x = meter_x + meter_width/2
                    ax.plot([center_x, center_x], [meter_y, meter_y + meter_height], 
                           color='white', linewidth=2, alpha=0.5, linestyle='--')
                    
                    # Info text
                    ax.text(0.5, 0.85, f"{st.session_state.selected_string}", 
                           ha='center', va='center', fontsize=16, color='#00d4ff', fontweight='bold')
                    
                    ax.text(0.5, 0.75, f"Detected: {dominant_freq:.2f} Hz", 
                           ha='center', va='center', fontsize=13, color='white', fontweight='bold')
                    ax.text(0.5, 0.68, f"Target: {target_freq:.2f} Hz", 
                           ha='center', va='center', fontsize=12, color='#888')
                    ax.text(0.5, 0.61, f"Difference: {diff:+.2f} Hz", 
                           ha='center', va='center', fontsize=11, color=needle_color, fontweight='bold')
                    
                    # Zone labels
                    ax.text(meter_x + 0.02, meter_y - 0.04, "FLAT", 
                           ha='left', va='top', fontsize=10, color='#4444ff', fontweight='bold')
                    ax.text(center_x, meter_y - 0.04, "IN TUNE", 
                           ha='center', va='top', fontsize=10, color='#00ff88', fontweight='bold')
                    ax.text(meter_x + meter_width - 0.02, meter_y - 0.04, "SHARP", 
                           ha='right', va='top', fontsize=10, color='#ff4444', fontweight='bold')
                    
                    # Status
                    ax.text(0.5, 0.35, status, 
                           ha='center', va='center', fontsize=26, color=needle_color, 
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=1', facecolor='#1a1f3a', 
                                    edgecolor=needle_color, linewidth=4))
                    
                    # Instructions
                    ax.text(0.5, 0.15, instruction, 
                           ha='center', va='center', fontsize=10, color=needle_color, 
                           style='italic', wrap=True)
                    
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    
                    st.pyplot(fig)
                    plt.close()

# Sidebar info
with st.sidebar:
    st.markdown("## 🎵 ABOUT")
    st.info("""
    **Professional Guitar Tuner Dashboard**
    
    This application uses advanced Digital Signal Processing (DSP) techniques:
    
    ✨ **Features:**
    - FFT Analysis
    - 50Hz Notch Filter
    - 500Hz Lowpass Filter
    - Real-time Tuning Feedback
    - Multi-format Audio Support
    - Visual Spectrum Analysis
    """)
    
    st.markdown("---")
    st.markdown("### 🎸 Standard Tuning")
    
    for string_name, freq in STRING_FREQUENCIES.items():
        color = STRING_COLORS[string_name]
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, {color}33 0%, transparent 100%); 
             padding: 8px; margin: 5px 0; border-radius: 8px; border-left: 4px solid {color};'>
            <b style='color: {color};'>{string_name}</b><br>
            <span style='color: white; font-size: 1.1em;'>{freq} Hz</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Select String**: Click on the guitar string you want to tune
    2. **Upload Audio**: Upload a recording of your guitar string
    3. **Analyze**: Click the "ANALYZE & TUNE" button
    4. **Adjust**: Follow the instructions to tune your string
    5. **Visualize**: Explore the detailed analysis tabs
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Technical Info")
    st.markdown("""
    **Signal Processing:**
    - Sample Rate: Auto-detected
    - Tolerance: ±2 Hz
    - Frequency Range: 70-400 Hz
    - Filter Type: IIR Butterworth
    
    **Supported Formats:**
    - WAV, MP3, FLAC, OGG, M4A
    
    **EEX7434 Mini Project**
    © 2024
    """)

# Welcome screen when no audio is loaded
if st.session_state.audio_data is None:
    st.markdown("---")
    st.markdown("## 👋 WELCOME!")
    
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown("""
        <div class='info-box'>
            <h3 style='color: #00d4ff; text-align: center;'>🎸 Get Started</h3>
            <p style='color: white; font-size: 1.1em; text-align: center;'>
                Upload an audio file of your guitar string to begin tuning analysis
            </p>
            <br>
            <h4 style='color: #6bcf7f;'>✨ Features:</h4>
            <ul style='color: white; line-height: 1.8;'>
                <li>🎵 Supports all 6 guitar strings</li>
                <li>🔊 Advanced noise filtering</li>
                <li>📊 Real-time FFT spectrum analysis</li>
                <li>🎯 ±2 Hz tuning accuracy</li>
                <li>📈 Before/after filter comparison</li>
                <li>🌊 Interactive waveform display</li>
                <li>🎨 Beautiful visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🎼 STANDARD GUITAR TUNING REFERENCE")
    
    cols = st.columns(6)
    for idx, (string_name, freq) in enumerate(STRING_FREQUENCIES.items()):
        color = STRING_COLORS[string_name]
        with cols[idx]:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color}33 0%, {color}11 100%); 
                 padding: 20px; border-radius: 15px; text-align: center;
                 border: 3px solid {color}; box-shadow: 0 0 20px {color}44;'>
                <h4 style='color: {color}; margin: 0; font-size: 1em;'>{string_name}</h4>
                <h2 style='color: white; font-size: 1.8em; margin: 10px 0;'>{freq:.2f}</h2>
                <p style='color: #888; margin: 0;'>Hz</p>
            </div>
            """, unsafe_allow_html=True)
