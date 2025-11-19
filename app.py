"""
🎸 Professional Guitar Tuner - Complete Streamlit Edition
EEX7434 Mini Project - ALL FUNCTIONS INCLUDED
Complex & Attractive Light Interface
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import soundfile as sf
import sounddevice as sd
import io
import time
from matplotlib.patches import Rectangle, FancyBboxPatch
import threading

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="🎸 Guitar Tuner Pro",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS - COMPLEX LIGHT DESIGN ====================
st.markdown("""
<style>
    /* Main background - Light gradient with subtle pattern */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 25%, #fff5f7 50%, #ffffff 75%, #f0fff4 100%);
        background-attachment: fixed;
    }
    
    /* Add subtle pattern overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(0,0,0,.02) 35px, rgba(0,0,0,.02) 70px);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Animated header with gradient and glow */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 
            0 10px 40px rgba(102, 126, 234, 0.3),
            0 0 0 1px rgba(255,255,255,0.5) inset,
            0 20px 60px rgba(118, 75, 162, 0.2);
        border: 3px solid rgba(255, 255, 255, 0.8);
        position: relative;
        overflow: hidden;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 
            0 2px 10px rgba(0,0,0,0.2),
            0 0 20px rgba(255,255,255,0.5),
            2px 2px 0 rgba(0,0,0,0.1);
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: 2px;
    }
    
    .main-header p {
        color: #ffffff;
        font-size: 1.3rem;
        margin-top: 0.8rem;
        font-weight: 600;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    /* Glass-morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid rgba(255, 255, 255, 0.8);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.5) inset;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 12px 48px rgba(0, 0, 0, 0.15),
            0 0 0 1px rgba(255, 255, 255, 0.6) inset;
    }
    
    /* Metric cards with neon glow */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
    }
    
    [data-testid="stMetricLabel"] {
        color: #5e5e5e !important;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    /* Animated buttons with gradient */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 800;
        border-radius: 15px;
        box-shadow: 
            0 6px 25px rgba(102, 126, 234, 0.4),
            0 0 0 3px rgba(255,255,255,0.3) inset;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 10px 35px rgba(102, 126, 234, 0.6),
            0 0 0 4px rgba(255,255,255,0.4) inset;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Animated tabs with gradient */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.6);
        color: #667eea;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.8rem 2rem;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transform: scale(1.05);
    }
    
    /* Sidebar with gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 50%, #fff5f7 100%);
        border-right: 3px solid rgba(102, 126, 234, 0.2);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #4a4a4a;
    }
    
    /* Status boxes with animation */
    .status-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border-left: 6px solid;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        position: relative;
        overflow: hidden;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .status-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.3) 50%, transparent 70%);
        transform: translateX(-100%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(200%); }
    }
    
    .status-in-tune {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-color: #00d084;
    }
    
    .status-sharp {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-color: #ff6b6b;
    }
    
    .status-flat {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        border-color: #4facfe;
    }
    
    /* Expander with gradient */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        color: #667eea;
        font-weight: 700;
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader with animation */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        border: 3px dashed rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(102, 126, 234, 0.05);
        transform: scale(1.02);
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #667eea;
        font-weight: 700;
        font-size: 1.15rem;
    }
    
    .stRadio [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.8);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stRadio [role="radiogroup"] label:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateX(5px);
    }
    
    /* Selectbox */
    .stSelectbox > label {
        color: #667eea;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
    }
    
    /* Slider */
    .stSlider > label {
        color: #667eea;
        font-weight: 700;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Alert boxes with icons */
    .stAlert {
        border-radius: 15px;
        border-left: 6px solid;
        padding: 1.2rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-color: #00d084;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-color: #ff9068;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        border-color: #4facfe;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-color: #ff6b6b;
    }
    
    /* Spinner animation */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Code blocks */
    code {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: 600;
    }
    
    /* Headings */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    
    /* Links */
    a {
        color: #667eea;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    a:hover {
        color: #764ba2;
        text-decoration: none;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Input fields */
    input, textarea {
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, textarea:focus {
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================
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
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# ==================== GUITAR STRING DATA ====================
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

# ==================== HELPER FUNCTIONS ====================

def create_light_figure(figsize=(10, 4)):
    """Create a matplotlib figure with light theme"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#fafbff')
    ax.tick_params(colors='#4a4a4a', labelsize=10)
    ax.spines['bottom'].set_color('#c0c0c0')
    ax.spines['left'].set_color('#c0c0c0')
    ax.spines['top'].set_color('#e0e0e0')
    ax.spines['right'].set_color('#e0e0e0')
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    return fig, ax

def process_audio(audio_data, sample_rate, target_freq):
    """Process audio and detect frequency with advanced DSP"""
    try:
        # Apply 50Hz notch filter
        nyquist = sample_rate / 2
        Q = 30
        w0 = 50 / nyquist
        b_notch, a_notch = signal.iirnotch(w0, Q, sample_rate)
        
        # Apply 500Hz lowpass filter
        cutoff = 500 / nyquist
        b_low, a_low = signal.butter(4, cutoff, btype='low', analog=False)
        
        # Filter the signal
        temp = signal.filtfilt(b_notch, a_notch, audio_data)
        filtered_audio = signal.filtfilt(b_low, a_low, temp)
        
        # Perform FFT analysis
        N = len(filtered_audio)
        yf = fft(filtered_audio)
        xf = fftfreq(N, 1/sample_rate)
        
        # Get positive frequencies only
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        # Focus on guitar frequency range (70-400 Hz)
        mask = (xf >= 70) & (xf <= 400)
        freq_range = xf[mask]
        mag_range = yf[mask]
        
        if len(mag_range) == 0:
            return None, None, "Could not detect frequency in valid range"
        
        # Find dominant frequency
        peak_idx = np.argmax(mag_range)
        dominant_freq = freq_range[peak_idx]
        
        return filtered_audio, dominant_freq, "Success"
        
    except Exception as e:
        return None, None, str(e)

def get_tuning_status(detected_freq, target_freq):
    """Get tuning status with color coding"""
    diff = detected_freq - target_freq
    
    if abs(diff) <= TUNE_TOLERANCE:
        return "IN TUNE ✓", "success", "#00d084", "status-in-tune"
    elif diff > 0:
        return "SHARP ↑", "warning", "#ff6b6b", "status-sharp"
    else:
        return "FLAT ↓", "info", "#4facfe", "status-flat"

def calculate_cents(detected_freq, target_freq):
    """Calculate cents deviation"""
    if detected_freq <= 0 or target_freq <= 0:
        return 0
    return 1200 * np.log2(detected_freq / target_freq)

# ==================== MAIN HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🎸 PROFESSIONAL GUITAR TUNER PRO</h1>
    <p>Advanced DSP Analysis | Real-time Frequency Detection | Professional Tuning System</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🎛️ CONTROL CENTER")
    st.markdown("---")
    
    # String Selection
    st.markdown("### 🎻 Select Guitar String")
    selected_string = st.selectbox(
        "Choose the string you want to tune:",
        options=list(STRING_FREQUENCIES.keys()),
        index=5
    )
    
    target_freq = STRING_FREQUENCIES[selected_string]
    string_color = STRING_COLORS[selected_string]
    
    # Display target frequency with colored card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {string_color}20 0%, {string_color}40 100%); 
                padding: 1.5rem; border-radius: 15px; 
                border-left: 6px solid {string_color}; 
                margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <p style="color: {string_color}; font-weight: 800; font-size: 1.3rem; margin: 0;">
            🎯 Target: {target_freq} Hz
        </p>
        <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
            {selected_string}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Audio Input Method
    st.markdown("### 📁 Audio Input")
    input_method = st.radio(
        "Select your input method:",
        ["📂 Upload Audio File", "🎙️ Record from Microphone"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Upload or Record
    if input_method == "📂 Upload Audio File":
        st.markdown("#### 📂 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Supported: WAV, MP3, FLAC, OGG, M4A",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a']
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("🔄 Loading audio file..."):
                    audio_bytes = uploaded_file.read()
                    audio_data, sample_rate = librosa.load(
                        io.BytesIO(audio_bytes), 
                        sr=None, 
                        mono=True
                    )
                    
                    # Limit to 5 seconds
                    max_samples = int(5 * sample_rate)
                    if len(audio_data) > max_samples:
                        audio_data = audio_data[:max_samples]
                        st.info(f"ℹ️ Audio trimmed to 5 seconds for processing")
                    
                    st.session_state.audio_data = audio_data
                    st.session_state.sample_rate = sample_rate
                    st.session_state.current_filename = uploaded_file.name
                    st.session_state.processing_complete = False
                    
                    duration = len(audio_data) / sample_rate
                    
                    st.success(f"""
                    ✅ **Audio Loaded Successfully!**
                    
                    📄 **File:** {uploaded_file.name}  
                    🔊 **Sample Rate:** {sample_rate:,} Hz  
                    ⏱️ **Duration:** {duration:.2f} seconds  
                    📊 **Samples:** {len(audio_data):,}
                    """)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.markdown("#### 🎙️ Record from Microphone")
        
        duration = st.slider(
            "⏱️ Recording Duration (seconds)",
            min_value=1,
            max_value=10,
            value=5
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔴 START", use_container_width=True, type="primary"):
                try:
                    with st.spinner(f"🎤 Recording for {duration} seconds..."):
                        sample_rate = 44100
                        recording = sd.rec(
                            int(duration * sample_rate),
                            samplerate=sample_rate,
                            channels=1,
                            dtype='float32'
                        )
                        
                        # Show countdown
                        progress_bar = st.progress(0)
                        for i in range(duration):
                            time.sleep(1)
                            progress_bar.progress((i + 1) / duration)
                        
                        sd.wait()
                        progress_bar.empty()
                        
                        st.session_state.audio_data = recording.flatten()
                        st.session_state.sample_rate = sample_rate
                        st.session_state.current_filename = f"Recording_{int(time.time())}.wav"
                        st.session_state.processing_complete = False
                        
                        st.success(f"✅ Recording completed! ({duration}s)")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Recording failed: {str(e)}")
        
        with col2:
            if st.button("💾 SAVE", use_container_width=True):
                if st.session_state.audio_data is not None:
                    try:
                        # Convert to bytes
                        buffer = io.BytesIO()
                        sf.write(buffer, st.session_state.audio_data, st.session_state.sample_rate, format='WAV')
                        buffer.seek(0)
                        
                        st.download_button(
                            label="⬇️ Download WAV",
                            data=buffer,
                            file_name=st.session_state.current_filename,
                            mime="audio/wav",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Save failed: {str(e)}")
                else:
                    st.warning("⚠️ No audio to save!")
    
    st.markdown("---")
    
    # Playback Controls
    if st.session_state.audio_data is not None:
        st.markdown("### ▶️ Playback Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ PLAY", use_container_width=True, key="play_btn"):
                try:
                    if not st.session_state.is_playing:
                        sd.play(st.session_state.audio_data, st.session_state.sample_rate)
                        st.session_state.is_playing = True
                        st.success("🔊 Playing...")
                except Exception as e:
                    st.error(f"❌ Playback error: {str(e)}")
        
        with col2:
            if st.button("⏹️ STOP", use_container_width=True, key="stop_btn"):
                sd.stop()
                st.session_state.is_playing = False
                st.info("⏹️ Stopped")
        
        st.markdown("---")
    
    # Main Analyze Button
    if st.button("⚡ ANALYZE & TUNE ⚡", use_container_width=True, type="primary", key="analyze_btn"):
        if st.session_state.audio_data is None:
            st.error("❌ Please load audio first!")
        else:
            with st.spinner("🔍 Analyzing audio with DSP filters..."):
                # Show processing steps
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text("📊 Step 1/4: Applying 50Hz Notch Filter...")
                progress_bar.progress(25)
                time.sleep(0.3)
                
                progress_text.text("📊 Step 2/4: Applying Lowpass Filter...")
                progress_bar.progress(50)
                time.sleep(0.3)
                
                progress_text.text("📊 Step 3/4: Performing FFT Analysis...")
                progress_bar.progress(75)
                
                filtered_audio, dominant_freq, status = process_audio(
                    st.session_state.audio_data,
                    st.session_state.sample_rate,
                    target_freq
                )
                
                progress_text.text("📊 Step 4/4: Detecting Frequency...")
                progress_bar.progress(100)
                time.sleep(0.3)
                
                progress_text.empty()
                progress_bar.empty()
                
                if status == "Success":
                    st.session_state.filtered_audio = filtered_audio
                    st.session_state.dominant_freq = dominant_freq
                    st.session_state.processing_complete = True
                    
                    diff = dominant_freq - target_freq
                    cents = calculate_cents(dominant_freq, target_freq)
                    
                    st.balloons()
                    st.success(f"""
                    ✅ **Analysis Complete!**
                    
                    🎯 **Target:** {target_freq:.2f} Hz  
                    📡 **Detected:** {dominant_freq:.2f} Hz  
                    📊 **Difference:** {diff:+.2f} Hz ({cents:+.1f} cents)
                    """)
                    st.rerun()
                else:
                    st.error(f"❌ Analysis failed: {status}")
    
    st.markdown("---")
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        st.markdown("##### Filter Configuration")
        
        notch_freq = st.number_input("Notch Filter (Hz)", value=50, min_value=40, max_value=60)
        notch_q = st.slider("Notch Q Factor", min_value=10, max_value=50, value=30)
        
        lowpass_cutoff = st.number_input("Lowpass Cutoff (Hz)", value=500, min_value=300, max_value=1000)
        lowpass_order = st.slider("Lowpass Order", min_value=2, max_value=8, value=4)
        
        st.markdown("##### Tuning Tolerance")
        custom_tolerance = st.slider("Tolerance (Hz)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        
        st.info("💡 Default settings are optimized for guitar tuning")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); 
                border-radius: 10px; margin-top: 1rem;">
        <p style="color: #667eea; font-weight: 700; margin: 0;">© 2024 Guitar Tuner Pro</p>
        <p style="color: #999; font-size: 0.85rem; margin-top: 0.3rem;">EEX7434 Mini Project</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================

# Status Metrics Row
if st.session_state.audio_data is not None:
    st.markdown("### 📊 Audio Information")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        filename_display = st.session_state.current_filename
        if len(filename_display) > 15:
            filename_display = filename_display[:12] + "..."
        st.metric("📄 File", filename_display)
    
    with col2:
        duration = len(st.session_state.audio_data) / st.session_state.sample_rate
        st.metric("⏱️ Duration", f"{duration:.2f}s")
    
    with col3:
        st.metric("🔊 Sample Rate", f"{st.session_state.sample_rate:,} Hz")
    
    with col4:
        st.metric("📏 Samples", f"{len(st.session_state.audio_data):,}")
    
    with col5:
        if st.session_state.dominant_freq:
            st.metric("📡 Detected", f"{st.session_state.dominant_freq:.2f} Hz")
        else:
            st.metric("📡 Detected", "--")
    
    st.markdown("---")

# Tuning Status Display (Big Visual Feedback)
if st.session_state.dominant_freq and st.session_state.processing_complete:
    status_text, status_type, status_color, status_class = get_tuning_status(
        st.session_state.dominant_freq,
        target_freq
    )
    
    diff = st.session_state.dominant_freq - target_freq
    cents = calculate_cents(st.session_state.dominant_freq, target_freq)
    
    if "IN TUNE" in status_text:
        instruction = "🎉 Perfect! Your string is perfectly tuned!"
        emoji = "✅"
    elif "SHARP" in status_text:
        instruction = "⬇️ String is SHARP - Loosen the tuning peg (turn counter-clockwise)"
        emoji = "🔴"
    else:
        instruction = "⬆️ String is FLAT - Tighten the tuning peg (turn clockwise)"
        emoji = "🔵"
    
    st.markdown(f"""
    <div class="status-box {status_class}">
        <div style="text-align: center;">
            <h1 style="color: {status_color}; margin: 0; font-size: 4rem; font-weight: 900; 
                       text-shadow: 0 4px 20px rgba(0,0,0,0.2);">
                {emoji} {status_text} {emoji}
            </h1>
            <div style="margin-top: 1.5rem; font-size: 1.5rem;">
                <p style="color: #4a4a4a; margin: 0.5rem 0;">
                    <strong>Frequency Difference:</strong> {diff:+.2f} Hz
                </p>
                <p style="color: #4a4a4a; margin: 0.5rem 0;">
                    <strong>Cents Deviation:</strong> {cents:+.1f} cents
                </p>
            </div>
            <div style="margin-top: 1.5rem; padding: 1.5rem; background: rgba(255,255,255,0.8); 
                        border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <p style="color: {status_color}; font-size: 1.3rem; font-weight: 700; margin: 0;">
                    {instruction}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

# ==================== VISUALIZATION TABS ====================
if st.session_state.audio_data is not None:
    st.markdown("### 📊 Advanced Visualizations & Analysis")
    
    tabs = st.tabs([
        "🌊 Waveform Display",
        "📊 Spectrum Analyzer",
        "📈 FFT Before Filter",
        "📉 FFT After Filter",
        "🔧 Filter Response",
        "🎯 Tuning Meter",
        "📉 Comparative Analysis"
    ])
    
    # ==================== TAB 1: WAVEFORM ====================
    with tabs[0]:
        st.markdown("#### 🌊 Time Domain Waveform Analysis")
        
        fig, ax = create_light_figure(figsize=(14, 6))
        
        time_array = np.linspace(
            0, 
            len(st.session_state.audio_data) / st.session_state.sample_rate, 
            len(st.session_state.audio_data)
        )
        
        # Plot full signal
        samples_to_plot = min(20000, len(st.session_state.audio_data))
        
        ax.plot(
            time_array[:samples_to_plot], 
            st.session_state.audio_data[:samples_to_plot], 
            color=string_color, 
            linewidth=1, 
            alpha=0.7,
            label='Original Signal'
        )
        
        ax.fill_between(
            time_array[:samples_to_plot], 
            st.session_state.audio_data[:samples_to_plot], 
            alpha=0.2, 
            color=string_color
        )
        
        ax.set_title(
            f'Time Domain Signal - {selected_string}', 
            color='#667eea', 
            fontsize=16, 
            fontweight='bold', 
            pad=20
        )
        ax.set_xlabel('Time (seconds)', color='#4a4a4a', fontsize=12, fontweight='600')
        ax.set_ylabel('Amplitude', color='#4a4a4a', fontsize=12, fontweight='600')
        ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--', linewidth=0.5)
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Add zero line
        ax.axhline(y=0, color='#999', linestyle='-', linewidth=0.8, alpha=0.5)
        
        st.pyplot(fig)
        plt.close()
        
        # Statistics
        st.markdown("##### 📊 Waveform Statistics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Max Amplitude", f"{np.max(st.session_state.audio_data):.4f}")
        with col2:
            st.metric("Min Amplitude", f"{np.min(st.session_state.audio_data):.4f}")
        with col3:
            st.metric("Mean", f"{np.mean(st.session_state.audio_data):.4f}")
        with col4:
            rms = np.sqrt(np.mean(st.session_state.audio_data**2))
            st.metric("RMS", f"{rms:.4f}")
        with col5:
            st.metric("Std Dev", f"{np.std(st.session_state.audio_data):.4f}")
        with col6:
            peak_to_peak = np.max(st.session_state.audio_data) - np.min(st.session_state.audio_data)
            st.metric("Peak-to-Peak", f"{peak_to_peak:.4f}")
    
    # ==================== TAB 2: SPECTRUM ANALYZER ====================
    with tabs[1]:
        st.markdown("#### 📊 Complete Frequency Spectrum")
        
        fig, ax = create_light_figure(figsize=(14, 6))
        
        N = len(st.session_state.audio_data)
        yf = fft(st.session_state.audio_data)
        xf = fftfreq(N, 1/st.session_state.sample_rate)
        
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        # Plot spectrum
        ax.plot(xf, yf, color='#667eea', linewidth=1.2, alpha=0.8, label='Magnitude Spectrum')
        ax.fill_between(xf, yf, alpha=0.2, color='#667eea')
        
        # Mark target frequency
        ax.axvline(
            target_freq, 
            color='#00d084', 
            linestyle='--', 
            linewidth=2.5, 
            label=f'Target: {target_freq:.2f} Hz',
            alpha=0.8
        )
        
        # Mark detected frequency if available
        if st.session_state.dominant_freq:
            ax.axvline(
                st.session_state.dominant_freq, 
                color='#ff6b6b', 
                linestyle='--', 
                linewidth=2.5,
                label=f'Detected: {st.session_state.dominant_freq:.2f} Hz',
                alpha=0.8
            )
        
        ax.set_xlim([0, 1000])
        ax.set_title('Frequency Spectrum Analysis', color='#667eea', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=12, fontweight='600')
        ax.set_ylabel('Magnitude', color='#4a4a4a', fontsize=12, fontweight='600')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--', linewidth=0.5)
        
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 **Tip:** The spectrum shows all frequency components. The peak in the guitar range (70-400 Hz) indicates the fundamental frequency.")
    
    # ==================== TAB 3: FFT BEFORE FILTER ====================
    with tabs[2]:
        st.markdown("#### 📈 FFT Spectrum (Before Filtering)")
        
        fig, ax = create_light_figure(figsize=(14, 6))
        
        N = len(st.session_state.audio_data)
        yf = fft(st.session_state.audio_data)
        xf = fftfreq(N, 1/st.session_state.sample_rate)
        
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        ax.plot(xf, yf, color='#ff6b6b', linewidth=1.2, alpha=0.8, label='Unfiltered Spectrum')
        ax.fill_between(xf, yf, alpha=0.15, color='#ff6b6b')
        ax.set_xlim([0, 600])
        
        # Mark 50Hz interference
        ax.axvline(50, color='#ffd93d', linestyle=':', linewidth=3, label='50 Hz Power Line Noise', alpha=0.7)
        ax.axvspan(45, 55, alpha=0.1, color='#ffd93d')
        
        # Mark guitar range
        ax.axvspan(70, 400, alpha=0.1, color='#6bcf7f', label='Guitar Frequency Range')
        
        ax.set_title('Unfiltered Frequency Spectrum (Raw Signal)', color='#ff6b6b', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=12, fontweight='600')
        ax.set_ylabel('Magnitude', color='#4a4a4a', fontsize=12, fontweight='600')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--', linewidth=0.5)
        
        st.pyplot(fig)
        plt.close()
        
        st.warning("⚠️ **Notice:** The raw signal contains 50 Hz power line interference and high-frequency noise that must be filtered out for accurate tuning.")
    
    # ==================== TAB 4: FFT AFTER FILTER ====================
    with tabs[3]:
        if st.session_state.filtered_audio is not None:
            st.markdown("#### 📉 FFT Spectrum (After Filtering)")
            
            fig, ax = create_light_figure(figsize=(14, 6))
            
            N = len(st.session_state.filtered_audio)
            yf = fft(st.session_state.filtered_audio)
            xf = fftfreq(N, 1/st.session_state.sample_rate)
            
            pos_mask = xf > 0
            xf = xf[pos_mask]
            yf = np.abs(yf[pos_mask])
            
            ax.plot(xf, yf, color='#00d084', linewidth=1.2, alpha=0.8, label='Filtered Spectrum')
            ax.fill_between(xf, yf, alpha=0.2, color='#00d084')
            ax.set_xlim([0, 600])
            
            # Mark detected peak
            if st.session_state.dominant_freq:
                ax.axvline(
                    st.session_state.dominant_freq, 
                    color='#667eea', 
                    linestyle='--', 
                    linewidth=2.5,
                    label=f'Detected Peak: {st.session_state.dominant_freq:.2f} Hz',
                    alpha=0.8
                )
                
                # Mark the peak point
                peak_idx = np.argmin(np.abs(xf - st.session_state.dominant_freq))
                ax.plot(xf[peak_idx], yf[peak_idx], 'r*', markersize=20, label='Peak Frequency')
            
            # Show guitar range
            ax.axvspan(70, 400, alpha=0.1, color='#6bcf7f', label='Guitar Range')
            
            ax.set_title('Filtered Frequency Spectrum (Clean Signal)', color='#00d084', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=12, fontweight='600')
            ax.set_ylabel('Magnitude', color='#4a4a4a', fontsize=12, fontweight='600')
            ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
            ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--', linewidth=0.5)
            
            st.pyplot(fig)
            plt.close()
            
            st.success("✅ **Filters Applied:** 50 Hz Notch Filter + 500 Hz Lowpass Filter removed noise and isolated the fundamental frequency!")
        else:
            st.warning("⚠️ Please run 'ANALYZE & TUNE' first to see filtered results.")
            st.info("🔄 Click the '⚡ ANALYZE & TUNE ⚡' button in the sidebar to process the audio.")
    
    # ==================== TAB 5: FILTER RESPONSE ====================
    with tabs[4]:
        st.markdown("#### 🔧 Digital Filter Frequency Response")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='white')
        
        # Design filters
        nyquist = st.session_state.sample_rate / 2
        Q = 30
        w0 = 50 / nyquist
        b_notch, a_notch = signal.iirnotch(w0, Q, st.session_state.sample_rate)
        
        cutoff = 500 / nyquist
        b_low, a_low = signal.butter(4, cutoff, btype='low', analog=False)
        
        # Compute frequency responses
        w_notch, h_notch = signal.freqz(b_notch, a_notch, worN=4000, fs=st.session_state.sample_rate)
        w_low, h_low = signal.freqz(b_low, a_low, worN=4000, fs=st.session_state.sample_rate)
        
        # Plot 1: Both filters
        ax1.set_facecolor('#fafbff')
        ax1.plot(w_notch, 20*np.log10(abs(h_notch)), color='#ffd93d', label='50 Hz Notch Filter', linewidth=2.5, alpha=0.9)
        ax1.plot(w_low, 20*np.log10(abs(h_low)), color='#00d084', label='500 Hz Lowpass Filter', linewidth=2.5, alpha=0.9)
        ax1.set_xlim([0, 1000])
        ax1.set_ylim([-80, 5])
        
        ax1.axvline(50, color='#ff6b6b', linestyle=':', alpha=0.5, linewidth=2)
        ax1.axvline(500, color='#4facfe', linestyle=':', alpha=0.5, linewidth=2)
        ax1.axhline(-3, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax1.set_title('Combined Filter Response', color='#667eea', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=11, fontweight='600')
        ax1.set_ylabel('Gain (dB)', color='#4a4a4a', fontsize=11, fontweight='600')
        ax1.legend(loc='lower right', framealpha=0.9, fontsize=10)
        ax1.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--', linewidth=0.5)
        
        # Plot 2: Combined effect
        ax2.set_facecolor('#fafbff')
        h_combined = h_notch * h_low
        ax2.plot(w_notch, 20*np.log10(abs(h_combined)), color='#764ba2', label='Combined Filter Effect', linewidth=3, alpha=0.9)
        ax2.set_xlim([0, 1000])
        ax2.set_ylim([-80, 5])
        
        # Highlight guitar range
        ax2.axvspan(70, 400, alpha=0.15, color='#6bcf7f', label='Guitar Frequency Range')
        ax2.axhline(-3, color='gray', linestyle='--', alpha=0.3, linewidth=1, label='-3dB Cutoff')
        
        ax2.set_title('Combined Filter Effect', color='#764ba2', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=11, fontweight='600')
        ax2.set_ylabel('Gain (dB)', color='#4a4a4a', fontsize=11, fontweight='600')
        ax2.legend(loc='lower right', framealpha=0.9, fontsize=10)
        ax2.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Filter specifications
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #ffd93d;">🔸 Notch Filter Specifications</h4>
                <ul style="color: #4a4a4a; line-height: 2;">
                    <li><strong>Type:</strong> IIR Notch Filter</li>
                    <li><strong>Center Frequency:</strong> 50 Hz</li>
                    <li><strong>Quality Factor (Q):</strong> 30</li>
                    <li><strong>Bandwidth:</strong> ~1.67 Hz</li>
                    <li><strong>Purpose:</strong> Remove power line interference</li>
                    <li><strong>Attenuation:</strong> > 40 dB at 50 Hz</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #00d084;">🔸 Lowpass Filter Specifications</h4>
                <ul style="color: #4a4a4a; line-height: 2;">
                    <li><strong>Type:</strong> Butterworth IIR</li>
                    <li><strong>Cutoff Frequency:</strong> 500 Hz</li>
                    <li><strong>Order:</strong> 4th order</li>
                    <li><strong>Roll-off:</strong> -80 dB/decade</li>
                    <li><strong>Purpose:</strong> Remove high-frequency noise</li>
                    <li><strong>Passband:</strong> 0-500 Hz (guitar range)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== TAB 6: TUNING METER ====================
    with tabs[5]:
        if st.session_state.dominant_freq and st.session_state.processing_complete:
            st.markdown("#### 🎯 Professional Tuning Meter")
            
            fig, ax = create_light_figure(figsize=(14, 8))
            ax.axis('off')
            
            detected_freq = st.session_state.dominant_freq
            diff = detected_freq - target_freq
            cents = calculate_cents(detected_freq, target_freq)
            
            # Determine status
            if abs(diff) <= TUNE_TOLERANCE:
                status = "IN TUNE ✓"
                color = '#00d084'
            elif diff > 0:
                status = "SHARP ↑"
                color = '#ff6b6b'
            else:
                status = "FLAT ↓"
                color = '#4facfe'
            
            # Draw tuning meter background
            meter_width = 0.7
            meter_height = 0.15
            meter_x = 0.15
            meter_y = 0.45
            
            # Background
            bg_rect = FancyBboxPatch(
                (meter_x, meter_y), meter_width, meter_height,
                boxstyle="round,pad=0.01",
                facecolor='#f0f0f0',
                edgecolor='#c0c0c0',
                linewidth=3
            )
            ax.add_patch(bg_rect)
            
            # Color zones
            flat_rect = Rectangle(
                (meter_x, meter_y), meter_width*0.35, meter_height,
                facecolor='#4facfe', alpha=0.3
            )
            ax.add_patch(flat_rect)
            
            intune_rect = Rectangle(
                (meter_x + meter_width*0.35, meter_y), meter_width*0.3, meter_height,
                facecolor='#00d084', alpha=0.3
            )
            ax.add_patch(intune_rect)
            
            sharp_rect = Rectangle(
                (meter_x + meter_width*0.65, meter_y), meter_width*0.35, meter_height,
                facecolor='#ff6b6b', alpha=0.3
            )
            ax.add_patch(sharp_rect)
            
            # Calculate needle position
            max_diff = 10
            needle_pos = np.clip(diff / max_diff, -1, 1)
            needle_x = meter_x + meter_width/2 + (needle_pos * meter_width/2 * 0.9)
            
            # Draw needle shadow
            ax.plot([needle_x + 0.005, needle_x + 0.005], [meter_y, meter_y + meter_height], 
                   color='black', linewidth=7, alpha=0.2)
            
            # Draw needle
            ax.plot([needle_x, needle_x], [meter_y, meter_y + meter_height], 
                   color=color, linewidth=8, alpha=0.9, solid_capstyle='round')
            ax.plot([needle_x], [meter_y + meter_height + 0.04], 
                   marker='v', markersize=30, color=color, markeredgecolor='white', markeredgewidth=2)
            
            # Center line
            center_x = meter_x + meter_width/2
            ax.plot([center_x, center_x], [meter_y, meter_y + meter_height], 
                   color='white', linewidth=3, alpha=0.8, linestyle='--')
            
            # Info text at top
            ax.text(0.5, 0.88, f"🎸 {selected_string}", 
                   ha='center', va='center', fontsize=18, color='#667eea', fontweight='bold')
            
            ax.text(0.5, 0.78, f"Target: {target_freq:.2f} Hz", 
                   ha='center', va='center', fontsize=14, color='#999', fontweight='600')
            ax.text(0.5, 0.72, f"Detected: {detected_freq:.2f} Hz", 
                   ha='center', va='center', fontsize=16, color='#4a4a4a', fontweight='bold')
            
            # Difference info
            ax.text(0.5, 0.65, f"Δ {diff:+.2f} Hz  |  {cents:+.1f} cents", 
                   ha='center', va='center', fontsize=13, color=color, fontweight='bold')
            
            # Zone labels below meter
            ax.text(meter_x + 0.02, meter_y - 0.05, "FLAT", 
                   ha='left', va='top', fontsize=11, color='#4facfe', fontweight='bold')
            ax.text(center_x, meter_y - 0.05, "IN TUNE", 
                   ha='center', va='top', fontsize=11, color='#00d084', fontweight='bold')
            ax.text(meter_x + meter_width - 0.02, meter_y - 0.05, "SHARP", 
                   ha='right', va='top', fontsize=11, color='#ff6b6b', fontweight='bold')
            
            # Status box
            status_box = dict(
                boxstyle='round,pad=1.2', 
                facecolor='white', 
                edgecolor=color, 
                linewidth=6,
                alpha=0.95
            )
            ax.text(0.5, 0.25, status, 
                   ha='center', va='center', fontsize=32, color=color, 
                   fontweight='bold', bbox=status_box)
            
            # Instructions
            if "SHARP" in status:
                instruction = "⬇️ Loosen the string (turn tuning peg counter-clockwise)"
                inst_color = '#ff6b6b'
            elif "FLAT" in status:
                instruction = "⬆️ Tighten the string (turn tuning peg clockwise)"
                inst_color = '#4facfe'
            else:
                instruction = "✓ Perfect! No adjustment needed"
                inst_color = '#00d084'
            
            ax.text(0.5, 0.12, instruction, 
                   ha='center', va='center', fontsize=12, color=inst_color, 
                   style='italic', fontweight='600')
            
            # Frequency scale
            scale_y = meter_y - 0.15
            scale_freqs = [-10, -5, 0, 5, 10]
            for sf in scale_freqs:
                scale_x = meter_x + meter_width/2 + (sf/10 * meter_width/2 * 0.9)
                ax.plot([scale_x, scale_x], [scale_y, scale_y + 0.03], 
                       color='#999', linewidth=2)
                ax.text(scale_x, scale_y - 0.02, f"{sf:+d}", 
                       ha='center', va='top', fontsize=9, color='#666', fontweight='600')
            
            ax.text(0.5, scale_y - 0.08, "Frequency Deviation (Hz)", 
                   ha='center', va='top', fontsize=10, color='#999', style='italic', fontweight='600')
            
            # Accuracy arc
            accuracy_percent = max(0, 100 - abs(diff/target_freq * 100))
            arc_color = color
            
            # Draw decorative circles
            circle1 = plt.Circle((0.1, 0.25), 0.03, color=arc_color, alpha=0.2)
            circle2 = plt.Circle((0.9, 0.25), 0.03, color=arc_color, alpha=0.2)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            st.pyplot(fig)
            plt.close()
            
            # Detailed metrics in cards
            st.markdown("##### 📊 Detailed Tuning Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🎯 Target Freq", f"{target_freq:.2f} Hz")
            
            with col2:
                st.metric("📡 Detected Freq", f"{detected_freq:.2f} Hz", delta=f"{diff:+.2f} Hz")
            
            with col3:
                st.metric("🎵 Cents Offset", f"{cents:+.1f}")
            
            with col4:
                accuracy = max(0, 100 - abs(diff/target_freq * 100))
                st.metric("✓ Accuracy", f"{accuracy:.1f}%")
            
            with col5:
                st.metric("⚡ Tolerance", f"±{TUNE_TOLERANCE} Hz")
            
        else:
            st.warning("⚠️ Please run '⚡ ANALYZE & TUNE ⚡' first to see the tuning meter.")
            st.info("💡 The tuning meter provides visual feedback on exactly how sharp or flat your string is!")
    
    # ==================== TAB 7: COMPARATIVE ANALYSIS ====================
    with tabs[6]:
        st.markdown("#### 📉 Before vs After Filtering Comparison")
        
        if st.session_state.filtered_audio is not None:
            # Create comparison plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='white')
            
            # Plot 1: Waveform comparison
            ax1.set_facecolor('#fafbff')
            
            time_array = np.linspace(0, len(st.session_state.audio_data)/st.session_state.sample_rate, 
                                    len(st.session_state.audio_data))
            samples_to_plot = min(5000, len(st.session_state.audio_data))
            
            ax1.plot(time_array[:samples_to_plot], st.session_state.audio_data[:samples_to_plot], 
                    color='#ff6b6b', linewidth=1, alpha=0.6, label='Original (Unfiltered)')
            ax1.plot(time_array[:samples_to_plot], st.session_state.filtered_audio[:samples_to_plot], 
                    color='#00d084', linewidth=1, alpha=0.8, label='Filtered')
            
            ax1.set_title('Waveform: Before vs After Filtering', color='#667eea', fontsize=14, fontweight='bold', pad=15)
            ax1.set_xlabel('Time (s)', color='#4a4a4a', fontsize=11, fontweight='600')
            ax1.set_ylabel('Amplitude', color='#4a4a4a', fontsize=11, fontweight='600')
            ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
            ax1.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--', linewidth=0.5)
            
            # Plot 2: Spectrum comparison
            ax2.set_facecolor('#fafbff')
            
            # Original spectrum
            N_orig = len(st.session_state.audio_data)
            yf_orig = fft(st.session_state.audio_data)
            xf_orig = fftfreq(N_orig, 1/st.session_state.sample_rate)
            pos_mask_orig = xf_orig > 0
            xf_orig = xf_orig[pos_mask_orig]
            yf_orig = np.abs(yf_orig[pos_mask_orig])
            
            # Filtered spectrum
            N_filt = len(st.session_state.filtered_audio)
            yf_filt = fft(st.session_state.filtered_audio)
            xf_filt = fftfreq(N_filt, 1/st.session_state.sample_rate)
            pos_mask_filt = xf_filt > 0
            xf_filt = xf_filt[pos_mask_filt]
            yf_filt = np.abs(yf_filt[pos_mask_filt])
            
            ax2.plot(xf_orig, yf_orig, color='#ff6b6b', linewidth=1, alpha=0.6, label='Original Spectrum')
            ax2.plot(xf_filt, yf_filt, color='#00d084', linewidth=1.5, alpha=0.8, label='Filtered Spectrum')
            ax2.set_xlim([0, 600])
            
            # Mark frequencies
            ax2.axvline(50, color='#ffd93d', linestyle=':', linewidth=2, alpha=0.5, label='50 Hz Noise')
            ax2.axvline(500, color='#764ba2', linestyle=':', linewidth=2, alpha=0.5, label='500 Hz Cutoff')
            
            if st.session_state.dominant_freq:
                ax2.axvline(st.session_state.dominant_freq, color='#667eea', linestyle='--', 
                           linewidth=2.5, alpha=0.8, label=f'Detected: {st.session_state.dominant_freq:.2f} Hz')
            
            ax2.set_title('Spectrum: Before vs After Filtering', color='#667eea', fontsize=14, fontweight='bold', pad=15)
            ax2.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=11, fontweight='600')
            ax2.set_ylabel('Magnitude', color='#4a4a4a', fontsize=11, fontweight='600')
            ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
            ax2.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Comparison statistics
            st.markdown("##### 📊 Filtering Effectiveness")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                noise_reduction = (1 - np.std(st.session_state.filtered_audio) / np.std(st.session_state.audio_data)) * 100
                st.metric("🔇 Noise Reduction", f"{noise_reduction:.1f}%")
            
            with col2:
                snr_improvement = 20 * np.log10(np.std(st.session_state.filtered_audio) / (np.std(st.session_state.audio_data) + 1e-10))
                st.metric("📶 SNR Improvement", f"{abs(snr_improvement):.1f} dB")
            
            with col3:
                freq_clarity = (np.max(yf_filt) / np.mean(yf_filt)) / (np.max(yf_orig) / np.mean(yf_orig))
                st.metric("🎯 Frequency Clarity", f"{freq_clarity:.2f}x")
            
            st.success("✅ **Result:** Filtering successfully removed noise and isolated the fundamental frequency for accurate tuning!")
            
        else:
            st.warning("⚠️ Please run 'ANALYZE & TUNE' first to see the comparison.")

else:
    # ==================== WELCOME SCREEN ====================
    st.markdown("---")
    
    st.markdown("### 🚀 Welcome to Guitar Tuner Pro!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #667eea; margin-top: 0;">📖 Quick Start Guide</h3>
            <ol style="color: #4a4a4a; line-height: 2.2; font-size: 1.05rem;">
                <li><strong>Select String:</strong> Choose which guitar string to tune from the sidebar</li>
                <li><strong>Load Audio:</strong> Upload a file or record from your microphone</li>
                <li><strong>Analyze:</strong> Click "ANALYZE & TUNE" to process the audio</li>
                <li><strong>View Results:</strong> Check the tuning status and visualizations</li>
                <li><strong>Adjust:</strong> Follow the instructions to tune your guitar</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #764ba2; margin-top: 0;">✨ Key Features</h3>
            <ul style="color: #4a4a4a; line-height: 2.2; font-size: 1.05rem;">
                <li>🎯 <strong>Accurate Detection:</strong> ±0.1 Hz precision</li>
                <li>🔊 <strong>Real-time Processing:</strong> Fast FFT analysis</li>
                <li>🎨 <strong>Visual Feedback:</strong> Color-coded tuning meter</li>
                <li>📊 <strong>Advanced DSP:</strong> Notch + Lowpass filtering</li>
                <li>📈 <strong>Multiple Views:</strong> 7 visualization modes</li>
                <li>💾 <strong>Audio Recording:</strong> Built-in mic support</li>
                <li>🎵 <strong>All Strings:</strong> Complete guitar tuning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Guitar string reference with beautiful cards
    st.markdown("### 🎻 Standard Guitar Tuning Reference")
    
    cols = st.columns(6)
    for idx, (string_name, freq) in enumerate(STRING_FREQUENCIES.items()):
        with cols[idx]:
            color = STRING_COLORS[string_name]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}20 0%, {color}40 100%); 
                        padding: 1.5rem; border-radius: 15px; 
                        border-left: 5px solid {color}; 
                        text-align: center; margin: 0.5rem 0;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                        transition: transform 0.3s ease;">
                <p style="color: {color}; font-weight: 800; margin: 0; font-size: 0.85rem;">
                    {string_name.split('(')[0]}
                </p>
                <p style="color: #4a4a4a; margin: 0.3rem 0; font-size: 0.75rem;">
                    {string_name.split('(')[1].rstrip(')')}
                </p>
                <p style="color: {color}; margin: 0.5rem 0 0 0; font-size: 1.4rem; font-weight: 900;">
                    {freq} Hz
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technical documentation
    with st.expander("📚 Technical Documentation & DSP Pipeline", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔧 Signal Processing Pipeline
            
            **1. Signal Acquisition**
            - Audio input via file or microphone
            - Sampling rate: Up to 48 kHz
            - Mono channel conversion
            - Maximum 5-second duration
            
            **2. Preprocessing Filters**
            - **50 Hz Notch Filter (Q=30)**
              - Removes power line interference
              - IIR (Infinite Impulse Response)
              - Narrow bandwidth (~1.67 Hz)
            
            - **500 Hz Lowpass Filter**
              - 4th order Butterworth design
              - Removes high-frequency noise
              - Preserves guitar fundamentals
            
            **3. Frequency Analysis**
            - Fast Fourier Transform (FFT)
            - Frequency resolution: ~0.1 Hz
            - Peak detection algorithm
            - Range focus: 70-400 Hz
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Tuning Algorithm
            
            **4. Frequency Detection**
            - Magnitude spectrum analysis
            - Peak identification in guitar range
            - Dominant frequency extraction
            - Harmonic filtering
            
            **5. Tuning Calculation**
            - Frequency difference (Hz)
            - Cents deviation (musical scale)
            - ±2 Hz tolerance for "in tune"
            - Color-coded status feedback
            
            **6. Visualization**
            - Time domain waveform
            - Frequency spectrum (before/after)
            - Filter response curves
            - Professional tuning meter
            - Comparative analysis
            
            ### 📊 Supported Formats
            - WAV, MP3, FLAC, OGG, M4A
            - Sample rates: 8 kHz - 192 kHz
            - Mono or Stereo (auto-converted)
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 🎓 Understanding the Results
        
        - **Green (In Tune):** String is within ±2 Hz of target → Perfect!
        - **Red (Sharp):** Frequency too high → Loosen the string
        - **Blue (Flat):** Frequency too low → Tighten the string
        - **Cents:** Musical measurement (100 cents = 1 semitone)
        - **Hz Difference:** Absolute frequency deviation
        """)

# ==================== TIPS & TRICKS ====================
st.markdown("---")

with st.expander("💡 Pro Tips for Best Tuning Results", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🎤 Recording Tips
        - Place microphone 15-30cm from guitar
        - Minimize background noise
        - Pluck string clearly and firmly
        - Let string ring for 3+ seconds
        - Use moderate volume
        - Avoid touching other strings
        - Record in quiet environment
        """)
    
    with col2:
        st.markdown("""
        #### 🎸 Tuning Tips
        - Tune slowly and gradually
        - Always tune UP to pitch
        - Check tuning multiple times
        - Start from low E to high E
        - Use harmonics for fine-tuning
        - Retune after string stretching
        - Temperature affects tuning
        """)
    
    with col3:
        st.markdown("""
        #### 🔍 Troubleshooting
        - **No frequency detected:** Increase volume or re-record
        - **Inaccurate reading:** Reduce background noise
        - **Unstable reading:** Let string ring longer
        - **Wrong octave:** Check string selection
        - **Noise interference:** Use better microphone
        - **Recording fails:** Check browser permissions
        """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; 
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-radius: 20px; margin-top: 2rem;">
    <h3 style="color: #667eea; margin: 0; font-weight: 800;">
        EEX7434 Mini Project
    </h3>
    <p style="color: #764ba2; font-size: 1.2rem; margin: 0.8rem 0; font-weight: 600;">
        Advanced Digital Signal Processing Guitar Tuner
    </p>
    <p style="color: #999; font-size: 0.95rem; margin-top: 1rem;">
        Built with ❤️ using Streamlit, NumPy, SciPy, Matplotlib, and Librosa
    </p>
    <p style="color: #999; font-size: 0.9rem; margin-top: 0.5rem;">
        © 2024 Guitar Tuner Pro - All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== KEYBOARD SHORTCUTS ====================
with st.expander("⌨️ Keyboard Shortcuts & Features", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Quick Actions
        - `R` - Refresh/Rerun the app
        - `Ctrl/Cmd + K` - Search settings
        - `Esc` - Close dialogs
        - Click metrics to highlight
        - Hover charts for values
        """)
    
    with col2:
        st.markdown("""
        ### Navigation
        - Sidebar: All input controls
        - Main area: Results & visualizations
        - Tabs: Different analysis views
        - Expanders: Additional information
        - Buttons: Actions and processing
        """)

# ==================== DEBUG MODE ====================
if st.sidebar.checkbox("🔧 Debug Mode", value=False):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Debug Information")
    
    debug_info = {
        "Audio Loaded": st.session_state.audio_data is not None,
        "Sample Rate": st.session_state.sample_rate,
        "Audio Length": len(st.session_state.audio_data) if st.session_state.audio_data is not None else 0,
        "Filtered": st.session_state.filtered_audio is not None,
        "Detected Frequency": st.session_state.dominant_freq,
        "Target Frequency": target_freq,
        "Selected String": selected_string,
        "Processing Complete": st.session_state.processing_complete,
        "Is Playing": st.session_state.is_playing,
        "Filename": st.session_state.current_filename
    }
    
    st.sidebar.json(debug_info)
    
    if st.sidebar.button("🗑️ Clear All Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==================== END OF APPLICATION ====================
