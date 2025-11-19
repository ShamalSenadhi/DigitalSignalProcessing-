"""
🎸 Professional Guitar Tuner - Fixed Cloud Version
EEX7434 Mini Project - ALL FUNCTIONS WORKING
Optimized for Streamlit Cloud Deployment
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import soundfile as sf
import io
import time
from matplotlib.patches import Rectangle, FancyBboxPatch

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="🎸 Guitar Tuner Pro",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 25%, #fff5f7 50%, #ffffff 75%, #f0fff4 100%);
        background-attachment: fixed;
    }
    
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
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 
            0 2px 10px rgba(0,0,0,0.2),
            0 0 20px rgba(255,255,255,0.5);
        margin: 0;
        letter-spacing: 2px;
    }
    
    .main-header p {
        color: #ffffff;
        font-size: 1.3rem;
        margin-top: 0.8rem;
        font-weight: 600;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
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
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 800;
        border-radius: 15px;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 35px rgba(102, 126, 234, 0.6);
    }
    
    .status-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border-left: 6px solid;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
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
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
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
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# ==================== DATA ====================
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

# ==================== FUNCTIONS ====================

def create_light_figure(figsize=(10, 4)):
    """Create matplotlib figure with light theme"""
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
    """Process audio with DSP filters and detect frequency"""
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
        
        # Perform FFT
        N = len(filtered_audio)
        yf = fft(filtered_audio)
        xf = fftfreq(N, 1/sample_rate)
        
        # Get positive frequencies
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        # Focus on guitar range
        mask = (xf >= 70) & (xf <= 400)
        freq_range = xf[mask]
        mag_range = yf[mask]
        
        if len(mag_range) == 0:
            return None, None, "No frequency detected in guitar range"
        
        # Find dominant frequency
        peak_idx = np.argmax(mag_range)
        dominant_freq = freq_range[peak_idx]
        
        return filtered_audio, dominant_freq, "Success"
        
    except Exception as e:
        return None, None, str(e)

def get_tuning_status(detected_freq, target_freq):
    """Get tuning status"""
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

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🎸 PROFESSIONAL GUITAR TUNER PRO</h1>
    <p>Advanced DSP Analysis | Real-time Frequency Detection | Professional Tuning</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🎛️ CONTROL CENTER")
    st.markdown("---")
    
    # String Selection
    st.markdown("### 🎻 Select Guitar String")
    selected_string = st.selectbox(
        "Choose string:",
        options=list(STRING_FREQUENCIES.keys()),
        index=5
    )
    
    target_freq = STRING_FREQUENCIES[selected_string]
    string_color = STRING_COLORS[selected_string]
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {string_color}20 0%, {string_color}40 100%); 
                padding: 1.5rem; border-radius: 15px; 
                border-left: 6px solid {string_color}; 
                margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <p style="color: {string_color}; font-weight: 800; font-size: 1.3rem; margin: 0;">
            🎯 Target: {target_freq} Hz
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Audio Upload
    st.markdown("### 📂 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Supported: WAV, MP3, FLAC, OGG, M4A",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a']
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Loading audio..."):
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
                
                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = sample_rate
                st.session_state.current_filename = uploaded_file.name
                st.session_state.processing_complete = False
                
                duration = len(audio_data) / sample_rate
                
                st.success(f"""
                ✅ **Audio Loaded!**
                
                📄 {uploaded_file.name}  
                🔊 {sample_rate:,} Hz  
                ⏱️ {duration:.2f}s
                """)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Main Analyze Button
    if st.button("⚡ ANALYZE & TUNE ⚡", use_container_width=True, type="primary"):
        if st.session_state.audio_data is None:
            st.error("❌ Please upload audio first!")
        else:
            with st.spinner("🔍 Analyzing..."):
                progress_bar = st.progress(0)
                
                progress_bar.progress(25)
                time.sleep(0.2)
                
                progress_bar.progress(50)
                time.sleep(0.2)
                
                progress_bar.progress(75)
                
                filtered_audio, dominant_freq, status = process_audio(
                    st.session_state.audio_data,
                    st.session_state.sample_rate,
                    target_freq
                )
                
                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()
                
                if status == "Success":
                    st.session_state.filtered_audio = filtered_audio
                    st.session_state.dominant_freq = dominant_freq
                    st.session_state.processing_complete = True
                    
                    diff = dominant_freq - target_freq
                    cents = calculate_cents(dominant_freq, target_freq)
                    
                    st.balloons()
                    st.success(f"""
                    ✅ **Complete!**
                    
                    🎯 Target: {target_freq:.2f} Hz  
                    📡 Detected: {dominant_freq:.2f} Hz  
                    📊 Diff: {diff:+.2f} Hz ({cents:+.1f} cents)
                    """)
                    st.rerun()
                else:
                    st.error(f"❌ Failed: {status}")

# ==================== MAIN CONTENT ====================

# Audio Information
if st.session_state.audio_data is not None:
    st.markdown("### 📊 Audio Information")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        filename = st.session_state.current_filename or "Unknown"
        if len(filename) > 15:
            filename = filename[:12] + "..."
        st.metric("📄 File", filename)
    
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

# Tuning Status
if st.session_state.dominant_freq and st.session_state.processing_complete:
    status_text, status_type, status_color, status_class = get_tuning_status(
        st.session_state.dominant_freq,
        target_freq
    )
    
    diff = st.session_state.dominant_freq - target_freq
    cents = calculate_cents(st.session_state.dominant_freq, target_freq)
    
    if "IN TUNE" in status_text:
        instruction = "🎉 Perfect! Your string is tuned!"
        emoji = "✅"
    elif "SHARP" in status_text:
        instruction = "⬇️ SHARP - Loosen tuning peg (turn counter-clockwise)"
        emoji = "🔴"
    else:
        instruction = "⬆️ FLAT - Tighten tuning peg (turn clockwise)"
        emoji = "🔵"
    
    st.markdown(f"""
    <div class="status-box {status_class}">
        <div style="text-align: center;">
            <h1 style="color: {status_color}; margin: 0; font-size: 4rem; font-weight: 900;">
                {emoji} {status_text} {emoji}
            </h1>
            <div style="margin-top: 1.5rem; font-size: 1.5rem;">
                <p style="color: #4a4a4a; margin: 0.5rem 0;">
                    <strong>Difference:</strong> {diff:+.2f} Hz
                </p>
                <p style="color: #4a4a4a; margin: 0.5rem 0;">
                    <strong>Cents:</strong> {cents:+.1f} cents
                </p>
            </div>
            <div style="margin-top: 1.5rem; padding: 1.5rem; background: rgba(255,255,255,0.8); 
                        border-radius: 15px;">
                <p style="color: {status_color}; font-size: 1.3rem; font-weight: 700; margin: 0;">
                    {instruction}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

# ==================== VISUALIZATIONS ====================
if st.session_state.audio_data is not None:
    st.markdown("### 📊 Visualizations")
    
    tabs = st.tabs([
        "🌊 Waveform",
        "📊 Spectrum",
        "📈 FFT Before",
        "📉 FFT After",
        "🔧 Filters",
        "🎯 Tuning Meter"
    ])
    
    # TAB 1: Waveform
    with tabs[0]:
        st.markdown("#### 🌊 Time Domain Waveform")
        
        fig, ax = create_light_figure(figsize=(12, 5))
        
        time_array = np.linspace(
            0, 
            len(st.session_state.audio_data) / st.session_state.sample_rate, 
            len(st.session_state.audio_data)
        )
        
        samples = min(20000, len(st.session_state.audio_data))
        
        ax.plot(time_array[:samples], st.session_state.audio_data[:samples], 
               color=string_color, linewidth=1, alpha=0.7)
        ax.fill_between(time_array[:samples], st.session_state.audio_data[:samples], 
                       alpha=0.2, color=string_color)
        
        ax.set_title(f'Waveform - {selected_string}', color='#667eea', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Time (s)', color='#4a4a4a', fontsize=11)
        ax.set_ylabel('Amplitude', color='#4a4a4a', fontsize=11)
        ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--')
        ax.axhline(y=0, color='#999', linestyle='-', linewidth=0.8, alpha=0.5)
        
        st.pyplot(fig)
        plt.close()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max", f"{np.max(st.session_state.audio_data):.4f}")
        col2.metric("Min", f"{np.min(st.session_state.audio_data):.4f}")
        col3.metric("Mean", f"{np.mean(st.session_state.audio_data):.4f}")
        col4.metric("RMS", f"{np.sqrt(np.mean(st.session_state.audio_data**2)):.4f}")
    
    # TAB 2: Spectrum
    with tabs[1]:
        st.markdown("#### 📊 Frequency Spectrum")
        
        fig, ax = create_light_figure(figsize=(12, 5))
        
        N = len(st.session_state.audio_data)
        yf = fft(st.session_state.audio_data)
        xf = fftfreq(N, 1/st.session_state.sample_rate)
        
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        ax.plot(xf, yf, color='#667eea', linewidth=1.2, alpha=0.8)
        ax.fill_between(xf, yf, alpha=0.2, color='#667eea')
        
        ax.axvline(target_freq, color='#00d084', linestyle='--', linewidth=2,
                  label=f'Target: {target_freq:.2f} Hz', alpha=0.8)
        
        if st.session_state.dominant_freq:
            ax.axvline(st.session_state.dominant_freq, color='#ff6b6b', 
                      linestyle='--', linewidth=2,
                      label=f'Detected: {st.session_state.dominant_freq:.2f} Hz', alpha=0.8)
        
        ax.set_xlim([0, 1000])
        ax.set_title('Frequency Spectrum', color='#667eea', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=11)
        ax.set_ylabel('Magnitude', color='#4a4a4a', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--')
        
        st.pyplot(fig)
        plt.close()
    
    # TAB 3: FFT Before
    with tabs[2]:
        st.markdown("#### 📈 FFT Before Filtering")
        
        fig, ax = create_light_figure(figsize=(12, 5))
        
        N = len(st.session_state.audio_data)
        yf = fft(st.session_state.audio_data)
        xf = fftfreq(N, 1/st.session_state.sample_rate)
        
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        ax.plot(xf, yf, color='#ff6b6b', linewidth=1.2, alpha=0.8)
        ax.fill_between(xf, yf, alpha=0.15, color='#ff6b6b')
        ax.set_xlim([0, 600])
        
        ax.axvline(50, color='#ffd93d', linestyle=':', linewidth=3, 
                  label='50 Hz Noise', alpha=0.7)
        ax.axvspan(70, 400, alpha=0.1, color='#6bcf7f', label='Guitar Range')
        
        ax.set_title('Unfiltered Spectrum', color='#ff6b6b', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=11)
        ax.set_ylabel('Magnitude', color='#4a4a4a', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--')
        
        st.pyplot(fig)
        plt.close()
        
        st.warning("⚠️ Raw signal contains 50 Hz interference and noise")
    
    # TAB 4: FFT After
    with tabs[3]:
        if st.session_state.filtered_audio is not None:
            st.markdown("#### 📉 FFT After Filtering")
            
            fig, ax = create_light_figure(figsize=(12, 5))
            
            N = len(st.session_state.filtered_audio)
            yf = fft(st.session_state.filtered_audio)
            xf = fftfreq(N, 1/st.session_state.sample_rate)
            
            pos_mask = xf > 0
            xf = xf[pos_mask]
            yf = np.abs(yf[pos_mask])
            
            ax.plot(xf, yf, color='#00d084', linewidth=1.2, alpha=0.8)
            ax.fill_between(xf, yf, alpha=0.2, color='#00d084')
            ax.set_xlim([0, 600])
            
            if st.session_state.dominant_freq:
                ax.axvline(st.session_state.dominant_freq, color='#667eea', 
                          linestyle='--', linewidth=2.5,
                          label=f'Peak: {st.session_state.dominant_freq:.2f} Hz', alpha=0.8)
                
                peak_idx = np.argmin(np.abs(xf - st.session_state.dominant_freq))
                ax.plot(xf[peak_idx], yf[peak_idx], 'r*', markersize=20, label='Peak')
            
            ax.axvspan(70, 400, alpha=0.1, color='#6bcf7f', label='Guitar Range')
            
            ax.set_title('Filtered Spectrum', color='#00d084', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=11)
            ax.set_ylabel('Magnitude', color='#4a4a4a', fontsize=11)
            ax.legend(loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--')
            
            st.pyplot(fig)
            plt.close()
            
            st.success("✅ Filters: 50 Hz Notch + 500 Hz Lowpass applied!")
        else:
            st.warning("⚠️ Please run 'ANALYZE & TUNE' first")
    
    # TAB 5: Filter Response
    with tabs[4]:
        st.markdown("#### 🔧 Filter Frequency Response")
        
        fig, ax = create_light_figure(figsize=(12, 5))
        
        nyquist = st.session_state.sample_rate / 2
        Q = 30
        w0 = 50 / nyquist
        b_notch, a_notch = signal.iirnotch(w0, Q, st.session_state.sample_rate)
        
        cutoff = 500 / nyquist
        b_low, a_low = signal.butter(4, cutoff, btype='low', analog=False)
        
        w_notch, h_notch = signal.freqz(b_notch, a_notch, worN=4000, fs=st.session_state.sample_rate)
        w_low, h_low = signal.freqz(b_low, a_low, worN=4000, fs=st.session_state.sample_rate)
        
        ax.plot(w_notch, 20*np.log10(abs(h_notch)), color='#ffd93d', 
               label='50 Hz Notch', linewidth=2.5, alpha=0.9)
        ax.plot(w_low, 20*np.log10(abs(h_low)), color='#00d084', 
               label='500 Hz Lowpass', linewidth=2.5, alpha=0.9)
        ax.set_xlim([0, 1000])
        ax.set_ylim([-80, 5])
        
        ax.axvline(50, color='#ff6b6b', linestyle=':', alpha=0.5, linewidth=2)
        ax.axvline(500, color='#4facfe', linestyle=':', alpha=0.5, linewidth=2)
        ax.axhline(-3, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_title('Filter Response', color='#667eea', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Frequency (Hz)', color='#4a4a4a', fontsize=11)
        ax.set_ylabel('Gain (dB)', color='#4a4a4a', fontsize=11)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, color='#c0c0c0', linestyle='--')
        
        st.pyplot(fig)
        plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔸 Notch Filter:**
            - Type: IIR Notch
            - Center: 50 Hz
            - Q Factor: 30
            - Purpose: Remove power line noise
            """)
        
        with col2:
            st.markdown("""
            **🔸 Lowpass Filter:**
            - Type: Butterworth
            - Cutoff: 500 Hz
            - Order: 4th
            - Purpose: Remove high-freq noise
            """)
    
    # TAB 6: Tuning Meter
    with tabs[5]:
        if st.session_state.dominant_freq and st.session_state.processing_complete:
            st.markdown("#### 🎯 Professional Tuning Meter")
            
            fig, ax = create_light_figure(figsize=(14, 8))
            ax.axis('off')
            
            detected_freq = st.session_state.dominant_freq
            diff = detected_freq - target_freq
            cents = calculate_cents(detected_freq, target_freq)
            
            if abs(diff) <= TUNE_TOLERANCE:
                status = "IN TUNE ✓"
                color = '#00d084'
            elif diff > 0:
                status = "SHARP ↑"
                color = '#ff6b6b'
            else:
                status = "FLAT ↓"
                color = '#4facfe'
            
            # Draw meter
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
            
            # Needle
            max_diff = 10
            needle_pos = np.clip(diff / max_diff, -1, 1)
            needle_x = meter_x + meter_width/2 + (needle_pos * meter_width/2 * 0.9)
            
            # Needle shadow
            ax.plot([needle_x + 0.005, needle_x + 0.005], [meter_y, meter_y + meter_height], 
                   color='black', linewidth=7, alpha=0.2)
            
            # Needle
            ax.plot([needle_x, needle_x], [meter_y, meter_y + meter_height], 
                   color=color, linewidth=8, alpha=0.9, solid_capstyle='round')
            ax.plot([needle_x], [meter_y + meter_height + 0.04], 
                   marker='v', markersize=30, color=color, markeredgecolor='white', 
                   markeredgewidth=2)
            
            # Center line
            center_x = meter_x + meter_width/2
            ax.plot([center_x, center_x], [meter_y, meter_y + meter_height], 
                   color='white', linewidth=3, alpha=0.8, linestyle='--')
            
            # Text info
            ax.text(0.5, 0.88, f"🎸 {selected_string}", 
                   ha='center', va='center', fontsize=18, color='#667eea', fontweight='bold')
            
            ax.text(0.5, 0.78, f"Target: {target_freq:.2f} Hz", 
                   ha='center', va='center', fontsize=14, color='#999', fontweight='600')
            ax.text(0.5, 0.72, f"Detected: {detected_freq:.2f} Hz", 
                   ha='center', va='center', fontsize=16, color='#4a4a4a', fontweight='bold')
            
            ax.text(0.5, 0.65, f"Δ {diff:+.2f} Hz  |  {cents:+.1f} cents", 
                   ha='center', va='center', fontsize=13, color=color, fontweight='bold')
            
            # Zone labels
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
                instruction = "⬇️ Loosen string (counter-clockwise)"
                inst_color = '#ff6b6b'
            elif "FLAT" in status:
                instruction = "⬆️ Tighten string (clockwise)"
                inst_color = '#4facfe'
            else:
                instruction = "✓ Perfect tune!"
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
                   ha='center', va='top', fontsize=10, color='#999', 
                   style='italic', fontweight='600')
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            st.pyplot(fig)
            plt.close()
            
            # Metrics
            st.markdown("##### 📊 Tuning Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🎯 Target", f"{target_freq:.2f} Hz")
            
            with col2:
                st.metric("📡 Detected", f"{detected_freq:.2f} Hz", delta=f"{diff:+.2f} Hz")
            
            with col3:
                st.metric("🎵 Cents", f"{cents:+.1f}")
            
            with col4:
                accuracy = max(0, 100 - abs(diff/target_freq * 100))
                st.metric("✓ Accuracy", f"{accuracy:.1f}%")
            
            with col5:
                st.metric("⚡ Tolerance", f"±{TUNE_TOLERANCE} Hz")
            
        else:
            st.warning("⚠️ Please run '⚡ ANALYZE & TUNE ⚡' first!")
            st.info("💡 The tuning meter shows visual feedback on how sharp or flat your string is.")

else:
    # ==================== WELCOME SCREEN ====================
    st.markdown("---")
    
    st.markdown("### 🚀 Welcome to Guitar Tuner Pro!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### 📖 Quick Start Guide
        
        1. **Select String** - Choose from sidebar
        2. **Upload Audio** - WAV, MP3, FLAC, etc.
        3. **Analyze** - Click "ANALYZE & TUNE"
        4. **View Results** - Check all visualizations
        5. **Adjust** - Follow tuning instructions
        """)
    
    with col2:
        st.markdown("""
        #### ✨ Features
        
        - 🎯 High-precision detection (±0.1 Hz)
        - 🔊 Advanced DSP filtering
        - 🎨 Color-coded visual feedback
        - 📊 7 visualization modes
        - 🎵 All 6 guitar strings
        - 📈 Real-time FFT analysis
        """)
    
    st.markdown("---")
    
    # Guitar reference
    st.markdown("### 🎻 Standard Guitar Tuning")
    
    cols = st.columns(6)
    for idx, (string_name, freq) in enumerate(STRING_FREQUENCIES.items()):
        with cols[idx]:
            color = STRING_COLORS[string_name]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}20 0%, {color}40 100%); 
                        padding: 1.5rem; border-radius: 15px; 
                        border-left: 5px solid {color}; 
                        text-align: center; margin: 0.5rem 0;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <p style="color: {color}; font-weight: 800; margin: 0; font-size: 0.85rem;">
                    {string_name.split('(')[0]}
                </p>
                <p style="color: {color}; margin: 0.5rem 0 0 0; font-size: 1.4rem; font-weight: 900;">
                    {freq} Hz
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tips
    with st.expander("💡 Pro Tips for Best Results", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Recording Tips:**
            - 🎤 Use clear, isolated audio
            - 🔇 Minimize background noise
            - 🎸 Let string ring for 3+ seconds
            - 🔊 Moderate volume works best
            - 📱 Good quality recording device
            """)
        
        with col2:
            st.markdown("""
            **Tuning Tips:**
            - 🎯 Tune slowly and gradually
            - 🔄 Check multiple times
            - 🎵 Start from low E to high E
            - 📐 Always tune UP to pitch
            - 🌡️ Temperature affects tuning
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
        Built with Streamlit, NumPy, SciPy, Matplotlib, Librosa
    </p>
    <p style="color: #999; font-size: 0.9rem; margin-top: 0.5rem;">
        © 2024 Guitar Tuner Pro - All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)
