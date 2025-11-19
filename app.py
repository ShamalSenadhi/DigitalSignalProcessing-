"""
🎸 Professional Guitar Tuner - Streamlit Edition
EEX7434 Mini Project - Complex & Attractive UI
All features included with modern design
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
from PIL import Image

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
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #7c3aed 50%, #db2777 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.3);
        border: 2px solid rgba(0, 212, 255, 0.2);
    }
    
    .main-header h1 {
        color: #00d4ff;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.8);
        margin: 0;
    }
    
    .main-header p {
        color: #6bcf7f;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .css-1r6slb0 {
        background: rgba(26, 31, 58, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid rgba(0, 212, 255, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00d4ff;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6bcf7f !important;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 41, 0.6);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(45, 53, 97, 0.6);
        color: #00d4ff;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.2);
    }
    
    /* Status boxes */
    .status-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .status-in-tune {
        background: rgba(0, 255, 136, 0.1);
        border-color: #00ff88;
    }
    
    .status-sharp {
        background: rgba(255, 68, 68, 0.1);
        border-color: #ff4444;
    }
    
    .status-flat {
        background: rgba(68, 68, 255, 0.1);
        border-color: #4444ff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(45, 53, 97, 0.6);
        color: #00d4ff;
        font-weight: bold;
        border-radius: 8px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(26, 31, 58, 0.8);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed rgba(0, 212, 255, 0.3);
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #00d4ff;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* Selectbox */
    .stSelectbox > label {
        color: #6bcf7f;
        font-weight: bold;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
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
if 'recording_duration' not in st.session_state:
    st.session_state.recording_duration = 5
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

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

def create_dark_figure(figsize=(10, 4)):
    """Create a matplotlib figure with dark theme"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0f1729')
    ax.set_facecolor('#0f1729')
    ax.tick_params(colors='#888', labelsize=9)
    ax.spines['bottom'].set_color('#2d3561')
    ax.spines['left'].set_color('#2d3561')
    ax.spines['top'].set_color('#2d3561')
    ax.spines['right'].set_color('#2d3561')
    return fig, ax

def process_audio(audio_data, sample_rate, target_freq):
    """Process audio and detect frequency"""
    try:
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
            return None, None, "Could not detect frequency"
        
        peak_idx = np.argmax(mag_range)
        dominant_freq = freq_range[peak_idx]
        
        return filtered_audio, dominant_freq, "Success"
        
    except Exception as e:
        return None, None, str(e)

def get_tuning_status(detected_freq, target_freq):
    """Get tuning status"""
    diff = detected_freq - target_freq
    
    if abs(diff) <= TUNE_TOLERANCE:
        return "IN TUNE ✓", "success", "#00ff88"
    elif diff > 0:
        return "SHARP ↑", "warning", "#ff4444"
    else:
        return "FLAT ↓", "info", "#4444ff"

# ==================== MAIN HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🎸 PROFESSIONAL GUITAR TUNER</h1>
    <p>EEX7434 Mini Project | Advanced DSP Analysis | Real-time Tuning</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🎛️ CONTROL CENTER")
    
    st.markdown("---")
    
    # String Selection
    st.markdown("#### 🎻 Select Guitar String")
    selected_string = st.selectbox(
        "Choose string to tune:",
        options=list(STRING_FREQUENCIES.keys()),
        index=5,
        label_visibility="collapsed"
    )
    
    target_freq = STRING_FREQUENCIES[selected_string]
    string_color = STRING_COLORS[selected_string]
    
    st.markdown(f"""
    <div style="background: rgba(45, 53, 97, 0.6); padding: 1rem; border-radius: 8px; 
                border-left: 4px solid {string_color}; margin: 1rem 0;">
        <p style="color: {string_color}; font-weight: bold; font-size: 1.1rem; margin: 0;">
            Target Frequency: {target_freq} Hz
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Audio Input Method
    st.markdown("#### 📁 Audio Input Method")
    input_method = st.radio(
        "Select input:",
        ["Upload Audio File", "Record from Microphone"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if input_method == "Upload Audio File":
        st.markdown("#### 📂 Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                audio_bytes = uploaded_file.read()
                audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                
                # Limit to 5 seconds
                max_samples = int(5 * sample_rate)
                if len(audio_data) > max_samples:
                    audio_data = audio_data[:max_samples]
                
                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = sample_rate
                st.session_state.current_filename = uploaded_file.name
                
                st.success(f"✓ Loaded: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    else:
        st.markdown("#### 🎙️ Microphone Recording")
        
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=1,
            max_value=10,
            value=5,
            label_visibility="visible"
        )
        
        if st.button("🔴 START RECORDING", use_container_width=True):
            with st.spinner(f"Recording for {duration} seconds..."):
                try:
                    sample_rate = 44100
                    recording = sd.rec(
                        int(duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype='float32'
                    )
                    sd.wait()
                    
                    st.session_state.audio_data = recording.flatten()
                    st.session_state.sample_rate = sample_rate
                    st.session_state.current_filename = f"Recording_{int(time.time())}.wav"
                    
                    st.success("✓ Recording completed!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Recording failed: {str(e)}")
    
    st.markdown("---")
    
    # Playback Controls
    if st.session_state.audio_data is not None:
        st.markdown("#### ▶️ Playback Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Play", use_container_width=True):
                try:
                    sd.play(st.session_state.audio_data, st.session_state.sample_rate)
                    st.session_state.is_playing = True
                except Exception as e:
                    st.error(f"Playback error: {str(e)}")
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                sd.stop()
                st.session_state.is_playing = False
    
    st.markdown("---")
    
    # Main Analyze Button
    if st.button("⚡ ANALYZE & TUNE ⚡", use_container_width=True, type="primary"):
        if st.session_state.audio_data is None:
            st.error("Please load audio first!")
        else:
            with st.spinner("Processing audio..."):
                filtered_audio, dominant_freq, status = process_audio(
                    st.session_state.audio_data,
                    st.session_state.sample_rate,
                    target_freq
                )
                
                if status == "Success":
                    st.session_state.filtered_audio = filtered_audio
                    st.session_state.dominant_freq = dominant_freq
                    st.success("✓ Analysis complete!")
                    st.rerun()
                else:
                    st.error(f"Analysis failed: {status}")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <p>© 2024 Guitar Tuner Pro</p>
        <p>EEX7434 Mini Project</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================

# Status Metrics
if st.session_state.audio_data is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📄 Audio File",
            st.session_state.current_filename[:20] + "..." if len(st.session_state.current_filename or "") > 20 else st.session_state.current_filename
        )
    
    with col2:
        duration = len(st.session_state.audio_data) / st.session_state.sample_rate
        st.metric("⏱️ Duration", f"{duration:.2f}s")
    
    with col3:
        st.metric("🔊 Sample Rate", f"{st.session_state.sample_rate} Hz")
    
    with col4:
        if st.session_state.dominant_freq:
            st.metric("📡 Detected Freq", f"{st.session_state.dominant_freq:.2f} Hz")
        else:
            st.metric("📡 Detected Freq", "--")

# Tuning Status Display
if st.session_state.dominant_freq:
    status_text, status_type, status_color = get_tuning_status(
        st.session_state.dominant_freq,
        target_freq
    )
    
    diff = st.session_state.dominant_freq - target_freq
    
    if "IN TUNE" in status_text:
        status_class = "status-in-tune"
        instruction = "🎉 Perfect! Your string is tuned!"
    elif "SHARP" in status_text:
        status_class = "status-sharp"
        instruction = "⬇️ String is SHARP - Loosen slightly (turn tuning peg counter-clockwise)"
    else:
        status_class = "status-flat"
        instruction = "⬆️ String is FLAT - Tighten slightly (turn tuning peg clockwise)"
    
    st.markdown(f"""
    <div class="status-box {status_class}">
        <h2 style="color: {status_color}; margin: 0; font-size: 2.5rem; text-align: center;">
            {status_text}
        </h2>
        <p style="color: white; text-align: center; font-size: 1.2rem; margin-top: 1rem;">
            Difference: {diff:+.2f} Hz
        </p>
        <p style="color: {status_color}; text-align: center; font-size: 1.1rem; margin-top: 0.5rem;">
            {instruction}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== VISUALIZATION TABS ====================
if st.session_state.audio_data is not None:
    
    tabs = st.tabs([
        "🌊 Waveform",
        "📊 Spectrum Analyzer",
        "📈 FFT Before Filter",
        "📉 FFT After Filter",
        "🔧 Filter Response",
        "🎯 Tuning Meter"
    ])
    
    # Tab 1: Waveform
    with tabs[0]:
        st.markdown("### 🌊 Time Domain Waveform")
        
        fig, ax = create_dark_figure(figsize=(12, 5))
        
        time_array = np.linspace(0, len(st.session_state.audio_data)/st.session_state.sample_rate, 
                                len(st.session_state.audio_data))
        samples_to_plot = min(10000, len(st.session_state.audio_data))
        
        ax.plot(time_array[:samples_to_plot], st.session_state.audio_data[:samples_to_plot], 
               color=string_color, linewidth=0.8, alpha=0.8)
        ax.fill_between(time_array[:samples_to_plot], st.session_state.audio_data[:samples_to_plot], 
                       alpha=0.3, color=string_color)
        
        ax.set_title('Time Domain Signal', color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time (s)', color='#888', fontsize=11)
        ax.set_ylabel('Amplitude', color='#888', fontsize=11)
        ax.grid(True, alpha=0.2, color='#2d3561', linestyle='--')
        
        st.pyplot(fig)
        plt.close()
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Amplitude", f"{np.max(st.session_state.audio_data):.4f}")
        col2.metric("Min Amplitude", f"{np.min(st.session_state.audio_data):.4f}")
        col3.metric("Mean", f"{np.mean(st.session_state.audio_data):.4f}")
        col4.metric("RMS", f"{np.sqrt(np.mean(st.session_state.audio_data**2)):.4f}")
    
    # Tab 2: Spectrum Analyzer
    with tabs[1]:
        st.markdown("### 📊 Frequency Spectrum Analysis")
        
        fig, ax = create_dark_figure(figsize=(12, 5))
        
        N = len(st.session_state.audio_data)
        yf = fft(st.session_state.audio_data)
        xf = fftfreq(N, 1/st.session_state.sample_rate)
        
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        ax.plot(xf, yf, color='#6bcf7f', linewidth=1, alpha=0.8)
        ax.fill_between(xf, yf, alpha=0.3, color='#6bcf7f')
        ax.set_xlim([0, 1000])
        
        # Mark target frequency
        ax.axvline(target_freq, color='#ffd93d', linestyle='--', linewidth=2, 
                  label=f'Target: {target_freq:.2f} Hz', alpha=0.8)
        
        if st.session_state.dominant_freq:
            ax.axvline(st.session_state.dominant_freq, color='#ff6b6b', linestyle='--', 
                      linewidth=2, label=f'Detected: {st.session_state.dominant_freq:.2f} Hz', alpha=0.8)
        
        ax.set_title('Frequency Spectrum', color='#6bcf7f', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Frequency (Hz)', color='#888', fontsize=11)
        ax.set_ylabel('Magnitude', color='#888', fontsize=11)
        ax.legend(facecolor='#0f1729', edgecolor='#2d3561', labelcolor='white', fontsize=10)
        ax.grid(True, alpha=0.2, color='#2d3561', linestyle='--')
        
        st.pyplot(fig)
        plt.close()
        
        # Filter specifications
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔸 Notch Filter Specs:**
            - Center Frequency: 50 Hz
            - Quality Factor (Q): 30
            - Type: IIR Notch
            - Purpose: Remove power line interference
            """)
        
        with col2:
            st.markdown("""
            **🔸 Lowpass Filter Specs:**
            - Cutoff Frequency: 500 Hz
            - Order: 4th order
            - Type: Butterworth
            - Purpose: Remove high-frequency noise
            """)
    
    # Tab 6: Tuning Meter
    with tabs[5]:
        if st.session_state.dominant_freq:
            st.markdown("### 🎯 Visual Tuning Meter")
            
            fig, ax = create_dark_figure(figsize=(12, 6))
            ax.axis('off')
            
            detected_freq = st.session_state.dominant_freq
            diff = detected_freq - target_freq
            
            # Determine status
            if abs(diff) <= TUNE_TOLERANCE:
                status = "IN TUNE ✓"
                color = '#00ff88'
            elif diff > 0:
                status = "SHARP ↑"
                color = '#ff4444'
            else:
                status = "FLAT ↓"
                color = '#4444ff'
            
            # Draw tuning meter
            meter_width = 0.8
            meter_height = 0.12
            meter_x = 0.1
            meter_y = 0.5
            
            # Background meter
            meter_bg = plt.Rectangle((meter_x, meter_y), meter_width, meter_height,
                                      facecolor='#1a1f3a', edgecolor='#2d3561', linewidth=3)
            ax.add_patch(meter_bg)
            
            # Color zones
            flat_zone = plt.Rectangle((meter_x, meter_y), meter_width*0.35, meter_height,
                                       facecolor='#4444ff', alpha=0.3)
            ax.add_patch(flat_zone)
            
            intune_zone = plt.Rectangle((meter_x + meter_width*0.35, meter_y), 
                                         meter_width*0.3, meter_height,
                                         facecolor='#00ff88', alpha=0.3)
            ax.add_patch(intune_zone)
            
            sharp_zone = plt.Rectangle((meter_x + meter_width*0.65, meter_y), 
                                        meter_width*0.35, meter_height,
                                        facecolor='#ff4444', alpha=0.3)
            ax.add_patch(sharp_zone)
            
            # Needle
            max_diff = 10
            needle_pos = np.clip(diff / max_diff, -1, 1)
            needle_x = meter_x + meter_width/2 + (needle_pos * meter_width/2 * 0.9)
            
            ax.plot([needle_x, needle_x], [meter_y, meter_y + meter_height], 
                   color=color, linewidth=6, alpha=0.9)
            ax.plot([needle_x], [meter_y + meter_height + 0.03], 
                   marker='v', markersize=25, color=color)
            
            # Center line
            center_x = meter_x + meter_width/2
            ax.plot([center_x, center_x], [meter_y, meter_y + meter_height], 
                   color='white', linewidth=2, alpha=0.5, linestyle='--')
            
            # Info text
            ax.text(0.5, 0.85, f"String: {selected_string}", 
                   ha='center', va='center', fontsize=14, color='#00d4ff', fontweight='bold')
            
            ax.text(0.5, 0.75, f"Detected: {detected_freq:.2f} Hz", 
                   ha='center', va='center', fontsize=13, color='white', fontweight='bold')
            ax.text(0.5, 0.68, f"Target: {target_freq:.2f} Hz", 
                   ha='center', va='center', fontsize=11, color='#888')
            ax.text(0.5, 0.61, f"Difference: {diff:+.2f} Hz", 
                   ha='center', va='center', fontsize=11, color=color, fontweight='bold')
            
            # Zone labels
            ax.text(meter_x + 0.02, meter_y - 0.04, "FLAT", 
                   ha='left', va='top', fontsize=10, color='#4444ff', fontweight='bold')
            ax.text(center_x, meter_y - 0.04, "IN TUNE", 
                   ha='center', va='top', fontsize=10, color='#00ff88', fontweight='bold')
            ax.text(meter_x + meter_width - 0.02, meter_y - 0.04, "SHARP", 
                   ha='right', va='top', fontsize=10, color='#ff4444', fontweight='bold')
            
            # Status box
            status_box = dict(boxstyle='round,pad=1', facecolor='#1a1f3a', 
                            edgecolor=color, linewidth=5)
            ax.text(0.5, 0.30, status, 
                   ha='center', va='center', fontsize=28, color=color, 
                   fontweight='bold', bbox=status_box)
            
            # Instructions
            if "SHARP" in status:
                instruction = "⬇️ Loosen the string (turn tuning peg counter-clockwise)"
                inst_color = '#ff6b6b'
            elif "FLAT" in status:
                instruction = "⬆️ Tighten the string (turn tuning peg clockwise)"
                inst_color = '#4444ff'
            else:
                instruction = "✓ Perfect! No adjustment needed"
                inst_color = '#00ff88'
            
            ax.text(0.5, 0.15, instruction, 
                   ha='center', va='center', fontsize=10, color=inst_color, 
                   style='italic')
            
            # Frequency scale
            scale_y = meter_y - 0.12
            scale_freqs = [-10, -5, 0, 5, 10]
            for sf in scale_freqs:
                scale_x = meter_x + meter_width/2 + (sf/10 * meter_width/2 * 0.9)
                ax.plot([scale_x, scale_x], [scale_y, scale_y + 0.02], 
                       color='#666', linewidth=1)
                ax.text(scale_x, scale_y - 0.02, f"{sf:+d}", 
                       ha='center', va='top', fontsize=8, color='#666')
            
            ax.text(0.5, scale_y - 0.08, "Cents from Target", 
                   ha='center', va='top', fontsize=9, color='#888', style='italic')
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            st.pyplot(fig)
            plt.close()
            
            # Detailed metrics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cents = 1200 * np.log2(detected_freq / target_freq)
                st.metric("Cents Offset", f"{cents:+.1f}")
            
            with col2:
                st.metric("Frequency Error", f"{diff:+.2f} Hz")
            
            with col3:
                accuracy = max(0, 100 - abs(diff/target_freq * 100))
                st.metric("Accuracy", f"{accuracy:.1f}%")
            
            with col4:
                st.metric("Tolerance", f"±{TUNE_TOLERANCE} Hz")
        
        else:
            st.warning("⚠️ Please run 'ANALYZE & TUNE' first to see the tuning meter.")
            st.info("💡 The tuning meter will show you exactly how sharp or flat your string is!")

else:
    # Welcome screen when no audio is loaded
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style="background: rgba(26, 31, 58, 0.8); padding: 2rem; border-radius: 15px; 
                    border: 2px solid rgba(0, 212, 255, 0.2); margin: 1rem 0;">
            <h2 style="color: #00d4ff; margin-top: 0;">🚀 Getting Started</h2>
            <ol style="color: #888; line-height: 2;">
                <li>Select your guitar string from the sidebar</li>
                <li>Upload an audio file or record from microphone</li>
                <li>Click "ANALYZE & TUNE" to process</li>
                <li>View results in the visualization tabs</li>
                <li>Use the tuning meter to fine-tune your string</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(26, 31, 58, 0.8); padding: 2rem; border-radius: 15px; 
                    border: 2px solid rgba(107, 207, 127, 0.2); margin: 1rem 0;">
            <h2 style="color: #6bcf7f; margin-top: 0;">✨ Features</h2>
            <ul style="color: #888; line-height: 2;">
                <li>Real-time frequency detection</li>
                <li>Advanced DSP filtering (Notch + Lowpass)</li>
                <li>Visual tuning meter with color zones</li>
                <li>FFT spectrum analysis</li>
                <li>Waveform visualization</li>
                <li>Audio playback controls</li>
                <li>Microphone recording support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Guitar string reference
    st.markdown("### 🎻 Standard Guitar Tuning Reference")
    
    cols = st.columns(6)
    for idx, (string_name, freq) in enumerate(STRING_FREQUENCIES.items()):
        with cols[idx]:
            color = STRING_COLORS[string_name]
            st.markdown(f"""
            <div style="background: rgba(45, 53, 97, 0.6); padding: 1rem; border-radius: 10px; 
                        border-left: 4px solid {color}; text-align: center; margin: 0.5rem 0;">
                <p style="color: {color}; font-weight: bold; margin: 0; font-size: 0.9rem;">
                    {string_name}
                </p>
                <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">
                    {freq} Hz
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technical info
    with st.expander("📚 Technical Information", expanded=False):
        st.markdown("""
        ### Digital Signal Processing Pipeline
        
        **1. Signal Acquisition**
        - Audio input via file upload or microphone recording
        - Sampling rate: Typically 44.1 kHz
        - Mono channel processing
        
        **2. Preprocessing**
        - 50 Hz Notch Filter (Q=30) - Removes power line interference
        - 500 Hz Butterworth Lowpass Filter (4th order) - Removes high-frequency noise
        
        **3. Frequency Detection**
        - Fast Fourier Transform (FFT) analysis
        - Peak detection in 70-400 Hz range (guitar fundamental frequencies)
        - Dominant frequency identification
        
        **4. Tuning Analysis**
        - Frequency comparison with target
        - ±2 Hz tolerance for "in tune" status
        - Cent calculation for musical accuracy
        
        **5. Visualization**
        - Time domain waveform
        - Frequency spectrum (before/after filtering)
        - Filter response curves
        - Visual tuning meter with color-coded zones
        
        ### Supported Audio Formats
        - WAV, MP3, FLAC, OGG, M4A
        - Mono or Stereo (converted to mono)
        - Maximum 5 seconds processing duration
        
        ### Accuracy
        - Frequency resolution: ~0.1 Hz
        - Detection range: 70-400 Hz
        - Tuning tolerance: ±2 Hz
        """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p style="font-size: 1.1rem; margin: 0;">
        <strong style="color: #00d4ff;">EEX7434 Mini Project</strong> | 
        Advanced DSP Guitar Tuner
    </p>
    <p style="margin-top: 0.5rem; font-size: 0.9rem;">
        Built with Streamlit, NumPy, SciPy, and Librosa
    </p>
    <p style="margin-top: 1rem; font-size: 0.9rem;">
        © 2024 Guitar Tuner Pro - All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== ADDITIONAL INFO ====================
st.markdown("---")

with st.expander("💡 Tips for Best Results", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Recording Tips:**
        - 🎤 Place microphone close to guitar
        - 🔇 Minimize background noise
        - 🎸 Pluck string clearly and let it ring
        - ⏱️ Record for at least 3 seconds
        - 🔊 Use moderate volume (not too loud/quiet)
        """)
    
    with col2:
        st.markdown("""
        **Tuning Tips:**
        - 🎯 Tune slowly and gradually
        - 🔄 Check tuning multiple times
        - 🎵 Start from lowest string (E2) to highest
        - 📐 Use harmonic tuning for fine adjustment
        - 🌡️ Temperature affects tuning - retune as needed
        """)

with st.expander("🎓 Understanding the Display", expanded=False):
    st.markdown("""
    ### Color Code Guide
    
    - **🟢 Green (In Tune):** String is within ±2 Hz of target frequency
    - **🔴 Red (Sharp):** String frequency is higher than target - loosen the string
    - **🔵 Blue (Flat):** String frequency is lower than target - tighten the string
    
    ### Frequency vs Cents
    
    - **Frequency (Hz):** Absolute measurement of vibrations per second
    - **Cents:** Relative musical measurement (100 cents = 1 semitone)
    - Musicians typically use cents for fine-tuning accuracy
    
    ### Reading the Tuning Meter
    
    - **Needle Position:** Shows how far off-tune you are
    - **Color Zones:** Visual guide for tuning status
    - **Center Line:** Perfect tune position
    - **Scale Below:** Shows deviation in cents from target
    
    ### Filter Purpose
    
    - **Notch Filter (50 Hz):** Removes electrical hum from power lines
    - **Lowpass Filter (500 Hz):** Removes high-frequency noise and harmonics
    - **Result:** Clean fundamental frequency for accurate detection
    """)

# ==================== KEYBOARD SHORTCUTS INFO ====================
with st.expander("⌨️ Keyboard Shortcuts & Quick Actions", expanded=False):
    st.markdown("""
    ### Quick Actions
    
    - Press `R` to refresh/rerun the app
    - Use `Ctrl/Cmd + K` to search settings
    - Click on metrics to highlight them
    - Hover over charts for detailed values
    
    ### Navigation
    
    - Use sidebar for all input controls
    - Main area shows results and visualizations
    - Tabs organize different analysis views
    - Expanders hide/show additional information
    """)

# ==================== ERROR HANDLING INFO ====================
st.sidebar.markdown("---")
with st.sidebar.expander("❓ Troubleshooting", expanded=False):
    st.markdown("""
    **Common Issues:**
    
    1. **No frequency detected**
       - Increase recording volume
       - Ensure clear string pluck
       - Check audio file quality
    
    2. **Inaccurate readings**
       - Reduce background noise
       - Let string ring longer
       - Try different microphone
    
    3. **Cannot record**
       - Check microphone permissions
       - Select correct audio device
       - Restart browser if needed
    
    4. **Playback issues**
       - Verify audio output device
       - Check system volume
       - Reload audio file
    """)

# Debug mode (optional - can be removed in production)
if st.sidebar.checkbox("🔧 Debug Mode", value=False):
    st.sidebar.markdown("### Debug Information")
    st.sidebar.json({
        "audio_loaded": st.session_state.audio_data is not None,
        "sample_rate": st.session_state.sample_rate,
        "audio_length": len(st.session_state.audio_data) if st.session_state.audio_data is not None else 0,
        "filtered": st.session_state.filtered_audio is not None,
        "detected_freq": st.session_state.dominant_freq,
        "target_freq": target_freq,
        "selected_string": selected_string
    })2d3561', labelcolor='white', fontsize=10)
        ax.grid(True, alpha=0.2, color='#2d3561', linestyle='--')
        
        st.pyplot(fig)
        plt.close()
    
    # Tab 3: FFT Before Filter
    with tabs[2]:
        st.markdown("### 📈 FFT Spectrum (Before Filtering)")
        
        fig, ax = create_dark_figure(figsize=(12, 5))
        
        N = len(st.session_state.audio_data)
        yf = fft(st.session_state.audio_data)
        xf = fftfreq(N, 1/st.session_state.sample_rate)
        
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        ax.plot(xf, yf, color='#ff6b6b', linewidth=1, alpha=0.8)
        ax.fill_between(xf, yf, alpha=0.2, color='#ff6b6b')
        ax.set_xlim([0, 600])
        
        ax.axvline(50, color='#ffd93d', linestyle=':', linewidth=2, 
                  label='50 Hz Noise', alpha=0.6)
        
        ax.set_title('Unfiltered Spectrum (Raw Signal)', color='#ff6b6b', fontsize=14, 
                    fontweight='bold', pad=20)
        ax.set_xlabel('Frequency (Hz)', color='#888', fontsize=11)
        ax.set_ylabel('Magnitude', color='#888', fontsize=11)
        ax.legend(facecolor='#0f1729', edgecolor='#2d3561', labelcolor='white', fontsize=10)
        ax.grid(True, alpha=0.2, color='#2d3561', linestyle='--')
        
        st.pyplot(fig)
        plt.close()
        
        st.info("📌 Note: This shows the raw frequency spectrum before applying filters. "
               "Notice the 50 Hz power line interference and high-frequency noise.")
    
    # Tab 4: FFT After Filter
    with tabs[3]:
        if st.session_state.filtered_audio is not None:
            st.markdown("### 📉 FFT Spectrum (After Filtering)")
            
            fig, ax = create_dark_figure(figsize=(12, 5))
            
            N = len(st.session_state.filtered_audio)
            yf = fft(st.session_state.filtered_audio)
            xf = fftfreq(N, 1/st.session_state.sample_rate)
            
            pos_mask = xf > 0
            xf = xf[pos_mask]
            yf = np.abs(yf[pos_mask])
            
            ax.plot(xf, yf, color='#6bcf7f', linewidth=1, alpha=0.8)
            ax.fill_between(xf, yf, alpha=0.2, color='#6bcf7f')
            ax.set_xlim([0, 600])
            
            if st.session_state.dominant_freq:
                ax.axvline(st.session_state.dominant_freq, color='#00d4ff', linestyle='--', 
                          linewidth=2, label=f'Detected Peak: {st.session_state.dominant_freq:.2f} Hz', 
                          alpha=0.8)
            
            ax.set_title('Filtered Spectrum (Clean Signal)', color='#6bcf7f', fontsize=14, 
                        fontweight='bold', pad=20)
            ax.set_xlabel('Frequency (Hz)', color='#888', fontsize=11)
            ax.set_ylabel('Magnitude', color='#888', fontsize=11)
            ax.legend(facecolor='#0f1729', edgecolor='#2d3561', labelcolor='white', fontsize=10)
            ax.grid(True, alpha=0.2, color='#2d3561', linestyle='--')
            
            st.pyplot(fig)
            plt.close()
            
            st.success("✓ Filters applied: 50 Hz Notch Filter + 500 Hz Lowpass Filter")
        else:
            st.warning("Please run 'ANALYZE & TUNE' first to see filtered results.")
    
    # Tab 5: Filter Response
    with tabs[4]:
        st.markdown("### 🔧 Digital Filter Frequency Response")
        
        fig, ax = create_dark_figure(figsize=(12, 5))
        
        # Design filters
        nyquist = st.session_state.sample_rate / 2
        Q = 30
        w0 = 50 / nyquist
        b_notch, a_notch = signal.iirnotch(w0, Q, st.session_state.sample_rate)
        
        cutoff = 500 / nyquist
        b_low, a_low = signal.butter(4, cutoff, btype='low', analog=False)
        
        # Compute responses
        w_notch, h_notch = signal.freqz(b_notch, a_notch, worN=2000, fs=st.session_state.sample_rate)
        w_low, h_low = signal.freqz(b_low, a_low, worN=2000, fs=st.session_state.sample_rate)
        
        ax.plot(w_notch, 20*np.log10(abs(h_notch)), 
               color='#ffd93d', label='50 Hz Notch Filter', linewidth=2, alpha=0.9)
        ax.plot(w_low, 20*np.log10(abs(h_low)), 
               color='#6bcf7f', label='500 Hz Lowpass Filter', linewidth=2, alpha=0.9)
        ax.set_xlim([0, 600])
        ax.set_ylim([-60, 5])
        
        ax.axvline(50, color='#ff6b6b', linestyle=':', alpha=0.5)
        ax.axvline(500, color='#4ecdc4', linestyle=':', alpha=0.5)
        
        ax.set_title('Filter Frequency Response', color='#ffd93d', fontsize=14, 
                    fontweight='bold', pad=20)
        ax.set_xlabel('Frequency (Hz)', color='#888', fontsize=11)
        ax.set_ylabel('Gain (dB)', color='#888', fontsize=11)
        ax.legend(facecolor='#0f1729', edgecolor='#
