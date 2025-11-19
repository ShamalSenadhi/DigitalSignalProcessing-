import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import soundfile as sf
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Digital Guitar Tuner",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive interface
st.markdown("""
    <style>
    .main {
        background-color: #1a1a2e;
    }
    .stApp {
        background-color: #1a1a2e;
    }
    h1 {
        color: #00d4ff;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0;
    }
    h2 {
        color: #00d4ff;
        text-align: center;
    }
    h3 {
        color: #00d4ff;
    }
    .subtitle {
        color: #888;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #e94560;
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 2em;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        border: none;
    }
    .info-box {
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00d4ff;
        margin: 10px 0;
    }
    .success-box {
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00ff88;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffaa00;
        margin: 10px 0;
    }
    .error-box {
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 10px 0;
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

# Guitar string frequencies
STRING_FREQUENCIES = {
    'E2 (6th)': 82.41,
    'A2 (5th)': 110.00,
    'D3 (4th)': 146.83,
    'G3 (3rd)': 196.00,
    'B3 (2nd)': 246.94,
    'E4 (1st)': 329.63
}

TUNE_TOLERANCE = 2.0

# Functions
def design_analog_filter(sample_rate, filter_type='notch', freq=50, order=4):
    """Design analog filter (notch or lowpass)"""
    nyquist = sample_rate / 2
    
    if filter_type == 'notch':
        Q = 30
        w0 = freq / nyquist
        b, a = signal.iirnotch(w0, Q, sample_rate)
        return b, a
    elif filter_type == 'lowpass':
        cutoff = 500 / nyquist
        b, a = signal.butter(order, cutoff, btype='low', analog=False)
        return b, a

def apply_filters(audio_data, sample_rate):
    """Apply bilinear transformation and create digital filter"""
    b_notch, a_notch = design_analog_filter(sample_rate, 'notch', freq=50)
    b_low, a_low = design_analog_filter(sample_rate, 'lowpass')
    
    temp = signal.filtfilt(b_notch, a_notch, audio_data)
    filtered_audio = signal.filtfilt(b_low, a_low, temp)
    
    return filtered_audio, b_notch, a_notch, b_low, a_low

def compute_fft(audio_data, sample_rate):
    """Compute FFT and return frequencies and magnitudes"""
    N = len(audio_data)
    yf = fft(audio_data)
    xf = fftfreq(N, 1/sample_rate)
    
    pos_mask = xf > 0
    xf = xf[pos_mask]
    yf = np.abs(yf[pos_mask])
    
    return xf, yf

def find_dominant_frequency(frequencies, magnitudes, min_freq=70, max_freq=400):
    """Find dominant frequency within guitar range"""
    mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    freq_range = frequencies[mask]
    mag_range = magnitudes[mask]
    
    if len(mag_range) == 0:
        return None
    
    peak_idx = np.argmax(mag_range)
    dominant_freq = freq_range[peak_idx]
    
    return dominant_freq

def determine_tuning_status(detected_freq, target_freq):
    """Determine if string is in tune, sharp, or flat"""
    diff = detected_freq - target_freq
    
    if abs(diff) <= TUNE_TOLERANCE:
        return "IN TUNE ✓", "#00ff88", diff
    elif diff > 0:
        return "SHARP ↑", "#ff4444", diff
    else:
        return "FLAT ↓", "#4444ff", diff

# Header
st.markdown("<h1>🎸 Digital Guitar Tuner</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>FFT Analysis & Noise Filtering | EEX7434 Mini Project</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎵 Controls")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Supports WAV, MP3, FLAC, OGG, M4A formats"
    )
    
    # String selection
    selected_string = st.selectbox(
        "Select Guitar String",
        options=list(STRING_FREQUENCIES.keys()),
        index=5  # Default to E4 (1st string)
    )
    
    target_freq = STRING_FREQUENCIES[selected_string]
    
    st.markdown("---")
    st.markdown("### 📊 String Frequencies")
    for string, freq in STRING_FREQUENCIES.items():
        st.text(f"{string}: {freq:.2f} Hz")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This app analyzes guitar strings using FFT and applies noise filtering to determine tuning accuracy.")

# Main content
if uploaded_file is not None:
    try:
        # Load audio
        audio_bytes = uploaded_file.read()
        audio_data, sample_rate = librosa.load(BytesIO(audio_bytes), sr=None, mono=True)
        
        # Limit to first 3 seconds
        max_samples = int(3 * sample_rate)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        st.session_state.audio_data = audio_data
        st.session_state.sample_rate = sample_rate
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='info-box'><b>File:</b> {uploaded_file.name}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='info-box'><b>Sample Rate:</b> {sample_rate} Hz</div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='info-box'><b>Duration:</b> {len(audio_data)/sample_rate:.2f}s</div>", unsafe_allow_html=True)
        
        # Process button
        if st.button("⚡ Analyze & Tune", key="analyze_btn"):
            with st.spinner("Processing audio..."):
                
                # Apply filters
                filtered_audio, b_notch, a_notch, b_low, a_low = apply_filters(audio_data, sample_rate)
                st.session_state.filtered_audio = filtered_audio
                
                # Compute FFTs
                freq_before, mag_before = compute_fft(audio_data, sample_rate)
                freq_after, mag_after = compute_fft(filtered_audio, sample_rate)
                
                # Find dominant frequency
                dominant_freq = find_dominant_frequency(freq_after, mag_after)
                st.session_state.dominant_freq = dominant_freq
                
                if dominant_freq is None:
                    st.error("❌ Could not detect frequency. Please check the audio file.")
                else:
                    # Determine tuning status
                    status, color, diff = determine_tuning_status(dominant_freq, target_freq)
                    
                    # Display result
                    st.markdown("---")
                    st.markdown("## 🎯 Tuning Result")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.markdown(f"""
                        <div class='info-box'>
                            <h3 style='color: white; text-align: center;'>Detected Frequency</h3>
                            <h2 style='color: #00d4ff; text-align: center;'>{dominant_freq:.2f} Hz</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.markdown(f"""
                        <div class='info-box'>
                            <h3 style='color: white; text-align: center;'>Target Frequency</h3>
                            <h2 style='color: #00d4ff; text-align: center;'>{target_freq:.2f} Hz</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col3:
                        st.markdown(f"""
                        <div class='info-box'>
                            <h3 style='color: white; text-align: center;'>Difference</h3>
                            <h2 style='color: {color}; text-align: center;'>{diff:+.2f} Hz</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Status message
                    if "IN TUNE" in status:
                        st.markdown(f"""
                        <div class='success-box'>
                            <h2 style='color: {color}; text-align: center;'>🎉 {status}</h2>
                            <p style='color: white; text-align: center; font-size: 1.2em;'>Perfect! The string is tuned correctly!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif "SHARP" in status:
                        st.markdown(f"""
                        <div class='error-box'>
                            <h2 style='color: {color}; text-align: center;'>⬇️ {status}</h2>
                            <p style='color: white; text-align: center; font-size: 1.2em;'>String is too high. Loosen the string slightly.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='warning-box'>
                            <h2 style='color: {color}; text-align: center;'>⬆️ {status}</h2>
                            <p style='color: white; text-align: center; font-size: 1.2em;'>String is too low. Tighten the string slightly.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualization
                    st.markdown("---")
                    st.markdown("## 📈 Signal Analysis")
                    
                    # Create figure with dark theme
                    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
                    fig.subplots_adjust(hspace=0.4, wspace=0.3)
                    
                    # Plot 1: Original Signal
                    ax1 = plt.subplot(2, 3, 1)
                    time = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
                    ax1.plot(time[:2000], audio_data[:2000], color='#00d4ff', linewidth=0.5)
                    ax1.set_xlabel('Time (s)', color='white', fontsize=10)
                    ax1.set_ylabel('Amplitude', color='white', fontsize=10)
                    ax1.set_title('Original Audio Signal', color='#00d4ff', fontsize=12, fontweight='bold')
                    ax1.set_facecolor('#0f3460')
                    ax1.tick_params(colors='white', labelsize=8)
                    ax1.grid(True, alpha=0.3)
                    for spine in ax1.spines.values():
                        spine.set_color('white')
                    
                    # Plot 2: FFT Before
                    ax2 = plt.subplot(2, 3, 2)
                    ax2.plot(freq_before, mag_before, color='#ff6b6b', linewidth=0.8)
                    ax2.set_xlabel('Frequency (Hz)', color='white', fontsize=10)
                    ax2.set_ylabel('Magnitude', color='white', fontsize=10)
                    ax2.set_xlim([0, 600])
                    ax2.set_title('FFT - Before Filtering', color='#00d4ff', fontsize=12, fontweight='bold')
                    ax2.set_facecolor('#0f3460')
                    ax2.tick_params(colors='white', labelsize=8)
                    ax2.grid(True, alpha=0.3)
                    for spine in ax2.spines.values():
                        spine.set_color('white')
                    
                    # Plot 3: Filter Response
                    ax3 = plt.subplot(2, 3, 3)
                    w_notch, h_notch = signal.freqz(b_notch, a_notch, worN=2000, fs=sample_rate)
                    w_low, h_low = signal.freqz(b_low, a_low, worN=2000, fs=sample_rate)
                    ax3.plot(w_notch, 20*np.log10(abs(h_notch)), color='#ffd93d', label='Notch (50Hz)', linewidth=1.5)
                    ax3.plot(w_low, 20*np.log10(abs(h_low)), color='#6bcf7f', label='Lowpass', linewidth=1.5)
                    ax3.set_xlabel('Frequency (Hz)', color='white', fontsize=10)
                    ax3.set_ylabel('Gain (dB)', color='white', fontsize=10)
                    ax3.set_xlim([0, 600])
                    ax3.set_title('Filter Frequency Response', color='#00d4ff', fontsize=12, fontweight='bold')
                    ax3.set_facecolor('#0f3460')
                    ax3.tick_params(colors='white', labelsize=8)
                    ax3.legend(facecolor='#16213e', edgecolor='white', labelcolor='white', fontsize=8)
                    ax3.grid(True, alpha=0.3)
                    for spine in ax3.spines.values():
                        spine.set_color('white')
                    
                    # Plot 4: Filtered Signal
                    ax4 = plt.subplot(2, 3, 4)
                    time_filtered = np.linspace(0, len(filtered_audio)/sample_rate, len(filtered_audio))
                    ax4.plot(time_filtered[:2000], filtered_audio[:2000], color='#6bcf7f', linewidth=0.5)
                    ax4.set_xlabel('Time (s)', color='white', fontsize=10)
                    ax4.set_ylabel('Amplitude', color='white', fontsize=10)
                    ax4.set_title('Filtered Audio Signal', color='#00d4ff', fontsize=12, fontweight='bold')
                    ax4.set_facecolor('#0f3460')
                    ax4.tick_params(colors='white', labelsize=8)
                    ax4.grid(True, alpha=0.3)
                    for spine in ax4.spines.values():
                        spine.set_color('white')
                    
                    # Plot 5: FFT After
                    ax5 = plt.subplot(2, 3, 5)
                    ax5.plot(freq_after, mag_after, color='#6bcf7f', linewidth=0.8)
                    ax5.set_xlabel('Frequency (Hz)', color='white', fontsize=10)
                    ax5.set_ylabel('Magnitude', color='white', fontsize=10)
                    ax5.set_xlim([0, 600])
                    ax5.set_title('FFT - After Filtering', color='#00d4ff', fontsize=12, fontweight='bold')
                    ax5.set_facecolor('#0f3460')
                    ax5.tick_params(colors='white', labelsize=8)
                    ax5.grid(True, alpha=0.3)
                    for spine in ax5.spines.values():
                        spine.set_color('white')
                    
                    # Plot 6: Tuning Status
                    ax6 = plt.subplot(2, 3, 6)
                    ax6.axis('off')
                    ax6.text(0.5, 0.75, f"Detected: {dominant_freq:.2f} Hz", 
                            ha='center', va='center', fontsize=14, color='white', fontweight='bold')
                    ax6.text(0.5, 0.6, f"Target: {target_freq:.2f} Hz", 
                            ha='center', va='center', fontsize=14, color='white')
                    ax6.text(0.5, 0.45, f"Difference: {diff:+.2f} Hz", 
                            ha='center', va='center', fontsize=12, color='white')
                    ax6.text(0.5, 0.25, status, 
                            ha='center', va='center', fontsize=24, color=color, 
                            fontweight='bold', bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
                    ax6.set_facecolor('#0f3460')
                    
                    st.pyplot(fig)
                    plt.close()
    
    except Exception as e:
        st.error(f"❌ Error loading audio file: {str(e)}")

else:
    # Welcome message
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #00d4ff;'>👋 Welcome to the Digital Guitar Tuner!</h3>
        <p style='color: white; font-size: 1.1em;'>
            This application uses FFT (Fast Fourier Transform) analysis and digital signal processing 
            to help you tune your guitar strings accurately.
        </p>
        <h4 style='color: #00d4ff; margin-top: 20px;'>How to use:</h4>
        <ol style='color: white; font-size: 1em;'>
            <li>Upload an audio file of your guitar string</li>
            <li>Select which string you're tuning</li>
            <li>Click "Analyze & Tune" to get results</li>
            <li>Adjust your string based on the feedback</li>
        </ol>
        <h4 style='color: #00d4ff; margin-top: 20px;'>Features:</h4>
        <ul style='color: white; font-size: 1em;'>
            <li>🎵 Supports multiple audio formats (WAV, MP3, FLAC, OGG, M4A)</li>
            <li>🔊 Advanced noise filtering (50Hz notch + lowpass filters)</li>
            <li>📊 Real-time FFT analysis visualization</li>
            <li>🎯 Accurate frequency detection (±2 Hz tolerance)</li>
            <li>📈 Before/after filtering comparison</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Display example
    st.markdown("---")
    st.markdown("### 🎸 Standard Guitar Tuning Reference")
    
    cols = st.columns(6)
    for idx, (string, freq) in enumerate(STRING_FREQUENCIES.items()):
        with cols[idx]:
            st.markdown(f"""
            <div style='background-color: #16213e; padding: 15px; border-radius: 10px; text-align: center;'>
                <h4 style='color: #00d4ff; margin: 0;'>{string}</h4>
                <p style='color: white; font-size: 1.2em; margin: 5px 0;'>{freq:.2f} Hz</p>
            </div>
            """, unsafe_allow_html=True)
