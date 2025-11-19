import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import io

# Page configuration
st.set_page_config(
    page_title="Digital Guitar Tuner",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #1a1a2e;
    }
    .stApp {
        background-color: #1a1a2e;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .stButton>button {
        background-color: #e94560;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    </style>
""", unsafe_allow_html=True)

class DigitalGuitarTuner:
    def __init__(self):
        # Guitar string frequencies (Hz)
        self.string_frequencies = {
            'E2 (6th)': 82.41,
            'A2 (5th)': 110.00,
            'D3 (4th)': 146.83,
            'G3 (3rd)': 196.00,
            'B3 (2nd)': 246.94,
            'E4 (1st)': 329.63
        }
        
        # Tolerance for tuning (in Hz)
        self.tune_tolerance = 2.0
        
        # Audio data
        self.audio_data = None
        self.sample_rate = None
        self.filtered_audio = None
        self.dominant_freq = None
    
    def load_audio(self, audio_file):
        """Load audio file"""
        try:
            # Load audio using librosa
            self.audio_data, self.sample_rate = librosa.load(audio_file, sr=None, mono=True)
            
            # Limit to first 3 seconds for faster processing
            max_samples = int(3 * self.sample_rate)
            if len(self.audio_data) > max_samples:
                self.audio_data = self.audio_data[:max_samples]
            
            return True, f"Audio loaded successfully! Sample Rate: {self.sample_rate} Hz | Duration: {len(self.audio_data)/self.sample_rate:.2f}s"
        except Exception as e:
            return False, f"Error loading audio: {str(e)}"
    
    def design_analog_filter(self, filter_type='notch', freq=50, order=4):
        """Design analog filter (notch or lowpass)"""
        nyquist = self.sample_rate / 2
        
        if filter_type == 'notch':
            Q = 30
            w0 = freq / nyquist
            b, a = signal.iirnotch(w0, Q, self.sample_rate)
            return b, a
        elif filter_type == 'lowpass':
            cutoff = 500 / nyquist
            b, a = signal.butter(order, cutoff, btype='low', analog=False)
            return b, a
    
    def apply_bilinear_transform(self):
        """Apply bilinear transformation and create digital filter"""
        b_notch, a_notch = self.design_analog_filter('notch', freq=50)
        b_low, a_low = self.design_analog_filter('lowpass')
        
        temp = signal.filtfilt(b_notch, a_notch, self.audio_data)
        self.filtered_audio = signal.filtfilt(b_low, a_low, temp)
        
        return b_notch, a_notch, b_low, a_low
    
    def compute_fft(self, audio_data):
        """Compute FFT and return frequencies and magnitudes"""
        N = len(audio_data)
        yf = fft(audio_data)
        xf = fftfreq(N, 1/self.sample_rate)
        
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])
        
        return xf, yf
    
    def find_dominant_frequency(self, frequencies, magnitudes, min_freq=70, max_freq=400):
        """Find dominant frequency within guitar range"""
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        freq_range = frequencies[mask]
        mag_range = magnitudes[mask]
        
        if len(mag_range) == 0:
            return None
        
        peak_idx = np.argmax(mag_range)
        dominant_freq = freq_range[peak_idx]
        
        return dominant_freq
    
    def determine_tuning_status(self, detected_freq, target_freq):
        """Determine if string is in tune, sharp, or flat"""
        diff = detected_freq - target_freq
        
        if abs(diff) <= self.tune_tolerance:
            return "IN TUNE ✓", "#00ff88", diff
        elif diff > 0:
            return "SHARP ↑", "#ff4444", diff
        else:
            return "FLAT ↓", "#4444ff", diff
    
    def process_audio(self, target_freq):
        """Main processing pipeline"""
        if self.audio_data is None:
            return None
        
        results = {}
        
        # Original signal data
        time = np.linspace(0, len(self.audio_data)/self.sample_rate, len(self.audio_data))
        results['time_original'] = time[:2000]
        results['audio_original'] = self.audio_data[:2000]
        
        # FFT before filtering
        freq_before, mag_before = self.compute_fft(self.audio_data)
        results['freq_before'] = freq_before
        results['mag_before'] = mag_before
        
        # Apply filter
        b_notch, a_notch, b_low, a_low = self.apply_bilinear_transform()
        
        # Filter response
        w_notch, h_notch = signal.freqz(b_notch, a_notch, worN=2000, fs=self.sample_rate)
        w_low, h_low = signal.freqz(b_low, a_low, worN=2000, fs=self.sample_rate)
        results['filter_notch'] = (w_notch, h_notch)
        results['filter_low'] = (w_low, h_low)
        
        # Filtered signal
        time_filtered = np.linspace(0, len(self.filtered_audio)/self.sample_rate, len(self.filtered_audio))
        results['time_filtered'] = time_filtered[:2000]
        results['audio_filtered'] = self.filtered_audio[:2000]
        
        # FFT after filtering
        freq_after, mag_after = self.compute_fft(self.filtered_audio)
        results['freq_after'] = freq_after
        results['mag_after'] = mag_after
        
        # Find dominant frequency
        self.dominant_freq = self.find_dominant_frequency(freq_after, mag_after)
        
        if self.dominant_freq is None:
            return None
        
        # Determine tuning status
        status, color, diff = self.determine_tuning_status(self.dominant_freq, target_freq)
        results['dominant_freq'] = self.dominant_freq
        results['status'] = status
        results['color'] = color
        results['diff'] = diff
        
        return results

# Initialize session state
if 'tuner' not in st.session_state:
    st.session_state.tuner = DigitalGuitarTuner()

# Main UI
st.title("🎸 Digital Guitar Tuner")
st.markdown("### FFT Analysis & Noise Filtering")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Upload a guitar string audio recording"
    )
    
    # String selection
    selected_string = st.selectbox(
        "Select Guitar String",
        options=list(st.session_state.tuner.string_frequencies.keys()),
        index=5
    )
    
    target_freq = st.session_state.tuner.string_frequencies[selected_string]
    st.info(f"Target Frequency: **{target_freq:.2f} Hz**")
    
    # Process button
    process_button = st.button("⚡ Analyze & Tune", type="primary", use_container_width=True)

# Load audio if uploaded
if uploaded_file is not None:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Loading audio file..."):
            success, message = st.session_state.tuner.load_audio(uploaded_file)
            if success:
                st.success(message)
                st.session_state.current_file = uploaded_file.name
            else:
                st.error(message)

# Process audio
if process_button and uploaded_file is not None:
    with st.spinner("🎵 Analyzing audio..."):
        results = st.session_state.tuner.process_audio(target_freq)
        
        if results is None:
            st.error("Could not detect frequency. Please check the audio file.")
        else:
            # Display tuning status
            status = results['status']
            color = results['color']
            detected_freq = results['dominant_freq']
            diff = results['diff']
            
            st.markdown("---")
            st.markdown(f"## Tuning Result")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected Frequency", f"{detected_freq:.2f} Hz")
            with col2:
                st.metric("Target Frequency", f"{target_freq:.2f} Hz")
            with col3:
                st.metric("Difference", f"{diff:+.2f} Hz")
            
            # Status message
            if "IN TUNE" in status:
                st.success(f"### 🎉 {status}")
                st.balloons()
            elif "SHARP" in status:
                st.error(f"### {status}")
                st.warning("⬇️ String is too high - Loosen the string slightly")
            else:
                st.info(f"### {status}")
                st.warning("⬆️ String is too low - Tighten the string slightly")
            
            st.markdown("---")
            
            # Create plots
            st.markdown("## 📊 Analysis Results")
            
            # Row 1
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor='#1a1a2e')
                ax1.set_facecolor('#0f3460')
                ax1.plot(results['time_original'], results['audio_original'], color='#00d4ff', linewidth=0.8)
                ax1.set_xlabel('Time (s)', color='white')
                ax1.set_ylabel('Amplitude', color='white')
                ax1.set_title('Original Audio Signal', color='#00d4ff', fontweight='bold')
                ax1.tick_params(colors='white')
                ax1.grid(True, alpha=0.3, color='white')
                for spine in ax1.spines.values():
                    spine.set_color('white')
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor='#1a1a2e')
                ax2.set_facecolor('#0f3460')
                ax2.plot(results['freq_before'], results['mag_before'], color='#ff6b6b', linewidth=0.8)
                ax2.set_xlabel('Frequency (Hz)', color='white')
                ax2.set_ylabel('Magnitude', color='white')
                ax2.set_xlim([0, 600])
                ax2.set_title('FFT - Before Filtering', color='#00d4ff', fontweight='bold')
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.3, color='white')
                for spine in ax2.spines.values():
                    spine.set_color('white')
                st.pyplot(fig2)
            
            with col3:
                fig3, ax3 = plt.subplots(figsize=(6, 4), facecolor='#1a1a2e')
                ax3.set_facecolor('#0f3460')
                w_notch, h_notch = results['filter_notch']
                w_low, h_low = results['filter_low']
                ax3.plot(w_notch, 20*np.log10(abs(h_notch)), color='#ffd93d', label='Notch (50Hz)', linewidth=1.5)
                ax3.plot(w_low, 20*np.log10(abs(h_low)), color='#6bcf7f', label='Lowpass', linewidth=1.5)
                ax3.set_xlabel('Frequency (Hz)', color='white')
                ax3.set_ylabel('Gain (dB)', color='white')
                ax3.set_xlim([0, 600])
                ax3.set_title('Filter Frequency Response', color='#00d4ff', fontweight='bold')
                ax3.tick_params(colors='white')
                ax3.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
                ax3.grid(True, alpha=0.3, color='white')
                for spine in ax3.spines.values():
                    spine.set_color('white')
                st.pyplot(fig3)
            
            # Row 2
            col4, col5, col6 = st.columns(3)
            
            with col4:
                fig4, ax4 = plt.subplots(figsize=(6, 4), facecolor='#1a1a2e')
                ax4.set_facecolor('#0f3460')
                ax4.plot(results['time_filtered'], results['audio_filtered'], color='#6bcf7f', linewidth=0.8)
                ax4.set_xlabel('Time (s)', color='white')
                ax4.set_ylabel('Amplitude', color='white')
                ax4.set_title('Filtered Audio Signal', color='#00d4ff', fontweight='bold')
                ax4.tick_params(colors='white')
                ax4.grid(True, alpha=0.3, color='white')
                for spine in ax4.spines.values():
                    spine.set_color('white')
                st.pyplot(fig4)
            
            with col5:
                fig5, ax5 = plt.subplots(figsize=(6, 4), facecolor='#1a1a2e')
                ax5.set_facecolor('#0f3460')
                ax5.plot(results['freq_after'], results['mag_after'], color='#6bcf7f', linewidth=0.8)
                ax5.set_xlabel('Frequency (Hz)', color='white')
                ax5.set_ylabel('Magnitude', color='white')
                ax5.set_xlim([0, 600])
                ax5.set_title('FFT - After Filtering', color='#00d4ff', fontweight='bold')
                ax5.tick_params(colors='white')
                ax5.grid(True, alpha=0.3, color='white')
                for spine in ax5.spines.values():
                    spine.set_color('white')
                st.pyplot(fig5)
            
            with col6:
                fig6, ax6 = plt.subplots(figsize=(6, 4), facecolor='#1a1a2e')
                ax6.set_facecolor('#0f3460')
                ax6.axis('off')
                ax6.text(0.5, 0.75, f"Detected:\n{detected_freq:.2f} Hz", 
                        ha='center', va='center', fontsize=14, color='white', fontweight='bold')
                ax6.text(0.5, 0.5, f"Target:\n{target_freq:.2f} Hz", 
                        ha='center', va='center', fontsize=14, color='white')
                ax6.text(0.5, 0.25, f"Difference:\n{diff:+.2f} Hz", 
                        ha='center', va='center', fontsize=12, color='white')
                ax6.text(0.5, 0.05, status, 
                        ha='center', va='center', fontsize=18, color=color, 
                        fontweight='bold', bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
                st.pyplot(fig6)

elif uploaded_file is None:
    st.info("👆 Please upload an audio file to get started!")
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload an audio file** (WAV, MP3, FLAC, OGG, or M4A format)
    2. **Select the guitar string** you want to tune
    3. **Click 'Analyze & Tune'** to process the audio
    4. View the results and adjust your guitar string accordingly
    
    #### Supported Audio Formats:
    - WAV, MP3, FLAC, OGG, M4A
    
    #### Guitar String Frequencies:
    """)
    
    for string, freq in st.session_state.tuner.string_frequencies.items():
        st.markdown(f"- **{string}**: {freq:.2f} Hz")