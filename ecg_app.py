# ecg_app.py - Complete ECG Signal Processor
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# Page configuration
st.set_page_config(
    page_title="ECG Signal Processor",
    page_icon="🧬",
    layout="wide"
)

# Title and description
st.title("🧬 ECG Signal Processor - DSP Assignment")
st.markdown("**EEE 3207 - Digital Signal Processing** | *Signal Acquisition → Sampling → Reconstruction → Noise Elimination → FIR/IIR Filtering*")

def generate_realistic_ecg(duration=10, fs=125, heart_rate=60):
    """Generate a realistic ECG signal with P, QRS, T waves"""
    t = np.linspace(0, duration, int(duration * fs))
    ecg_signal = np.zeros_like(t)
    
    # Heart period in seconds
    heart_period = 60.0 / heart_rate
    
    # Create multiple heartbeats
    for beat in range(int(duration / heart_period) + 1):
        beat_start = beat * heart_period
        
        # P Wave (atrial depolarization)
        p_start = beat_start + 0.1
        p_duration = 0.08
        p_indices = (t >= p_start) & (t < p_start + p_duration)
        if np.any(p_indices):
            p_wave = 0.15 * np.sin(np.pi * (t[p_indices] - p_start) / p_duration)
            ecg_signal[p_indices] += p_wave
        
        # QRS Complex (ventricular depolarization)
        qrs_start = beat_start + 0.2
        qrs_duration = 0.1
        qrs_indices = (t >= qrs_start) & (t < qrs_start + qrs_duration)
        if np.any(qrs_indices):
            qrs_wave = 1.2 * np.sin(np.pi * (t[qrs_indices] - qrs_start) / qrs_duration)
            ecg_signal[qrs_indices] += qrs_wave
        
        # T Wave (ventricular repolarization)
        t_start = beat_start + 0.4
        t_duration = 0.16
        t_indices = (t >= t_start) & (t < t_start + t_duration)
        if np.any(t_indices):
            t_wave = 0.3 * np.sin(np.pi * (t[t_indices] - t_start) / t_duration)
            ecg_signal[t_indices] += t_wave
    
    # Add some baseline wander
    baseline = 0.1 * np.sin(2 * np.pi * 0.2 * t)
    ecg_signal += baseline
    
    return t, ecg_signal

def sampling_reconstruction(signal_data, t_original, target_fs, method='cubic'):
    """Perform sampling and reconstruction"""
    duration = t_original[-1]
    num_downsampled = max(2, int(duration * target_fs))
    
    # Time arrays
    t_down = np.linspace(0, duration, num_downsampled)
    
    # Downsample by selecting points
    downsampled = np.interp(t_down, t_original, signal_data)
    
    # Reconstruct using interpolation
    f_interp = interp1d(t_down, downsampled, kind=method, 
                       bounds_error=False, fill_value='extrapolate')
    reconstructed = f_interp(t_original)
    
    return t_down, downsampled, reconstructed

def add_noise(signal_data, noise_level):
    """Add Gaussian noise to the signal"""
    if noise_level == 0:
        return signal_data
    noise = np.random.normal(0, noise_level * np.std(signal_data), len(signal_data))
    return signal_data + noise

def apply_filter(signal_data, filter_type, fs):
    """Apply FIR or IIR filter for noise removal"""
    if filter_type == 'none':
        return signal_data
    
    # Normalize frequencies
    nyquist = fs / 2
    low_freq = 0.5  # Hz
    high_freq = 40.0  # Hz
    
    if filter_type == 'fir':
        # FIR filter design
        numtaps = 101
        taps = signal.firwin(numtaps, [low_freq, high_freq], fs=fs, pass_zero=False)
        filtered = signal.lfilter(taps, 1.0, signal_data)
    else:  # IIR filter
        order = 4
        b, a = signal.butter(order, [low_freq, high_freq], btype='band', fs=fs)
        filtered = signal.lfilter(b, a, signal_data)
    
    return filtered

def calculate_metrics(original, reconstructed):
    """Calculate performance metrics"""
    min_len = min(len(original), len(reconstructed))
    orig_trim = original[:min_len]
    recon_trim = reconstructed[:min_len]
    
    mse = np.mean((orig_trim - recon_trim) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig_trim - recon_trim))
    
    # Cross-correlation
    correlation = np.correlate(orig_trim, recon_trim, mode='valid')[0] / min_len
    
    # Signal-to-Noise Ratio
    noise_power = np.mean((orig_trim - recon_trim) ** 2)
    signal_power = np.mean(orig_trim ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'SNR': snr
    }

# Sidebar - Controls
st.sidebar.header("🎛️ Control Panel")

# Signal Generation Parameters
st.sidebar.subheader("📊 Signal Parameters")
duration = st.sidebar.slider("Duration (seconds)", 5.0, 30.0, 10.0)
heart_rate = st.sidebar.slider("Heart Rate (BPM)", 40, 120, 72)
original_fs = st.sidebar.slider("Original Sampling Rate (Hz)", 100, 500, 250)

# Processing Parameters
st.sidebar.subheader("⚙️ Processing Parameters")
target_fs = st.sidebar.slider("Target Sampling Rate (Hz)", 1.0, 1000.0, 125.0)
interp_method = st.sidebar.selectbox("Interpolation Method", 
                                   ["linear", "cubic", "nearest"])
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2)
filter_type = st.sidebar.selectbox("Filter Type", 
                                 ["none", "fir", "iir"])

# Process button
process_clicked = st.sidebar.button("🚀 Process Signal", type="primary", use_container_width=True)

# Main content area
if process_clicked:
    with st.spinner("Processing ECG signal..."):
        # Generate ECG signal
        t_original, original_signal = generate_realistic_ecg(duration, original_fs, heart_rate)
        
        # Add noise
        noisy_signal = add_noise(original_signal, noise_level)
        
        # Apply filter
        filtered_signal = apply_filter(noisy_signal, filter_type, original_fs)
        
        # Sampling and reconstruction
        t_down, sampled, reconstructed = sampling_reconstruction(
            filtered_signal, t_original, target_fs, interp_method
        )
        
        # Calculate performance metrics
        metrics = calculate_metrics(original_signal, reconstructed)
        
    # Display results in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Signal Visualization")
        
        # Plot 1: Original vs Reconstructed
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top plot: Signal comparison
        ax1.plot(t_original, original_signal, 'b-', alpha=0.8, linewidth=1.5, label='Original ECG')
        ax1.plot(t_original, reconstructed, 'r--', alpha=0.8, linewidth=1.5, label='Reconstructed')
        if len(t_down) <= 50:  # Only show points if not too many
            ax1.plot(t_down, sampled, 'go', markersize=4, label='Sampled Points', alpha=0.7)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (mV)')
        ax1.set_title('ECG Signal: Original vs Reconstructed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Error
        error = original_signal - reconstructed
        ax2.plot(t_original, error, 'purple', alpha=0.7, linewidth=1)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Error')
        ax2.set_title('Reconstruction Error')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Frequency spectrum plot
        if target_fs >= 20:  # Only show for reasonable sampling rates
            fig2, ax3 = plt.subplots(figsize=(12, 4))
            f_orig, Pxx_orig = signal.welch(original_signal, original_fs, nperseg=256)
            f_recon, Pxx_recon = signal.welch(reconstructed, original_fs, nperseg=256)
            
            ax3.semilogy(f_orig, Pxx_orig, 'b-', alpha=0.7, label='Original')
            ax3.semilogy(f_recon, Pxx_recon, 'r--', alpha=0.7, label='Reconstructed')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power Spectral Density')
            ax3.set_title('Frequency Spectrum Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, min(100, original_fs/2))
            st.pyplot(fig2)
    
    with col2:
        st.subheader("📊 Performance Metrics")
        
        # Display metrics in nice boxes
        st.metric("Mean Squared Error (MSE)", f"{metrics['MSE']:.6f}")
        st.metric("Root MSE (RMSE)", f"{metrics['RMSE']:.6f}")
        st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.6f}")
        st.metric("Cross-Correlation", f"{metrics['Correlation']:.4f}")
        
        snr_display = f"{metrics['SNR']:.2f} dB" if not np.isinf(metrics['SNR']) else "∞ dB"
        st.metric("Signal-to-Noise Ratio", snr_display)
        
        st.subheader("📋 System Information")
        st.write(f"**Original Sampling Rate:** {original_fs} Hz")
        st.write(f"**Target Sampling Rate:** {target_fs:.1f} Hz")
        st.write(f"**Signal Duration:** {duration} seconds")
        st.write(f"**Heart Rate:** {heart_rate} BPM")
        st.write(f"**Total Samples:** {len(original_signal):,}")
        st.write(f"**Downsampled Points:** {len(sampled):,}")
        st.write(f"**Interpolation:** {interp_method.title()}")
        st.write(f"**Noise Level:** {noise_level}")
        st.write(f"**Filter Type:** {filter_type.upper()}")
        
        # Nyquist information
        nyquist_limit = target_fs / 2
        st.info(f"🎯 **Nyquist Frequency:** {nyquist_limit:.1f} Hz")

else:
    # Default view when not processed
    st.info("👆 **Click the 'Process Signal' button in the sidebar to start the analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Assignment Objectives")
        st.markdown("""
        - **Signal Acquisition**: Generate realistic ECG signals
        - **Sampling & Reconstruction**: Test different sampling rates
        - **Noise Elimination**: Add and remove Gaussian noise
        - **FIR/IIR Filtering**: Compare filter performance
        - **Performance Analysis**: Calculate MSE, correlation, SNR
        """)
    
    with col2:
        st.subheader("📖 Instructions")
        st.markdown("""
        1. Adjust parameters in the sidebar
        2. Click **'Process Signal'**
        3. Observe reconstruction quality
        4. Analyze performance metrics
        5. Experiment with different settings
        """)

# Footer
st.markdown("---")
st.markdown("**EEE 3207 - Digital Signal Processing Assignment** | " +
           "**Course:** EEE 3207 | **Teacher:** Dr. Mohammed Abdul Motin, Md. Rakibul Islam | " +
           "**Series:** 2021")
