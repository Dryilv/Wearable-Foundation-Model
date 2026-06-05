import numpy as np
from scipy.signal import find_peaks, welch, butter, filtfilt
from scipy.stats import skew, kurtosis

def extract_ecg_features(signal, fs=100):
    """
    Extracts statistical, HRV, and morphological features from an ECG signal.
    Returns a fixed-length numpy array.
    """
    # 1. R-peak detection
    # Simple thresholding based on signal amplitude
    # Assuming signal is normalized or at least centered
    std_val = np.std(signal)
    if std_val < 1e-6:
        return np.zeros(15, dtype=np.float32)
        
    peaks, _ = find_peaks(signal, distance=int(0.4 * fs), height=0.5 * std_val)
    
    if len(peaks) < 2:
        return np.zeros(15, dtype=np.float32)
        
    rr_intervals = np.diff(peaks) / fs  # in seconds
    
    # 1. HR & HRV
    hr = 60.0 / np.mean(rr_intervals)
    rr_ms = rr_intervals * 1000
    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2)) if len(rr_ms) > 1 else 0.0
    pnn50 = np.mean(np.abs(np.diff(rr_ms)) > 50) * 100 if len(rr_ms) > 1 else 0.0
    
    # 2. Frequency domain (Welch's method)
    # Simple power estimation if we have enough peaks (usually need much more for accurate LF/HF)
    try:
        f, pxx = welch(signal, fs, nperseg=min(len(signal), 256))
        vlf_power = np.sum(pxx[(f >= 0.0033) & (f < 0.04)])
        lf_power = np.sum(pxx[(f >= 0.04) & (f < 0.15)])
        hf_power = np.sum(pxx[(f >= 0.15) & (f < 0.4)])
        total_power = vlf_power + lf_power + hf_power
        lf_hf_ratio = lf_power / (hf_power + 1e-6)
    except:
        vlf_power = lf_power = hf_power = total_power = lf_hf_ratio = 0.0

    # 3. Morphological
    r_amps = signal[peaks]
    r_amp_mean = np.mean(r_amps)
    r_amp_std = np.std(r_amps)
    
    # 4. Signal Quality
    signal_power = np.sum(signal**2) / len(signal)
    
    features = [
        hr, sdnn, rmssd, pnn50, 
        vlf_power, lf_power, hf_power, lf_hf_ratio, total_power,
        r_amp_mean, r_amp_std,
        signal_power,
        len(peaks),  # peak count
        np.max(signal), # max val
        np.min(signal)  # min val
    ]
    
    return np.nan_to_num(np.array(features, dtype=np.float32))

def extract_ppg_features(signal, fs=100):
    """
    Extracts PRV, morphological, and APG features from a PPG signal.
    Returns a fixed-length numpy array (same length as ECG for unified head).
    """
    std_val = np.std(signal)
    if std_val < 1e-6:
        return np.zeros(15, dtype=np.float32)
        
    peaks, _ = find_peaks(signal, distance=int(0.4 * fs), height=0.2 * std_val)
    
    if len(peaks) < 2:
        return np.zeros(15, dtype=np.float32)
        
    pp_intervals = np.diff(peaks) / fs  # in seconds
    
    # 1. PR & PRV
    pr = 60.0 / np.mean(pp_intervals)
    pp_ms = pp_intervals * 1000
    sdnn_pr = np.std(pp_ms)
    rmssd_pr = np.sqrt(np.mean(np.diff(pp_ms)**2)) if len(pp_ms) > 1 else 0.0
    pnn50_pr = np.mean(np.abs(np.diff(pp_ms)) > 50) * 100 if len(pp_ms) > 1 else 0.0
    
    # 2. Frequency domain
    try:
        f, pxx = welch(signal, fs, nperseg=min(len(signal), 256))
        vlf_power = np.sum(pxx[(f >= 0.0033) & (f < 0.04)])
        lf_power = np.sum(pxx[(f >= 0.04) & (f < 0.15)])
        hf_power = np.sum(pxx[(f >= 0.15) & (f < 0.4)])
        total_power = vlf_power + lf_power + hf_power
        lf_hf_ratio = lf_power / (hf_power + 1e-6)
    except:
        vlf_power = lf_power = hf_power = total_power = lf_hf_ratio = 0.0

    # 3. Morphological & APG
    sp_amps = signal[peaks]
    sp_amp_mean = np.mean(sp_amps)
    sp_amp_std = np.std(sp_amps)
    
    # 2nd derivative for APG features (simple approx)
    apg = np.gradient(np.gradient(signal))
    apg_power = np.sum(apg**2) / len(apg)
    
    features = [
        pr, sdnn_pr, rmssd_pr, pnn50_pr,
        vlf_power, lf_power, hf_power, lf_hf_ratio, total_power,
        sp_amp_mean, sp_amp_std,
        apg_power,
        len(peaks),
        np.max(signal),
        np.min(signal)
    ]
    
    return np.nan_to_num(np.array(features, dtype=np.float32))

def extract_acc_features(signal, fs=100):
    """
    Extracts features from accelerometer signals based on SARP system methodology.
    References:
    - Moatamed et al. (BSN 2016): Activity recognition with statistical features
    - Ramezani et al. (JMIR 2019): MAD-based energy quantification, bandpass filtering
    
    Features (15 total to match ECG/PPG):
    1. mean - filtered signal mean
    2. median - filtered signal median
    3. variance - filtered signal variance
    4. skewness - distribution asymmetry
    5. kurtosis - distribution tail heaviness
    6. peak_frequency - dominant frequency from PSD
    7. peak_power - power at dominant frequency
    8. total_power - total spectral power (0.5-8 Hz band)
    9. mad - Mean Absolute Deviation (energy proxy, Ramezani 2019)
    10. activity_level - MAD / 0.02 threshold (>=1 means active)
    11. rms - root mean square
    12. zero_crossing_rate - signal oscillation frequency
    13. signal_energy - sum of squared values
    14. max_val - maximum amplitude
    15. min_val - minimum amplitude
    """
    std_val = np.std(signal)
    if std_val < 1e-6:
        return np.zeros(15, dtype=np.float32)
    
    nyquist = 0.5 * fs
    lowcut = 0.5
    highcut = min(8.0, nyquist * 0.95)
    
    if lowcut < nyquist and highcut > lowcut:
        try:
            b, a = butter(5, [lowcut / nyquist, highcut / nyquist], btype='band')
            filtered = filtfilt(b, a, signal)
        except:
            filtered = signal
    else:
        filtered = signal
    
    mean_val = np.mean(filtered)
    median_val = np.median(filtered)
    var_val = np.var(filtered)
    skew_val = skew(filtered) if len(filtered) > 2 else 0.0
    kurt_val = kurtosis(filtered) if len(filtered) > 3 else 0.0
    
    mad = np.mean(np.abs(filtered - mean_val))
    activity_level = mad / 0.02
    
    rms = np.sqrt(np.mean(filtered**2))
    signal_energy = np.sum(filtered**2) / len(filtered)
    
    zero_crossings = np.sum(np.diff(np.sign(filtered)) != 0) / len(filtered)
    
    max_val = np.max(filtered)
    min_val = np.min(filtered)
    
    try:
        nperseg = min(len(filtered), 256)
        f, pxx = welch(filtered, fs, nperseg=nperseg)
        
        band_mask = (f >= lowcut) & (f <= highcut)
        if np.any(band_mask):
            f_band = f[band_mask]
            pxx_band = pxx[band_mask]
            peak_idx = np.argmax(pxx_band)
            peak_freq = f_band[peak_idx]
            peak_power = pxx_band[peak_idx]
            total_power = np.sum(pxx_band)
        else:
            peak_freq = f[np.argmax(pxx)]
            peak_power = np.max(pxx)
            total_power = np.sum(pxx)
    except:
        peak_freq = peak_power = total_power = 0.0
    
    features = [
        mean_val,
        median_val,
        var_val,
        skew_val,
        kurt_val,
        peak_freq,
        peak_power,
        total_power,
        mad,
        activity_level,
        rms,
        zero_crossings,
        signal_energy,
        max_val,
        min_val
    ]
    
    return np.nan_to_num(np.array(features, dtype=np.float32))

def extract_features(signal_np, channel_id, fs=100, item_info=None):
    """
    signal_np: (1, L)
    channel_id: 0=ECG, 1=ACC_X, 2=ACC_Y, 3=ACC_Z, 4=PPG
    item_info: metadata dictionary from index json, which may contain age etc.
    """
    sig = signal_np.flatten()
    if channel_id == 0:
        feats = extract_ecg_features(sig, fs=fs)
    elif channel_id == 4:
        feats = extract_ppg_features(sig, fs=fs)
    else:
        feats = extract_acc_features(sig, fs=fs)
        
    age = 0.0
    if item_info is not None and 'age' in item_info:
        age = float(item_info['age'])
        
    final_feats = np.append(feats, [age])
    return final_feats
