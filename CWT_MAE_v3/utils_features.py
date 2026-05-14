import numpy as np
from scipy.signal import find_peaks, welch

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

def extract_features(signal_np, channel_id, fs=100, item_info=None):
    """
    signal_np: (1, L)
    channel_id: 0 for ECG, 1 for PPG
    item_info: metadata dictionary from index json, which may contain age etc.
    """
    sig = signal_np.flatten()
    if channel_id == 0:
        feats = extract_ecg_features(sig, fs=fs)
    else:
        feats = extract_ppg_features(sig, fs=fs)
        
    # Append external stats if requested (e.g., age)
    # The target feature vector size must be constant.
    # Let's add 'age' as the 16th feature if available, else 0
    age = 0.0
    if item_info is not None and 'age' in item_info:
        age = float(item_info['age'])
        
    final_feats = np.append(feats, [age])
    return final_feats
