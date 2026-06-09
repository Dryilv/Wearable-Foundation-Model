"""
生理信号预处理模块 — 纯 NumPy FFT 实现，零依赖高性能。

针对 5 通道数据 (ECG, ACC×3, PPG) 的带通滤波。
使用 FFT 频域滤波 + Tukey 窗平滑过渡，避免 Gibbs 振铃。
"""

import numpy as np


# 通道类型定义
CH_ECG = 0
CH_ACC = [1, 2, 3]
CH_PPG = 4

# 滤波参数
FILTER_PARAMS = {
    'ecg': {'low': 0.05, 'high': 40.0},
    'acc': {'low': 0.5, 'high': 20.0},
    'ppg': {'low': 0.1, 'high': 20.0},
}

# 过渡带宽度 (Hz) — 决定频域窗口的平滑程度
TRANSITION_BW = 0.5


def _build_bandpass_mask(n_fft: int, fs: float, low: float, high: float, transition: float = 0.5) -> np.ndarray:
    """
    构造频域带通掩码，带平滑过渡 (raised cosine 形)。
    返回长度为 n_fft//2+1 的实数数组。
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mask = np.zeros(len(freqs), dtype=np.float64)

    # 通带
    passband = (freqs >= low) & (freqs <= high)
    mask[passband] = 1.0

    # 低端过渡带: [low - transition, low]
    low_trans_start = max(0, low - transition)
    low_trans = (freqs >= low_trans_start) & (freqs < low)
    if np.any(low_trans):
        t = (freqs[low_trans] - low_trans_start) / (low - low_trans_start + 1e-10)
        mask[low_trans] = 0.5 * (1 - np.cos(np.pi * t))

    # 高端过渡带: [high, high + transition]
    high_trans_end = min(fs / 2, high + transition)
    high_trans = (freqs > high) & (freqs <= high_trans_end)
    if np.any(high_trans):
        t = (freqs[high_trans] - high) / (high_trans_end - high + 1e-10)
        mask[high_trans] = 0.5 * (1 + np.cos(np.pi * t))

    return mask.astype(np.float32)


# 预计算掩码缓存
_MASK_CACHE: dict = {}


def _get_cached_mask(n_fft: int, fs: float, low: float, high: float) -> np.ndarray:
    key = (n_fft, fs, low, high)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _build_bandpass_mask(n_fft, fs, low, high, TRANSITION_BW)
    return _MASK_CACHE[key]


def bandpass_filter_fft(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """
    FFT 频域带通滤波 (零相位，线性相位)。
    输入输出均为 1D float32 数组。
    """
    n = signal.shape[-1]
    if n < 10:
        return (signal - np.mean(signal)).astype(np.float32)

    mask = _get_cached_mask(n, fs, low, high)
    spectrum = np.fft.rfft(signal)
    spectrum *= mask
    filtered = np.fft.irfft(spectrum, n=n)
    return filtered.astype(np.float32)


def preprocess_signal(signal: np.ndarray, fs: float = 100.0, num_channels: int = 5) -> np.ndarray:
    """
    对多通道生理信号进行带通滤波预处理。

    Args:
        signal: (M, L) 多通道信号，通道顺序 [ECG, ACC_x, ACC_y, ACC_z, PPG]
        fs: 采样率 (Hz)
        num_channels: 实际通道数

    Returns:
        (M, L) 滤波后的信号，float32
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    M, L = signal.shape
    out = np.empty_like(signal, dtype=np.float32)

    for ch in range(M):
        ch_data = signal[ch].astype(np.float32)

        if num_channels == 5:
            if ch == CH_ECG:
                params = FILTER_PARAMS['ecg']
            elif ch in CH_ACC:
                params = FILTER_PARAMS['acc']
            elif ch == CH_PPG:
                params = FILTER_PARAMS['ppg']
            else:
                out[ch] = ch_data
                continue
        elif num_channels == 1:
            params = FILTER_PARAMS['ecg']
        else:
            params = FILTER_PARAMS['ecg']

        out[ch] = bandpass_filter_fft(ch_data, params['low'], params['high'], fs)

    return out


def preprocess_signal_batch(signals: np.ndarray, fs: float = 100.0, num_channels: int = 5) -> np.ndarray:
    """
    批量预处理 — 利用 FFT 向量化一次处理同类通道。

    Args:
        signals: (B, M, L) 批量多通道信号
        fs: 采样率
        num_channels: 通道数

    Returns:
        (B, M, L) 滤波后信号
    """
    B, M, L = signals.shape
    out = np.empty_like(signals, dtype=np.float32)

    if num_channels == 5:
        channel_groups = {
            'ecg': [CH_ECG],
            'acc': CH_ACC,
            'ppg': [CH_PPG],
        }
    else:
        channel_groups = {'ecg': list(range(M))}

    for ch_type, ch_indices in channel_groups.items():
        params = FILTER_PARAMS[ch_type]
        mask = _get_cached_mask(L, fs, params['low'], params['high'])

        for ch_idx in ch_indices:
            if ch_idx >= M:
                continue
            batch_ch = signals[:, ch_idx, :].astype(np.float32)
            spectra = np.fft.rfft(batch_ch, axis=-1)
            spectra *= mask[np.newaxis, :]
            out[:, ch_idx, :] = np.fft.irfft(spectra, n=L, axis=-1).astype(np.float32)

    return out
