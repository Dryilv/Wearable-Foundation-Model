"""
GPU 批量预处理模块 — 使用 torch.fft 实现带通滤波 + 纯 PyTorch 特征提取。
替代原有 numpy/scipy 实现，在 GPU 上批量执行以消除 CPU 瓶颈。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TRANSITION_BW = 0.5


def _build_bandpass_mask(n_fft: int, fs: float, low: float, high: float, transition: float = 0.5) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs)
    mask = torch.zeros_like(freqs)

    passband = (freqs >= low) & (freqs <= high)
    mask[passband] = 1.0

    low_trans_start = max(0, low - transition)
    low_trans = (freqs >= low_trans_start) & (freqs < low)
    if low_trans.any():
        t = (freqs[low_trans] - low_trans_start) / (low - low_trans_start + 1e-10)
        mask[low_trans] = 0.5 * (1 - torch.cos(math.pi * t))

    high_trans_end = min(fs / 2, high + transition)
    high_trans = (freqs > high) & (freqs <= high_trans_end)
    if high_trans.any():
        t = (freqs[high_trans] - high) / (high_trans_end - high + 1e-10)
        mask[high_trans] = 0.5 * (1 + torch.cos(math.pi * t))

    return mask


class GPUPreprocessor(nn.Module):
    def __init__(self, signal_len: int = 1000, fs: float = 100.0, num_channels: int = 5,
                 min_std: float = 1e-4, max_std: float = 5000.0):
        super().__init__()
        self.signal_len = signal_len
        self.fs = fs
        self.num_channels = num_channels
        self.min_std = min_std
        self.max_std = max_std

        self.register_buffer('ecg_mask', _build_bandpass_mask(signal_len, fs, 0.05, 40.0, TRANSITION_BW))
        self.register_buffer('acc_mask', _build_bandpass_mask(signal_len, fs, 0.5, 20.0, TRANSITION_BW))
        self.register_buffer('ppg_mask', _build_bandpass_mask(signal_len, fs, 0.1, 20.0, TRANSITION_BW))

        nperseg = min(256, signal_len)
        self.nperseg = nperseg
        self.register_buffer('welch_window', torch.hann_window(nperseg))

        self.register_buffer('psd_freqs', torch.fft.rfftfreq(nperseg, d=1.0 / fs))

    def forward(self, signals: torch.Tensor, channel_mask: torch.Tensor,
                ages: torch.Tensor = None) -> tuple:
        """
        Args:
            signals: (B, M, L) raw signals on GPU, float32
            channel_mask: (B, M) bool mask for valid channels
            ages: (B,) float ages or None
        Returns:
            processed: (B, M, L) filtered + normalized signals
            stats: (B, 16) feature vector
        """
        filtered = self.bandpass_filter(signals, channel_mask)
        stats = self.extract_features(filtered, channel_mask, ages)
        processed = self.normalize(filtered, channel_mask)
        return processed, stats

    def bandpass_filter(self, signals: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """
        对 (B, M, L) 信号按通道类型进行频域带通滤波。
        """
        B, M, L = signals.shape
        out = torch.zeros_like(signals)

        if L != self.signal_len:
            ecg_mask = _build_bandpass_mask(L, self.fs, 0.05, 40.0, TRANSITION_BW).to(signals.device)
            acc_mask = _build_bandpass_mask(L, self.fs, 0.5, 20.0, TRANSITION_BW).to(signals.device)
            ppg_mask = _build_bandpass_mask(L, self.fs, 0.1, 20.0, TRANSITION_BW).to(signals.device)
        else:
            ecg_mask = self.ecg_mask
            acc_mask = self.acc_mask
            ppg_mask = self.ppg_mask

        if self.num_channels == 5:
            channel_masks_map = [
                ([0], ecg_mask),
                ([1, 2, 3], acc_mask),
                ([4], ppg_mask),
            ]
        elif self.num_channels == 1:
            channel_masks_map = [([0], ecg_mask)]
        else:
            channel_masks_map = [(list(range(M)), ecg_mask)]

        for ch_indices, freq_mask in channel_masks_map:
            valid_indices = [i for i in ch_indices if i < M]
            if not valid_indices:
                continue

            x = signals[:, valid_indices, :]  # (B, num_ch, L)
            flat = x.reshape(-1, L)           # (B*num_ch, L)

            spectrum = torch.fft.rfft(flat)
            spectrum = spectrum * freq_mask.unsqueeze(0)
            filtered = torch.fft.irfft(spectrum, n=L)

            out[:, valid_indices, :] = filtered.reshape(B, len(valid_indices), L)

        valid_mask = channel_mask.unsqueeze(-1).float()  # (B, M, 1)
        out = out * valid_mask

        return out

    def normalize(self, signals: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """逐通道 Z-score 归一化 + 质量检查"""
        mean = signals.mean(dim=-1, keepdim=True)
        std = signals.std(dim=-1, keepdim=True).clamp(min=1e-5)

        normalized = (signals - mean) / std
        normalized = normalized.clamp(-10, 10)

        bad_std = (std.squeeze(-1) < self.min_std) | (std.squeeze(-1) > self.max_std)
        bad_channels = bad_std & channel_mask
        if bad_channels.any():
            normalized[bad_channels] = 0.01 * torch.randn_like(normalized[bad_channels])

        valid_mask = channel_mask.unsqueeze(-1).float()
        normalized = normalized * valid_mask

        return normalized

    def extract_features(self, signals: torch.Tensor, channel_mask: torch.Tensor,
                         ages: torch.Tensor = None) -> torch.Tensor:
        """
        GPU 批量特征提取。输出 (B, 16)。
        对所有有效通道提取 15 个特征，跨通道取平均后拼接 age。
        """
        B, M, L = signals.shape

        per_channel_feats = self._compute_channel_features(signals)  # (B, M, 15)

        mask_expanded = channel_mask.unsqueeze(-1).float()  # (B, M, 1)
        num_valid = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        aggregated = (per_channel_feats * mask_expanded).sum(dim=1) / num_valid  # (B, 15)

        if ages is not None:
            age_feat = ages.unsqueeze(-1).float()  # (B, 1)
        else:
            age_feat = torch.zeros(B, 1, device=signals.device)

        stats = torch.cat([aggregated, age_feat], dim=-1)  # (B, 16)
        return stats

    def _compute_channel_features(self, signals: torch.Tensor) -> torch.Tensor:
        """
        对每个通道提取 15 个特征。
        统一用统计 + PSD + 自相关法，不区分 ECG/ACC/PPG（都能提取有意义统计量）。
        """
        B, M, L = signals.shape
        x = signals.reshape(B * M, L)  # (N, L)
        N = x.shape[0]

        feat_mean = x.mean(dim=-1)
        feat_var = x.var(dim=-1)
        feat_max = x.max(dim=-1).values
        feat_min = x.min(dim=-1).values
        feat_rms = (x.pow(2).mean(dim=-1)).sqrt()
        feat_energy = x.pow(2).mean(dim=-1)

        x_centered = x - feat_mean.unsqueeze(-1)
        feat_mad = x_centered.abs().mean(dim=-1)

        std = feat_var.sqrt().clamp(min=1e-8)
        x_norm = x_centered / std.unsqueeze(-1)
        feat_skew = x_norm.pow(3).mean(dim=-1)
        feat_kurt = x_norm.pow(4).mean(dim=-1) - 3.0

        sign_changes = torch.diff(torch.sign(x), dim=-1)
        feat_zcr = (sign_changes != 0).float().sum(dim=-1) / L

        psd, psd_freqs = self._welch_psd(x)  # (N, n_freq)
        feat_total_power = psd.sum(dim=-1)
        feat_peak_power, peak_idx = psd.max(dim=-1)
        feat_peak_freq = psd_freqs[peak_idx]

        feat_hr = self._autocorrelation_hr(x)  # (N,)

        feat_peak_count = self._estimate_peak_count(x)  # (N,)

        features = torch.stack([
            feat_hr,
            feat_var,
            feat_skew,
            feat_kurt,
            feat_peak_freq,
            feat_peak_power,
            feat_total_power,
            feat_mad,
            feat_mad / 0.02,  # activity level
            feat_rms,
            feat_zcr,
            feat_energy,
            feat_peak_count,
            feat_max,
            feat_min,
        ], dim=-1)  # (N, 15)

        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.reshape(B, M, 15)

    def _welch_psd(self, x: torch.Tensor) -> tuple:
        """
        批量 Welch PSD 估计。
        输入: (N, L), 输出: (N, n_freq), freqs
        """
        N, L = x.shape
        nperseg = self.nperseg
        window = self.welch_window

        hop = nperseg // 2
        n_segments = max(1, (L - nperseg) // hop + 1)

        segments = x.unfold(dimension=-1, size=nperseg, step=hop)  # (N, n_seg, nperseg)
        segments = segments[:, :n_segments, :]

        segments = segments * window.unsqueeze(0).unsqueeze(0)

        spectrum = torch.fft.rfft(segments, dim=-1)
        psd = (spectrum.real.pow(2) + spectrum.imag.pow(2)) / (self.fs * (window.pow(2).sum()))

        psd = psd.mean(dim=1)  # (N, n_freq)

        return psd, self.psd_freqs

    def _autocorrelation_hr(self, x: torch.Tensor) -> torch.Tensor:
        """
        通过自相关估计心率/脉搏率。
        输出单位: BPM (beats per minute)。
        """
        N, L = x.shape
        fs = self.fs

        n_fft = 2 ** (2 * L - 1).bit_length()
        X = torch.fft.rfft(x, n=n_fft)
        power_spec = X.real.pow(2) + X.imag.pow(2)
        autocorr = torch.fft.irfft(power_spec, n=n_fft)[..., :L]

        autocorr_norm = autocorr / (autocorr[:, 0:1].clamp(min=1e-10))

        min_lag = int(0.3 * fs)  # 最小间隔 0.3s → 200 BPM
        max_lag = min(int(2.0 * fs), L - 1)  # 最大间隔 2.0s → 30 BPM

        if min_lag >= max_lag or max_lag >= L:
            return torch.full((N,), 75.0, device=x.device)

        search_region = autocorr_norm[:, min_lag:max_lag]  # (N, search_len)
        peak_idx = search_region.argmax(dim=-1)  # (N,)
        peak_lag = peak_idx + min_lag  # (N,)

        hr = 60.0 * fs / peak_lag.float().clamp(min=1.0)
        hr = hr.clamp(30.0, 200.0)

        return hr

    def _estimate_peak_count(self, x: torch.Tensor) -> torch.Tensor:
        """
        利用局部极大值 + 阈值估计峰值数量。
        """
        N, L = x.shape
        min_dist = max(1, int(0.3 * self.fs))

        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        threshold = 0.5 * std  # (N, 1)

        above_thresh = (x > threshold).float()

        kernel_size = min_dist * 2 + 1
        padding = kernel_size // 2
        x_pad = F.pad(x, (padding, padding), mode='reflect')
        local_max = F.max_pool1d(x_pad.unsqueeze(1), kernel_size=kernel_size, stride=1).squeeze(1)
        local_max = local_max[:, :L]

        is_peak = ((x >= local_max - 1e-7) & (x > threshold)).float()

        peak_count = is_peak.sum(dim=-1)
        return peak_count
