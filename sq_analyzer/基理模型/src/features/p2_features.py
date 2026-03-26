"""
P2阶段特征提取器

包含2个特定特征:
1. 增强频谱峰值 ESP (Enhanced Spectrum Peak)
2. 非平稳性指标 NSI (NonStationarity Index)
"""
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Optional, Tuple

from ..core.base_extractor import BaseFeatureExtractor


class EnhancedSpectrumPeakExtractor(BaseFeatureExtractor):
    """
    增强频谱峰值提取器 (P2)
    
    参考标准: Randall & Antoni 2011 (基于频谱峭度的峰值增强)
    单位: dB
    
    描述: 
    结合频谱峭度(SK)的频谱峰值检测方法。
    通过频谱峭度定位包含冲击成分的频率带，
    在该频带内检测频谱峰值，用于识别轴承、齿轮等故障特征。
    
    应用场景: 轴承故障、齿轮啮合异常、周期性冲击检测
    """
    
    def __init__(self):
        super().__init__(
            name="增强频谱峰值",
            symbol="ESP",
            unit="dB",
            stage="P2",
            description="Randall增强频谱峰值，结合频谱峭度的峰值检测"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取增强频谱峰值
        
        算法步骤:
        1. 计算信号的STFT得到时变频谱
        2. 计算各频率点的频谱峭度(SK)
        3. 选择SK值最高的频带作为关注区域
        4. 在该频带内检测频谱峰值
        5. 返回峰值相对于背景噪声的dB值
        """
        # 参数设置
        nperseg = min(2048, len(sig) // 4)
        noverlap = nperseg // 2
        
        # 1. 计算STFT
        f, t, Zxx = signal.stft(sig, sr, nperseg=nperseg, noverlap=noverlap)
        
        if Zxx.shape[1] < 4:  # 需要至少几个时间帧
            return 0.0
        
        # 2. 计算各频率的频谱峭度
        power = np.abs(Zxx) ** 2
        sk_values = []
        
        for i in range(power.shape[0]):
            freq_power = power[i, :]
            if np.std(freq_power) > 0:
                # 计算该频率的峭度
                mean_p = np.mean(freq_power)
                std_p = np.std(freq_power)
                if std_p > 0:
                    # 标准化后计算四阶矩
                    normalized = (freq_power - mean_p) / std_p
                    kurt = np.mean(normalized ** 4)
                    sk_values.append(kurt)
                else:
                    sk_values.append(3.0)  # 正态分布默认值
            else:
                sk_values.append(3.0)
        
        sk_values = np.array(sk_values)
        
        # 3. 找到SK值最高的频带 (关注冲击成分)
        # 只考虑中高频 (低频通常是正常旋转)
        min_freq_idx = np.argmax(f >= 500)  # 从500Hz开始
        if min_freq_idx == 0 and f[0] < 500:
            min_freq_idx = len(f) // 4  # 如果没有500Hz，取1/4处
        
        sk_highfreq = sk_values[min_freq_idx:]
        f_highfreq = f[min_freq_idx:]
        
        if len(sk_highfreq) == 0:
            return 0.0
        
        # 找到SK最高的频率区域
        sk_threshold = np.percentile(sk_values, 75)  # 上四分位数
        high_sk_mask = sk_values >= sk_threshold
        
        # 4. 在关注区域计算平均功率谱
        avg_power = np.mean(power, axis=1)
        avg_power_db = 10 * np.log10(avg_power + 1e-10)
        
        # 5. 在SK高的区域寻找峰值
        if np.any(high_sk_mask):
            # 在SK高的频率中取平均功率谱的最大值
            esp_power = np.max(avg_power_db[high_sk_mask])
            
            # 计算背景噪声 (非高SK区域的中位数)
            low_sk_mask = ~high_sk_mask
            if np.any(low_sk_mask):
                noise_floor = np.median(avg_power_db[low_sk_mask])
            else:
                noise_floor = np.percentile(avg_power_db, 25)
            
            # 峰值相对于噪声的dB值
            ESP = esp_power - noise_floor
        else:
            # 如果没有明显的高SK区域，使用常规峰值
            ESP = np.max(avg_power_db) - np.percentile(avg_power_db, 25)
        
        return float(ESP)


class NonStationarityIndexExtractor(BaseFeatureExtractor):
    """
    非平稳性指标提取器 (P2)
    
    参考标准: 基于Ravindra & Halim 2008的时频分析方法
    单位: 无 (通常0-10)
    
    描述:
    量化信号非平稳程度的指标。
    通过分析信号的时频分布随时间的变化来评估信号的平稳性。
    高值表示信号特性随时间显著变化（如摩擦声、间歇性噪声）。
    
    应用场景: 间歇性噪声检测、摩擦声、时变信号分析
    """
    
    def __init__(self):
        super().__init__(
            name="非平稳性指标",
            symbol="NSI",
            unit="-",
            stage="P2",
            description="信号非平稳性程度，时频分布变化率"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取非平稳性指标
        
        算法步骤:
        1. 将信号分段
        2. 计算每段的频谱特征
        3. 计算相邻段频谱的差异
        4. 综合差异程度得到NSI
        
        数学定义:
        NSI = mean(|S(i,f) - S(i-1,f)|) / mean(S(i,f))
        其中S(i,f)是第i段在频率f处的功率谱
        """
        # 参数设置
        segment_duration = 0.05  # 50ms每段
        segment_samples = int(segment_duration * sr)
        
        if segment_samples < 256:
            segment_samples = 256
        
        # 确保有足够的段
        min_segments = 4
        if len(sig) < segment_samples * min_segments:
            # 信号太短，返回低非平稳性
            return 0.5
        
        # 1. 分段并计算每段的频谱
        num_segments = len(sig) // segment_samples
        spectra = []
        
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = sig[start:end]
            
            # 计算功率谱
            fft_vals = np.abs(rfft(segment))
            power = fft_vals ** 2
            spectra.append(power)
        
        spectra = np.array(spectra)  # shape: (num_segments, freq_bins)
        
        # 2. 计算相邻段频谱的差异
        spectral_changes = []
        for i in range(1, num_segments):
            prev_spec = spectra[i-1]
            curr_spec = spectra[i]
            
            # 避免除零
            prev_spec = np.maximum(prev_spec, 1e-10)
            curr_spec = np.maximum(curr_spec, 1e-10)
            
            # 计算相对变化 (可以使用对数差异或线性差异)
            # 方法1: 对数差异 (dB)
            log_diff = np.abs(10 * np.log10(curr_spec / prev_spec))
            
            # 方法2: 归一化线性差异
            linear_diff = np.abs(curr_spec - prev_spec) / (curr_spec + prev_spec)
            
            # 综合两种方法
            combined_diff = (log_diff / 10 + linear_diff) / 2
            spectral_changes.append(np.mean(combined_diff))
        
        if not spectral_changes:
            return 0.0
        
        # 3. 综合非平稳性指标
        mean_change = np.mean(spectral_changes)
        max_change = np.max(spectral_changes)
        std_change = np.std(spectral_changes)
        
        # 综合计算NSI
        # 考虑平均变化、最大变化和变化的标准差
        NSI = mean_change + 0.5 * max_change + 0.3 * std_change
        
        # 缩放到0-10范围 (经验值)
        NSI_scaled = NSI * 5
        
        # 限制范围
        NSI_scaled = max(0.0, min(10.0, NSI_scaled))
        
        return float(NSI_scaled)


class FrequencyDriftExtractor(BaseFeatureExtractor):
    """
    频率漂移提取器 (附加特征)
    
    描述:
    检测主导频率随时间的漂移程度。
    用于评估音调的稳定性。
    """
    
    def __init__(self):
        super().__init__(
            name="频率漂移",
            symbol="FD",
            unit="Hz/s",
            stage="P2",
            description="主导频率的时变漂移率"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """提取频率漂移"""
        # 分段计算主导频率
        segment_duration = 0.1  # 100ms
        segment_samples = int(segment_duration * sr)
        
        if len(sig) < segment_samples * 2:
            return 0.0
        
        num_segments = len(sig) // segment_samples
        dominant_freqs = []
        
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = sig[start:end]
            
            # 计算频谱
            fft_vals = np.abs(rfft(segment))
            freqs = rfftfreq(len(segment), 1/sr)
            
            # 找到主导频率 (排除直流)
            if len(fft_vals) > 1:
                dominant_idx = np.argmax(fft_vals[1:]) + 1
                dominant_freqs.append(freqs[dominant_idx])
        
        if len(dominant_freqs) < 2:
            return 0.0
        
        # 计算频率变化率
        freq_changes = np.diff(dominant_freqs)
        drift_rate = np.mean(np.abs(freq_changes)) / segment_duration
        
        return float(drift_rate)
