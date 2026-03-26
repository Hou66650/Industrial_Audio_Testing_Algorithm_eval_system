"""
P1阶段特征提取器

包含7个进阶特征:
1. 波动强度 F (Fluctuation Strength)
2. 音调指数 TI (Tonality Index)
3. 烦恼度 A (Annoyance)
4. 音噪比 TNR (Tone-to-Noise Ratio)
5. 频谱不规则度 SI (Spectral Irregularity)
6. 频谱峭度 SK (Spectral Kurtosis)
7. 幅度调制深度 AMD (Amplitude Modulation Depth)
"""
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Optional, Tuple

from ..core.base_extractor import BaseFeatureExtractor


class FluctuationStrengthExtractor(BaseFeatureExtractor):
    """
    波动强度提取器 (P1)
    
    参考标准: Zwicker & Fastl
    单位: vacil
    
    描述: 感知低于20Hz调制的强度，如发动机转速波动
    """
    
    def __init__(self):
        super().__init__(
            name="波动强度",
            symbol="F",
            unit="vacil",
            stage="P1",
            description="Zwicker波动强度，感知低频调制"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取波动强度
        
        算法:
        1. 提取信号包络 (Hilbert变换)
        2. 分析包络在<20Hz的调制能量
        3. 考虑调制频率的影响 (4Hz时感知最强)
        """
        # 提取包络
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(sig))
        
        # 去除直流分量
        envelope = envelope - np.mean(envelope)
        
        # 包络频谱
        env_fft = np.abs(rfft(envelope))
        env_freqs = rfftfreq(len(envelope), 1/sr)
        
        # 只考虑<20Hz的调制
        mod_mask = env_freqs < 20
        mod_energy = np.sum(env_fft[mod_mask] ** 2)
        
        # 找到主导调制频率
        if np.any(mod_mask) and np.sum(env_fft[mod_mask]) > 0:
            peak_idx = np.argmax(env_fft[mod_mask])
            f_mod = env_freqs[mod_mask][peak_idx]
        else:
            f_mod = 4.0  # 默认4Hz (人耳最敏感)
        
        # 波动强度公式
        # F = mod_depth / (f_mod/4 + 4/f_mod)
        total_energy = np.sum(env_fft ** 2)
        if total_energy == 0:
            return 0.0
        
        mod_depth = np.sqrt(mod_energy / total_energy)
        
        # 频率加权 (4Hz时最大)
        freq_weight = 1.0 / (f_mod/4 + 4/f_mod + 0.1)
        
        F = mod_depth * freq_weight * 5.0  # 缩放到合理范围
        
        return float(F)


class TonalityIndexExtractor(BaseFeatureExtractor):
    """
    音调指数提取器 (P1)
    
    参考标准: ECMA-418-2 (Hearing Model)
    单位: 无 (tu)
    
    描述: 感知音调的程度，反映信号中纯音成分的强度
    """
    
    def __init__(self):
        super().__init__(
            name="音调指数",
            symbol="TI",
            unit="-",
            stage="P1",
            description="ECMA-418-2音调指数，感知纯音程度"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取音调指数
        
        算法:
        1. 计算频谱
        2. 检测频谱中的峰值 (纯音)
        3. 计算峰值的尖锐度和突出度
        4. 综合为音调指数
        """
        # FFT
        spec = np.abs(rfft(sig))
        freqs = rfftfreq(len(sig), 1/sr)
        
        # 转换为dB
        spec_db = 20 * np.log10(spec + 1e-10)
        
        # 检测频谱峰值
        peaks, properties = signal.find_peaks(spec_db, height=np.max(spec_db)-30, distance=5)
        
        if len(peaks) == 0:
            return 0.0
        
        # 计算音调指数
        # 基于峰值数量、高度和尖锐度
        tonality = 0.0
        
        for peak in peaks:
            # 峰值高度 (相对最大峰值)
            peak_height = spec_db[peak] - np.max(spec_db)
            
            # 峰值尖锐度 (半高宽)
            if peak > 0 and peak < len(spec_db) - 1:
                left = spec_db[peak] - spec_db[peak-1]
                right = spec_db[peak] - spec_db[peak+1]
                sharpness = (left + right) / 2
            else:
                sharpness = 0
            
            # 加权求和
            tonality += (1 + peak_height/30) * (1 + sharpness/10)
        
        # 归一化
        TI = np.log1p(tonality) / 2
        
        return float(TI)


class AnnoyanceExtractor(BaseFeatureExtractor):
    """
    烦恼度提取器 (P1)
    
    参考标准: Aures (综合模型)
    单位: 无
    
    描述: 综合响度N、尖锐度S、粗糙度R的主观烦恼程度
    公式: A = N * (1 + |S|)^0.5 * (1 + R)^0.5
    """
    
    def __init__(self):
        super().__init__(
            name="烦恼度",
            symbol="A",
            unit="-",
            stage="P1",
            description="Aures综合烦恼度模型"
        )
        # 依赖其他特征
        self._loudness_extractor = None
        self._sharpness_extractor = None
        self._roughness_extractor = None
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取烦恼度
        
        需要先计算N、S、R三个特征
        """
        # 导入依赖的提取器
        from .psychoacoustic import LoudnessExtractor, SharpnessExtractor, RoughnessExtractor
        
        # 计算依赖特征
        if self._loudness_extractor is None:
            self._loudness_extractor = LoudnessExtractor()
            self._sharpness_extractor = SharpnessExtractor()
            self._roughness_extractor = RoughnessExtractor()
        
        N = self._loudness_extractor.safe_extract(sig, sr)
        S = self._sharpness_extractor.safe_extract(sig, sr)
        R = self._roughness_extractor.safe_extract(sig, sr)
        
        # 如果任一特征失败，使用默认值
        if N is None:
            N = 100
        if S is None:
            S = 1.0
        if R is None:
            R = 0.5
        
        # Aures烦恼度公式
        # A = N * (1 + |S|)^0.5 * (1 + R)^0.5
        A = N * np.sqrt(1 + abs(S)) * np.sqrt(1 + R)
        
        # 缩放到合理范围 (0-10)
        A_scaled = A / 50
        
        return float(A_scaled)


class ToneToNoiseRatioExtractor(BaseFeatureExtractor):
    """
    音噪比提取器 (P1)
    
    参考标准: ISO 7779 / ECMA-74
    单位: dB
    
    描述: 纯音成分相对于噪声成分的能量比
    """
    
    def __init__(self):
        super().__init__(
            name="音噪比",
            symbol="TNR",
            unit="dB",
            stage="P1",
            description="ISO 7779音噪比"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取音噪比
        
        算法:
        1. 计算功率谱
        2. 检测频谱峰值 (纯音)
        3. 估计背景噪声
        4. 计算峰值与噪声的差值
        """
        # 计算功率谱
        n_fft = min(8192, len(sig))
        freqs, psd = signal.welch(sig, sr, nperseg=n_fft)
        
        # 转换为dB
        psd_db = 10 * np.log10(psd + 1e-10)
        
        # 检测峰值 (纯音)
        peaks, properties = signal.find_peaks(psd_db, height=np.max(psd_db)-20, distance=10)
        
        if len(peaks) == 0:
            return 0.0
        
        # 找到最大峰值
        max_peak_idx = peaks[np.argmax(psd_db[peaks])]
        max_peak_db = psd_db[max_peak_idx]
        
        # 估计背景噪声 (使用局部最小值或平滑谱)
        # 使用移动平均平滑
        smoothed = np.convolve(psd_db, np.ones(10)/10, mode='same')
        
        # 峰值周围的噪声 (排除峰值本身)
        noise_region = np.concatenate([psd_db[:max(0, max_peak_idx-5)], 
                                       psd_db[max_peak_idx+6:]])
        noise_floor = np.percentile(noise_region, 50)
        
        # 音噪比
        TNR = max_peak_db - noise_floor
        
        return float(TNR)


class SpectralIrregularityExtractor(BaseFeatureExtractor):
    """
    频谱不规则度提取器 (P1)
    
    参考标准: Krimphoff et al. 1994
    单位: 无
    
    描述: 频谱偏离平滑包络的程度，反映频谱的不规则性
    """
    
    def __init__(self):
        super().__init__(
            name="频谱不规则度",
            symbol="SI",
            unit="-",
            stage="P1",
            description="Krimphoff频谱不规则度"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取频谱不规则度
        
        算法:
        1. 计算频谱
        2. 计算频谱包络 (平滑处理)
        3. 计算实际频谱与包络的差异
        """
        # 计算频谱
        spec = np.abs(rfft(sig))
        
        # 计算频谱包络 (使用移动平均)
        window_size = max(5, len(spec) // 50)
        envelope = np.convolve(spec, np.ones(window_size)/window_size, mode='same')
        
        # 避免除零
        envelope = np.maximum(envelope, 1e-10)
        
        # 计算不规则度 (实际与包络的差异)
        irregularity = np.mean(np.abs(spec - envelope) / envelope)
        
        # 另一种计算: 相邻频带的差分
        diff = np.diff(spec)
        irregularity_alt = np.sum(np.abs(diff)) / np.sum(spec)
        
        # 综合
        SI = (irregularity + irregularity_alt) / 2
        
        return float(SI)


class SpectralKurtosisExtractor(BaseFeatureExtractor):
    """
    频谱峭度提取器 (P1)
    
    参考标准: Antoni & Randall 2006
    单位: 无
    
    描述: 频域信号的峭度，检测频域中的冲击成分
    """
    
    def __init__(self):
        super().__init__(
            name="频谱峭度",
            symbol="SK",
            unit="-",
            stage="P1",
            description="Antoni频谱峭度"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取频谱峭度
        
        算法:
        1. 计算STFT得到时变频谱
        2. 对每个频率点计算峭度
        3. 加权平均得到整体频谱峭度
        """
        from scipy.stats import kurtosis
        
        # 计算STFT
        f, t, Zxx = signal.stft(sig, sr, nperseg=min(2048, len(sig)//4))
        
        # 功率谱
        power = np.abs(Zxx) ** 2
        
        # 对每个频率点计算峭度
        kurtosis_values = []
        for i in range(power.shape[0]):
            freq_power = power[i, :]
            if np.std(freq_power) > 0:
                k_val = kurtosis(freq_power, fisher=False)
                # 加权 (高频权重更大，通常包含更多故障信息)
                weight = (i + 1) / power.shape[0]
                kurtosis_values.append(k_val * weight)
        
        if not kurtosis_values:
            return 3.0  # 正态分布默认值
        
        # 加权平均
        SK = np.sum(kurtosis_values) / np.sum([(i + 1) / power.shape[0] 
                                               for i in range(len(kurtosis_values))])
        
        return float(SK)


class AmplitudeModulationDepthExtractor(BaseFeatureExtractor):
    """
    幅度调制深度提取器 (P1)
    
    参考标准: Zwicker & Fastl
    单位: %
    
    描述: 信号幅度调制的程度，反映AM调制深度
    """
    
    def __init__(self):
        super().__init__(
            name="幅度调制深度",
            symbol="AMD",
            unit="%",
            stage="P1",
            description="幅度调制深度百分比"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取幅度调制深度
        
        算法:
        1. 提取包络
        2. 计算包络的变化范围
        3. 计算调制深度 = (max-min)/(max+min)
        """
        # 提取包络
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(sig))
        
        # 去除趋势 (使用高通滤波或去均值)
        envelope_detrended = envelope - np.mean(envelope)
        
        # 计算调制深度
        env_max = np.max(envelope)
        env_min = np.max([np.min(envelope), 0.001])  # 避免负值或零
        env_mean = np.mean(envelope)
        
        if env_mean == 0:
            return 0.0
        
        # 调制深度定义: (max - min) / (2 * mean) * 100%
        modulation_depth = (env_max - env_min) / (2 * env_mean) * 100
        
        # 使用标准差作为备选
        cv = np.std(envelope) / env_mean * 100  # 变异系数
        
        # 综合 (取较小值，避免异常)
        AMD = min(abs(modulation_depth), cv * 2)
        
        # 限制在合理范围
        AMD = min(100.0, max(0.0, AMD))
        
        return float(AMD)
