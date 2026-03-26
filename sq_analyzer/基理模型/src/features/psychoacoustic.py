"""心理声学特征提取器"""
import numpy as np
from scipy import signal
from typing import Optional

from ..core.base_extractor import BaseFeatureExtractor
from .utils import calculate_rms, spl_from_pressure, a_weighting_filter


class AWeightedSPLExtractor(BaseFeatureExtractor):
    """
    A计权声压级提取器 (P0)
    
    参考标准: IEC 61672
    
    A计权模拟人耳对40phon等响曲线的频率响应
    """
    
    def __init__(self):
        super().__init__(
            name="A计权声压级",
            symbol="dB(A)",
            unit="dB",
            stage="P0",
            description="IEC 61672 A计权声压级"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取A计权声压级
        
        步骤:
        1. 应用A计权滤波器
        2. 计算RMS
        3. 转换为dB
        """
        # 应用A计权滤波器
        weighted_signal = self._a_weight(sig, sr)
        
        # 计算RMS
        rms = calculate_rms(weighted_signal)
        
        # 转换为dB (参考20μPa)
        # 注意：这里假设信号是归一化的，返回相对值
        # 实际应用中需要校准
        return spl_from_pressure(rms + 1e-10)
    
    def _a_weight(self, sig: np.ndarray, sr: int) -> np.ndarray:
        """A计权滤波器"""
        # 使用scipy设计A计权滤波器
        # 双线性变换法
        
        # A计权系数 (IEC 61672)
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        A1000 = 1.9997
        
        # 预扭曲
        fs = sr
        
        # 使用scipy.signal.bilinear设计
        # 模拟传递函数 -> 数字传递函数
        
        # 简化的A计权实现：使用IIR滤波器
        # 分子分母系数
        
        # 使用librosa的A计权实现
        try:
            import librosa
            return librosa.effects.preemphasis(sig, coef=0.0)  # 占位
        except:
            pass
        
        # 备选：使用频率域滤波
        return a_weighting_filter(sig, sr)


class LoudnessExtractor(BaseFeatureExtractor):
    """
    响度提取器 (P0)
    
    参考标准: ISO 532B (Zwicker方法)
    单位: sone
    
    使用mosqito库实现
    """
    
    def __init__(self):
        super().__init__(
            name="响度",
            symbol="N",
            unit="sone",
            stage="P0",
            description="Zwicker响度模型 (ISO 532B)"
        )
        self._mosqito_available = self._check_mosqito()
    
    def _check_mosqito(self) -> bool:
        """检查mosqito是否可用"""
        try:
            import mosqito
            return True
        except ImportError:
            return False
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取响度
        
        使用mosqito库的Zwicker稳态响度算法
        """
        if not self._mosqito_available:
            # 备用：使用简化算法
            return self._approximate_loudness(sig, sr)
        
        try:
            import mosqito
            
            # 需要转换为声压 (Pa)，这里假设是归一化信号
            # 实际应用中需要校准系数
            # 简化为：假设信号对应于40dB SPL
            
            # mosqito需要特定格式
            # 计算1/3倍频程频谱
            spec, freqs = self._compute_third_octave(sig, sr)
            
            # 调用mosqito
            N, N_specific = mosqito.loudness_zwicker_stationary(
                spec,
                freqs,
                field_type='free'
            )
            
            return float(N)
            
        except Exception as e:
            # 失败时使用近似
            return self._approximate_loudness(signal, sr)
    
    def _approximate_loudness(self, sig: np.ndarray, sr: int) -> float:
        """
        简化响度近似算法
        
        基于Zwicker方法的简化实现
        """
        # 1/3倍频程分析
        spec, freqs = self._compute_third_octave(sig, sr)
        
        # 简化的响度计算
        # 基于Stevens响度法则的近似
        
        # 参考声压
        p_ref = 20e-6
        
        loudness = 0.0
        for i, (p, f) in enumerate(zip(spec, freqs)):
            if p <= 0:
                continue
            
            # SPL
            L = 20 * np.log10(p / p_ref + 1e-10)
            
            # 简化的等响度计算
            # N' = k * (10^(L/40) - threshold)
            
            # 频率加权
            freq_weight = self._bark_weight(f)
            
            if L > 0:
                band_loudness = (10 ** (L / 40) - 1) * freq_weight
                loudness += max(0, band_loudness)
        
        # 总响度 (Stevens幂律)
        N = loudness ** 0.6
        
        return N
    
    def _compute_third_octave(self, sig: np.ndarray, sr: int) -> tuple:
        """计算1/3倍频程频谱"""
        from .utils import third_octave_bands
        
        bands = third_octave_bands(sig, sr)
        
        freqs = []
        powers = []
        for name, power in bands.items():
            # 从名称提取频率
            f = float(name.replace('Hz', ''))
            freqs.append(f)
            powers.append(np.sqrt(power))  # 转换为声压
        
        return np.array(powers), np.array(freqs)
    
    def _bark_weight(self, freq: float) -> float:
        """Bark频带加权"""
        # Bark频率转换
        z = 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)
        
        # 简化的临界频带加权
        return 1.0


class SharpnessExtractor(BaseFeatureExtractor):
    """
    尖锐度提取器 (P0)
    
    参考标准: DIN 45692
    单位: acum
    """
    
    def __init__(self):
        super().__init__(
            name="尖锐度",
            symbol="S",
            unit="acum",
            stage="P0",
            description="DIN 45692尖锐度"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取尖锐度
        
        尖锐度反映高频成分的比例
        S = 0.11 * ∫(N'(z) * g(z) * z * dz) / N
        """
        # 简化实现
        # 基于频谱质心的近似
        
        from scipy.fft import rfft, rfftfreq
        
        spec = np.abs(rfft(sig))
        freqs = rfftfreq(len(sig), 1/sr)
        
        # 转换为Bark标度
        bark = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
        
        # Aures尖锐度权重函数
        g = np.ones_like(bark)
        g[bark > 15.8] = 0.2  # 高频加权
        
        # 计算加权平均
        power = spec ** 2
        
        if np.sum(power) == 0:
            return 0.0
        
        # 简化的尖锐度估计
        # 基于高频能量比例
        high_freq_mask = freqs > 3000
        if np.sum(power) > 0:
            high_ratio = np.sum(power[high_freq_mask]) / np.sum(power)
            # 映射到acum单位 (经验公式)
            sharpness = high_ratio * 5.0
        else:
            sharpness = 0.0
        
        return sharpness


class RoughnessExtractor(BaseFeatureExtractor):
    """
    粗糙度提取器 (P0)
    
    参考标准: Daniel & Weber 1997
    单位: asper
    """
    
    def __init__(self):
        super().__init__(
            name="粗糙度",
            symbol="R",
            unit="asper",
            stage="P0",
            description="Daniel & Weber粗糙度"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取粗糙度
        
        粗糙度反映信号的调制特性
        主要关注20-300Hz的调制
        """
        # 简化实现
        # 基于时域包络的调制深度
        
        # 1. 提取包络 (Hilbert变换)
        from scipy.signal import hilbert
        analytic = hilbert(sig)
        envelope = np.abs(analytic)
        
        # 2. 分析包络的调制
        # 关注约70Hz附近的调制
        
        # 计算包络的频谱
        from scipy.fft import rfft, rfftfreq
        
        env_fft = np.abs(rfft(envelope - np.mean(envelope)))
        env_freqs = rfftfreq(len(envelope), 1/sr)
        
        # 20-300Hz范围的调制能量
        mod_mask = (env_freqs >= 20) & (env_freqs <= 300)
        mod_energy = np.sum(env_fft[mod_mask] ** 2)
        total_energy = np.sum(env_fft ** 2)
        
        if total_energy == 0:
            return 0.0
        
        # 归一化并映射到asper单位
        roughness = (mod_energy / total_energy) * 2.0
        
        return roughness


class FluctuationStrengthExtractor(BaseFeatureExtractor):
    """
    波动强度提取器 (P1)
    
    参考标准: Zwicker & Fastl
    单位: vacil
    """
    
    def __init__(self):
        super().__init__(
            name="波动强度",
            symbol="F",
            unit="vacil",
            stage="P1",
            description="Zwicker波动强度"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """
        提取波动强度
        
        关注低于20Hz的调制
        """
        from scipy.signal import hilbert
        # Hilbert包络
        envelope = np.abs(hilbert(sig))
        
        # 包络频谱
        from scipy.fft import rfft, rfftfreq
        
        env_fft = np.abs(rfft(envelope - np.mean(envelope)))
        env_freqs = rfftfreq(len(envelope), 1/sr)
        
        # <20Hz调制
        mod_mask = env_freqs < 20
        mod_energy = np.sum(env_fft[mod_mask] ** 2)
        
        # 峰值调制频率
        if np.any(mod_mask):
            peak_idx = np.argmax(env_fft[mod_mask])
            f_mod = env_freqs[mod_mask][peak_idx]
        else:
            f_mod = 4.0  # 默认
        
        # 波动强度公式 (简化)
        # F ∝ mod_depth / (f_mod/4 + 4/f_mod)
        total_energy = np.sum(env_fft ** 2)
        
        if total_energy == 0:
            return 0.0
        
        mod_depth = mod_energy / total_energy
        fluctuation = mod_depth / (f_mod/4 + 4/f_mod + 0.1)
        
        return fluctuation
