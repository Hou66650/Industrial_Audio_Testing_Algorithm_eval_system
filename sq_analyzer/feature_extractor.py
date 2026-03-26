"""
SmartQuality 音频特征提取器 - 基理模型架构
整合P0/P1/P2三阶段共18个特征

特征清单:
P0 (9个): SPL_peak, K, E_low, E_mid, E_high, dB(A), N, S, R
P1 (7个): F, TI, A, TNR, SI, SK, AMD
P2 (2个): ESP, NSI
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert, find_peaks
import librosa

# 尝试导入loguru，如不可用则使用标准logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)

# =============================================================================
# 工具函数
# =============================================================================

def a_weighting_filter(sig: np.ndarray, sr: int) -> np.ndarray:
    """
    应用A计权滤波器 (IEC 61672-1:2013)
    
    Args:
        signal: 输入信号
        sr: 采样率
        
    Returns:
        A计权后的信号
    """
    freqs = np.fft.rfftfreq(len(sig), 1/sr)
    
    # A计权公式系数
    c1 = 12194.217 ** 2
    c2 = 20.598997 ** 2
    c3 = 107.65265 ** 2
    c4 = 737.86223 ** 2
    
    f_sq = freqs ** 2
    
    # A计权公式
    num = c1 * f_sq ** 2
    den = (f_sq + c2) * np.sqrt((f_sq + c3) * (f_sq + c4)) * (f_sq + c1)
    
    # 避免除零
    den = np.where(den == 0, 1e-10, den)
    
    A_weight = num / den
    A_weight = 20 * np.log10(A_weight) + 2.0  # 2dB偏移修正
    
    # 转换回线性
    A_linear = 10 ** (A_weight / 20)
    
    # FFT -> 应用权重 -> IFFT
    sig_fft = rfft(sig)
    sig_fft *= A_linear
    
    return np.fft.irfft(sig_fft, n=len(sig))


def calculate_rms(signal: np.ndarray) -> float:
    """计算RMS（均方根）值"""
    return np.sqrt(np.mean(signal ** 2))


def calculate_peak(signal: np.ndarray) -> float:
    """计算峰值"""
    return np.max(np.abs(signal))


def spl_from_pressure(p_rms: float, p_ref: float = 20e-6) -> float:
    """
    从声压计算声压级 (dB)
    
    Args:
        p_rms: RMS声压 (Pa)
        p_ref: 参考声压 (默认20μPa)
        
    Returns:
        声压级 (dB)
    """
    if p_rms <= 0:
        return -np.inf
    return 20 * np.log10(p_rms / p_ref)


def third_octave_bands(
    sig: np.ndarray,
    sr: int,
    freq_range: Tuple[float, float] = (20, 20000)
) -> Dict[str, np.ndarray]:
    """
    计算1/3倍频程频带能量 (IEC 61260-1:2014)
    
    Args:
        signal: 输入信号
        sr: 采样率
        freq_range: 频率范围 (Hz)
        
    Returns:
        {频段名称: 能量数组}
    """
    # IEC 61260-1:2014 标准1/3倍频程中心频率
    center_freqs = [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ]
    
    # 确保不超过奈奎斯特频率
    nyquist = sr / 2
    max_freq = min(freq_range[1], nyquist * 0.95)
    
    # 过滤频率范围
    center_freqs = [f for f in center_freqs if freq_range[0] <= f <= max_freq]
    
    bands = {}
    
    for fc in center_freqs:
        # 1/3倍频程带宽
        f_lower = fc / (2 ** (1/6))
        f_upper = fc * (2 ** (1/6))
        
        # 确保不超过奈奎斯特频率
        if f_upper >= nyquist:
            f_upper = nyquist * 0.99
        if f_lower >= f_upper:
            continue
        
        # 设计带通滤波器
        sos = signal.butter(
            N=4,
            Wn=[f_lower, f_upper],
            btype='band',
            fs=sr,
            output='sos'
        )
        
        # 滤波
        filtered = signal.sosfilt(sos, sig)
        
        # 计算能量
        bands[f"{fc}Hz"] = np.sum(filtered ** 2)
    
    return bands


def calculate_kurtosis(sig: np.ndarray) -> float:
    """
    计算峭度 (Kurtosis)
    
    Args:
        signal: 输入信号
        
    Returns:
        峭度值 (正态分布=3，超额峭度=0)
    """
    return stats.kurtosis(sig, fisher=False)


# =============================================================================
# 特征提取器基类
# =============================================================================

class BaseFeatureExtractor(ABC):
    """
    音频特征提取器基类
    
    所有特征提取器必须继承此类并实现 extract 方法
    """
    
    def __init__(
        self,
        name: str,
        symbol: str,
        unit: str,
        stage: str,
        description: str = "",
        version: str = "1.0"
    ):
        """
        初始化提取器
        
        Args:
            name: 特征中文名称
            symbol: 特征符号（如 dB(A), N, SPL_peak）
            unit: 单位
            stage: 实施阶段 P0/P1/P2
            description: 描述
            version: 实现版本
        """
        self.name = name
        self.symbol = symbol
        self.unit = unit
        self.stage = stage
        self.description = description
        self.version = version
        
    @abstractmethod
    def extract(self, signal: np.ndarray, sr: int) -> float:
        """
        提取特征值
        
        Args:
            signal: 音频信号数组 (1D numpy array)
            sr: 采样率 (Hz)
            
        Returns:
            单一浮点数值特征
        """
        pass
    
    def validate(self, value: float) -> bool:
        """
        验证特征值是否有效
        
        Args:
            value: 特征值
            
        Returns:
            是否有效
        """
        if value is None:
            return False
        if not isinstance(value, (int, float, np.number)):
            return False
        if np.isnan(value) or np.isinf(value):
            return False
        return True
    
    def safe_extract(self, signal: np.ndarray, sr: int) -> Optional[float]:
        """
        安全提取特征值（带异常处理）
        
        Args:
            signal: 音频信号
            sr: 采样率
            
        Returns:
            特征值，失败返回 None
        """
        try:
            # 验证输入
            if signal is None or len(signal) == 0:
                logger.warning(f"[{self.symbol}] 输入信号为空")
                return None
            
            if sr <= 0:
                logger.warning(f"[{self.symbol}] 采样率无效: {sr}")
                return None
            
            # 执行提取
            value = self.extract(signal, sr)
            
            # 验证输出
            if not self.validate(value):
                logger.warning(f"[{self.symbol}] 提取结果无效: {value}")
                return None
                
            return float(value)
            
        except Exception as e:
            logger.error(f"[{self.symbol}] 特征提取失败: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """获取提取器信息"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "unit": self.unit,
            "stage": self.stage,
            "description": self.description,
            "version": self.version
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.symbol}: {self.name})"


# =============================================================================
# P0 特征提取器
# =============================================================================

class PeakSPLExtractor(BaseFeatureExtractor):
    """
    峰值声压级提取器 (P0)
    
    参考标准: IEC 61672
    """
    
    def __init__(self):
        super().__init__(
            name="峰值声压级",
            symbol="SPL_peak",
            unit="dB",
            stage="P0",
            description="IEC 61672标准峰值声压级"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """提取峰值声压级"""
        peak = calculate_peak(sig)
        return 20 * np.log10(max(peak, 1e-10))


class TemporalKurtosisExtractor(BaseFeatureExtractor):
    """
    时域峭度提取器 (P0)
    
    参考标准: ISO 20286
    """
    
    def __init__(self):
        super().__init__(
            name="时域峭度",
            symbol="K",
            unit="-",
            stage="P0",
            description="时域信号峭度，正态分布时为3"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """提取时域峭度"""
        return float(stats.kurtosis(sig, fisher=False))


class EnergyRatioExtractor(BaseFeatureExtractor):
    """
    频带能量占比提取器 (P0)
    
    参考标准: IEC 61260 (1/3倍频程)
    """
    
    BAND_RANGES = {
        'low': (20, 200),      # 低频 20-200Hz
        'mid': (200, 2000),    # 中频 200Hz-2kHz
        'high': (2000, 20000)  # 高频 2k-20kHz
    }
    
    def __init__(self, band: str = 'low'):
        """
        Args:
            band: 'low', 'mid', 或 'high'
        """
        if band not in self.BAND_RANGES:
            raise ValueError(f"无效的频段: {band}")
        
        self.band = band
        band_names = {'low': '低频', 'mid': '中频', 'high': '高频'}
        symbols = {'low': 'E_low', 'mid': 'E_mid', 'high': 'E_high'}
        
        super().__init__(
            name=f"{band_names[band]}能量占比",
            symbol=symbols[band],
            unit="%",
            stage="P0",
            description=f"{self.BAND_RANGES[band][0]}-{self.BAND_RANGES[band][1]}Hz能量占比"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """提取指定频段能量占比"""
        # FFT
        fft_vals = np.abs(rfft(sig))
        freqs = rfftfreq(len(sig), 1/sr)
        
        # 计算功率谱
        power = fft_vals ** 2
        
        # 频率范围
        f_low, f_high = self.BAND_RANGES[self.band]
        
        # 找到对应频段的索引
        idx = (freqs >= f_low) & (freqs <= f_high)
        
        # 计算能量占比
        band_energy = np.sum(power[idx])
        total_energy = np.sum(power)
        
        if total_energy == 0:
            return 0.0
        
        return (band_energy / total_energy) * 100


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
        weighted_signal = a_weighting_filter(sig, sr)
        
        # 计算RMS
        rms = calculate_rms(weighted_signal)
        
        # 转换为dB (参考20μPa)
        return spl_from_pressure(rms + 1e-10)


class LoudnessExtractor(BaseFeatureExtractor):
    """
    响度提取器 (P0)
    
    参考标准: ISO 532B (Zwicker方法)
    单位: sone
    
    使用mosqito库实现，如不可用则使用简化算法
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
        
        使用mosqito库的Zwicker稳态响度算法，失败时使用近似算法
        """
        if not self._mosqito_available:
            return self._approximate_loudness(sig, sr)
        
        try:
            import mosqito
            
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
            logger.warning(f"mosqito响度计算失败，使用近似算法: {e}")
            return self._approximate_loudness(sig, sr)
    
    def _approximate_loudness(self, sig: np.ndarray, sr: int) -> float:
        """
        简化响度近似算法
        
        基于Zwicker方法的简化实现 (Stevens响度法则)
        """
        # 1/3倍频程分析
        spec, freqs = self._compute_third_octave(sig, sr)
        
        # 参考声压
        p_ref = 20e-6
        
        loudness = 0.0
        for i, (p, f) in enumerate(zip(spec, freqs)):
            if p <= 0:
                continue
            
            # SPL
            L = 20 * np.log10(p / p_ref + 1e-10)
            
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
        # 1. 提取包络 (Hilbert变换)
        analytic = hilbert(sig)
        envelope = np.abs(analytic)
        
        # 2. 分析包络的调制
        # 计算包络的频谱
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


# =============================================================================
# P1 特征提取器
# =============================================================================

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
        # 延迟初始化依赖的提取器
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
        # 计算STFT
        f, t, Zxx = signal.stft(sig, sr, nperseg=min(2048, len(sig)//4))
        
        # 功率谱
        power = np.abs(Zxx) ** 2
        
        # 对每个频率点计算峭度
        kurtosis_values = []
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
                    # 加权 (高频权重更大，通常包含更多故障信息)
                    weight = (i + 1) / power.shape[0]
                    kurtosis_values.append(kurt * weight)
                else:
                    kurtosis_values.append(3.0)
            else:
                kurtosis_values.append(3.0)
        
        if not kurtosis_values:
            return 3.0
        
        # 加权平均
        weights = [(i + 1) / power.shape[0] for i in range(len(kurtosis_values))]
        SK = np.sum(kurtosis_values) / np.sum(weights)
        
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


# =============================================================================
# P2 特征提取器
# =============================================================================

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
                    sk_values.append(3.0)
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


# =============================================================================
# 主特征提取器 (对外接口)
# =============================================================================

@dataclass
class FeatureResult:
    """特征提取结果"""
    symbol: str
    name: str
    value: Optional[float]
    unit: str
    stage: str
    success: bool
    error: Optional[str] = None


class SQFeatureExtractor:
    """
    SmartQuality 音频特征提取器主类 (对外接口)
    
    管理全部18个特征的提取 (P0:9个 + P1:7个 + P2:2个)
    
    特征清单:
    P0 (9个): SPL_peak, K, E_low, E_mid, E_high, dB(A), N, S, R
    P1 (7个): F, TI, A, TNR, SI, SK, AMD  
    P2 (2个): ESP, NSI
    """
    
    # 特征名称映射 (新符号 -> 原名称，用于兼容)
    FEATURE_NAME_MAP = {
        'SPL_peak': 'SPL_peak',
        'K': 'K_kurtosis',
        'E_low': 'E_low_ratio',
        'E_mid': 'E_mid_ratio',
        'E_high': 'E_high_ratio',
        'dB(A)': 'SPL_A',
        'N': 'Loudness_N',
        'S': 'Sharpness_S',
        'R': 'Roughness_R',
        'F': 'Fluctuation_F',
        'TI': 'TonalityIndex_TI',
        'A': 'Annoyance_A',
        'TNR': 'TNR',
        'SI': 'Spectral_Irregularity_SI',
        'SK': 'Spectral_Kurtosis_SK',
        'AMD': 'AMD',
        'ESP': 'EnhancedSpectrum_ESP',
        'NSI': 'NonStationarity_NSI',
    }
    
    def __init__(self, sr: int = 44100, stage: str = "P2", audio_root_dir: str = None):
        """
        初始化提取器
        
        Args:
            sr: 采样率，默认44100
            stage: 实施阶段 "P0"/"P1"/"P2"，默认"P2"提取全部特征
            audio_root_dir: 音频根目录（可选，保留兼容）
        """
        self.sr = sr
        self.stage = stage
        self.audio_root_dir = audio_root_dir
        self.p_ref = 2e-5
        
        # 构建提取器字典
        self.extractors: Dict[str, BaseFeatureExtractor] = {}
        self._build_extractors()
        
        logger.info(f"SQFeatureExtractor初始化完成 (阶段: {stage}, 特征数: {len(self.extractors)})")
    
    def _build_extractors(self):
        """构建提取器字典"""
        # P0 特征 (9个)
        p0_extractors = {
            'SPL_peak': PeakSPLExtractor(),
            'K': TemporalKurtosisExtractor(),
            'E_low': EnergyRatioExtractor('low'),
            'E_mid': EnergyRatioExtractor('mid'),
            'E_high': EnergyRatioExtractor('high'),
            'dB(A)': AWeightedSPLExtractor(),
            'N': LoudnessExtractor(),
            'S': SharpnessExtractor(),
            'R': RoughnessExtractor(),
        }
        
        # P1 特征 (7个)
        p1_extractors = {
            'F': FluctuationStrengthExtractor(),
            'TI': TonalityIndexExtractor(),
            'A': AnnoyanceExtractor(),
            'TNR': ToneToNoiseRatioExtractor(),
            'SI': SpectralIrregularityExtractor(),
            'SK': SpectralKurtosisExtractor(),
            'AMD': AmplitudeModulationDepthExtractor(),
        }
        
        # P2 特征 (2个)
        p2_extractors = {
            'ESP': EnhancedSpectrumPeakExtractor(),
            'NSI': NonStationarityIndexExtractor(),
        }
        
        # 根据阶段选择特征
        stage_order = {'P0': 0, 'P1': 1, 'P2': 2}
        current_level = stage_order.get(self.stage, 2)
        
        self.extractors.update(p0_extractors)
        
        if current_level >= 1:
            self.extractors.update(p1_extractors)
            
        if current_level >= 2:
            self.extractors.update(p2_extractors)
    
    def extract_slice_features(
        self, 
        wav_path: str, 
        start_time: float, 
        end_time: float
    ) -> tuple:
        """
        提取音频切片特征 (兼容原接口)
        
        Args:
            wav_path: WAV文件路径
            start_time: 开始时间 (秒)
            end_time: 结束时间 (秒)
            
        Returns:
            (features_dict, error_msg)
            features_dict: 特征字典，键为特征名，值为特征值
            error_msg: 错误信息，成功为None
        """
        try:
            s_start, s_end = float(start_time), float(end_time)
            duration = s_end - s_start
            
            if duration <= 0:
                return None, "非法切片时长"
            
            # 加载音频切片
            y, sr = librosa.load(wav_path, sr=self.sr, offset=s_start, duration=duration)
            
            if len(y) == 0:
                return None, "空音频切片"
            
            # 提取所有特征
            results = self.extract(y, sr)
            
            # 转换为简单的字典格式 (兼容原接口)
            features = {}
            for symbol, data in results.items():
                # 使用映射后的名称或原始符号
                feature_name = self.FEATURE_NAME_MAP.get(symbol, symbol)
                features[feature_name] = data['value']
            
            return features, None
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None, str(e)
    
    def extract(
        self,
        signal: np.ndarray,
        sr: int,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        提取所有特征
        
        Args:
            signal: 音频信号
            sr: 采样率
            return_dict: 是否返回字典格式
            
        Returns:
            特征结果字典
        """
        results = {}
        
        for symbol, extractor in self.extractors.items():
            try:
                value = extractor.safe_extract(signal, sr)
                results[symbol] = {
                    'value': value,
                    'name': extractor.name,
                    'unit': extractor.unit,
                    'stage': extractor.stage,
                    'success': value is not None
                }
            except Exception as e:
                logger.error(f"特征提取异常 [{symbol}]: {e}")
                results[symbol] = {
                    'value': None,
                    'name': extractor.name,
                    'unit': extractor.unit,
                    'stage': extractor.stage,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def extract_single(
        self,
        symbol: str,
        signal: np.ndarray,
        sr: int
    ) -> Optional[float]:
        """
        提取单个特征
        
        Args:
            symbol: 特征符号 (如 'K', 'SPL_peak', 'N')
            signal: 音频信号
            sr: 采样率
            
        Returns:
            特征值
        """
        if symbol not in self.extractors:
            logger.error(f"未知的特征符号: {symbol}")
            return None
        
        return self.extractors[symbol].safe_extract(signal, sr)
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称列表"""
        return list(self.extractors.keys())
    
    def get_feature_info(self, symbol: Optional[str] = None) -> Dict:
        """
        获取特征信息
        
        Args:
            symbol: 特征符号，None则返回所有
            
        Returns:
            特征信息字典
        """
        if symbol:
            if symbol in self.extractors:
                return self.extractors[symbol].get_info()
            return {}
        
        return {sym: ext.get_info() for sym, ext in self.extractors.items()}
    
    def extract_p0_physical_features(self, filename: str) -> Optional[Dict]:
        """
        提取P0级机理特征 (兼容旧接口)
        
        Args:
            filename: 音频文件名（相对路径）
            
        Returns:
            特征字典，失败返回None
        """
        import os
        
        if self.audio_root_dir is None:
            logger.error("未设置audio_root_dir")
            return None
        
        wav_path = os.path.join(self.audio_root_dir, filename)
        
        if not os.path.exists(wav_path):
            logger.error(f"文件不存在: {wav_path}")
            return None
        
        try:
            # 加载完整音频
            y, sr = librosa.load(wav_path, sr=self.sr)
            
            if len(y) == 0:
                return None
            
            # 临时切换到P0阶段
            original_stage = self.stage
            self.stage = "P0"
            self._build_extractors()
            
            # 提取特征
            results = self.extract(y, sr)
            
            # 恢复阶段
            self.stage = original_stage
            self._build_extractors()
            
            # 转换为简单字典
            features = {}
            for symbol, data in results.items():
                feature_name = self.FEATURE_NAME_MAP.get(symbol, symbol)
                features[feature_name] = data['value']
            
            return features
            
        except Exception as e:
            logger.error(f"P0特征提取失败: {e}")
            return None


# =============================================================================
# 向后兼容的别名
# =============================================================================

# 保持旧接口兼容
AudioFeatureExtractor = SQFeatureExtractor


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("SmartQuality 特征提取器 - 基理模型架构")
    print("=" * 60)
    
    extractor = SQFeatureExtractor(stage="P2")
    
    print(f"\n共加载 {len(extractor.get_feature_names())} 个特征:")
    
    # 按阶段分组显示
    for stage in ["P0", "P1", "P2"]:
        stage_features = [
            (sym, ext.name) 
            for sym, ext in extractor.extractors.items() 
            if ext.stage == stage
        ]
        print(f"\n【{stage}阶段】({len(stage_features)}个):")
        for sym, name in stage_features:
            print(f"  - {sym}: {name}")
    
    print("\n" + "=" * 60)
    print("初始化完成！")
    print("=" * 60)
