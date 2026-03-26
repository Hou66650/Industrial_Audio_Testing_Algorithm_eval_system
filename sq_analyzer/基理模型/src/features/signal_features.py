"""信号特征提取器 - 物理/信号维度"""
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Dict

from ..core.base_extractor import BaseFeatureExtractor
from .utils import (
    calculate_rms, calculate_peak, spl_from_pressure,
    third_octave_bands, calculate_kurtosis as calc_kurtosis
)


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
        """
        提取峰值声压级
        
        注意：这里假设输入信号已经是声压值（相对于20μPa归一化）
        如果输入是归一化的[-1, 1]信号，需要知道校准值
        """
        peak = calculate_peak(sig)
        # 假设信号归一化到满量程，转换为声压级
        # 这里简化处理，使用相对dB值
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
        # 使用scipy的kurtosis，fisher=False返回标准峭度（正态=3）
        from scipy.stats import kurtosis
        return float(kurtosis(sig, fisher=False))


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


class ZeroCrossingRateExtractor(BaseFeatureExtractor):
    """
    过零率提取器 (附加特征)
    """
    
    def __init__(self):
        super().__init__(
            name="过零率",
            symbol="ZCR",
            unit="-",
            stage="P0",
            description="信号过零率"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """提取过零率"""
        zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
        return zero_crossings / len(sig)


class RMSExtractor(BaseFeatureExtractor):
    """
    RMS能量提取器 (附加特征)
    """
    
    def __init__(self):
        super().__init__(
            name="RMS能量",
            symbol="RMS",
            unit="-",
            stage="P0",
            description="信号RMS能量"
        )
    
    def extract(self, sig: np.ndarray, sr: int) -> float:
        """提取RMS"""
        return calculate_rms(sig)
