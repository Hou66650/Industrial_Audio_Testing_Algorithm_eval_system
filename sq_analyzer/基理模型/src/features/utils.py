"""特征计算工具函数"""
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Tuple, Dict, List


# A计权滤波器系数 (IEC 61672-1:2013)
def a_weighting_filter(sig: np.ndarray, sr: int) -> np.ndarray:
    """
    应用A计权滤波器
    
    Args:
        signal: 输入信号
        sr: 采样率
        
    Returns:
        A计权后的信号
    """
    # A计权滤波器 - 使用双线性变换设计的IIR滤波器
    # 参考: IEC 61672-1:2013
    
    # 定义A计权系数
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    
    # 模拟滤波器系数
    A1000 = 1.9997  # 1kHz增益
    
    # 数字滤波器设计
    # 使用scipy的bilinear变换
    # 这里使用简化的A计权近似
    
    # 预扭曲频率
    fs = sr
    
    # A计权近似 - 使用双二阶带通滤波器
    # 10^((A(f))/20) 的频率响应
    
    # 定义极点/零点 (模拟域)
    zeros = [0, 0, 2 * np.pi * f4]
    poles = [2 * np.pi * f1, 2 * np.pi * f2, 2 * np.pi * f3, 2 * np.pi * f4]
    
    # 双线性变换到数字域
    # 简化实现：使用已有的A计权曲线拟合
    
    # 使用FIR近似实现（更稳定）
    # A计权曲线的FIR滤波器系数
    
    # 替代方案：使用scipy的频率响应直接计算
    freqs = np.fft.rfftfreq(len(sig), 1/sr)
    
    # 计算A计权曲线
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
    计算1/3倍频程频带能量
    
    Args:
        signal: 输入信号
        sr: 采样率
        freq_range: 频率范围 (Hz)
        
    Returns:
        {频段名称: 能量数组}
    """
    # IEC 61260-1:2014 标准1/3倍频程中心频率
    # 20Hz - 20kHz 范围
    
    center_freqs = [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ]
    
    # 确保不超过奈奎斯特频率 (fs/2)
    nyquist = sr / 2
    max_freq = min(freq_range[1], nyquist * 0.95)  # 留5%余量
    
    # 过滤频率范围
    center_freqs = [f for f in center_freqs if freq_range[0] <= f <= max_freq]
    
    bands = {}
    
    nyquist = sr / 2
    
    for fc in center_freqs:
        # 1/3倍频程带宽: fc * (2^(1/6) - 2^(-1/6))
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
    from scipy.stats import kurtosis
    return kurtosis(sig, fisher=False)  # fisher=False返回标准峭度


def calculate_spectral_centroid(sig: np.ndarray, sr: int) -> float:
    """计算频谱质心"""
    spec = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), 1/sr)
    
    return np.sum(freqs * spec) / np.sum(spec)


def calculate_spectral_rolloff(sig: np.ndarray, sr: int, roll_percent: float = 0.85) -> float:
    """计算频谱滚降点"""
    spec = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), 1/sr)
    
    cumsum = np.cumsum(spec)
    threshold = roll_percent * cumsum[-1]
    
    idx = np.where(cumsum >= threshold)[0]
    if len(idx) > 0:
        return freqs[idx[0]]
    return 0.0


def calculate_zero_crossing_rate(sig: np.ndarray) -> float:
    """计算过零率"""
    return np.mean(np.diff(np.sign(sig)) != 0)
