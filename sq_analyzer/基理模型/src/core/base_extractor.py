"""特征提取器基类"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from loguru import logger


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
        # 支持numpy数值类型
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
