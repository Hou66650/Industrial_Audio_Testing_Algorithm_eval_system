"""主特征提取器"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from loguru import logger

from .base_extractor import BaseFeatureExtractor
from .config import Config


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


class AudioFeatureExtractor:
    """
    音频特征提取器主类
    
    管理全部18个特征的提取
    """
    
    def __init__(self, stage: str = "P0", config: Optional[Config] = None):
        """
        初始化提取器
        
        Args:
            stage: 实施阶段 "P0"/"P1"/"P2"
            config: 配置对象
        """
        self.stage = stage
        self.config = config or Config(stage=stage)
        
        # 构建提取器字典
        self.extractors: Dict[str, BaseFeatureExtractor] = {}
        self._build_extractors()
        
        logger.info(f"特征提取器初始化完成 (阶段: {stage}, 特征数: {len(self.extractors)})")
    
    def _build_extractors(self):
        """构建提取器字典"""
        # 导入所有特征提取器
        from ..features.signal_features import (
            PeakSPLExtractor,
            TemporalKurtosisExtractor,
            EnergyRatioExtractor
        )
        from ..features.psychoacoustic import (
            AWeightedSPLExtractor,
            LoudnessExtractor,
            SharpnessExtractor,
            RoughnessExtractor
        )
        from ..features.p1_features import (
            FluctuationStrengthExtractor,
            TonalityIndexExtractor,
            AnnoyanceExtractor,
            ToneToNoiseRatioExtractor,
            SpectralIrregularityExtractor,
            SpectralKurtosisExtractor,
            AmplitudeModulationDepthExtractor
        )
        from ..features.p2_features import (
            EnhancedSpectrumPeakExtractor,
            NonStationarityIndexExtractor
        )
        
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
        current_level = stage_order.get(self.stage, 0)
        
        self.extractors.update(p0_extractors)
        
        if current_level >= 1:
            self.extractors.update(p1_extractors)
            
        if current_level >= 2:
            self.extractors.update(p2_extractors)
    
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
            symbol: 特征符号
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
    
    def extract_batch(
        self,
        signals: Dict[str, tuple],
        progress_callback=None
    ) -> Dict[str, Dict]:
        """
        批量提取特征
        
        Args:
            signals: {名称: (信号, 采样率)}
            progress_callback: 进度回调函数
            
        Returns:
            {名称: 特征结果}
        """
        results = {}
        total = len(signals)
        
        for i, (name, (signal, sr)) in enumerate(signals.items()):
            try:
                results[name] = self.extract(signal, sr)
                
                if progress_callback:
                    progress_callback(i + 1, total, name)
                    
            except Exception as e:
                logger.error(f"批量处理失败 [{name}]: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def to_dataframe(self, results: Dict[str, Dict]) -> 'pd.DataFrame':
        """
        将结果转换为DataFrame
        
        Args:
            results: extract_batch的结果
            
        Returns:
            DataFrame
        """
        import pandas as pd
        
        rows = []
        for sample_name, features in results.items():
            row = {'sample_name': sample_name}
            
            # 添加标签 (从文件名推断)
            if 'ng' in sample_name.lower():
                row['label'] = 'NG'
            elif 'ok' in sample_name.lower():
                row['label'] = 'OK'
            else:
                row['label'] = 'UNKNOWN'
            
            # 添加特征值
            for symbol, data in features.items():
                if isinstance(data, dict) and 'value' in data:
                    row[symbol] = data['value']
                else:
                    row[symbol] = None
            
            rows.append(row)
        
        return pd.DataFrame(rows)
