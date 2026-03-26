"""
特征归一化与量化模块

支持两种归一化模式：
1. 自适应模式 (adaptive): 基于已有数据计算Min-Max进行归一化
2. 固定模式 (fixed): 使用预设的通用范围进行归一化

输出：0-100的量化分数
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class FeatureRange:
    """特征范围定义"""
    min_val: float
    max_val: float
    name: str = ""


class FeatureNormalizer:
    """
    特征归一化器
    
    将原始特征值转换为0-100的量化分数
    """
    
    # 预设的固定范围（基于声学常识和文档参考值）
    DEFAULT_RANGES = {
        # 信号特征
        'SPL_peak': FeatureRange(-30, 10, '峰值声压级'),  # dB
        'K': FeatureRange(0, 100, '时域峭度'),  # 正常信号3，冲击可达几十
        'E_low': FeatureRange(0, 100, '低频能量占比'),  # %
        'E_mid': FeatureRange(0, 100, '中频能量占比'),  # %
        'E_high': FeatureRange(0, 100, '高频能量占比'),  # %
        
        # 心理声学特征
        'dB(A)': FeatureRange(40, 100, 'A计权声压级'),  # dB(A), 正常40-100
        'N': FeatureRange(0, 500, '响度'),  # sone, 正常0-500
        'S': FeatureRange(0, 5, '尖锐度'),  # acum, 正常0-5
        'R': FeatureRange(0, 2, '粗糙度'),  # asper, 正常0-2
        
        # P1阶段特征（预留）
        'F': FeatureRange(0, 5, '波动强度'),  # vacil
        'TI': FeatureRange(0, 5, '音调指数'),  # -
        'A': FeatureRange(0, 10, '烦恼度'),  # -
        'TNR': FeatureRange(-10, 30, '音噪比'),  # dB
        'SI': FeatureRange(0, 5, '频谱不规则度'),  # -
        'SK': FeatureRange(0, 50, '频谱峭度'),  # -
        'AMD': FeatureRange(0, 100, '幅度调制深度'),  # %
        'ESP': FeatureRange(0, 30, '增强频谱峰值'),  # dB (相对于噪声的峰值)
        'NSI': FeatureRange(0, 10, '非平稳性指标'),  # -
    }
    
    def __init__(self, mode: str = 'both'):
        """
        初始化归一化器
        
        Args:
            mode: 'adaptive'(自适应), 'fixed'(固定), 'both'(两者)
        """
        self.mode = mode
        self.adaptive_ranges: Dict[str, FeatureRange] = {}
        self.fixed_ranges = self.DEFAULT_RANGES.copy()
        
    def fit(self, data: Dict[str, List[float]]) -> 'FeatureNormalizer':
        """
        基于数据计算自适应范围（Min-Max）
        
        Args:
            data: {特征名: [值列表]}
        """
        for feature_name, values in data.items():
            if not values:
                continue
                
            valid_values = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]
            if not valid_values:
                continue
            
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            # 添加10%的余量
            margin = (max_val - min_val) * 0.1 if max_val != min_val else 1.0
            self.adaptive_ranges[feature_name] = FeatureRange(
                min_val=min_val - margin,
                max_val=max_val + margin,
                name=feature_name
            )
            
        logger.info(f"自适应归一化已拟合: {len(self.adaptive_ranges)} 个特征")
        return self
    
    def _normalize_to_0_100(self, value: Optional[float], range_def: FeatureRange) -> Optional[float]:
        """
        将值归一化到0-100范围
        
        Args:
            value: 原始值
            range_def: 范围定义
            
        Returns:
            0-100的分数，或None
        """
        if value is None or np.isnan(value) or np.isinf(value):
            return None
        
        min_val = range_def.min_val
        max_val = range_def.max_val
        
        if max_val == min_val:
            return 50.0  # 范围为零时返回中间值
        
        # Min-Max归一化到0-1，然后映射到0-100
        normalized = (value - min_val) / (max_val - min_val)
        
        # 限制在0-100范围内
        normalized = max(0.0, min(1.0, normalized))
        
        return normalized * 100
    
    def transform(
        self,
        feature_name: str,
        raw_value: Optional[float]
    ) -> Dict[str, Optional[float]]:
        """
        转换单个特征值
        
        Args:
            feature_name: 特征名称
            raw_value: 原始值
            
        Returns:
            {
                'raw': 原始值,
                'adaptive_score': 自适应分数(0-100),
                'fixed_score': 固定分数(0-100)
            }
        """
        result = {'raw': raw_value}
        
        # 自适应归一化
        if self.mode in ('adaptive', 'both'):
            if feature_name in self.adaptive_ranges:
                result['adaptive_score'] = self._normalize_to_0_100(
                    raw_value, self.adaptive_ranges[feature_name]
                )
            else:
                result['adaptive_score'] = None
        
        # 固定归一化
        if self.mode in ('fixed', 'both'):
            if feature_name in self.fixed_ranges:
                result['fixed_score'] = self._normalize_to_0_100(
                    raw_value, self.fixed_ranges[feature_name]
                )
            else:
                result['fixed_score'] = None
        
        return result
    
    def transform_batch(
        self,
        results: Dict[str, Dict[str, any]]
    ) -> Dict[str, Dict[str, any]]:
        """
        批量转换所有样本的所有特征
        
        Args:
            results: {样本名: {特征名: {value, unit, ...}}}
            
        Returns:
            添加score字段的结果
        """
        # 如果是自适应模式，先拟合范围
        if self.mode in ('adaptive', 'both'):
            # 收集所有特征值
            feature_values: Dict[str, List[float]] = {}
            
            for sample_name, features in results.items():
                for feature_name, data in features.items():
                    if isinstance(data, dict) and 'value' in data:
                        if feature_name not in feature_values:
                            feature_values[feature_name] = []
                        val = data['value']
                        if val is not None:
                            feature_values[feature_name].append(val)
            
            self.fit(feature_values)
        
        # 转换所有样本
        normalized_results = {}
        
        for sample_name, features in results.items():
            normalized_results[sample_name] = {}
            
            for feature_name, data in features.items():
                if isinstance(data, dict) and 'value' in data:
                    raw_value = data['value']
                    scores = self.transform(feature_name, raw_value)
                    
                    # 保留原始数据，添加分数
                    normalized_results[sample_name][feature_name] = {
                        **data,  # 保留原始字段
                        **scores  # 添加分数字段
                    }
                else:
                    # 非特征数据直接复制
                    normalized_results[sample_name][feature_name] = data
        
        return normalized_results
    
    def get_range_info(self, feature_name: str) -> Dict:
        """获取特征范围信息"""
        info = {}
        
        if feature_name in self.adaptive_ranges:
            ar = self.adaptive_ranges[feature_name]
            info['adaptive'] = {'min': ar.min_val, 'max': ar.max_val}
        
        if feature_name in self.fixed_ranges:
            fr = self.fixed_ranges[feature_name]
            info['fixed'] = {'min': fr.min_val, 'max': fr.max_val}
        
        return info


def create_normalizer(mode: str = 'both') -> FeatureNormalizer:
    """
    创建归一化器工厂函数
    
    Args:
        mode: 'adaptive', 'fixed', 或 'both'
        
    Returns:
        FeatureNormalizer实例
    """
    return FeatureNormalizer(mode=mode)
