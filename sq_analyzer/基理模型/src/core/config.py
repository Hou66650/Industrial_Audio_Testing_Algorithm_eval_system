"""配置管理模块"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class FeatureConfig:
    """特征配置"""
    name: str
    symbol: str
    unit: str
    stage: str  # P0, P1, P2
    description: str = ""
    enabled: bool = True


@dataclass
class PreprocessConfig:
    """预处理配置"""
    target_sr: int = 48000
    slice_duration: float = 0.3  # 秒
    overlap: float = 0.0  # 重叠比例 0-1
    normalize: bool = True
    remove_dc: bool = True


@dataclass
class Config:
    """主配置类"""
    # 阶段设置 P0/P1/P2
    stage: str = "P0"
    
    # 预处理配置
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    
    # 数据路径
    raw_audio_dir: Path = field(default_factory=lambda: Path("data/raw"))
    sliced_audio_dir: Path = field(default_factory=lambda: Path("data/sliced"))
    results_dir: Path = field(default_factory=lambda: Path("data/results"))
    
    # 特征配置列表 (18个特征)
    features: List[FeatureConfig] = field(default_factory=lambda: [
        # P0 - 心理声学 (4个)
        FeatureConfig("A计权声压级", "dB(A)", "dB", "P0", "IEC 61672标准A计权声压级"),
        FeatureConfig("响度", "N", "sone", "P0", "Zwicker响度模型"),
        FeatureConfig("尖锐度", "S", "acum", "P0", "DIN 45692尖锐度"),
        FeatureConfig("粗糙度", "R", "asper", "P0", "Daniel & Weber粗糙度"),
        
        # P1 - 心理声学 (3个)
        FeatureConfig("波动强度", "F", "vacil", "P1", "Zwicker波动强度"),
        FeatureConfig("音调指数", "TI", "-", "P1", "ECMA-418-2音调指数"),
        FeatureConfig("烦恼度", "A", "-", "P1", "Aures综合烦恼度"),
        
        # P0 - 信号特征 (7个)
        FeatureConfig("峰值声压级", "SPL_peak", "dB", "P0", "峰值声压级"),
        FeatureConfig("低频能量占比", "E_low", "%", "P0", "20-200Hz能量占比"),
        FeatureConfig("中频能量占比", "E_mid", "%", "P0", "200Hz-2kHz能量占比"),
        FeatureConfig("高频能量占比", "E_high", "%", "P0", "2k-20kHz能量占比"),
        FeatureConfig("时域峭度", "K", "-", "P0", "时域信号峭度"),
        
        # P1 - 信号特征 (4个)
        FeatureConfig("音噪比", "TNR", "dB", "P1", "ISO 7779音噪比"),
        FeatureConfig("频谱不规则度", "SI", "-", "P1", "Krimphoff频谱不规则度"),
        FeatureConfig("频谱峭度", "SK", "-", "P1", "Antoni频谱峭度"),
        FeatureConfig("幅度调制深度", "AMD", "%", "P1", "幅度调制深度"),
        
        # P2 - 信号特征 (2个)
        FeatureConfig("增强频谱峰值", "ESP", "dB", "P2", "Randall增强频谱峰值"),
        FeatureConfig("非平稳性指标", "NSI", "-", "P2", "非平稳性指标"),
    ])
    
    def get_enabled_features(self) -> List[FeatureConfig]:
        """获取当前阶段启用的特征"""
        stage_order = {"P0": 0, "P1": 1, "P2": 2}
        current_level = stage_order.get(self.stage, 0)
        
        enabled = []
        for feat in self.features:
            feat_level = stage_order.get(feat.stage, 0)
            if feat_level <= current_level and feat.enabled:
                enabled.append(feat)
        return enabled
    
    def get_feature_by_symbol(self, symbol: str) -> Optional[FeatureConfig]:
        """通过符号获取特征配置"""
        for feat in self.features:
            if feat.symbol == symbol:
                return feat
        return None
