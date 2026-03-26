"""音频文件加载模块"""
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import soundfile as sf
import librosa
from loguru import logger


class AudioLoader:
    """音频加载器"""
    
    # 支持的音频格式
    SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    
    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        target_sr: Optional[int] = None,
        mono: bool = True,
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            filepath: 音频文件路径
            target_sr: 目标采样率，None则保持原始采样率
            mono: 是否转换为单声道
            offset: 起始偏移（秒）
            duration: 加载时长（秒），None则加载全部
            
        Returns:
            (信号数组, 采样率)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"音频文件不存在: {filepath}")
        
        if filepath.suffix.lower() not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的音频格式: {filepath.suffix}")
        
        try:
            # 使用librosa加载（支持更多格式和重采样）
            signal, sr = librosa.load(
                filepath,
                sr=target_sr,
                mono=mono,
                offset=offset,
                duration=duration,
                res_type='kaiser_best'
            )
            
            logger.debug(f"加载音频: {filepath.name}, 采样率: {sr}, 长度: {len(signal)}")
            return signal, sr
            
        except Exception as e:
            logger.error(f"加载音频失败 {filepath}: {e}")
            raise
    
    @classmethod
    def load_batch(
        cls,
        filepaths: List[Union[str, Path]],
        target_sr: Optional[int] = None,
        mono: bool = True
    ) -> Dict[str, Tuple[np.ndarray, int]]:
        """
        批量加载音频文件
        
        Args:
            filepaths: 音频文件路径列表
            target_sr: 目标采样率
            mono: 是否单声道
            
        Returns:
            {文件名: (信号, 采样率)}
        """
        results = {}
        for filepath in filepaths:
            try:
                signal, sr = cls.load(filepath, target_sr, mono)
                results[Path(filepath).name] = (signal, sr)
            except Exception as e:
                logger.error(f"跳过文件 {filepath}: {e}")
                continue
        return results
    
    @classmethod
    def scan_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "*.wav",
        recursive: bool = False
    ) -> List[Path]:
        """
        扫描目录中的音频文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归子目录
            
        Returns:
            音频文件路径列表
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        # 过滤掉临时文件
        files = [f for f in files if not f.name.startswith('~')]
        
        logger.info(f"扫描到 {len(files)} 个音频文件")
        return sorted(files)
    
    @classmethod
    def get_audio_info(cls, filepath: Union[str, Path]) -> Dict:
        """
        获取音频文件信息
        
        Args:
            filepath: 音频文件路径
            
        Returns:
            音频信息字典
        """
        filepath = Path(filepath)
        info = sf.info(filepath)
        
        return {
            "filename": filepath.name,
            "duration": info.duration,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "format": info.format
        }
    
    @classmethod
    def classify_by_filename(cls, filepath: Union[str, Path]) -> str:
        """
        根据文件名分类音频类型 (OK/NG/UNKNOWN)
        
        Args:
            filepath: 音频文件路径
            
        Returns:
            'OK', 'NG', 或 'UNKNOWN'
        """
        name = Path(filepath).stem.lower()
        
        if 'ng' in name:
            return 'NG'
        elif 'ok' in name:
            return 'OK'
        else:
            return 'UNKNOWN'


# 便捷函数
def load_audio(
    filepath: Union[str, Path],
    target_sr: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """加载单个音频文件"""
    return AudioLoader.load(filepath, target_sr, mono)


def load_audio_batch(
    filepaths: List[Union[str, Path]],
    target_sr: Optional[int] = None,
    mono: bool = True
) -> Dict[str, Tuple[np.ndarray, int]]:
    """批量加载音频文件"""
    return AudioLoader.load_batch(filepaths, target_sr, mono)
