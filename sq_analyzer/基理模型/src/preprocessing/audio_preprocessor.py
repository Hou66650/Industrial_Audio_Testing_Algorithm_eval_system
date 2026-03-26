"""音频预处理模块 - 切片和预处理"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import soundfile as sf
from loguru import logger


@dataclass
class AudioSlice:
    """音频切片数据类"""
    slice_id: int
    data: np.ndarray
    start_time: float
    end_time: float
    sr: int
    source_file: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """切片时长"""
        return len(self.data) / self.sr
    
    @property
    def rms(self) -> float:
        """RMS能量"""
        return np.sqrt(np.mean(self.data ** 2))
    
    def save(self, output_path: Path) -> Path:
        """保存切片到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, self.data, self.sr)
        return output_path


class AudioPreprocessor:
    """音频预处理器"""
    
    def __init__(
        self,
        target_sr: int = 48000,
        slice_duration: float = 0.3,
        overlap: float = 0.0,
        normalize: bool = True,
        remove_dc: bool = True
    ):
        """
        初始化预处理器
        
        Args:
            target_sr: 目标采样率
            slice_duration: 切片时长（秒）
            overlap: 重叠比例（0-1）
            normalize: 是否归一化
            remove_dc: 是否去除直流分量
        """
        self.target_sr = target_sr
        self.slice_duration = slice_duration
        self.overlap = overlap
        self.normalize = normalize
        self.remove_dc = remove_dc
        
        # 计算切片参数
        self.slice_samples = int(slice_duration * target_sr)
        self.hop_samples = int(self.slice_samples * (1 - overlap))
        
        logger.info(
            f"预处理器初始化: sr={target_sr}, "
            f"slice_duration={slice_duration}s, "
            f"overlap={overlap}"
        )
    
    def process_signal(
        self,
        signal: np.ndarray,
        sr: int,
        source_file: Optional[str] = None
    ) -> List[AudioSlice]:
        """
        处理音频信号并切片
        
        Args:
            signal: 原始信号
            sr: 原始采样率
            source_file: 源文件名
            
        Returns:
            AudioSlice列表
        """
        # 1. 重采样
        if sr != self.target_sr:
            signal = self._resample(signal, sr, self.target_sr)
            sr = self.target_sr
        
        # 2. 预处理
        signal = self._preprocess(signal)
        
        # 3. 切片
        slices = self._slice(signal, sr, source_file)
        
        logger.debug(f"生成 {len(slices)} 个切片")
        return slices
    
    def _resample(
        self,
        signal: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """重采样"""
        import librosa
        return librosa.resample(
            signal,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type='kaiser_best'
        )
    
    def _preprocess(self, signal: np.ndarray) -> np.ndarray:
        """预处理信号"""
        # 去除直流分量
        if self.remove_dc:
            signal = signal - np.mean(signal)
        
        # 归一化
        if self.normalize:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val
        
        return signal
    
    def _slice(
        self,
        signal: np.ndarray,
        sr: int,
        source_file: Optional[str] = None
    ) -> List[AudioSlice]:
        """切片"""
        slices = []
        total_samples = len(signal)
        
        slice_id = 0
        start = 0
        
        while start + self.slice_samples <= total_samples:
            end = start + self.slice_samples
            slice_data = signal[start:end]
            
            audio_slice = AudioSlice(
                slice_id=slice_id,
                data=slice_data,
                start_time=start / sr,
                end_time=end / sr,
                sr=sr,
                source_file=source_file
            )
            
            slices.append(audio_slice)
            
            # 下一个切片的起始位置
            start += self.hop_samples
            slice_id += 1
        
        return slices
    
    def process_file(
        self,
        filepath: Path,
        output_dir: Optional[Path] = None,
        save_slices: bool = False
    ) -> List[AudioSlice]:
        """
        处理音频文件
        
        Args:
            filepath: 音频文件路径
            output_dir: 输出目录（保存切片时用）
            save_slices: 是否保存切片到文件
            
        Returns:
            AudioSlice列表
        """
        from ..io.audio_loader import AudioLoader
        
        # 加载音频
        signal, sr = AudioLoader.load(filepath, target_sr=self.target_sr)
        
        # 处理
        slices = self.process_signal(signal, sr, source_file=filepath.name)
        
        # 保存切片（可选）
        if save_slices and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(filepath).stem
            for s in slices:
                filename = f"{base_name}_slice{s.slice_id:04d}.wav"
                output_path = output_dir / filename
                s.save(output_path)
                logger.debug(f"保存切片: {output_path}")
        
        return slices
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.wav"
    ) -> Dict[str, List[AudioSlice]]:
        """
        批量处理目录中的音频
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
            
        Returns:
            {文件名: [切片列表]}
        """
        from ..io.audio_loader import AudioLoader
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 扫描文件
        files = AudioLoader.scan_directory(input_dir, pattern)
        logger.info(f"找到 {len(files)} 个音频文件待处理")
        
        results = {}
        for filepath in files:
            try:
                slices = self.process_file(filepath, output_dir, save_slices=True)
                results[filepath.name] = slices
                logger.info(f"处理完成: {filepath.name} -> {len(slices)} 切片")
            except Exception as e:
                logger.error(f"处理失败 {filepath.name}: {e}")
                continue
        
        return results
