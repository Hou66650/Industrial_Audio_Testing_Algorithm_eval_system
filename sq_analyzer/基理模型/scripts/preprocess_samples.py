"""
样本音频预处理脚本

将完整的样本音频切分为固定时长的片段

使用方法:
    python scripts/preprocess_samples.py --input_dir sample_wav --output_dir data/sliced --duration 0.3

参数:
    --input_dir: 输入音频目录
    --output_dir: 输出切片目录
    --duration: 切片时长（秒）
    --overlap: 重叠比例 (0-1)
    --target_sr: 目标采样率
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from tqdm import tqdm

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.io.audio_loader import AudioLoader


def preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    duration: float = 0.3,
    overlap: float = 0.0,
    target_sr: int = 48000,
    pattern: str = "*.wav"
):
    """
    预处理目录中的音频
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        duration: 切片时长
        overlap: 重叠比例
        target_sr: 目标采样率
        pattern: 文件匹配模式
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info(f"开始预处理音频")
    logger.info(f"输入: {input_dir}")
    logger.info(f"输出: {output_dir}")
    logger.info(f"切片时长: {duration}s, 重叠: {overlap}")
    logger.info(f"目标采样率: {target_sr}Hz")
    logger.info(f"=" * 60)
    
    # 扫描文件
    audio_files = AudioLoader.scan_directory(input_dir, pattern)
    
    if not audio_files:
        logger.error("未找到音频文件")
        return
    
    # 初始化预处理器
    preprocessor = AudioPreprocessor(
        target_sr=target_sr,
        slice_duration=duration,
        overlap=overlap,
        normalize=True,
        remove_dc=True
    )
    
    total_slices = 0
    
    # 处理每个文件
    for filepath in tqdm(audio_files, desc="预处理"):
        try:
            slices = preprocessor.process_file(
                filepath,
                output_dir=output_dir,
                save_slices=True
            )
            total_slices += len(slices)
            
            label = AudioLoader.classify_by_filename(filepath)
            logger.debug(f"{filepath.name} ({label}) -> {len(slices)} 切片")
            
        except Exception as e:
            logger.error(f"处理失败 [{filepath.name}]: {e}")
            continue
    
    logger.info("=" * 60)
    logger.info(f"预处理完成！")
    logger.info(f"总切片数: {total_slices}")
    logger.info(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="音频预处理工具 - 将音频切分为固定长度片段"
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='sample_wav',
        help='输入音频目录 (默认: sample_wav)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/sliced',
        help='输出切片目录 (默认: data/sliced)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=0.3,
        help='切片时长（秒）(默认: 0.3)'
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.0,
        help='重叠比例 0-1 (默认: 0)'
    )
    
    parser.add_argument(
        '--target_sr',
        type=int,
        default=48000,
        help='目标采样率 (默认: 48000)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.wav',
        help='文件匹配模式 (默认: *.wav)'
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    preprocess_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        duration=args.duration,
        overlap=args.overlap,
        target_sr=args.target_sr,
        pattern=args.pattern
    )


if __name__ == '__main__':
    main()
