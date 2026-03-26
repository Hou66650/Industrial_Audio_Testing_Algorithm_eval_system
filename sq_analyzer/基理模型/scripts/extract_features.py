"""
特征提取验证脚本 - 支持量化分数输出

使用方法:
    python scripts/extract_features.py --input_dir sample_wav --output_dir data/results --stage P0

参数:
    --input_dir: 输入音频目录
    --output_dir: 输出结果目录
    --stage: 阶段 P0/P1/P2 (默认P0)
    --normalize_mode: 归一化模式 adaptive/fixed/both (默认both)
    --format: 输出格式 csv/excel/json/all (默认all)
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from tqdm import tqdm

from src.io.audio_loader import AudioLoader
from src.core.feature_extractor import AudioFeatureExtractor
from src.core.normalizer import create_normalizer
from src.io.result_exporter import ResultExporter


def setup_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )


def extract_features_from_directory(
    input_dir: Path,
    output_dir: Path,
    stage: str = "P0",
    normalize_mode: str = "both",
    output_format: str = "all",
    pattern: str = "*.wav"
) -> None:
    """
    从目录提取特征并量化
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        stage: 阶段
        normalize_mode: 归一化模式 (adaptive/fixed/both)
        output_format: 输出格式
        pattern: 文件匹配模式
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 70)
    logger.info(f"SmartQuality 音频特征提取系统")
    logger.info(f"=" * 70)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"阶段: {stage}")
    logger.info(f"归一化模式: {normalize_mode}")
    logger.info(f"=" * 70)
    
    # 1. 扫描音频文件
    audio_files = AudioLoader.scan_directory(input_dir, pattern)
    
    if not audio_files:
        logger.error(f"未找到音频文件: {input_dir}")
        return
    
    # 统计OK/NG
    labels = [AudioLoader.classify_by_filename(f) for f in audio_files]
    ok_count = labels.count('OK')
    ng_count = labels.count('NG')
    unknown_count = labels.count('UNKNOWN')
    
    logger.info(f"找到 {len(audio_files)} 个音频文件")
    logger.info(f"  OK: {ok_count}, NG: {ng_count}, UNKNOWN: {unknown_count}")
    
    # 2. 初始化提取器
    extractor = AudioFeatureExtractor(stage=stage)
    feature_names = extractor.get_feature_names()
    logger.info(f"已加载 {len(feature_names)} 个特征提取器")
    logger.info(f"特征列表: {', '.join(feature_names)}")
    
    # 3. 批量处理 - 提取原始特征
    raw_results = {}
    errors = []
    
    for filepath in tqdm(audio_files, desc="提取特征"):
        try:
            # 加载音频
            signal, sr = AudioLoader.load(filepath)
            
            # 提取特征
            feature_result = extractor.extract(signal, sr)
            
            raw_results[filepath.name] = feature_result
            
        except Exception as e:
            logger.error(f"处理失败 [{filepath.name}]: {e}")
            errors.append((filepath.name, str(e)))
            continue
    
    logger.info(f"特征提取完成: {len(raw_results)} 成功, {len(errors)} 失败")
    
    # 4. 归一化/量化
    logger.info(f"正在进行归一化 (模式: {normalize_mode})...")
    normalizer = create_normalizer(mode=normalize_mode)
    normalized_results = normalizer.transform_batch(raw_results)
    
    # 5. 导出结果
    logger.info("正在导出结果...")
    
    base_name = f"features_{stage}_{normalize_mode}"
    
    if output_format in ('csv', 'all'):
        csv_path = output_dir / f"{base_name}.csv"
        ResultExporter.to_csv(normalized_results, csv_path, include_scores=True)
    
    if output_format in ('excel', 'all'):
        excel_path = output_dir / f"{base_name}.xlsx"
        ResultExporter.to_excel(normalized_results, excel_path, normalizer=normalizer)
    
    if output_format in ('json', 'all'):
        json_path = output_dir / f"{base_name}.json"
        ResultExporter.to_json(normalized_results, json_path)
    
    # 6. 输出示例
    logger.info("=" * 70)
    logger.info("特征提取与量化完成！")
    logger.info(f"结果保存在: {output_dir}")
    logger.info("")
    
    # 显示部分结果示例
    if normalized_results:
        sample_name = list(normalized_results.keys())[0]
        sample_result = normalized_results[sample_name]
        logger.info(f"示例结果 [{sample_name}]:")
        
        for symbol, data in list(sample_result.items())[:5]:
            if isinstance(data, dict):
                raw = data.get('value')
                adaptive = data.get('adaptive_score')
                fixed = data.get('fixed_score')
                unit = data.get('unit', '')
                
                raw_str = f"{raw:.2f}" if raw is not None else "None"
                adaptive_str = f"{adaptive:.1f}" if adaptive is not None else "None"
                fixed_str = f"{fixed:.1f}" if fixed is not None else "None"
                
                logger.info(f"  {symbol}: 原始={raw_str}{unit}, 自适应分数={adaptive_str}, 固定分数={fixed_str}")
    
    logger.info("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="音频特征提取与量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法 - 两种归一化模式
  python scripts/extract_features.py --input_dir sample_wav --output_dir data/results
  
  # 仅使用固定范围归一化
  python scripts/extract_features.py --input_dir sample_wav --normalize_mode fixed
  
  # 仅使用自适应归一化
  python scripts/extract_features.py --input_dir sample_wav --normalize_mode adaptive
  
  # 仅输出Excel
  python scripts/extract_features.py --input_dir sample_wav --format excel
        """
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
        default='data/results',
        help='输出结果目录 (默认: data/results)'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        default='P0',
        choices=['P0', 'P1', 'P2'],
        help='特征提取阶段 (默认: P0)'
    )
    
    parser.add_argument(
        '--normalize_mode',
        type=str,
        default='both',
        choices=['adaptive', 'fixed', 'both'],
        help='归一化模式: adaptive=自适应, fixed=固定范围, both=两者 (默认: both)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='all',
        choices=['csv', 'excel', 'json', 'all'],
        help='输出格式 (默认: all)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.wav',
        help='文件匹配模式 (默认: *.wav)'
    )
    
    args = parser.parse_args()
    
    setup_logger()
    
    extract_features_from_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        stage=args.stage,
        normalize_mode=args.normalize_mode,
        output_format=args.format,
        pattern=args.pattern
    )


if __name__ == '__main__':
    main()
