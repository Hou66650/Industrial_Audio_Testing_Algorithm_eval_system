"""
特征分析脚本 - 对比OK和NG样本的差异

使用方法:
    python scripts/analyze_features.py --input data/results/features_P0.csv --output data/results/analysis.png
"""
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def analyze_features(csv_path: Path, output_path: Path = None):
    """
    分析特征差异
    
    Args:
        csv_path: 特征CSV文件路径
        output_path: 输出图表路径
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    logger.info(f"加载数据: {len(df)} 样本")
    logger.info(f"OK: {len(df[df['label']=='OK'])}, NG: {len(df[df['label']=='NG'])}")
    
    # 获取特征列
    feature_cols = [c for c in df.columns if c not in ['sample_name', 'label'] and not c.endswith('_unit')]
    
    logger.info(f"特征维度: {len(feature_cols)}")
    logger.info(f"特征列表: {feature_cols}")
    
    # 统计分析
    logger.info("\n" + "="*60)
    logger.info("统计分析")
    logger.info("="*60)
    
    stats = []
    for col in feature_cols:
        ok_values = df[df['label'] == 'OK'][col].dropna()
        ng_values = df[df['label'] == 'NG'][col].dropna()
        
        ok_mean = ok_values.mean()
        ng_mean = ng_values.mean()
        ok_std = ok_values.std()
        ng_std = ng_values.std()
        
        # Cohen's d
        if ok_std > 0:
            cohens_d = (ng_mean - ok_mean) / ok_std
        else:
            cohens_d = 0
        
        stats.append({
            'feature': col,
            'ok_mean': ok_mean,
            'ok_std': ok_std,
            'ng_mean': ng_mean,
            'ng_std': ng_std,
            'difference': ng_mean - ok_mean,
            'cohens_d': cohens_d
        })
        
        logger.info(f"\n{col}:")
        logger.info(f"  OK: {ok_mean:.4f} ± {ok_std:.4f}")
        logger.info(f"  NG: {ng_mean:.4f} ± {ng_std:.4f}")
        logger.info(f"  差异: {ng_mean - ok_mean:.4f}")
        logger.info(f"  Cohen's d: {cohens_d:.4f}")
    
    # 创建可视化
    if output_path:
        create_visualization(df, feature_cols, output_path)
    
    return stats


def create_visualization(df, feature_cols, output_path):
    """创建特征对比图"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        
        # 箱线图
        data_to_plot = [df[df['label']=='OK'][col].dropna(), 
                        df[df['label']=='NG'][col].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['OK', 'NG'], patch_artist=True)
        
        # 设置颜色
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n图表已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="特征分析工具")
    parser.add_argument('--input', type=str, default='data/results/features_P0.csv',
                        help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='data/results/analysis.png',
                        help='输出图表路径')
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    analyze_features(Path(args.input), Path(args.output))


if __name__ == '__main__':
    main()
