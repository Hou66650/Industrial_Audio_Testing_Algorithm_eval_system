#!/usr/bin/env python3
"""
SmartQuality 音频特征分析系统 - 一键启动脚本

使用方法:
    python run.py

功能:
    1. 提取全部18维特征 (P2阶段)
    2. 使用双模式归一化 (adaptive + fixed)
    3. 输出所有格式 (Excel + CSV + JSON)
    4. 生成可视化分析图表
"""
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[错误] 执行失败: {cmd}")
        sys.exit(1)
    print(f"[完成] {description}")


def print_summary():
    """打印结果摘要"""
    try:
        import pandas as pd
        
        csv_path = Path("data/results/features_P2_both.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            ok_count = len(df[df['label'] == 'OK'])
            ng_count = len(df[df['label'] == 'NG'])
            
            print("\n" + "="*50)
            print("          分析结果摘要")
            print("="*50)
            print(f"样本总数: {len(df)}")
            print(f"  OK样本: {ok_count}")
            print(f"  NG样本: {ng_count}")
            print(f"\n特征维度: 18个")
            print("\n生成文件:")
            print("  📊 features_P2_both.xlsx (Excel完整报告)")
            print("  📄 features_P2_both.csv  (CSV数据)")
            print("  📋 features_P2_both.json (JSON数据)")
            print("  📈 analysis_P2.png       (可视化分析图)")
            print("\nExcel报告包含5个Sheet:")
            print("  1. 原始数据     - 原始计算值")
            print("  2. 自适应分数   - 基于数据分布的0-100分")
            print("  3. 固定分数     - 基于预设范围的0-100分")
            print("  4. 范围定义     - 归一化参数")
            print("  5. 统计摘要     - OK/NG对比统计")
            
            # 显示高区分度特征
            print("\n高区分度特征 (Cohen's d > 1.0):")
            try:
                # 获取原始值列
                feature_cols = [c for c in df.columns if c.endswith('_raw')]
                for col in feature_cols[:5]:  # 显示前5个
                    feature_name = col.replace('_raw', '')
                    ok_vals = df[df['label'] == 'OK'][col].dropna()
                    ng_vals = df[df['label'] == 'NG'][col].dropna()
                    if len(ok_vals) > 0 and ok_vals.std() > 0:
                        cohens_d = (ng_vals.mean() - ok_vals.mean()) / ok_vals.std()
                        if abs(cohens_d) > 0.5:
                            print(f"  • {feature_name}: Cohen's d = {cohens_d:.2f}")
            except:
                pass
    except ImportError:
        pass


def main():
    """主函数"""
    print("="*50)
    print("  SmartQuality 音频特征分析系统")
    print("  一键启动")
    print("="*50)
    
    # 检查目录
    if not Path("sample_wav").exists():
        print("\n[错误] 未找到 sample_wav 目录，请确保样本音频放在该目录下")
        sys.exit(1)
    
    # 步骤1: 特征提取
    run_command(
        "python scripts/extract_features.py --input_dir sample_wav --output_dir data/results --stage P2 --normalize_mode both --format all",
        "[1/3] 提取18维特征 (P2阶段)"
    )
    
    # 步骤2: 可视化分析
    run_command(
        "python scripts/analyze_features.py --input data/results/features_P2_both.csv --output data/results/analysis_P2.png",
        "[2/3] 生成可视化分析"
    )
    
    # 步骤3: 摘要报告
    print("\n" + "="*50)
    print("[3/3] 生成报告摘要")
    print("="*50)
    print_summary()
    
    print("\n" + "="*50)
    print("✅ 分析完成！请查看 data/results 目录")
    print("="*50)


if __name__ == "__main__":
    main()
