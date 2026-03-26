@echo off
chcp 65001 >nul
echo ========================================
echo  SmartQuality 音频特征分析系统
echo  一键启动脚本
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请确保Python已安装并添加到环境变量
    pause
    exit /b 1
)

echo [1/3] 正在提取18维特征（P2阶段）...
python scripts/extract_features.py --input_dir sample_wav --output_dir data/results --stage P2 --normalize_mode both --format all
if errorlevel 1 (
    echo [错误] 特征提取失败
    pause
    exit /b 1
)
echo [1/3] 特征提取完成！
echo.

echo [2/3] 正在生成可视化分析...
python scripts/analyze_features.py --input data/results/features_P2_both.csv --output data/results/analysis_P2.png >nul 2>&1
echo [2/3] 可视化分析完成！
echo.

echo [3/3] 生成报告摘要...
python -c "
import pandas as pd
import json
from pathlib import Path

# 读取结果
csv_path = Path('data/results/features_P2_both.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
    ok_count = len(df[df['label'] == 'OK'])
    ng_count = len(df[df['label'] == 'NG'])
    
    print('')
    print('========================================')
    print('          分析结果摘要')
    print('========================================')
    print(f'样本总数: {len(df)}')
    print(f'  OK样本: {ok_count}')
    print(f'  NG样本: {ng_count}')
    print('')
    print('生成文件:')
    print('  - features_P2_both.xlsx (Excel报告)')
    print('  - features_P2_both.csv  (CSV数据)')
    print('  - features_P2_both.json (JSON数据)')
    print('  - analysis_P2.png       (可视化分析)')
    print('')
    print('Excel包含以下Sheet:')
    print('  1. 原始数据     - 原始计算值')
    print('  2. 自适应分数   - 基于数据分布的0-100分')
    print('  3. 固定分数     - 基于预设范围的0-100分')
    print('  4. 范围定义     - 归一化参数')
    print('  5. 统计摘要     - OK/NG对比统计')
    print('')
"

echo ========================================
echo  分析完成！请查看 data/results 目录
echo ========================================
echo.
pause
