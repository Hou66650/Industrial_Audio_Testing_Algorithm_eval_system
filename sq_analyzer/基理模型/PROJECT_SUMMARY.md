# SmartQuality 音频特征分析系统 - 项目总结

## 项目概述

本项目实现了基于文档《SmartQuality_机理特征维度_v1.0.docx》的18维音频特征提取系统，用于汽车零部件NVH声学检测。

## 实现状态

### ✅ 已完成

1. **全部18个特征提取器**
   - P0阶段：9个基础特征
   - P1阶段：7个进阶特征
   - P2阶段：2个特定特征

2. **归一化与量化系统**
   - 自适应归一化（基于数据分布）
   - 固定范围归一化（基于声学标准）
   - 输出0-100量化分数

3. **多种输出格式**
   - Excel（多Sheet结构）
   - CSV
   - JSON

4. **完整文档**
   - 设计思路文档
   - 使用文档与特征计算原理
   - 代码注释

## 项目结构

```
基理模型/
├── README.md                          # 项目主文档
├── requirements.txt                   # Python依赖
├── PROJECT_SUMMARY.md                 # 本文件
│
├── docs/                              # 文档目录
│   ├── 设计思路.md                     # 系统架构设计
│   └── 使用文档_特征计算原理.md         # 详细使用说明与计算原理
│
├── src/                               # 源代码
│   ├── core/                          # 核心模块
│   │   ├── base_extractor.py          # 特征提取器基类
│   │   ├── feature_extractor.py       # 主提取器（管理18个特征）
│   │   ├── normalizer.py              # 归一化/量化模块
│   │   └── config.py                  # 配置管理
│   │
│   ├── features/                      # 特征实现
│   │   ├── signal_features.py         # P0信号特征(5个)
│   │   ├── psychoacoustic.py          # P0心理声学(4个)
│   │   ├── p1_features.py             # P1进阶特征(7个)
│   │   ├── p2_features.py             # P2特定特征(2个)
│   │   └── utils.py                   # 工具函数
│   │
│   ├── preprocessing/                 # 预处理
│   │   └── audio_preprocessor.py      # 音频切片/预处理
│   │
│   └── io/                            # 输入输出
│       ├── audio_loader.py            # 音频加载
│       └── result_exporter.py         # 结果导出
│
├── scripts/                           # 脚本工具
│   ├── extract_features.py            # 特征提取主脚本
│   ├── preprocess_samples.py          # 样本预处理
│   └── analyze_features.py            # 分析可视化
│
├── data/                              # 数据目录
│   └── results/                       # 输出结果
│       ├── features_P0_both.xlsx      # Excel报告示例
│       ├── features_P0_both.csv       # CSV示例
│       └── analysis.png               # 可视化分析
│
└── sample_wav/                        # 样本音频
    └── (23个测试音频文件)
```

## 18个特征清单

### P0阶段 - 基础特征 (9个)

| # | 符号 | 名称 | 单位 | 标准 | 计算复杂度 | 区分度 |
|---|------|------|------|------|------------|--------|
| 1 | SPL_peak | 峰值声压级 | dB | IEC 61672 | 低 | 中 |
| 2 | K | 时域峭度 | - | ISO 20286 | 低 | **极高** |
| 3 | E_low | 低频能量占比 | % | IEC 61260 | 中 | 低 |
| 4 | E_mid | 中频能量占比 | % | IEC 61260 | 中 | 低 |
| 5 | E_high | 高频能量占比 | % | IEC 61260 | 中 | 低 |
| 6 | dB(A) | A计权声压级 | dB(A) | IEC 61672 | 中 | 高 |
| 7 | N | 响度 | sone | ISO 532B | 高 | 高 |
| 8 | S | 尖锐度 | acum | DIN 45692 | 中 | 低 |
| 9 | R | 粗糙度 | asper | Daniel & Weber | 中 | **高** |

### P1阶段 - 进阶特征 (7个)

| # | 符号 | 名称 | 单位 | 标准 | 计算复杂度 | 应用场景 |
|---|------|------|------|------|------------|----------|
| 10 | F | 波动强度 | vacil | Zwicker & Fastl | 中 | 转速波动 |
| 11 | TI | 音调指数 | - | ECMA-418-2 | 中 | 啸叫检测 |
| 12 | A | 烦恼度 | - | Aures | 高 | 主观评价 |
| 13 | TNR | 音噪比 | dB | ISO 7779 | 中 | 纯音检测 |
| 14 | SI | 频谱不规则度 | - | Krimphoff | 低 | 频谱异常 |
| 15 | SK | 频谱峭度 | - | Antoni 2006 | **高** | 轴承故障 |
| 16 | AMD | 幅度调制深度 | % | Zwicker & Fastl | 低 | AM调制 |

### P2阶段 - 特定特征 (2个)

| # | 符号 | 名称 | 单位 | 标准 | 计算复杂度 | 应用场景 |
|---|------|------|------|------|------------|----------|
| 17 | ESP | 增强频谱峰值 | dB | Randall & Antoni | **高** | 故障特征 |
| 18 | NSI | 非平稳性指标 | - | Ravindra & Halim | 中 | 间歇噪声 |

## 核心算法评估

### 高度推荐算法（⭐⭐⭐⭐⭐）

1. **K (时域峭度)**
   - 优点：对冲击极其敏感，实测Cohen's d = 4.19
   - 应用：轴承、齿轮故障检测
   - 算法：scipy.stats.kurtosis

2. **dB(A) (A计权声压级)**
   - 优点：IEC国际标准，与人耳感知高度相关
   - 应用：通用声压测量
   - 算法：频域A权滤波 + RMS

3. **SK (频谱峭度)**
   - 优点：轴承故障检测金标准
   - 应用：频域故障定位
   - 算法：STFT + 频域峭度计算

4. **ESP (增强频谱峰值)**
   - 优点：结合SK优势，抑制噪声
   - 应用：微弱故障特征提取
   - 算法：SK加权频谱峰值检测

### 推荐使用算法（⭐⭐⭐⭐）

- **R (粗糙度)**：Cohen's d = 1.63，对调制声敏感
- **TNR (音噪比)**：ISO 7779，纯音检测准确
- **N (响度)**：ISO 532B，主观响度预测
- **NSI (非平稳性)**：时变信号分析

## 使用方法

### 快速提取

```bash
# P0阶段（9个特征，快速）
python scripts/extract_features.py --input_dir sample_wav --stage P0

# P2阶段（18个特征，完整）
python scripts/extract_features.py --input_dir sample_wav --stage P2
```

### 归一化模式

```bash
# 同时输出两种分数（推荐）
python scripts/extract_features.py --normalize_mode both

# 仅固定范围（跨数据集对比）
python scripts/extract_features.py --normalize_mode fixed

# 仅自适应（单批次内对比）
python scripts/extract_features.py --normalize_mode adaptive
```

### Python API

```python
from src.core.feature_extractor import AudioFeatureExtractor
from src.core.normalizer import create_normalizer

# 提取特征
extractor = AudioFeatureExtractor(stage="P2")
features = extractor.extract(signal, sr)

# 量化分数
normalizer = create_normalizer(mode="both")
results = normalizer.transform_batch({"sample": features})

# 访问结果
print(results["sample"]["K"])
# {'value': 11.62, 'adaptive_score': 45.2, 'fixed_score': 11.6}
```

## 输出格式

### Excel多Sheet结构

| Sheet | 内容 | 说明 |
|-------|------|------|
| 原始数据 | 原始计算值 | 保留物理单位的原始特征值 |
| 自适应分数 | 0-100分 | 基于数据分布的相对评分 |
| 固定分数 | 0-100分 | 基于预设范围的绝对评分 |
| 范围定义 | 归一化参数 | Min-Max或预设范围 |
| 统计摘要 | OK/NG对比 | 均值、标准差、Cohen's d |

### 分数解读

**自适应分数：**
- 基于当前数据集的Min-Max
- 50分 = 数据集中位数
- 适合同一批次内相对比较

**固定分数：**
- 基于声学标准预设范围
- 50分 = 一般工业水平
- 适合跨批次绝对比较

## 验证结果

使用23个样本音频验证（11个OK，12个NG）：

### 高区分度特征（Cohen's d > 1.0）

| 特征 | Cohen's d | 说明 |
|------|-----------|------|
| K | 4.19 | 时域峭度，冲击敏感 |
| R | 1.63 | 粗糙度，调制敏感 |

### 中等区分度特征（Cohen's d 0.5-1.0）

| 特征 | Cohen's d | 说明 |
|------|-----------|------|
| SPL_peak | 0.94 | 峰值声压 |
| N | 0.91 | 响度 |
| dB(A) | 0.75 | A计权声压级 |

## 计算性能

在测试环境（单线程）下：

| 阶段 | 特征数 | 单文件耗时 | 23文件耗时 |
|------|--------|------------|------------|
| P0 | 9 | ~0.6s | ~13s |
| P1 | 16 | ~1.5s | ~35s |
| P2 | 18 | ~3.5s | ~80s |

*注：P2阶段包含STFT等复杂计算，耗时较长*

## 后续建议

### 1. 算法优化

- **E_low/mid/high**：改用Bark频带更符合人耳感知
- **S (尖锐度)**：结合频谱质心提高稳定性
- **N (响度)**：集成mosqito库提高精度

### 2. 特征工程

- 组合特征：K + SK 联合检测冲击
- 时序特征：多切片间特征变化趋势
- 频带特征：针对特定零部件的频带能量

### 3. 系统集成

- REST API封装
- 实时流处理
- 阈值自动学习
- 可视化Dashboard

## 文档索引

| 文档 | 内容 |
|------|------|
| README.md | 快速开始指南 |
| docs/设计思路.md | 系统架构设计 |
| docs/使用文档_特征计算原理.md | 详细使用说明、18个特征的计算原理与合理性评估 |
| PROJECT_SUMMARY.md | 本文件，项目总体总结 |

---

**项目状态**: 核心功能完成 ✅  
**最后更新**: 2026-03-23  
**版本**: 1.0.0
