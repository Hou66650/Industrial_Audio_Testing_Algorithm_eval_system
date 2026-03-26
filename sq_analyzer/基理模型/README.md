# SmartQuality 音频特征分析系统

用于汽车零部件NVH声学检测的18维音频特征提取与量化系统。

## 功能特性

- **18维音频特征提取**: P0(9个) + P1(7个) + P2(2个)
- **双模式量化**: 自适应归一化 + 固定范围归一化 → 0-100分
- **输出格式**: CSV/Excel/JSON
- **自动分类**: 根据文件名自动识别OK/NG样本

## 快速开始

### 一键启动（推荐）

**Windows:**
```bash
# 双击运行
run.bat

# 或命令行
python run.py
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh

# 或使用Python
python run.py
```

一键启动会自动：
1. 提取全部18维特征（P2阶段）
2. 使用双模式归一化（自适应 + 固定）
3. 输出所有格式（Excel + CSV + JSON）
4. 生成可视化分析图表

**输出文件：**
- `data/results/features_P2_both.xlsx` - Excel完整报告（5个Sheet）
- `data/results/features_P2_both.csv` - CSV数据
- `data/results/features_P2_both.json` - JSON数据
- `data/results/analysis_P2.png` - 可视化分析图

### 手动运行

```bash
# 安装依赖
pip install -r requirements.txt

# P0阶段 - 9个基础特征（快速，约13秒）
python scripts/extract_features.py --input_dir sample_wav --stage P0

# P1阶段 - 16个特征（更全面，约35秒）
python scripts/extract_features.py --input_dir sample_wav --stage P1

# P2阶段 - 18个特征（完整，约80秒）
python scripts/extract_features.py --input_dir sample_wav --stage P2
```

### 归一化模式选择

```bash
# 同时输出两种分数（推荐）
python scripts/extract_features.py --normalize_mode both

# 仅固定范围（适合跨数据集对比）
python scripts/extract_features.py --normalize_mode fixed

# 仅自适应（适合单批次内对比）
python scripts/extract_features.py --normalize_mode adaptive
```

## 项目结构

```
基理模型/
├── run.bat / run.sh / run.py     # 一键启动脚本
├── README.md                      # 项目主文档
├── requirements.txt               # Python依赖
├── docs/                          # 文档目录
│   ├── 设计思路.md
│   └── 使用文档_特征计算原理.md   # 详细计算原理
├── src/                           # 源代码
│   ├── core/                      # 核心模块
│   ├── features/                  # 18个特征实现
│   ├── preprocessing/             # 预处理
│   └── io/                        # 输入输出
├── scripts/                       # 脚本工具
│   ├── extract_features.py        # 特征提取
│   ├── preprocess_samples.py      # 音频切片
│   └── analyze_features.py        # 分析可视化
└── data/results/                  # 输出结果
```

## 特征维度

### P0阶段 (9个基础特征)

| 符号 | 名称 | 单位 | 说明 | 推荐度 |
|------|------|------|------|--------|
| SPL_peak | 峰值声压级 | dB | 瞬时最大声压 | ⭐⭐⭐ |
| K | 时域峭度 | - | 冲击检测 | ⭐⭐⭐⭐⭐ |
| E_low/mid/high | 频带能量 | % | 低/中/高频占比 | ⭐⭐⭐ |
| dB(A) | A计权声压级 | dB(A) | 人耳感知声压 | ⭐⭐⭐⭐⭐ |
| N | 响度 | sone | 主观响度 | ⭐⭐⭐⭐ |
| S | 尖锐度 | acum | 高频尖锐程度 | ⭐⭐⭐ |
| R | 粗糙度 | asper | 调制粗糙感 | ⭐⭐⭐⭐ |

### P1阶段 (7个进阶特征)

| 符号 | 名称 | 单位 | 说明 | 推荐度 |
|------|------|------|------|--------|
| F | 波动强度 | vacil | 低频调制感知 | ⭐⭐⭐⭐ |
| TI | 音调指数 | - | 纯音成分强度 | ⭐⭐⭐ |
| A | 烦恼度 | - | 综合主观烦恼 | ⭐⭐⭐⭐ |
| TNR | 音噪比 | dB | 纯音vs噪声 | ⭐⭐⭐⭐ |
| SI | 频谱不规则度 | - | 频谱不平滑度 | ⭐⭐⭐ |
| SK | 频谱峭度 | - | 频域冲击检测 | ⭐⭐⭐⭐⭐ |
| AMD | 幅度调制深度 | % | AM调制程度 | ⭐⭐⭐ |

### P2阶段 (2个特定特征)

| 符号 | 名称 | 单位 | 说明 | 推荐度 |
|------|------|------|------|--------|
| ESP | 增强频谱峰值 | dB | 结合SK的峰值检测 | ⭐⭐⭐⭐⭐ |
| NSI | 非平稳性指标 | - | 时变程度 | ⭐⭐⭐⭐ |

## Excel输出格式

包含5个Sheet：

| Sheet | 内容 | 说明 |
|-------|------|------|
| 原始数据 | 原始计算值 | 各特征的原始计算值（带物理单位） |
| 自适应分数 | 0-100分 | 基于数据Min-Max的相对评分 |
| 固定分数 | 0-100分 | 基于预设范围的绝对评分 |
| 范围定义 | 归一化参数 | 各特征的归一化范围参数 |
| 统计摘要 | OK/NG对比 | 均值、标准差、Cohen's d效应量 |

## 文档

- **[设计思路](docs/设计思路.md)** - 系统架构设计
- **[使用文档_特征计算原理](docs/使用文档_特征计算原理.md)** - 详细使用说明、18个特征的计算原理、合理性评估

## Python API

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

## 样本分类

系统自动根据文件名分类：
- 包含 `ng` 或 `NG` → NG（不良品）
- 包含 `ok` 或 `OK` → OK（良品）

## 开发状态

- [x] P0阶段基础特征 (9个)
- [x] P1阶段进阶特征 (7个)
- [x] P2阶段特定特征 (2个)
- [x] 双模式归一化量化
- [x] 一键启动脚本
- [ ] 阈值自动学习
- [ ] REST API接口
