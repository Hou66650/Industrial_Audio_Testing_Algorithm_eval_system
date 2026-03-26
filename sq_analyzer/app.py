import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import librosa
import soundfile as sf
import io
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from data_loader import DataLoader
from analyzer import SQAnalyzer
from feature_extractor import SQFeatureExtractor

from sklearn.metrics import confusion_matrix
import numpy as np

import io
import zipfile
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from concurrent.futures import ProcessPoolExecutor, as_completed
import base64
import os
import requests
import json



# 工具函数
# 机理解释表（放在app.py顶部工具区）
MECH_TABLE = pd.DataFrame([
    {"特征": "SPL_A", "含义": "A计权声压级，人耳加权后的整体响度基准", "典型缺陷": "所有异响"},
    {"特征": "Loudness_N", "含义": "响度（sone），接近人耳主观感知强度", "典型缺陷": "所有异响"},
    {"特征": "Sharpness_S", "含义": "尖锐度，高频刺耳程度", "典型缺陷": "啸叫、金属摩擦"},
    {"特征": "Roughness_R", "含义": "粗糙度，20–300Hz摩擦调制感", "典型缺陷": "摩擦/刮擦"},
    {"特征": "Fluctuation_F", "含义": "波动强度，<20Hz缓慢起伏", "典型缺陷": "不稳定运转"},
    {"特征": "SPL_peak", "含义": "峰值声压级，瞬时冲击强度", "典型缺陷": "敲击/冲击"},
    {"特征": "E_low_ratio", "含义": "低频能量比(20–200Hz)", "典型缺陷": "低频共振/轰鸣"},
    {"特征": "E_mid_ratio", "含义": "中频能量比(200–2kHz)", "典型缺陷": "摩擦/齿轮啮合"},
    {"特征": "E_high_ratio", "含义": "高频能量比(2k–20kHz)", "典型缺陷": "啸叫/摩擦"},
    {"特征": "K_kurtosis", "含义": "时域峭度，冲击尖峰程度", "典型缺陷": "冲击/敲击"},
    {"特征": "TNR", "含义": "音调突出比，纯音峰值突出程度", "典型缺陷": "啸叫纯音"},
    {"特征": "Spectral_Irregularity_SI", "含义": "频谱不规则度，频谱凹凸程度", "典型缺陷": "摩擦/刮擦"},
    {"特征": "Spectral_Kurtosis_SK", "含义": "谱峭度，频段冲击性分布", "典型缺陷": "轴承/齿轮缺陷"},
    {"特征": "AMD", "含义": "调制深度，周期性幅度变化", "典型缺陷": "偏心/周期摩擦"},
    {"特征": "Annoyance_A", "含义": "烦恼度，综合严重度指标", "典型缺陷": "所有异响"}
])

def generate_action_summary(fp_report, fn_conf_report, fn_drift_report):
    actions = []
    if 'clustering_result' in fp_report:
        if "系统性漏检" in fp_report['clustering_result']:
            actions.append("🔴 FP 系统性漏检：补充该类缺陷子样本")
        elif "随机漏检" in fp_report['clustering_result']:
            actions.append("🔴 FP 随机漏检：增加 FP 惩罚权重或调阈值")
    for feat, res in fp_report.items():
        if isinstance(res, dict) and res.get('signal') == '弱信号':
            actions.append(f"⚠️ 特征 {feat} 信号弱：需新增机理特征")
    if fn_conf_report.get("type") == "边界模糊":
        actions.append("🟡 FN 边界样本不足：补充边缘合格样本")
    elif fn_conf_report.get("type") == "模型过度自信":
        actions.append("🟡 FN 高置信误判：疑似漂移，排查工况")
    severe_drift = [k for k,v in fn_drift_report.items() if v['drift']=="严重漂移"]
    if severe_drift:
        actions.append(f"🚨 特征漂移严重: {', '.join(severe_drift)} → 排查产线工况")
    if not actions:
        actions.append("✅ 未发现明显结构性问题")
    return actions

def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()




def build_pdf_report(metrics, actions, cm, cm_norm):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    y = 800
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "SmartQuality Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Recall: {metrics['Recall']:.2%}")
    y -= 20
    c.drawString(50, y, f"Specificity: {metrics['Specificity']:.2%}")
    y -= 20
    c.drawString(50, y, f"FP: {metrics['FP']}   FN: {metrics['FN']}")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Action Summary:")
    y -= 20
    c.setFont("Helvetica", 11)
    for a in actions:
        c.drawString(60, y, "- " + a)
        y -= 18

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Confusion Matrix:")
    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(60, y, f"TP: {cm[0,0]}  FN: {cm[0,1]}")
    y -= 18
    c.drawString(60, y, f"FP: {cm[1,0]}  TN: {cm[1,1]}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def detect_production_shift(fn_conf_report, fn_drift_report):
    severe = any(v['drift']=="严重漂移" for v in fn_drift_report.values())
    if fn_conf_report.get("type")=="模型过度自信" and severe:
        return "🚨 高风险：疑似产线/设备工况变化"
    return "✅ 未检测到明显工况漂移"

def generate_markdown_report(metrics, actions, shift_warning):
    report = f"""
# SmartQuality 复盘报告

## 📊 核心指标
- 特异性: {metrics['Specificity']:.2%}
- 召回率: {metrics['Recall']:.2%}
- FP 漏检数: {metrics['FP']}
- FN 误判数: {metrics['FN']}

## 🚨 漂移预警
{shift_warning}

## ✅ 行动建议
"""
    for a in actions:
        report += f"- {a}\n"
    return report

# ==================== 页面配置 ====================
st.set_page_config(page_title="SmartQuality 智能声学复盘", page_icon="🌌", layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {vi24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 24px; border: 1px solid #f0f2f6;}
    .kpi-title { color: #6c757d; font-size: 1.1rem; font-weight: 600; margisibility: hidden;}
    .custom-card {background-color: #ffffff; border-radius: 12px; padding: n-bottom: 8px; }
    .kpi-value { font-size: 2.5rem; font-weight: 800; line-height: 1.2; }
    .text-danger { color: #e74c3c; } .text-success { color: #2ecc71; } .text-warning { color: #f39c12; }
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e74c3c', 'FN': '#f39c12'}

# ==================== Sidebar ====================
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "logo", "logo.png")
    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:20px; font-weight:700; line-height:1.2;">
                SmartQuality<br>算法复盘系统
            </div>
            <img src="data:image/png;base64,{logo_base64}" width="36">
        </div>
        """, unsafe_allow_html=True)
    # ===== 默认首页（必须有，否则空白）=====
    if 'analysis_done' not in st.session_state:
        st.info("请先在左侧填写参数，然后点击 🚀 启动分析")

    else:
        st.markdown("### SmartQuality<br>算法复盘系统", unsafe_allow_html=True)
    task_id_input = st.text_input("🎯 目标多个Task ID（使用，分隔开来）", value="519,531")
    audio_root_dir = st.text_input("音频根目录", value="/home/bestfunc/ai_train/ai_train_server/样本目录/NSK/训练数据/0.3s_0315")
    db_host = st.text_input("数据库 IP", value="192.168.2.176")
    start_btn = st.button("🚀 启动全量特征提取与 ML 分析", type="primary", use_container_width=True)
    st.markdown("### 🔑 AI 接口设置")
    ai_api_key = st.text_input(" API Key", type="password",value="")
    ai_api_url = st.text_input(" API URL", value="")
    with st.expander("📘 机理特征解释表（点击展开）", expanded=False):
        st.dataframe(MECH_TABLE, use_container_width=True, height=350)

DB_CONFIG = {"host": db_host, "port": 23306, "user": "root", "password": "SmartTpm.2023", "db_name": "smart_tools"}
@st.cache_data(ttl=600, show_spinner=False)
def fetch_base_data(task_id, audio_dir):
    loader = DataLoader(**DB_CONFIG)
    return loader.fetch_and_merge_data(task_id, audio_dir)

# ==================== 主流程 ====================
import os
task_ids = [int(x.strip()) for x in str(task_id_input).split(",") if x.strip().isdigit()]
if start_btn:
    for k in list(st.session_state.keys()):
        del st.session_state[k]

if start_btn and 'analysis_done' not in st.session_state:
    df_list = []
    for tid in task_ids:
        df_t = fetch_base_data(tid, audio_root_dir)
        if not df_t.empty:
            df_t['task_id'] = tid
            df_list.append(df_t)

    if not df_list:
        st.error("❌ 未找到数据。")
        st.stop()

    raw_df = pd.concat(df_list, ignore_index=True)
    raw_df['slice_id'] = raw_df['task_id'].astype(str) + "_" + raw_df['slice_id'].astype(str)

    if raw_df.empty:
        st.error("❌ 未找到数据。")
    else:
        progress_bar = st.progress(0, text="⏳ 初始化引擎...")

        # ====== 特征提取：缓存 + 并行 + 断点恢复 ======
        cache_path = f"feature_cache_task_{'_'.join(map(str,task_ids))}.parquet"
        feats_list, fail_log = [], []

        if os.path.exists(cache_path):
            feat_df_cached = pd.read_parquet(cache_path)
            cached_ids = set(feat_df_cached['slice_id'])
            feats_list = feat_df_cached.to_dict("records")
            st.info(f"✅ 已有缓存 {len(cached_ids)} 条，将只计算缺失部分")
        else:
            cached_ids = set()

        raw_df_todo = raw_df[~raw_df['slice_id'].isin(cached_ids)]
        st.write(f"待计算切片数: {len(raw_df_todo)}")

        def worker(row_dict):
            extractor = SQFeatureExtractor()
            f, err = extractor.extract_slice_features(
                row_dict['physical_wav_path'],
                row_dict['start_time'],
                row_dict['end_time']
            )
            if f:
                f['slice_id'] = row_dict['slice_id']
                return f, None
            else:
                return None, {"slice_id": row_dict['slice_id'], "file_id": row_dict.get('file_id', None), "reason": err}

        futures = []
        with ProcessPoolExecutor(max_workers=6) as exe:
            for _, row in raw_df_todo.iterrows():
                futures.append(exe.submit(worker, row.to_dict()))

            for i, f in enumerate(as_completed(futures), 1):
                feat, err = f.result()
                if feat: feats_list.append(feat)
                if err: fail_log.append(err)

                if len(futures) > 0:
                    progress_bar.progress(int(i/len(futures)*100), text=f"⏳ 提取特征 {i}/{len(futures)}")

        # ✅ 保存缓存（增量更新）
        if feats_list:
            feat_df_all = pd.DataFrame(feats_list).drop_duplicates('slice_id')
            feat_df_all.to_parquet(cache_path, index=False)
            st.success(f"✅ 缓存已更新，共 {len(feat_df_all)} 条")

        # ===== 合并特征 =====
        if feats_list:
            raw_df_adv = pd.merge(raw_df, pd.DataFrame(feats_list), on='slice_id', how='inner')
        else:
            raw_df_adv = raw_df.copy()

        # ===== 过程统计 =====
        total_slices = len(raw_df)
        success_count = len(feats_list)
        fail_count = len(fail_log)
        merged_count = len(raw_df_adv)

        # ===== 特征列 =====
        feature_cols = [
            'SPL_A', 'Loudness_N', 'Sharpness_S', 'Roughness_R', 'Fluctuation_F',
            'SPL_peak','E_low_ratio','E_mid_ratio','E_high_ratio','K_kurtosis',
            'Spectral_Irregularity_SI','TNR','Spectral_Kurtosis_SK','AMD','Annoyance_A'
        ]

        for col in feature_cols:
            if col in raw_df_adv.columns:
                raw_df_adv[col] = pd.to_numeric(raw_df_adv[col], errors='coerce')
        raw_df_adv[feature_cols] = raw_df_adv[feature_cols].fillna(0)

        # ===== PCA =====
        if all(col in raw_df_adv.columns for col in feature_cols):
            X_scaled = StandardScaler().fit_transform(raw_df_adv[feature_cols].values)
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(X_scaled)
            raw_df_adv['PCA_X'], raw_df_adv['PCA_Y'], raw_df_adv['PCA_Z'] = pca_result[:,0], pca_result[:,1], pca_result[:,2]

            labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_scaled)
            raw_df_adv['Cluster'] = "聚类簇 " + pd.Series(labels, index=raw_df_adv.index).astype(str)

        # ===== 分组统计 =====
        loader = DataLoader(**DB_CONFIG)
        grouped_data, metrics = loader.split_confusion_matrix_groups(raw_df_adv)
        df_tp = grouped_data.get('TP', pd.DataFrame())
        df_fp = grouped_data.get('FP', pd.DataFrame())
        df_fn = grouped_data.get('FN', pd.DataFrame())

        analyzer = SQAnalyzer()
        analyzer.fit_baseline(df_tp, feature_cols)

        noise_fn = analyzer.detect_label_noise(df_fn, 'FN')
        noise_fp = analyzer.detect_label_noise(df_fp, 'FP')

        cleaned_df = raw_df_adv.copy()
        if not noise_fn.empty: cleaned_df = cleaned_df.drop(noise_fn.index)
        if not noise_fp.empty: cleaned_df = cleaned_df.drop(noise_fp.index)

        # 清洗后统计
        cleaned_count = len(cleaned_df)

        # 保存统计 & 缺失列表
        st.session_state['process_stats'] = {
            "raw_total": total_slices,
            "feat_success": success_count,
            "feat_fail": fail_count,
            "merged_valid": merged_count,
            "cleaned_final": cleaned_count
        }
        st.session_state['fail_log'] = fail_log
        st.session_state['missing_df'] = raw_df[~raw_df['slice_id'].isin(raw_df_adv['slice_id'])]

        # ===== 重新分组 =====
        grouped_data, metrics = loader.split_confusion_matrix_groups(cleaned_df)
        cleaned_df = pd.concat(grouped_data.values())

        df_tp = grouped_data.get('TP', pd.DataFrame())
        df_fp = grouped_data.get('FP', pd.DataFrame())
        df_fn = grouped_data.get('FN', pd.DataFrame())

        # ===== 混淆矩阵 =====
        def to_binary(x):
            x = str(x).strip().upper()
            return 0 if x in ['OK','合格','PASS','NORMAL','0','0.0','正常'] else 1

        y_true = cleaned_df['label_name'].apply(to_binary)
        y_pred = cleaned_df['anomaly_output'].apply(to_binary)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        cm_norm = cm / cm.sum()

        # ===== SOP 分析 =====
        fp_report = analyzer.analyze_fp_mechanisms(df_fp, df_tp)
        fn_conf_report = analyzer.analyze_fn_confidence(df_fn)
        fn_drift_report = analyzer.analyze_fn_distribution(df_fn, df_tp)

        y_score = cleaned_df['anomaly_score']
        fpr, tpr, thresholds, roc_auc = analyzer.analyze_tradeoff(y_true, y_score)

        cleaned_df = analyzer.calculate_severity(cleaned_df)

        actions = generate_action_summary(fp_report, fn_conf_report, fn_drift_report)
        shift_warning = detect_production_shift(fn_conf_report, fn_drift_report)
        md_report = generate_markdown_report(metrics, actions, shift_warning)

        pdf_bytes = build_pdf_report(metrics, actions, cm, cm_norm)

        st.session_state.update({
            'raw_df_adv': cleaned_df,
            'grouped_data': grouped_data,
            'metrics': metrics,
            'noise_fn': noise_fn,
            'noise_fp': noise_fp,
            'fp_report': fp_report,
            'fn_conf_report': fn_conf_report,
            'fn_drift_report': fn_drift_report,
            'roc_data': (fpr, tpr, thresholds, roc_auc),
            'fail_log': fail_log,
            'actions': actions,
            'shift_warning': shift_warning,
            'md_report': md_report,
            'analysis_done': True,
            'pdf_bytes': pdf_bytes,
            'cm': cm,
            'cm_norm': cm_norm
        })


# ==================== 展示 ====================
if 'analysis_done' in st.session_state:
    raw_df_adv = st.session_state['raw_df_adv']
    grouped_data = st.session_state['grouped_data']
    metrics = st.session_state['metrics']

    st.markdown("### 📊 核心指标概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="custom-card"><div class="kpi-title">特异性</div><div class="kpi-value text-danger">{metrics["Specificity"]:.1%}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="custom-card"><div class="kpi-title">召回率</div><div class="kpi-value text-success">{metrics["Recall"]:.1%}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="custom-card"><div class="kpi-title">漏检 (FP)</div><div class="kpi-value text-danger">{metrics["FP"]}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="custom-card"><div class="kpi-title">误判 (FN)</div><div class="kpi-value text-warning">{metrics["FN"]}</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9,tab10 = st.tabs(
    ["🧹 Step1 标签噪声", "🔴 Step2a 漏检分析", "🟡 Step2b 误判分析",
     "⚖️ Step3 ROC & Tradeoff", "🌌 PCA 特征空间",
     "✅ Step5 严重度验证", "📄 Step4 行动总结", "📊 混淆矩阵", "🧭 3D PCA","📈AI智能分析"]
)

    with tab1:
        stats = st.session_state.get('process_stats', {})
        if stats:
            st.markdown("### ✅ 处理过程统计")
            st.write(f"原始切片数: {stats['raw_total']}")
            st.write(f"特征提取成功: {stats['feat_success']}")
            st.write(f"特征提取失败: {stats['feat_fail']}")
            st.write(f"合并后有效样本: {stats['merged_valid']}")
            st.write(f"有重复的样本数：{stats['merged_valid']-len(raw_df_adv)}条")
        st.write(f"✅ 成功提取 {len(raw_df_adv)} 条")
        if st.session_state['fail_log']:
            st.warning(f"⚠️ 提取失败 {len(st.session_state['fail_log'])} 条")
            df_fail = pd.DataFrame(st.session_state['fail_log'])
            st.dataframe(df_fail.head(20))
            st.write("❌ 失败原因统计")
            st.write(df_fail['reason'].value_counts())
        st.markdown("### 标签噪声清洗结果")
        st.write("FN 噪声", st.session_state['noise_fn'])
        st.write("FP 噪声", st.session_state['noise_fp'])
        # ===== 显示失败样本 =====
        if st.session_state.get('fail_log'):
            st.markdown("### ❌ 特征提取失败样本（前50）")
            st.dataframe(pd.DataFrame(st.session_state['fail_log']).head(50))

        # ===== 显示合并丢失样本 =====
        missing_df = st.session_state.get('missing_df')
        if missing_df is not None and not missing_df.empty:
            st.markdown("### ⚠️ 合并时丢失样本（前50）")
            st.dataframe(missing_df.head(50))


    with tab2:
        st.markdown("### 🔴 FP 漏检特征差异（Cohen’s d）")

        fp_report = st.session_state['fp_report']

        # 把 report 转成表格（排除聚类结果）
        rows = []
        for feat, res in fp_report.items():
            if isinstance(res, dict):
                rows.append({
                    "特征": feat,
                    "Cohen_d": res.get("cohen_d", None),
                    "信号强度": res.get("signal", "")
                })

        df_fp_table = pd.DataFrame(rows)

        # 按 Cohen_d 从大到小排序
        if not df_fp_table.empty:
            df_fp_table = df_fp_table.sort_values("Cohen_d", ascending=False)

        st.dataframe(df_fp_table, use_container_width=True)

        # 如果还有聚类结果，单独展示
        if "clustering_result" in fp_report:
            st.info(f"🧩 聚类结论：{fp_report['clustering_result']}")


    with tab3:
        st.json(st.session_state['fn_conf_report'])
        st.write(pd.DataFrame(st.session_state['fn_drift_report']).T)

    with tab4:
        fpr, tpr, thresholds, roc_auc = st.session_state['roc_data']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={roc_auc:.3f}"))
        fig.update_layout(title="ROC 曲线", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
    ## ✅ 指标解释

    - **TPR（True Positive Rate）**：真正例率，也叫召回率  
    公式：`TPR = TP / (TP + FN)`  
    含义：缺陷样本中被正确识别为缺陷的比例  

    - **FPR（False Positive Rate）**：假正例率  
    公式：`FPR = FP / (FP + TN)`  
    含义：正常样本中被误判为缺陷的比例  

    ---

    ## ✅ ROC 曲线说明
    ROC 曲线展示在不同阈值下 **TPR 与 FPR 的权衡关系**。  
    曲线越靠近左上角，模型区分能力越强。  
    AUC 越接近 1 越好。  

    ---

    ## ✅ Tradeoff 含义
    阈值越低 → **TPR 提高（少漏检）**，但 **FPR 增大（误报增加）**  
    阈值越高 → **FPR 降低（少误报）**，但 **TPR 降低（漏检增加）**
    """)

    with tab5:
        if 'PCA_X' in raw_df_adv.columns:
            fig2d = px.scatter(raw_df_adv, x='PCA_X', y='PCA_Y',
                               color='cm_group', color_discrete_map=COLOR_MAP,
                               hover_name='slice_id', opacity=0.7)
            st.plotly_chart(fig2d, use_container_width=True)
        else:
            st.warning("PCA 特征不足，无法展示。")

    with tab6:
        if 'mechanism_severity' in raw_df_adv.columns:
            fig_sev = px.box(raw_df_adv, x='cm_group', y='mechanism_severity',
                             color='cm_group', color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig_sev, use_container_width=True)

    with tab7:
        for a in st.session_state['actions']:
            if "✅" in a:
                st.success(a)
            elif "🚨" in a or "🔴" in a:
                st.error(a)
            else:
                st.warning(a)

        st.markdown("### 漂移预警")
        if "🚨" in st.session_state['shift_warning']:
            st.error(st.session_state['shift_warning'])
        else:
            st.info(st.session_state['shift_warning'])
        

         # ✅ 下载CSV
        csv_bytes = raw_df_adv.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下载完整结果 CSV",
                        csv_bytes,
                        file_name="SQ_full_results.csv",
                        mime="text/csv")

        # ✅ 下载PDF
        st.download_button("📥 下载完整报告 PDF",
                        st.session_state['pdf_bytes'],
                        file_name="SQ_report.pdf",
                        mime="application/pdf")


    with tab8:
        st.markdown("### 📊 混淆矩阵（数量）")
        cm = st.session_state['cm']
        cm_norm = st.session_state['cm_norm']
        fig_cm = px.imshow(cm,
                           x=['Pred OK','Pred NG'],
                           y=['True OK','True NG'],
                           text_auto=True, color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("### 📊 混淆矩阵（占比）")
        fig_cm2 = px.imshow(np.round(cm_norm*100,2),
                            x=['Pred OK','Pred NG'],
                            y=['True OK','True NG'],
                            text_auto=True, color_continuous_scale='Oranges')
        st.plotly_chart(fig_cm2, use_container_width=True)


    with tab9:
        if all(col in raw_df_adv.columns for col in ['PCA_X','PCA_Y','PCA_Z']):
            fig3d = px.scatter_3d(
                raw_df_adv,
                x='PCA_X', y='PCA_Y', z='PCA_Z',
                color='cm_group',
                color_discrete_map=COLOR_MAP,
                hover_name='slice_id',
                opacity=0.7
            )

            # ✅ 扩大视角 + 拉远
            fig3d.update_layout(
                scene=dict(
                    camera=dict(
                        eye=dict(x=2.5, y=2.5, z=1.8)   # 视角拉远
                    ),
                    aspectratio=dict(x=1.6, y=1.6, z=1.0),  # 增强空间感
                ),
                title="PCA 3D 特征空间（扩大视角）"
            )

            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.warning("PCA 三维特征不足")
    with tab10:
        st.markdown("### 🤖 AI 复盘助手（云雾接口）")

        # ====== 扩展 JSON Payload ======
        fp_report = st.session_state.get("fp_report", {})
        fn_conf_report = st.session_state.get("fn_conf_report", {})
        fn_drift_report = st.session_state.get("fn_drift_report", {})
        roc_data = st.session_state.get("roc_data", (None,None,None,None))
        metrics = st.session_state.get("metrics", {})
        cm = st.session_state.get("cm", None)
        cm_norm = st.session_state.get("cm_norm", None)
        df = st.session_state.get("raw_df_adv", pd.DataFrame())

        # Top5 机理差异特征（Step2a）
        fp_top = []
        for k, v in fp_report.items():
            if isinstance(v, dict) and "cohen_d" in v:
                fp_top.append({"feature": k, "cohen_d": v["cohen_d"], "signal": v.get("signal","")})
        fp_top = sorted(fp_top, key=lambda x: abs(x["cohen_d"]), reverse=True)[:5]

        # 严重度统计
        if "Severity" in df.columns:
            sev_stats = {
                "mean": float(df["Severity"].mean()),
                "max": float(df["Severity"].max()),
                "p90": float(df["Severity"].quantile(0.9))
            }
        else:
            sev_stats = {}

        ai_payload = {
            "step2a_fp_report": fp_report,
            "step2a_fp_top5": fp_top,
            "step2b_fn_conf": fn_conf_report,
            "step2b_fn_drift": fn_drift_report,
            "step3_roc_auc": roc_data[3],
            "metrics": metrics,
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "confusion_matrix_norm": cm_norm.tolist() if cm_norm is not None else None,
            "severity_stats": sev_stats
        }

        prompt = st.text_area(
            "Prompt（自动生成）",
            value=json.dumps(ai_payload, ensure_ascii=False, indent=2),
            height=400
        )

        if st.button("🚀 发送给 AI"):
            if not ai_api_key:
                st.warning("请输入 API Key")
            else:
                try:
                    payload = {
                        "model": "gpt-5.2",
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    headers = {"Authorization": f"Bearer {ai_api_key}"}
                    resp = requests.post(ai_api_url, json=payload, headers=headers, timeout=50)
                    resp.raise_for_status()
                    result = resp.json()
                    answer = result.get("choices",[{}])[0].get("message",{}).get("content","(无返回)")
                    st.success("✅ AI 返回结果")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"调用失败: {e}")

