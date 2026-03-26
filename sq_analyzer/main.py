import pandas as pd
from data_loader import DataLoader
from analyzer import SQAnalyzer
from feature_extractor import SQFeatureExtractor

# ==========================================
# 请在这里配置你的真实音频存放路径！
# ==========================================
AUDIO_ROOT_DIR = "/home/bestfunc/ai_train/test_system/data/audio/"

def run_advanced_analysis(task_id):
    print(f"\n🚀 开始执行 SmartQuality 深度复盘 (Task: {task_id})")
    print("="*60)

    # 1. 连接数据库加载数据
    DB_CONFIG = {
        "host": "192.168.2.176",
        "port": 23306,
        "user": "root",
        "password": "SmartTpm.2023",
        "db_name": "smart_tools"
    }
    loader = DataLoader(**DB_CONFIG)
    raw_df = loader.fetch_test_results(task_id)
    
    if raw_df.empty: return
    grouped_data, metrics = loader.split_confusion_matrix_groups(raw_df)

    # 2. 提取 TP (基准) 和 FP (漏检) 队列
    df_tp = grouped_data.get('TP', pd.DataFrame())
    df_fp = grouped_data.get('FP', pd.DataFrame())

    if df_fp.empty:
        print("✅ 无漏检样本，特异性达标，无需进行 FP 深度分析。")
        return
    if df_tp.empty:
        print("⚠️ 缺少 TP (正确合格) 样本作为基准，无法计算 Cohen's d。")
        return

    analyzer = SQAnalyzer()
    
    # ---------------------------------------------------------
    # 阶段 1：诊断数据库现有特征 (验证弱信号)
    # ---------------------------------------------------------
    print("\n[阶段 1] 诊断现有数据库特征 (db_value, diff_rms_value)...")
    old_report = analyzer.diagnose_fp(df_tp, df_fp, feature_cols=['db_value', 'diff_rms_value'])
    for feat, res in old_report.items():
        print(f"  - {feat}: Cohen's d = {res['cohen_d']} -> {res['conclusion']}")

    # ---------------------------------------------------------
    # 阶段 2：SOP Step 2b.1 提取 P0 级机理特征并验证
    # ---------------------------------------------------------
    print("\n[阶段 2] 启动音频引擎，提取 P0 级机理特征 (SOP Step 2b.1)...")
    extractor = SQFeatureExtractor(audio_root_dir=AUDIO_ROOT_DIR)
    
    # 辅助函数：为 DataFrame 批量提取特征
    def append_advanced_features(df):
        features_list = []
        for idx, row in df.iterrows():
            filename = row['original_filename']
            feats = extractor.extract_p0_physical_features(filename)
            if feats:
                feats['original_filename'] = filename
                features_list.append(feats)
        
        if not features_list: return df
        
        # 将新特征合并回原 DataFrame
        feat_df = pd.DataFrame(features_list)
        return pd.merge(df, feat_df, on='original_filename', how='inner')

    print(f"  ⏳ 正在提取 TP 基准组特征 (共 {len(df_tp)} 条)...")
    df_tp_adv = append_advanced_features(df_tp)
    
    print(f"  ⏳ 正在提取 FP 漏检组特征 (共 {len(df_fp)} 条)...")
    df_fp_adv = append_advanced_features(df_fp)

    # ---------------------------------------------------------
    # 阶段 3：计算新特征的 Cohen's d
    # ---------------------------------------------------------
    if not df_tp_adv.empty and not df_fp_adv.empty:
        print("\n[阶段 3] 新机理特征 Cohen's d 验证结果：")
        new_feature_cols = ['K_kurtosis', 'E_low_ratio', 'E_mid_ratio', 'E_high_ratio', 'SPL_peak_relative']
        
        new_report = analyzer.diagnose_fp(df_tp_adv, df_fp_adv, feature_cols=new_feature_cols)
        
        strong_signals_found = False
        for feat, res in new_report.items():
            # 高亮显示强信号
            if res['cohen_d'] > 0.8:
                print(f"  🌟 【突破!】 {feat}: Cohen's d = {res['cohen_d']} -> {res['conclusion']}")
                strong_signals_found = True
            else:
                print(f"  - {feat}: Cohen's d = {res['cohen_d']} -> {res['conclusion']}")
                
        print("\n" + "="*60)
        if strong_signals_found:
            print("🎯 最终决策 (SOP Step 4): ")
            print("已成功找到强信号特征！建议将上述高 Cohen's d 的机理特征加入算法模型的特征集中，重训模型，即可解决漏检问题！")
        else:
            print("⚠️ 最终决策 (SOP Step 4): ")
            print("P0 级物理特征依然为弱信号。建议：")
            print("1. 检查音频是否包含极强的环境噪声掩蔽了异响。")
            print("2. 升级至 P1 级心理声学特征（如粗糙度 R、尖锐度 S）进行更深度的解构。")
        print("="*60)
    else:
        print("\n❌ 特征提取失败，请检查 AUDIO_ROOT_DIR 路径是否正确，以及音频文件是否存在。")

if __name__ == "__main__":
    run_advanced_analysis(522)
