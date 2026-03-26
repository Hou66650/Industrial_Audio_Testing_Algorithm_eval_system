import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, roc_curve, auc

class SQAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.is_fitted = False

    def fit_baseline(self, df_tp, feature_cols):
        self.feature_cols = feature_cols
        if not df_tp.empty:
            self.scaler.fit(df_tp[feature_cols])
            self.is_fitted = True

    def get_z_scores(self, df):
        if not self.is_fitted or df.empty:
            return pd.DataFrame()
        z_vals = self.scaler.transform(df[self.feature_cols])
        return pd.DataFrame(z_vals, columns=[f"{c}_z" for c in self.feature_cols], index=df.index)

    def detect_label_noise(self, df_target, group_type):
        if df_target.empty or not self.is_fitted:
            return pd.DataFrame()
        z_df = self.get_z_scores(df_target)
        max_z = np.abs(z_df).max(axis=1)
        noise_mask = pd.Series(False, index=df_target.index)
        reason = ""
        if group_type=='FN':
            noise_mask = max_z > 2.5
            reason = "机理异常极强(Z>2.5)，疑似真实异响被误标为合格"
        elif group_type=='FP':
            noise_mask = max_z < 1.0
            reason = "机理完全正常(Z<1.0)，疑似合格被误标为异响"
        noise_df = df_target[noise_mask].copy()
        if not noise_df.empty:
            noise_df['noise_reason'] = reason
            noise_df['max_z_score'] = max_z[noise_mask]
        return noise_df

    def analyze_fp_mechanisms(self, df_fp, df_tp):
        report = {}
        if df_fp.empty or df_tp.empty:
            return report
        for col in self.feature_cols:
            mean_diff = df_fp[col].mean() - df_tp[col].mean()
            pooled_std = np.sqrt((df_fp[col].std()**2 + df_tp[col].std()**2)/2)
            d = abs(mean_diff)/pooled_std if pooled_std>0 else 0
            signal_type = '弱信号'
            if d>0.8: signal_type='强信号'
            elif d>0.5: signal_type='中等信号'
            report[col] = {'cohen_d': round(d,2), 'signal': signal_type}

        z_fp = self.get_z_scores(df_fp)
        if len(z_fp) >= 5:
            db = DBSCAN(eps=1.5, min_samples=3).fit(z_fp)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_info = "随机漏检"
            if n_clusters >= 1:
                try:
                    sil_score = silhouette_score(z_fp, labels) if n_clusters>0 else 0
                    if sil_score>0.4:
                        cluster_info = f"系统性漏检 (簇 {n_clusters}, 轮廓系数 {sil_score:.2f})"
                except: pass
            report['clustering_result'] = cluster_info
        return report

    def analyze_fn_confidence(self, df_fn):
        if df_fn.empty: return {}
        avg_conf = df_fn['anomaly_score'].mean()
        if avg_conf < 0.6:
            return {"type":"边界模糊","action":"补充边缘合格样本"}
        elif avg_conf > 0.8:
            return {"type":"模型过度自信","action":"疑似数据漂移"}
        return {"type":"中间区域","action":"结合漂移判断"}

    def analyze_fn_distribution(self, df_fn, df_tp):
        report = {}
        if df_fn.empty or df_tp.empty:
            for col in self.feature_cols:
                report[col] = {'ks_stat': None,'p_value':None,'drift':"无有效样本"}
            return report
        for col in self.feature_cols:
            stat,pval = ks_2samp(df_tp[col], df_fn[col])
            drift = "严重漂移" if pval<0.01 else ("轻微漂移" if pval<0.05 else "无漂移")
            report[col] = {'ks_stat': round(stat,3), 'p_value': round(pval,4), 'drift': drift}
        return report

    def analyze_tradeoff(self, y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def calculate_severity(self, df):
        if df.empty or not self.is_fitted: return df
        z_df = self.get_z_scores(df)
        df['mechanism_severity'] = np.abs(z_df).mean(axis=1)
        return df
