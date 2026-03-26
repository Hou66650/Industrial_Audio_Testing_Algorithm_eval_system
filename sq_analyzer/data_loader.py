import os
import json
import pandas as pd
from sqlalchemy import create_engine
import sys

class DataLoader:
    def __init__(self, host, port, user, password, db_name="smart_tools"):
        self.db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset=utf8mb4"
        try:
            self.engine = create_engine(self.db_url)
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            sys.exit(1)

    def fetch_and_merge_data(self, task_id, audio_root_dir):
        sql = f"""
            SELECT 
                file_id, split_start AS start_time, split_end AS end_time, 
                label_name, anomaly_output, anomaly_score, db_value
            FROM model_test_detail
            WHERE task_id = {task_id} AND isDelete = 0
        """
        df_db = pd.read_sql(sql, self.engine)
        if df_db.empty: return df_db

        df_db = df_db.drop_duplicates(subset=['file_id','start_time','end_time'], keep='first')

        file_mapping = []
        for dirpath, _, filenames in os.walk(audio_root_dir):
            for f in filenames:
                if f.endswith('.json'):
                    json_path = os.path.join(dirpath, f)
                    wav_path = json_path.replace('.json', '.wav')
                    if os.path.exists(wav_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as jf:
                                data = json.load(jf)
                                file_id = str(data.get('file_id','')).strip()
                                if file_id:
                                    file_mapping.append({'file_id': file_id, 'physical_wav_path': wav_path})
                        except:
                            pass

        df_files = pd.DataFrame(file_mapping)
        if df_files.empty: return pd.DataFrame()

        df_files = df_files.drop_duplicates(subset=['file_id'], keep='first')
        df_db['file_id'] = df_db['file_id'].astype(str).str.strip()
        df_merged = pd.merge(df_db, df_files, on='file_id', how='inner')

        df_merged['start_time'] = pd.to_numeric(df_merged['start_time'], errors='coerce')
        df_merged['end_time'] = pd.to_numeric(df_merged['end_time'], errors='coerce')
        df_merged = df_merged[df_merged['end_time'] > df_merged['start_time']]

        df_merged['slice_id'] = df_merged['file_id'] + "_T" + df_merged['start_time'].astype(str)
        
        # 按 slice_id 去重，只保留第一个（在机理计算前过滤重复片段）
        before_drop = len(df_merged)
        df_merged = df_merged.drop_duplicates(subset=['slice_id'], keep='first')
        after_drop = len(df_merged)
        if before_drop > after_drop:
            print(f"🧹 根据 slice_id 去重: 删除 {before_drop - after_drop} 条重复片段，剩余 {after_drop} 条")
        
        return df_merged

    def split_confusion_matrix_groups(self, df):
        if df.empty: return {}, {}

        def categorize(row):
            t = str(row['label_name']).strip().upper()
            p = str(row['anomaly_output']).strip().upper()
            is_true_ok = t in ['合格','OK','0','NORMAL']
            is_pred_ok = p in ['合格','OK','0','NORMAL']
            if is_true_ok and is_pred_ok: return 'TP'
            if not is_true_ok and not is_pred_ok: return 'TN'
            if is_true_ok and not is_pred_ok: return 'FN'
            if not is_true_ok and is_pred_ok: return 'FP'
            return 'Unknown'

        df = df.copy()
        df['cm_group'] = df.apply(categorize, axis=1)
        counts = df['cm_group'].value_counts().to_dict()
        TP, TN, FP, FN = counts.get('TP',0), counts.get('TN',0), counts.get('FP',0), counts.get('FN',0)

        metrics = {
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "Recall": TP/(TP+FN) if (TP+FN)>0 else 0.0,
            "Specificity": TN/(TN+FP) if (TN+FP)>0 else 0.0
        }

        grouped_dfs = {k: df[df['cm_group']==k].copy() for k in ['TP','TN','FP','FN']}
        return grouped_dfs, metrics
