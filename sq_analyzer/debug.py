import os
import json
import pandas as pd
from sqlalchemy import create_engine

TASK_ID = 519
AUDIO_DIR = "/home/bestfunc/ai_train/ai_train_server/样本目录/NSK/训练数据/0.3s_0315"
DB_URL = "mysql+pymysql://root:SmartTpm.2023@192.168.2.176:23306/smart_tools?charset=utf8mb4"

print("========================================")
print(" 🕵️‍♂️ 寻找失踪的 38 条数据")
print("========================================")

engine = create_engine(DB_URL)

# 1. 查数据库原始数据
sql = f"SELECT file_id, split_start, split_end FROM model_test_detail WHERE task_id = {TASK_ID} AND isDelete = 0"
df_db = pd.read_sql(sql, engine)
print(f"1️⃣ 数据库原始记录: {len(df_db)} 条")

# 2. 查重：看看有没有完全一样的 0.3s 切片
df_db_clean = df_db.drop_duplicates(subset=['file_id', 'split_start', 'split_end'])
dup_count = len(df_db) - len(df_db_clean)
print(f"2️⃣ 剔除完全重复的 0.3s 切片后: {len(df_db_clean)} 条 (发现了 {dup_count} 条纯 Bug 克隆数据)")

# 3. 扫描本地真实存在的音频
file_ids_on_disk = set()
for dirpath, _, filenames in os.walk(AUDIO_DIR):
    for f in filenames:
        if f.endswith('.json'):
            try:
                with open(os.path.join(dirpath, f), 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    file_ids_on_disk.add(str(data.get('file_id', '')).strip())
            except: pass

# 4. 找出数据库有，但本地没有的文件
db_file_ids = set(df_db_clean['file_id'].astype(str).str.strip())
missing_files = db_file_ids - file_ids_on_disk

# 计算这些缺失的文件，对应了多少个切片
missing_slices_df = df_db_clean[df_db_clean['file_id'].astype(str).str.strip().isin(missing_files)]
missing_slice_count = len(missing_slices_df)

print(f"3️⃣ 最终能匹配上物理音频的切片: {len(df_db_clean) - missing_slice_count} 条")

print("\n========================================")
print(" 🚨 破案报告")
print("========================================")
if missing_slice_count > 0:
    print(f"丢失的切片主要原因是：【本地音频文件缺失】！")
    print(f"数据库里有 {len(missing_files)} 个音频文件（共包含 {missing_slice_count} 个切片），")
    print(f"但在你的本地目录 {AUDIO_DIR} 里，根本找不到它们！")
    print("\n缺失的 file_id 列表如下（你可以去查查为什么没下载下来）：")
    print(list(missing_files)[:10], "..." if len(missing_files) > 10 else "")
else:
    print("丢失的数据全部是因为数据库里存在完全一模一样的重复记录。")
