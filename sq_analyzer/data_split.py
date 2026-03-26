import librosa
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

print("========================================")
print(" 阶段 2：测试内存级动态切片与特征提取")
print("========================================")

# 1. 使用你刚才跑出来的真实物理文件路径
wav_path = "/home/bestfunc/ai_train/ai_train_server/样本目录/NSK/训练数据/0.3s_0315/OK/2026-03-09/488106YU0AT426309004_0_3464609_350_2791146.wav"

# 2. 模拟从数据库拿到的切片时间 (取 0.198秒 到 0.505秒)
start_time = 0.19876952552278124
end_time = 0.5052514039056193
duration = end_time - start_time

print(f"🎯 目标文件: {wav_path.split('/')[-1]}")
print(f"⏱️ 截取区间: {start_time:.3f} 秒 -> {end_time:.3f} 秒 (时长: {duration:.3f} 秒)")

try:
    # 【核心操作】只把这 0.3 秒读进内存
    print("\n⏳ 正在执行 librosa 内存切片...")
    y, sr = librosa.load(wav_path, sr=44100, offset=start_time, duration=duration)
    
    print(f"✅ 切片成功！")
    print(f"📊 采样率: {sr} Hz")
    print(f"📊 截取到的数据点数: {len(y)} 个 (约等于 {len(y)/sr:.3f} 秒)")
    
    if len(y) > 0:
        print("\n⏳ 正在计算机理特征...")
        # 算一个时域峭度 (P0级特征)
        kurtosis_val = float(stats.kurtosis(y, fisher=True))
        print(f"✅ 计算完成！")
        print(f"👉 该切片的 时域峭度 (K) = {kurtosis_val:.4f}")
        
        if kurtosis_val > 3:
            print("💡 结论：该切片存在明显的冲击/敲击特征。")
        else:
            print("💡 结论：该切片信号平稳，无明显冲击。")
            
except Exception as e:
    print(f"❌ 切片或计算失败: {e}")
