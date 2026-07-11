import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
import glob
import gc

# ================= 配置区域 =================
INPUT_DIR = "./data_in"       # 存放多个录制CSV的文件夹
OUTPUT_FILE = "hxaim_recoil.csv"  # 最终输出文件
TARGET_HZ = 240               # 目标输出采样率
WINDOW_LENGTH = 15            # 必须为奇数
POLYORDER = 3
# ===========================================

def run_process():
    # 1. 获取所有csv文件
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not files:
        print(f"❌ 错误: 在 {INPUT_DIR} 下未找到任何 CSV 文件。")
        return

    print(f"🔍 发现 {len(files)} 个录制文件，开始处理...")

    # 2. 统一处理：读取 -> 去重 -> 对齐
    all_data = []
    max_time = 0.0
    dtype = np.float32

    for f in files:
        df = pd.read_csv(f, header=None, names=["t", "dx", "dy"], dtype=dtype)
        # 去除时间戳重复或回退的异常行
        df = df[df['t'].diff().fillna(1) > 0]
        if not df.empty:
            all_data.append(df)
            max_time = max(max_time, df['t'].max())

    # 3. 预分配连续内存空间
    master_time = np.arange(0, max_time + 1.0, 1.0, dtype=dtype)
    num_samples = len(master_time)
    
    # 使用 numpy 矩阵存储所有样本
    dx_matrix = np.zeros((len(all_data), num_samples), dtype=dtype)
    dy_matrix = np.zeros((len(all_data), num_samples), dtype=dtype)

    print("⚙️ 正在进行亚帧级数据对齐...")
    for i, df in enumerate(all_data):
        # np.interp 是性能最高且内存最稳的插值方案
        dx_matrix[i, :] = np.interp(master_time, df['t'], df['dx'])
        dy_matrix[i, :] = np.interp(master_time, df['t'], df['dy'])
    
    del all_data
    gc.collect()

    # 4. 计算中位数轨迹（去噪）
    master_dx = np.median(dx_matrix, axis=0)
    master_dy = np.median(dy_matrix, axis=0)
    
    del dx_matrix, dy_matrix
    gc.collect()

    # 5. 平滑处理
    print("📈 正在进行平滑滤波...")
    safe_window = WINDOW_LENGTH if WINDOW_LENGTH % 2 != 0 else WINDOW_LENGTH + 1
    master_dx = savgol_filter(master_dx, window_length=safe_window, polyorder=POLYORDER)
    master_dy = savgol_filter(master_dy, window_length=safe_window, polyorder=POLYORDER)

    # 6. 重采样至目标频率
    interval = 1000.0 / TARGET_HZ
    target_times = np.arange(0, max_time, interval, dtype=dtype)
    
    output_dx = np.interp(target_times, master_time, master_dx)
    output_dy = np.interp(target_times, master_time, master_dy)

    # 7. 保存结果
    final_df = pd.DataFrame({'t': target_times, 'dx': output_dx, 'dy': output_dy})
    final_df.to_csv(OUTPUT_FILE, index=False, header=False)
    
    print(f"✅ 处理完成！文件已保存至: {os.path.abspath(OUTPUT_FILE)}")
    print(f"📊 总数据行数: {len(final_df)}")

if __name__ == "__main__":
    try:
        run_process()
    except Exception as e:
        print(f"💥 发生严重错误: {e}")
