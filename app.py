import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import gc

st.set_page_config(page_title="HxAim 亚帧级弹道提取系统", layout="wide")

st.title("🚀 亚帧级全量弹道解析引擎 (纯数据处理版)")
st.markdown("已移除图形渲染，解决内存过载导致的崩溃问题。")

# 侧边栏参数控制
with st.sidebar:
    st.header("⚙️ 引擎输出配置")
    target_hz = st.number_input("目标回放频率 (Hz)", min_value=60, max_value=1000, value=240, step=1)
    
    st.header("🎛️ 滤波参数")
    window_length = st.slider("窗口长度", min_value=5, max_value=51, value=15, step=2)
    polyorder = st.slider("拟合阶数", min_value=2, max_value=4, value=3, step=1)

uploaded_files = st.file_uploader("拖入 CSV 文件", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    dfs = []
    max_time_all = 0.0

    for f in uploaded_files:
        df = pd.read_csv(f, header=None, names=["time_ms", "dx", "dy"])
        df = df[df['time_ms'].diff().fillna(1) > 0]
        if not df.empty:
            dfs.append(df)
            max_time_all = max(max_time_all, df['time_ms'].max())

    if len(dfs) > 0:
        master_time = np.arange(0, max_time_all + 1, 1.0)
        all_dx_interp = []
        all_dy_interp = []

        for df in dfs:
            f_x = interp1d(df['time_ms'], df['dx'], kind='linear', bounds_error=False, fill_value=(df['dx'].iloc[0], df['dx'].iloc[-1]))
            f_y = interp1d(df['time_ms'], df['dy'], kind='linear', bounds_error=False, fill_value=(df['dy'].iloc[0], df['dy'].iloc[-1]))
            all_dx_interp.append(f_x(master_time))
            all_dy_interp.append(f_y(master_time))

        master_dx = np.nan_to_num(np.nanmedian(all_dx_interp, axis=0))
        master_dy = np.nan_to_num(np.nanmedian(all_dy_interp, axis=0))
        
        # 显式清理内存
        del all_dx_interp, all_dy_interp, dfs
        gc.collect()

        # 确保 S-G 窗口合法
        safe_window = int(window_length) if int(window_length) % 2 != 0 else int(window_length) + 1
        safe_window = max(safe_window, polyorder + 2)

        smoothed_dx = savgol_filter(master_dx, window_length=safe_window, polyorder=polyorder)
        smoothed_dy = savgol_filter(master_dy, window_length=safe_window, polyorder=polyorder)

        # 生成 240Hz 采样数据
        interval_ms = 1000.0 / target_hz
        target_times = np.arange(0, max_time_all, interval_ms)
        
        f_smooth_x = interp1d(master_time, smoothed_dx, kind='linear', bounds_error=False, fill_value="extrapolate")
        f_smooth_y = interp1d(master_time, smoothed_dy, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        final_df = pd.DataFrame({
            'time_ms': np.round(target_times, 2), 
            'dx': np.round(f_smooth_x(target_times), 2),
            'dy': np.round(f_smooth_y(target_times), 2)
        })

        st.success(f"✅ 处理完成！共 {len(final_df)} 行数据。")
        st.dataframe(final_df, width='stretch')
        
        csv_bytes = final_df.to_csv(index=False, header=False).encode('utf-8')
        st.download_button("📥 下载 hxaim_recoil.csv", data=csv_bytes, file_name='hxaim_recoil.csv', mime='text/csv', width='stretch')
