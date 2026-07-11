import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import gc
import os

# ✨ 关键补丁：在所有库加载前限制线程，防止 Linux 下多线程并行计算导致的段错误
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(page_title="HxAim 亚帧级弹道提取系统", layout="wide")

st.title("🚀 亚帧级全量弹道解析引擎 (Web UI 稳定版)")
st.markdown("已通过绘图抽样与内存管理优化，兼容服务器环境。")

with st.sidebar:
    st.header("⚙️ 枪械核心参数")
    target_hz = st.number_input("输出频率 (Hz)", value=240, step=1)
    
    st.header("🎛️ 滤波参数 (S-G Filter)")
    window_length = st.slider("窗口长度", min_value=5, max_value=51, value=15, step=2)
    polyorder = st.slider("拟合阶数", min_value=2, max_value=4, value=3, step=1)

uploaded_files = st.file_uploader("拖入由 Recorder DLL 采集的高密度 CSV 文件", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    dfs = []
    max_time_all = 0.0

    for f in uploaded_files:
        df = pd.read_csv(f, header=None, names=["time_ms", "dx", "dy"])
        if not df.empty:
            dfs.append(df)
            max_time_all = max(max_time_all, df['time_ms'].max())

    # 核心处理逻辑 (基于之前讨论的优化内存版)
    master_time = np.arange(0, max_time_all + 1, 1.0)
    all_dx_interp = []
    all_dy_interp = []

    for df in dfs:
        # 使用 np.interp 替代 interp1d 以减少对象创建
        all_dx_interp.append(np.interp(master_time, df['time_ms'], df['dx']))
        all_dy_interp.append(np.interp(master_time, df['time_ms'], df['dy']))

    master_dx = np.nan_to_num(np.median(all_dx_interp, axis=0))
    master_dy = np.nan_to_num(np.median(all_dy_interp, axis=0))
    
    del all_dx_interp, all_dy_interp, dfs
    gc.collect()

    # 滤波
    safe_window = int(window_length) if int(window_length) % 2 != 0 else int(window_length) + 1
    smoothed_dx = savgol_filter(master_dx, window_length=safe_window, polyorder=polyorder)
    smoothed_dy = savgol_filter(master_dy, window_length=safe_window, polyorder=polyorder)

    # 240Hz 重采样
    target_times = np.arange(0, max_time_all, 1000.0 / target_hz)
    output_dx = np.interp(target_times, master_time, smoothed_dx)
    output_dy = np.interp(target_times, master_time, smoothed_dy)

    final_df = pd.DataFrame({'time_ms': np.round(target_times, 2), 'dx': np.round(output_dx, 2), 'dy': np.round(output_dy, 2)})

    # UI 显示
    col1, col2 = st.columns([1.2, 3])
    with col1:
        st.success(f"✅ 提取成功，共 {len(final_df)} 帧。")
        st.dataframe(final_df, width='stretch')
        csv_bytes = final_df.to_csv(index=False, header=False).encode('utf-8')
        st.download_button("📥 下载 hxaim_recoil.csv", data=csv_bytes, file_name='hxaim_recoil.csv', mime='text/csv', width='stretch')

    with col2:
        # ✨ 绘图降采样处理：如果数据超过 1000 点，则每隔 5 点取 1 点绘图，防止浏览器渲染崩溃
        plot_df = final_df.iloc[::5, :] if len(final_df) > 1000 else final_df
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['dx'], y=plot_df['dy'], mode='markers', name='240Hz采样', marker=dict(size=4)))
        fig.update_layout(yaxis=dict(autorange="reversed"), width=700, height=500)
        st.plotly_chart(fig, use_container_width=True) # 这里用 use_container_width=True 仍然支持，或者改成 width=...
