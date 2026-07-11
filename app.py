import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

st.set_page_config(page_title="HxAim 亚帧级弹道提取系统", layout="wide")

st.title("🚀 亚帧级全量弹道解析引擎 (240Hz直寻版)")
st.markdown("放弃基于射速的离散点提取，直接按 **固定刷新率 (如 240Hz)** 输出密集平滑轨迹，完美适配 C++ 端的 $O(1)$ 直寻算法。")

# 侧边栏参数控制
with st.sidebar:
    st.header("⚙️ 引擎输出配置")
    target_hz = st.number_input("目标回放频率 (Hz)", min_value=60, max_value=1000, value=240, step=1, 
                                help="与 C++ 回放端的循环频率保持一致，240Hz 即每 4.16ms 输出一帧坐标。")
    
    st.header("🎛️ 滤波算法参数 (S-G Filter)")
    window_length = st.slider("窗口长度 (Window Length)", min_value=5, max_value=51, value=15, step=2,
                              help="必须为奇数。值越大平滑度越高，越小保留细节越多。由于是高频采样，建议 15-25。")
    polyorder = st.slider("拟合多项式阶数 (Polyorder)", min_value=2, max_value=4, value=3, step=1)

uploaded_files = st.file_uploader("拖入由 Recorder DLL 采集的高密度 CSV 文件", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    dfs = []
    max_time_all = 0.0

    for f in uploaded_files:
        df = pd.read_csv(f, header=None, names=["time_ms", "dx", "dy"])
        # 去除重复时间戳的脏数据
        df = df[df['time_ms'].diff().fillna(1) > 0]
        if not df.empty:
            dfs.append(df)
            max_time_all = max(max_time_all, df['time_ms'].max())

    if len(dfs) == 0:
        st.error("没有解析到有效数据！")
    else:
        # 1. 建立全局 1ms 级极高精度中间时间轴 (用于对齐多份样本)
        master_time = np.arange(0, max_time_all + 1, 1.0)
        all_dx_interp = []
        all_dy_interp = []

        # 2. 插值对齐
        for df in dfs:
            f_x = interp1d(df['time_ms'], df['dx'], kind='linear', bounds_error=False, fill_value=(df['dx'].iloc[0], df['dx'].iloc[-1]))
            f_y = interp1d(df['time_ms'], df['dy'], kind='linear', bounds_error=False, fill_value=(df['dy'].iloc[0], df['dy'].iloc[-1]))
            all_dx_interp.append(f_x(master_time))
            all_dy_interp.append(f_y(master_time))

        master_dx = np.nanmedian(all_dx_interp, axis=0)
        master_dy = np.nanmedian(all_dy_interp, axis=0)

        # 3. S-G 滤波平滑 (滤除视觉捕捉的抖动毛刺)
        smoothed_dx = savgol_filter(master_dx, window_length=window_length, polyorder=polyorder)
        smoothed_dy = savgol_filter(master_dy, window_length=window_length, polyorder=polyorder)

        # 4. ✨ 核心修改：生成固定频率的回放时间轴 (例如 240Hz = 每 4.1666ms)
        interval_ms = 1000.0 / target_hz
        # 采集的总时间除以帧间隔，得出总帧数
        target_times = np.arange(0, max_time_all, interval_ms)
        
        # 5. 按固定频率密集采样
        f_smooth_x = interp1d(master_time, smoothed_dx, kind='linear', bounds_error=False, fill_value="extrapolate")
        f_smooth_y = interp1d(master_time, smoothed_dy, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        output_dx = f_smooth_x(target_times)
        output_dy = f_smooth_y(target_times)

        final_df = pd.DataFrame({
            'time_ms': np.round(target_times, 2), 
            'dx': np.round(output_dx, 2), # 保留两位小数，交由 C++ 端的微积分余数累加器处理
            'dy': np.round(output_dy, 2)
        })

        # 显示与绘图
        col1, col2 = st.columns([1.2, 3])
        with col1:
            st.success(f"✅ 提取完成！共生成 {len(final_df)} 帧密集数据 ({target_hz}Hz)。")
            st.dataframe(final_df, height=500)
            csv_bytes = final_df.to_csv(index=False, header=False).encode('utf-8')
            st.download_button("📥 下载 hxaim_recoil.csv", data=csv_bytes, file_name='hxaim_recoil.csv', mime='text/csv', use_container_width=True)

        with col2:
            fig = go.Figure()
            # 原始路径与平滑路径对照
            fig.add_trace(go.Scatter(x=master_dx, y=master_dy, mode='lines', name='1ms中位原始轨迹', line=dict(color='rgba(200,200,200,0.5)', width=1)))
            fig.add_trace(go.Scatter(x=smoothed_dx, y=smoothed_dy, mode='lines', name='S-G 平滑连续轨迹', line=dict(color='blue', width=2)))
            
            # 使用较小的点展示密集的 240Hz 采样帧
            fig.add_trace(go.Scatter(x=final_df['dx'], y=final_df['dy'], mode='markers', name=f'{target_hz}Hz 采样输出点', marker=dict(size=4, color='red', opacity=0.7)))
            
            fig.update_layout(yaxis=dict(autorange="reversed"), width=800, height=650)
            st.plotly_chart(fig, use_container_width=True)
