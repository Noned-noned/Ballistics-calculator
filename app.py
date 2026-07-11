import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter  # 核心：引入 S-G 滤波替代高斯滤波

st.set_page_config(page_title="HxAim 亚帧级弹道提取系统", layout="wide")

st.title("🚀 亚帧级全量弹道解析引擎 (已优化)")
st.markdown("使用 **S-G 滤波** 替代高斯模糊，在平滑轨迹的同时最大限度保留枪械的瞬间力度转折。")

# 侧边栏参数控制
with st.sidebar:
    st.header("⚙️ 枪械核心参数")
    target_rpm = st.number_input("枪械射速 (RPM)", min_value=100, max_value=2000, value=649, step=1)
    num_bullets = st.number_input("需要提取的子弹数 (行数)", min_value=10, max_value=200, value=30, step=1)
    
    st.header("🎛️ 滤波算法参数 (S-G Filter)")
    window_length = st.slider("窗口长度 (Window Length)", min_value=5, max_value=31, value=11, step=2,
                              help="必须为奇数。值越大平滑度越高，越小保留细节越多。建议 7-15。")
    polyorder = st.slider("拟合多项式阶数 (Polyorder)", min_value=2, max_value=4, value=3, step=1,
                          help="阶数越高，曲线拟合越贴合原始数据点。")

uploaded_files = st.file_uploader("拖入由 Recorder DLL 采集的高密度 CSV 文件", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    dfs = []
    max_time_all = 0.0

    for f in uploaded_files:
        df = pd.read_csv(f, header=None, names=["time_ms", "dx", "dy"])
        df = df[df['time_ms'].diff().fillna(1) > 0]
        if not df.empty:
            dfs.append(df)
            max_time_all = max(max_time_all, df['time_ms'].max())

    if len(dfs) == 0:
        st.error("没有解析到有效数据！")
    else:
        # 1. 建立全局高精度时间轴
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

        # 3. 核心改进：使用 S-G 滤波替代 gaussian_filter1d
        smoothed_dx = savgol_filter(master_dx, window_length=window_length, polyorder=polyorder)
        smoothed_dy = savgol_filter(master_dy, window_length=window_length, polyorder=polyorder)

        # 4. 计算子弹击发时间轴
        interval_ms = 60000.0 / target_rpm
        target_times = np.arange(0, num_bullets) * interval_ms
        
        # 5. 精准提取
        f_smooth_x = interp1d(master_time, smoothed_dx, kind='linear', bounds_error=False, fill_value="extrapolate")
        f_smooth_y = interp1d(master_time, smoothed_dy, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        bullet_dx = f_smooth_x(target_times)
        bullet_dy = f_smooth_y(target_times)

        final_df = pd.DataFrame({
            'time_ms': np.round(target_times, 2), 
            'dx': np.round(bullet_dx, 1), 
            'dy': np.round(bullet_dy, 1)
        })

        # 显示与绘图
        col1, col2 = st.columns([1.2, 3])
        with col1:
            st.success(f"✅ 提取完成！")
            st.dataframe(final_df, height=500)
            csv_bytes = final_df.to_csv(index=False, header=False).encode('utf-8')
            # ✨ 这里修改了导出的文件名，与 C++ 回放端一致
            st.download_button("📥 下载 .csv", data=csv_bytes, file_name='hxaim_recoil.csv', mime='text/csv', use_container_width=True)

        with col2:
            fig = go.Figure()
            # 原始路径与平滑路径对照
            fig.add_trace(go.Scatter(x=master_dx, y=master_dy, mode='lines', name='原始采集轨迹', line=dict(color='rgba(200,200,200,0.5)', width=1)))
            fig.add_trace(go.Scatter(x=smoothed_dx, y=smoothed_dy, mode='lines', name='S-G 平滑轨迹', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=final_df['dx'], y=final_df['dy'], mode='markers', name='子弹位点', marker=dict(size=8, color='red')))
            fig.update_layout(yaxis=dict(autorange="reversed"), width=800, height=650)
            st.plotly_chart(fig, use_container_width=True)
