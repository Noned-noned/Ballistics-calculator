import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="HxAim 亚帧级弹道提取系统", layout="wide")

st.title("🚀 亚帧级全量弹道解析引擎")
st.markdown("上传 240Hz 高密度全量采集数据，系统将使用 **高斯平滑滤波 + 高精样条插值** 为您降维提取极其纯净的子弹级宏文件！")

# 侧边栏参数控制
with st.sidebar:
    st.header("⚙️ 枪械核心参数")
    target_rpm = st.number_input("枪械射速 (RPM)", min_value=100, max_value=2000, value=649, step=1)
    num_bullets = st.number_input("需要提取的子弹数 (行数)", min_value=10, max_value=200, value=30, step=1)
    
    st.header("🎛️ 算法微调")
    smooth_sigma = st.slider("高斯滤波强度 (Sigma)", min_value=1.0, max_value=50.0, value=8.0, step=0.5,
                             help="值越大，曲线越平滑，但可能会丢失枪械极速变向时的尖锐细节。默认 8.0 通常最佳。")

uploaded_files = st.file_uploader("拖入由 Recorder DLL 采集的高密度 CSV 文件", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    dfs = []
    max_time_all = 0.0

    for f in uploaded_files:
        df = pd.read_csv(f, header=None, names=["time_ms", "dx", "dy"])
        # 去除时间倒退的异常帧（极少发生，为了安全）
        df = df[df['time_ms'].diff().fillna(1) > 0]
        if not df.empty:
            dfs.append(df)
            max_time_all = max(max_time_all, df['time_ms'].max())

    if len(dfs) == 0:
        st.error("没有解析到有效数据！")
    else:
        # 1. 建立全局高精度时间轴 (每 1ms 采样一个点)
        master_time = np.arange(0, max_time_all + 1, 1.0)
        
        all_dx_interp = []
        all_dy_interp = []

        # 2. 将所有高频采集的非标准时间轴，线性插值对齐到标准 1ms 时间轴上
        for df in dfs:
            f_x = interp1d(df['time_ms'], df['dx'], kind='linear', bounds_error=False, fill_value=(df['dx'].iloc[0], df['dx'].iloc[-1]))
            f_y = interp1d(df['time_ms'], df['dy'], kind='linear', bounds_error=False, fill_value=(df['dy'].iloc[0], df['dy'].iloc[-1]))
            all_dx_interp.append(f_x(master_time))
            all_dy_interp.append(f_y(master_time))

        # 3. 跨次录制的特征融合：使用中位数，彻底消除单次录制时的画面噪点
        master_dx = np.nanmedian(all_dx_interp, axis=0)
        master_dy = np.nanmedian(all_dy_interp, axis=0)

        # 4. 高斯平滑滤波：将毛刺曲线烫平成如丝般顺滑的物理阻尼轨迹
        smoothed_dx = gaussian_filter1d(master_dx, sigma=smooth_sigma)
        smoothed_dy = gaussian_filter1d(master_dy, sigma=smooth_sigma)

        # 5. 计算子弹击发的理论绝对时间轴
        interval_ms = 60000.0 / target_rpm
        target_times = np.arange(0, num_bullets) * interval_ms
        
        # 6. 终极提取：从顺滑曲线上，精准切出子弹射出那一毫秒的坐标
        f_smooth_x = interp1d(master_time, smoothed_dx, kind='linear', bounds_error=False, fill_value="extrapolate")
        f_smooth_y = interp1d(master_time, smoothed_dy, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        bullet_dx = f_smooth_x(target_times)
        bullet_dy = f_smooth_y(target_times)

        # 包装成 DataFrame
        final_df = pd.DataFrame({
            'time_ms': np.round(target_times, 2), 
            'dx': np.round(bullet_dx, 1), 
            'dy': np.round(bullet_dy, 1)
        })

        col1, col2 = st.columns([1.2, 3])

        with col1:
            st.success(f"✅ 提取成功！已结合 {len(dfs)} 份高频录制数据。")
            st.dataframe(final_df, height=500)
            
            csv_bytes = final_df.to_csv(index=False, header=False).encode('utf-8')
            st.download_button("📥 下载完美提取版 .csv", data=csv_bytes, file_name='optimized_recoil.csv', mime='text/csv', use_container_width=True)

        with col2:
            fig = go.Figure()
            
            # 绘制降采样后的高频轮廓 (为了性能和美观，图表上只画出采样密度)
            fig.add_trace(go.Scatter(
                x=smoothed_dx[::5], y=smoothed_dy[::5], # 每 5ms 画一个点
                mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.4)', width=3),
                name='高斯平滑底层轨迹',
                hoverinfo='skip'
            ))
            
            # 绘制最终切出的 30 发子弹红点
            fig.add_trace(go.Scatter(
                x=final_df['dx'], y=final_df['dy'],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=10, color='white', line=dict(color='red', width=2), symbol='circle'),
                name=f'🎯 降采样提取 ({target_rpm} RPM)',
                hovertemplate="击发时间: %{customdata} ms<br>X偏移: %{x}<br>Y偏移: %{y}<extra></extra>",
                customdata=final_df['time_ms']
            ))

            fig.update_layout(
                xaxis_title="X 轴偏移 (像素)", yaxis_title="Y 轴偏移 (像素)",
                yaxis=dict(autorange="reversed"), 
                width=800, height=650, hovermode="closest",
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            # 标出屏幕原点
            fig.add_shape(type="line", x0=-20, y0=0, x1=20, y1=0, line=dict(color="blue", width=1, dash="dash"))
            fig.add_shape(type="line", x0=0, y0=-20, x1=0, y1=20, line=dict(color="blue", width=1, dash="dash"))

            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 请在左侧配置好枪械射速 (RPM)，然后将高频录制的文件拖拽到上方区域！")
