import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 设置网页标题和宽屏布局
st.set_page_config(page_title="HxAim 弹道数据优化可视化", layout="wide")

st.title("🔫 弹道数据综合优化与可视化工具")
st.markdown("上传你在游戏中录制的多个 `hxaim_recoil.csv` 文件，系统将自动执行**时间轴对齐**与**中位数滤波**，生成最平滑的压枪数据。")

# 文件上传组件（支持多选）
uploaded_files = st.file_uploader("请选择采集的 CSV 文件（可框选多个）", accept_multiple_files=True, type=['csv'])

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        # 读取无表头的 CSV
        df = pd.read_csv(f, header=None, names=["time_ms", "dx", "dy"])
        dfs.append(df)

    st.success(f"✅ 成功读取 {len(dfs)} 份弹道数据！")

    # 确定最小行数，截断多余的空弹数据
    lengths = [len(df) for df in dfs]
    max_len = min(lengths)
    
    time_ms_avg = []
    dx_avg = []
    dy_avg = []

    # 执行核心优化算法 (均值时间 + 中位数去噪坐标)
    for i in range(max_len):
        row_data = [df.iloc[i] for df in dfs]
        times = [r['time_ms'] for r in row_data]
        dxs = [r['dx'] for r in row_data]
        dys = [r['dy'] for r in row_data]
        
        time_ms_avg.append(np.mean(times))
        dx_avg.append(np.median(dxs))
        dy_avg.append(np.median(dys))

    # 生成优化后的 DataFrame
    optimized_df = pd.DataFrame({
        "time_ms": time_ms_avg,
        "dx": dx_avg,
        "dy": dy_avg
    })

    # 保留小数位数
    optimized_df['time_ms'] = optimized_df['time_ms'].round(2)
    optimized_df['dx'] = optimized_df['dx'].round(1)
    optimized_df['dy'] = optimized_df['dy'].round(1)

    # UI 布局：分左右两栏
    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.subheader("📊 优化结果表")
        st.info(f"数据有效行数自动截断为：**{max_len} 行**")
        st.dataframe(optimized_df, height=500)
        
        # 将 DataFrame 转换为 CSV 格式字节流，供下载
        csv_bytes = optimized_df.to_csv(index=False, header=False).encode('utf-8')
        st.download_button(
            label="📥 一键下载 optimized_recoil.csv",
            data=csv_bytes,
            file_name='optimized_recoil.csv',
            mime='text/csv',
            use_container_width=True
        )

    with col2:
        st.subheader("📈 弹道轨迹散点图")
        fig = go.Figure()
        
        # 1. 绘制所有原始数据（半透明浅色线，展示采集误差）
        for idx, df in enumerate(dfs):
            fig.add_trace(go.Scatter(
                x=df['dx'][:max_len], y=df['dy'][:max_len],
                mode='lines+markers',
                line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
                marker=dict(size=4, opacity=0.3),
                name=f'原始采集 {idx+1}',
                hoverinfo='skip'
            ))
            
        # 2. 绘制最终优化后的数据（红色高亮粗线）
        fig.add_trace(go.Scatter(
            x=optimized_df['dx'], y=optimized_df['dy'],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=8, color='black', symbol='cross'),
            name='🎯 最终优化平滑弹道',
            hovertemplate="击发时间: %{customdata} ms<br>X轴偏移: %{x}<br>Y轴偏移: %{y}<extra></extra>",
            customdata=optimized_df['time_ms']
        ))

        # 3. 设置图表样式 (将Y轴反转以符合游戏屏幕坐标系)
        fig.update_layout(
            xaxis_title="X 轴偏移 (像素)",
            yaxis_title="Y 轴偏移 (屏幕向下为正，故反转显示更直观)",
            yaxis=dict(autorange="reversed"), # 反转 Y 轴，更贴合压枪视觉
            width=800,
            height=600,
            hovermode="closest",
            plot_bgcolor='rgba(240, 240, 240, 0.8)'
        )
        # 标出屏幕中心十字准星原点 (0,0)
        fig.add_shape(type="line", x0=-20, y0=0, x1=20, y1=0, line=dict(color="blue", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=-20, x1=0, y1=20, line=dict(color="blue", width=1, dash="dash"))

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 请在上方拖拽或点击上传你的 CSV 数据文件，上传后系统将立刻开始渲染。")
