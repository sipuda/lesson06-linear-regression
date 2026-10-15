import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 제목
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 예측기")
st.markdown("과거 서울 기온 데이터를 바탕으로 선형 회귀 분석을 수행하여 기온 변화 추세 및 특정 연도의 예상 기온을 예측합니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 불러오기 및 전처리
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding='utf-8')
    
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    df['평균기온'] = pd.to_numeric(df['평균기온'], errors='coerce')
    df_clean = df.dropna(subset=['평균기온'])
    
    yearly_summary = df_clean.groupby('연도').agg(
        관측일수=('평균기온', 'count'),
        연평균기온=('평균기온', 'mean')
    ).reset_index()
    
    filtered_df = yearly_summary[
        (yearly_summary['연도'] <= 2025) & 
        (yearly_summary['관측일수'] >= 300)
    ].copy()
    
    return filtered_df

try:
    df_yearly = load_and_preprocess_data()
except Exception as e:
    st.error(f"⚠️ 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 회귀 분석 계산 (전체 기간 vs 최근 20년)
# -----------------------------------------------------------------------------
# 1) 전체 기간 분석
X_all = df_yearly['연도'].values
Y_all = df_yearly['연평균기온'].values
slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)
corr_all = np.corrcoef(X_all, Y_all)[0, 1]

slope_100y_all = slope_all * 100  # 100년당 기온 변화량량

start_year = int(X_all.min())
end_year = int(X_all.max())
data_count = len(X_all)

# 2) 최근 20년 분석 (마지막 연도 기준 최근 20개 연도)
df_recent20 = df_yearly[df_yearly['연도'] > (end_year - 20)]
X_recent = df_recent20['연도'].values
Y_recent = df_recent20['연평균기온'].values
slope_recent, intercept_recent = np.polyfit(X_recent, Y_recent, 1)
corr_recent = np.corrcoef(X_recent, Y_recent)[0, 1]

slope_100y_recent = slope_recent * 100  # 100년당 기온 변화량

# 사이드바 데이터 요약 정보
st.sidebar.header("📊 분석 데이터 요약")
st.sidebar.write(f"- **분석 시작 연도:** {start_year}년")
st.sidebar.write(f"- **분석 끝 연도:** {end_year}년")
st.sidebar.write(f"- **분석에 사용된 연도 개수:** {data_count}개")
st.sidebar.write(f"- **전체 기간 상관계수:** {corr_all:.4f}")
st.sidebar.write(f"- **최근 20년 상관계수:** {corr_recent:.4f}")

# -----------------------------------------------------------------------------
# 4. 100년당 상승 기온 비교 (전체 기간 vs 최근 20년)
# -----------------------------------------------------------------------------
st.header("🔥 기온 상승 속도 비교 (100년당 상승 온도의 변화)")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"🌐 전체 기간 ({start_year}~{end_year}년)",
        value=f"{slope_100y_all:+.2f} °C / 100년",
        delta=f"연간 {slope_all:+.4f} °C"
    )

with col2:
    # 전체 기간 대비 최근 20년의 상승 속도 차이 계산
    diff_slope = slope_100y_recent - slope_100y_all
    st.metric(
        label=f"⚡ 최근 20년 ({end_year-19}~{end_year}년)",
        value=f"{slope_100y_recent:+.2f} °C / 100년",
        delta=f"전체 평균 대비 {diff_slope:+.2f} °C 빠른 속도",
        delta_color="inverse"
    )

st.caption(f"ℹ️ 직선을 만든 전체 해의 개수는 **{data_count}개** ({start_year}년 ~ {end_year}년)입니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 인터랙티브 슬라이더 및 예측 결과 표시
# -----------------------------------------------------------------------------
st.header("🔮 연도별 예상 기온 예측")

selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요:",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 전체 기간 회귀선 기준 예측값
predicted_temp_all = slope_all * selected_year + intercept_all
# 최근 20년 회귀선 기준 예측값
predicted_temp_recent = slope_recent * selected_year + intercept_recent

pred_col1, pred_col2 = st.columns(2)
with pred_col1:
    st.metric(
        label=f"📌 {selected_year}년 예상 기온 (전체 추세 기준)",
        value=f"{predicted_temp_all:.2f} °C"
    )
with pred_col2:
    st.metric(
        label=f"📌 {selected_year}년 예상 기온 (최근 20년 추세 기준)",
        value=f"{predicted_temp_recent:.2f} °C"
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Plotly 시각화 (산점도 + 회귀 직선 2개 + 선택 연도 포인트)
# -----------------------------------------------------------------------------
st.header("📈 연평균 기온 추세 및 회귀 분석 그래프")

x_line = np.linspace(1900, 2100, 201)
y_line_all = slope_all * x_line + intercept_all
y_line_recent = slope_recent * x_line + intercept_recent

fig = go.Figure()

# 1. 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_yearly['연도'],
    y=df_yearly['연평균기온'],
    mode='markers',
    name='관측 연평균기온',
    marker=dict(color='#1f77b4', size=7, opacity=0.7),
    hovertemplate="<b>%{x}년</b><br>연평균 기온: %{y:.2f} °C<extra></extra>"
))

# 2. 전체 기간 회귀 직선
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line_all,
    mode='lines',
    name=f'전체 기간 추세선 ({slope_100y_all:+.2f}°C/100년)',
    line=dict(color='#ff7f0e', width=2, dash='dash'),
    hovertemplate="<b>%{x}년 (전체 추세)</b><br>예상 기온: %{y:.2f} °C<extra></extra>"
))

# 3. 최근 20년 회귀 직선
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line_recent,
    mode='lines',
    name=f'최근 20년 추세선 ({slope_100y_recent:+.2f}°C/100년)',
    line=dict(color='#d62728', width=2.5),
    hovertemplate="<b>%{x}년 (최근 20년 추세)</b><br>예상 기온: %{y:.2f} °C<extra></extra>"
))

# 4. 선택된 연도 위치 표시 (전체 추세 기준)
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp_all],
    mode='markers',
    name=f'선택 연도 ({selected_year}년)',
    marker=dict(color='black', size=14, symbol='star'),
    hovertemplate=f"<b>선택 연도: {selected_year}년</b><br>전체 추세 예상: {predicted_temp_all:.2f} °C<br>최근 20년 추세 예상: {predicted_temp_recent:.2f} °C<extra></extra>"
))

fig.update_layout(
    title=dict(text="<b>서울 연도별 평균기온 추세선 비교 (전체 기간 vs 최근 20년)</b>", font=dict(size=18)),
    xaxis=dict(title="연도", range=[1895, 2105]),
    yaxis=dict(title="연평균 기온 (°C)"),
    template="plotly_white",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
