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
st.markdown("과거 서울 기온 데이터를 바탕으로 선형 회귀 분석을 수행하여 특정 연도의 예상 기온을 예측합니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 불러오기 및 전처리
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    # 데이터 로드 (UTF-8 인코딩)
    df = pd.read_csv(DATA_URL, encoding='utf-8')
    
    # 날짜 데이터 처리 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 평균기온 수치형 변환 및 결측치 제거
    df['평균기온'] = pd.to_numeric(df['평균기온'], errors='coerce')
    df_clean = df.dropna(subset=['평균기온'])
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_summary = df_clean.groupby('연도').agg(
        관측일수=('평균기온', 'count'),
        연평균기온=('평균기온', 'mean')
    ).reset_index()
    
    # 조건 필터링: 2025년 이하 & 관측일수 300일 이상
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
# 3. 회귀 분석 (선형 모델) 계산
# -----------------------------------------------------------------------------
X = df_yearly['연도'].values
Y = df_yearly['연평균기온'].values

# 1차 선형 회귀 (기울기, 절편)
slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 계산
corr_matrix = np.corrcoef(X, Y)
correlation = corr_matrix[0, 1]

# 정보 추출
start_year = int(X.min())
end_year = int(X.max())
data_count = len(X)

# 사이드바 데이터 요약 정보 표시
st.sidebar.header("📊 분석 데이터 요약")
st.sidebar.write(f"- **분석 시작 연도:** {start_year}년")
st.sidebar.write(f"- **분석 끝 연도:** {end_year}년")
st.sidebar.write(f"- **분석에 사용된 연도 개수:** {data_count}개")
st.sidebar.write(f"- **상관계수:** {correlation:.4f}")

# -----------------------------------------------------------------------------
# 4. 인터랙티브 슬라이더 및 예측 결과 표시
# -----------------------------------------------------------------------------
st.header("🔮 연도별 예상 기온 예측")

selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요:",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 회귀 방정식을 이용한 예상 기온 계산
predicted_temp = slope * selected_year + intercept

# 메트릭으로 크게 강조 표시
st.metric(
    label=f"📌 {selected_year}년 서울 예상 연평균 기온",
    value=f"{predicted_temp:.2f} °C"
)

st.caption(f"ℹ️ 이 예측값은 {start_year}년부터 {end_year}년까지 총 {data_count}개 연도의 데이터를 바탕으로 계산된 추세선(회귀 직선)에 기반합니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 5. Plotly 시각화 (산점도 + 회귀 직선 + 선택된 연도 포인트)
# -----------------------------------------------------------------------------
st.header("📈 연평균 기온 추세 및 회귀 분석 그래프")

# 회귀 직선을 그리기 위한 x, y 축 범위 설정 (1900년 ~ 2100년)
x_line = np.linspace(1900, 2100, 201)
y_line = slope * x_line + intercept

fig = go.Figure()

# 1. 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_yearly['연도'],
    y=df_yearly['연평균기온'],
    mode='markers',
    name='관측 연평균기온',
    marker=dict(color='#1f77b4', size=7, opacity=0.8),
    hovertemplate="<b>%{x}년</b><br>연평균 기온: %{y:.2f} °C<extra></extra>"
))

# 2. 회귀 직선
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line,
    mode='lines',
    name='회귀 직선 (추세선)',
    line=dict(color='#ff7f0e', width=2, dash='dash'),
    hovertemplate="<b>%{x}년 (추세)</b><br>예상 기온: %{y:.2f} °C<extra></extra>"
))

# 3. 슬라이더로 선택된 예측 연도 위치 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode='markers',
    name=f'선택한 연도 ({selected_year}년)',
    marker=dict(color='red', size=14, symbol='star'),
    hovertemplate=f"<b>선택 연도: {selected_year}년</b><br>예상 기온: {predicted_temp:.2f} °C<extra></extra>"
))

fig.update_layout(
    title=dict(text=f"<b>서울 연도별 평균기온과 선형 회귀 직선 (상관계수: {correlation:.4f})</b>", font=dict(size=18)),
    xaxis=dict(title="연도", range=[1895, 2105]),
    yaxis=dict(title="연평균 기온 (°C)"),
    template="plotly_white",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
