"""
핸드폰 데이터 군집분석: 거리 계산 방식(유클리드/맨해튼/코사인) 비교 앱
Model, Price, Storage, Battery, Weight 4개 특성 기반
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="거리 방식별 핸드폰 군집분석", layout="wide")

st.title("📱 거리 계산 방식별 핸드폰 군집분석 비교")
st.caption("유클리드(Euclidean) · 맨해튼(Manhattan) · 코사인(Cosine) 거리를 같은 데이터에 적용해 군집 결과를 비교합니다.")

# ---------------------------------------------------------
# 1. 데이터 불러오기
# ---------------------------------------------------------
st.sidebar.header("1. 데이터 입력")
uploaded = st.sidebar.file_uploader("CSV 업로드 (Model, Price, Storage, Battery, Weight)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("sample_phones.csv")
    st.sidebar.info("샘플 데이터(sample_phones.csv)를 사용 중입니다.")

required_cols = {"Model", "Price", "Storage", "Battery", "Weight"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV에는 다음 컬럼이 모두 있어야 합니다: {required_cols}")
    st.stop()

with st.expander("원본 데이터 미리보기", expanded=False):
    st.dataframe(df, use_container_width=True)

feature_cols = ["Price", "Storage", "Battery", "Weight"]
X_raw = df[feature_cols].copy()

# ---------------------------------------------------------
# 2. 전처리: 표준화 (거리 기반 알고리즘은 스케일에 매우 민감)
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ---------------------------------------------------------
# 3. 사용자 설정
# ---------------------------------------------------------
st.sidebar.header("2. 군집분석 설정")
k = st.sidebar.slider("군집 개수 (k)", min_value=2, max_value=min(8, len(df) - 1), value=3)
linkage = st.sidebar.selectbox("연결 방식 (Linkage)", ["average", "complete", "single"], index=0,
                                help="Ward 연결은 유클리드 거리 전용이라 다른 거리와 비교하려면 average/complete/single을 사용합니다.")

metrics = ["euclidean", "manhattan", "cosine"]
metric_labels = {"euclidean": "유클리드 거리", "manhattan": "맨해튼 거리", "cosine": "코사인 거리"}

# ---------------------------------------------------------
# 4. 거리별 군집분석 수행
# ---------------------------------------------------------
results = {}
for metric in metrics:
    model = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage)
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels, metric=metric)
    dbi = davies_bouldin_score(X_scaled, labels)

    results[metric] = {"labels": labels, "silhouette": sil, "davies_bouldin": dbi}

# ---------------------------------------------------------
# 5. 결과 비교 표
# ---------------------------------------------------------
st.header("📊 거리 방식별 군집 품질 비교")
st.markdown(
    "정답 라벨이 없는 비지도 군집분석이므로, 군집 품질(=흔히 말하는 '정확도')은 "
    "**실루엣 점수(Silhouette, 높을수록 좋음)** 와 **Davies-Bouldin 지수(낮을수록 좋음)** 로 평가합니다."
)

summary_df = pd.DataFrame({
    "거리 방식": [metric_labels[m] for m in metrics],
    "실루엣 점수 (↑ 좋음)": [round(results[m]["silhouette"], 4) for m in metrics],
    "Davies-Bouldin 지수 (↓ 좋음)": [round(results[m]["davies_bouldin"], 4) for m in metrics],
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)

best_metric = max(metrics, key=lambda m: results[m]["silhouette"])
st.success(f"✅ 이 데이터에서는 **{metric_labels[best_metric]}** 방식의 실루엣 점수가 가장 높습니다 "
           f"({results[best_metric]['silhouette']:.4f}).")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=[metric_labels[m] for m in metrics],
                          y=[results[m]["silhouette"] for m in metrics],
                          name="Silhouette"))
fig_bar.update_layout(title="거리 방식별 실루엣 점수", yaxis_title="Silhouette Score")
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# 6. 거리 방식별 군집 시각화 (PCA 2D)
# ---------------------------------------------------------
st.header("🗺️ 거리 방식별 군집 시각화 (PCA 2D)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

cols = st.columns(3)
for i, metric in enumerate(metrics):
    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": results[metric]["labels"].astype(str),
        "Model": df["Model"],
    })
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", text="Model",
                      title=metric_labels[metric])
    fig.update_traces(textposition="top center", marker=dict(size=10))
    cols[i].plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 7. 상세 군집 배정 결과
# ---------------------------------------------------------
st.header("📋 모델별 군집 배정 상세 결과")
detail_df = df[["Model"] + feature_cols].copy()
for metric in metrics:
    detail_df[f"{metric_labels[metric]} 군집"] = results[metric]["labels"]
st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.download_button(
    "결과 CSV 다운로드",
    detail_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="clustering_results.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("Made with Streamlit · scikit-learn AgglomerativeClustering · Orange3 워크플로우를 Streamlit으로 재현")
