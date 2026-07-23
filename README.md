# 📱 거리 계산 방식별 핸드폰 군집분석 (Distance Metric Comparison for Phone Clustering)

동일한 핸드폰 스펙 데이터(Model, Price, Storage, Battery, Weight)에 유클리드(Euclidean), 맨해튼(Manhattan),
코사인(Cosine) 거리를 각각 적용해 군집분석을 수행하고, 실루엣 점수(Silhouette Score)와
Davies-Bouldin 지수로 군집 품질을 비교하는 Streamlit 앱입니다.

Orange3에서 수행한 군집분석 워크플로우(Distances → Hierarchical Clustering → Silhouette Plot)를
Streamlit + scikit-learn으로 재현하여, 웹에서 인터랙티브하게 비교할 수 있도록 만들었습니다.

## 🔍 주요 기능
- CSV 업로드 또는 기본 제공 샘플 데이터(`sample_phones.csv`) 사용
- 군집 개수(k), 연결 방식(linkage) 조절
- 유클리드 / 맨해튼 / 코사인 거리 기반 계층적 군집분석(Agglomerative Clustering) 동시 수행
- 실루엣 점수, Davies-Bouldin 지수로 거리 방식별 군집 품질 정량 비교
- PCA 2D 산점도로 거리 방식별 군집 결과 시각화
- 군집 배정 결과 CSV 다운로드

## 🗂️ 프로젝트 구조
```
phone-clustering-app/
├── app.py               # Streamlit 메인 앱
├── sample_phones.csv    # 샘플 핸드폰 데이터
├── requirements.txt     # 의존 패키지 목록
└── README.md
```

## ▶️ 로컬 실행 방법
```bash
git clone https://github.com/<YOUR_USERNAME>/phone-clustering-app.git
cd phone-clustering-app
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Streamlit Community Cloud 배포 방법
1. 이 저장소를 본인 GitHub 계정으로 push
2. https://share.streamlit.io 접속 → GitHub 계정 연동
3. "New app" → 저장소/브랜치/`app.py` 선택 → Deploy
4. 몇 분 후 공개 URL(`https://<앱이름>.streamlit.app`) 생성

## 📊 데이터 형식
CSV는 다음 컬럼을 포함해야 합니다.

| Model | Price | Storage | Battery | Weight |
|---|---|---|---|---|
| iPhone15 | 1200000 | 128 | 3349 | 171 |
| GalaxyS24 | 1000000 | 128 | 4000 | 168 |

- `Model`: 텍스트 (군집분석에서는 제외, 표시용으로만 사용)
- `Price`, `Storage`, `Battery`, `Weight`: 숫자형 특성 (StandardScaler로 표준화 후 사용)

## 📐 평가지표 설명
정답 라벨이 없는 비지도 군집분석이므로 "정확도" 대신 다음 내부 평가지표를 사용합니다.
- **Silhouette Score**: -1~1, 높을수록 군집이 잘 분리됨
- **Davies-Bouldin Index**: 0 이상, 낮을수록 군집이 잘 분리됨

## 📄 라이선스
MIT License
