# 📱 거리 계산 방식별 핸드폰 군집분석 (Distance Metric Comparison for Phone Clustering)

핸드폰 스펙 데이터(Model, Price, Storage, Battery, Weight)에 유클리드(Euclidean), 맨해튼(Manhattan),
코사인(Cosine) 거리를 각각 적용해 군집분석을 수행하고, 실루엣 점수(Silhouette Score)와
Davies-Bouldin 지수로 군집 품질을 비교하는 Streamlit 앱입니다.

## 🗂️ 프로젝트 구조 (반드시 아래처럼 같은 폴더/루트에 있어야 함)
```
├── app.py            # Streamlit 메인 앱
├── phone_data.csv    # 실제 수집한 핸드폰 데이터 (12개 모델)
├── requirements.txt  # 의존 패키지 목록
└── README.md
```

> ⚠️ `app.py`가 `phone_data.csv`를 찾을 때 app.py 파일이 있는 폴더를 기준으로 경로를 계산하므로,
> 두 파일은 반드시 GitHub 저장소의 **같은 위치(루트)** 에 함께 있어야 합니다.

## ▶️ 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Streamlit Community Cloud 배포
1. Repository: `<본인아이디>/<저장소이름>`
2. Branch: `main` (저장소의 실제 기본 브랜치명과 일치해야 함)
3. Main file path: `app.py`
4. Deploy

## 📊 평가지표
정답 라벨이 없는 비지도 군집분석이므로 다음 내부 평가지표로 품질을 비교합니다.
- **Silhouette Score**: -1~1, 높을수록 좋음
- **Davies-Bouldin Index**: 0 이상, 낮을수록 좋음
