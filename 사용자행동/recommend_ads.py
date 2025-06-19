import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

### 1. 파일 불러오기
ads = pd.read_csv("AISAC_업종별분류.csv")        # 광고소재명, 중업종 분류, 대업종 분류 포함
logs = pd.read_csv("user_watch_log.csv")           # 사용자 ID, 광고소재명, 시청 여부 포함

ads.columns = ads.columns.str.strip()
logs.columns = logs.columns.str.strip()

### 2. 점수화: 완료(+1), 스킵(-1)
logs["점수"] = logs["시청 비율"] / 100

### 3. 광고소재명 기준 업종 정보 병합
merged = logs.merge(
    ads[["광고소재명", "중업종 분류", "대업종 분류"]],
    on="광고소재명", how="left"
)

### 4. 누락 확인
if merged["중업종 분류"].isnull().any():
    unmatched = merged[merged["중업종 분류"].isnull()]["광고소재명"].unique()
    print("❗ 매칭되지 않은 광고소재명:")
    print(unmatched)

### 5. 사용자 선호 중업종 계산
user_pref = merged.groupby(["사용자 ID", "중업종 분류"])["점수"].sum().reset_index()
user_pref = user_pref[user_pref["점수"] > 1.5]

### 6. 광고 클러스터링 (광고소재명 기반 TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(ads["광고소재명"].astype(str))

kmeans = KMeans(n_clusters=10, random_state=42)
ads["광고 클러스터"] = kmeans.fit_predict(X)

### 7. 추천 함수 정의
def recommend_ads(user_id, top_n=5):
    preferred = user_pref[user_pref["사용자 ID"] == user_id]["중업종 분류"].tolist()
    if not preferred:
        print(f"⚠️ 사용자 '{user_id}'의 선호 업종이 부족합니다.")
        return pd.DataFrame()
    
    candidates = ads[ads["중업종 분류"].isin(preferred)]
    recs = (
    candidates.groupby("광고 클러스터", group_keys=False)
    .apply(lambda df: df.sample(1))
    .reset_index(drop=True)
    )

    return recs[["광고소재명", "중업종 분류", "대업종 분류", "광고 클러스터"]].head(top_n)

### 8. 예시 추천
user_id = "user_001"
result = recommend_ads(user_id)
print(f"\n📢 사용자 '{user_id}' 추천 광고:")
print(result)
