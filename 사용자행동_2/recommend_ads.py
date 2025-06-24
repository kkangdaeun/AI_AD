
# ===== 0. 라이브러리 =====
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ===== 1. 데이터 로드 =====
ADS_PATH   = "AISAC_업종별분류.csv"
LOGS_PATH  = "user_watch_log.csv"        

ads  = pd.read_csv(ADS_PATH).astype(str)
logs = pd.read_csv(LOGS_PATH)

# 공백 제거 (컬럼도, 문자열도)
ads.columns  = ads.columns.str.strip()
logs.columns = logs.columns.str.strip()
ads = ads.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# ===== 2. 점수화: 시청 비율(%) → 0~1 =====
logs["점수"] = logs["시청 비율"] / 100

# ===== 3. 메타 병합 =====
merged = logs.merge(
    ads[["광고소재명", "중업종 분류", "대업종 분류"]],
    on="광고소재명", how="left"
)

# 병합 실패 확인
if merged["중업종 분류"].isnull().any():
    print("매칭되지 않은 광고소재명:")
    print(merged[merged["중업종 분류"].isnull()]["광고소재명"].unique())

# ===== 4. 사용자 선호 중업종 집계 =====
user_pref = (
    merged.groupby(["사용자 ID", "중업종 분류"])["점수"]
    .sum()
    .reset_index()
)
# ‘1.0’은 기준값. 취향에 맞춰 조절
user_pref = user_pref[user_pref["점수"] > 1.0] #검토

# ===== 5. 광고 클러스터링 (소재명 TF-IDF) =====
tfidf = TfidfVectorizer(max_features=1000)
X     = tfidf.fit_transform(ads["광고소재명"])

kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")  # sklearn 1.4+
ads["광고 클러스터"] = kmeans.fit_predict(X)

# ===== 6. 추천 함수 =====
def recommend_ads(user_id: str,
                  top_n: int = 10,
                  min_major_score: float = 2.0) -> pd.DataFrame: #2.0검
    """
    ① 사용자가 좋아한 중업종 확인
    ② 그 중업종이 속한 대업종 안에서 '관심 강도' 계산
    ③ 일정 기준 넘으면 같은 대업종의 다른 중업종 광고까지 확장
    ④ 클러스터 편중 막기 위해 클러스터별 1개 샘플
    """
    pref_rows = user_pref[user_pref["사용자 ID"] == user_id]
    if pref_rows.empty:
        print(f"⚠️ '{user_id}'는 선호 이력이 부족합니다.")
        return pd.DataFrame()

    preferred_mid = pref_rows["중업종 분류"].tolist()

    # --- 전체 대업종-중업종 구성표 ---
    total_mid_per_major = ads.groupby("대업종 분류")["중업종 분류"].nunique()

    # --- 사용자 관심도 기준 계산 ---
    major_stats = (
        merged[merged["중업종 분류"].isin(preferred_mid)]
        .groupby("대업종 분류")
        .agg(n_user_mid=("중업종 분류", "nunique"),
             sum_score=("점수", "sum"))
        .reset_index()
    )

    # --- 전체 중업종 개수 기준 병합 ---
    major_stats["total_mid"] = major_stats["대업종 분류"].map(total_mid_per_major)
    major_stats["user_ratio"] = major_stats["n_user_mid"] / major_stats["total_mid"]

    # --- 확장 가능한 대업종: 중업종 50% 이상 and 점수 기준 통과 ---
    strong_major = major_stats[
        (major_stats["user_ratio"] >= 0.5) &
        (major_stats["sum_score"] >= min_major_score)
    ]["대업종 분류"].tolist()
    
    # --- 후보군 만들기 ---
    mid_cand = ads[ads["중업종 분류"].isin(preferred_mid)]

    major_cand = ads[
        (ads["대업종 분류"].isin(strong_major)) &
        (~ads["중업종 분류"].isin(preferred_mid))
    ]

    # --- 클러스터별 1개 샘플 ---
    def sample_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return (df.groupby("광고 클러스터", group_keys=False)
                  .apply(lambda x: x.sample(1))
                  .reset_index(drop=True))

    mid_recs   = sample_by_cluster(mid_cand)
    major_recs = sample_by_cluster(major_cand)

    recs = pd.concat([mid_recs, major_recs], ignore_index=True)

    return recs[["광고소재명", "중업종 분류", "대업종 분류", "광고 클러스터"]].head(top_n)

# ===== 7. 사용 예시 =====
if __name__ == "__main__":
    sample_id = "user_1"
    print(f"\n📢 '{sample_id}' 추천 광고:")
    print(recommend_ads(sample_id, top_n=18))
