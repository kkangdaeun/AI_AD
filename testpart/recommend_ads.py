import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)  # 열 모두 표시
pd.set_option("display.max_rows", None)     # 행 모두 표시
pd.set_option("display.max_colwidth", None) # 셀 안 내용 전부 표시

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

1# ===== 1. 데이터 로드 =====
ADS_PATH = "./data/AISAC 광고소재명별 광고 정보.csv"
LOGS_PATH = "./recs/user_watch_log.csv"

ads = pd.read_csv(ADS_PATH).astype(str)
logs = pd.read_csv(LOGS_PATH).astype(str)

# 공백 제거
ads.columns = ads.columns.str.strip()
logs.columns = logs.columns.str.strip()
ads = ads.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
logs = logs.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

# ===== 2. 점수화: 시청 비율(%) → 0~1 =====
logs["시청 비율"] = logs["시청 비율"].astype(float)

conditions = [
    logs["시청 비율"] >= 50,
    logs["시청 비율"] < 50
]

choices = [
    logs["시청 비율"] / 100,
    -1 + logs["시청 비율"] / 100
]

logs["점수"] = np.select(conditions, choices)

# ===== 3. 광고 클러스터링 (소재명 TF-IDF) =====
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(ads["광고소재명"])

kmeans = KMeans(n_clusters=18, n_init=10)
ads["광고 클러스터"] = kmeans.fit_predict(X)

# ===== 4. 사용자 선호 중업종 집계 =====
user_pref = (
    logs.groupby(["사용자 ID", "중업종 분류"])["점수"]
    .sum()
    .reset_index()
)
user_pref = user_pref[user_pref["점수"] > 0.1]


# ===== 5. 전체 중업종 개수 (대업종별 기준) =====
total_mid_per_major = ads.groupby("대업종 분류")["중업종 분류"].nunique()

# ===== 6. 추천 함수 =====
def recommend_ads(user_id: str,
                  top_n: int = 18,
                  min_major_score: float = 0.3) -> pd.DataFrame:
    
    # 사용자 선호 중업종 추출
    pref_rows = user_pref[user_pref["사용자 ID"] == user_id]
    preferred_mid = pref_rows["중업종 분류"].tolist() if not pref_rows.empty else []

    # 사용자 로그 필터링
    user_logs = logs[logs["사용자 ID"] == user_id]

    # 사용자 대업종 통계 계산
    major_stats = (
        user_logs.groupby("대업종 분류")
        .agg(
            n_user_mid=("중업종 분류", "nunique"),
            sum_score=("점수", "sum")
        )
        .reset_index()
    )

    major_stats["total_mid"] = major_stats["대업종 분류"].map(total_mid_per_major)
    major_stats["user_ratio"] = major_stats["n_user_mid"] / major_stats["total_mid"]

    # 강한 선호 대업종 선택
    strong_major = major_stats[
        (major_stats["user_ratio"] >= 0.5) &
        (major_stats["sum_score"] >= min_major_score)
    ]["대업종 분류"].tolist()

# ===== 7. 추천 후보 =====
    mid_cand = ads[ads["중업종 분류"].isin(preferred_mid)].copy()
    mid_cand["추천 유형"] = "중업종 기반"

    major_cand = ads[
        (ads["대업종 분류"].isin(strong_major)) &
        (~ads["중업종 분류"].isin(preferred_mid))
    ].copy()
    major_cand["추천 유형"] = "대업종 확장"

    # 클러스터 기반 샘플링 함수
    def sample_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        sampled = (
        df.groupby("광고 클러스터", group_keys=False)   
          .apply(lambda x: x.sample(1))                
          .reset_index(drop=True)                   
        )
        return sampled


    # 최종 추천 목록 구성
    mid_recs = sample_by_cluster(mid_cand)
    major_recs = sample_by_cluster(major_cand)
    recs = pd.concat([mid_recs, major_recs], ignore_index=True)

    return recs[[ 
        "광고소재명", "광고소재등록일", "광고소재초수",
        "대업종 분류", "중업종 분류", "소업종 분류",
        "광고주명", "광고회사명", "광고제작사",
        "광고 클러스터", "추천 유형"
    ]].head(top_n)




