import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

ADS_PATH = "./data/AISAC 광고소재명별 광고 정보.csv"
ads = pd.read_csv(ADS_PATH).astype(str)
ads.columns = ads.columns.str.strip()
ads = ads.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

# 광고 클러스터링
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(ads["광고소재명"])
kmeans = KMeans(n_clusters=30, n_init=10)
ads["광고 클러스터"] = kmeans.fit_predict(X)

total_mid_per_major = ads.groupby("대업종 분류")["중업종 분류"].nunique()


def process_watch_log(watch_log):
    print("\n📊 [main.py] 시청 로그 전달 받음")

    # ▶ 시청 로그에 해당하는 광고 추출
    user_logs = []
    for log in watch_log:
        row_num = log["row"]
        sec = log["sec"]

        matched = ads[ads["행 번호"] == str(row_num)]
        if matched.empty:
            print(f"⚠️ 광고 정보 없음: 행 번호 {row_num}")
            continue

        ad = matched.iloc[0]
        ad_info = {
            "광고소재명": ad["광고소재명"],
            "실제재생초수": float(ad["실제재생초수"]),
            "대업종 분류": ad["대업종 분류"],
            "중업종 분류": ad["중업종 분류"],
            "시청초수": float(sec)
        }
        print(f"  - {ad_info['광고소재명']} ({row_num}) / 시청: {sec}초")
        user_logs.append(ad_info)

    # ▶ DataFrame으로 변환
    user_watch_df = pd.DataFrame(user_logs)
    if user_watch_df.empty:
        print("❌ 유효한 시청 로그 없음")
        return

    # ▶ 시청 비율 계산
    user_watch_df["시청 비율"] = (user_watch_df["시청초수"] / user_watch_df["실제재생초수"]) * 100
    user_watch_df["점수"] = np.where(
        user_watch_df["시청 비율"] >= 50,
        user_watch_df["시청 비율"] / 100,
        -1 + user_watch_df["시청 비율"] / 100
    )

    # ▶ 선호 중업종 추출
    user_pref = (
        user_watch_df.groupby("중업종 분류")["점수"]
        .sum()
        .reset_index()
    )
    preferred_mid = user_pref[user_pref["점수"] > 0.1]["중업종 분류"].tolist()

        # ▶ 점수 0.1 이상 중업종만 필터링된 데이터프레임
    high_score_mid_df = user_pref[user_pref["점수"] > 0.1].reset_index(drop=True)
    total_high = high_score_mid_df["점수"].sum()
    high_score_mid_df["비율"] = (high_score_mid_df["점수"] / total_high)*100 
    print("\n🔥 [0.1 초과 중업종 점수]")
    print(high_score_mid_df)

    # ▶ 강한 대업종 추출
    major_stats = (
        user_watch_df.groupby("대업종 분류")
        .agg(n_user_mid=("중업종 분류", "nunique"), sum_score=("점수", "sum"))
        .reset_index()
    )
    major_stats["total_mid"] = major_stats["대업종 분류"].map(total_mid_per_major)
    major_stats["user_ratio"] = major_stats["n_user_mid"] / major_stats["total_mid"]
    strong_major = major_stats[
        (major_stats["user_ratio"] >= 0.5) & (major_stats["sum_score"] >= 0.3)
    ]["대업종 분류"].tolist()

    # ▶ 추천 후보군 구성
    mid_cand = ads[ads["중업종 분류"].isin(preferred_mid)].copy()
    mid_cand["추천 유형"] = "중업종 기반"

    major_cand = ads[
        (ads["대업종 분류"].isin(strong_major)) &
        (~ads["중업종 분류"].isin(preferred_mid))
    ].copy()
    major_cand["추천 유형"] = "대업종 확장"

    def sample_by_cluster(df):
        if df.empty:
            return df
        return df.groupby("광고 클러스터", group_keys=False).apply(lambda x: x.sample(1)).reset_index(drop=True)

    recs = pd.concat([
        sample_by_cluster(mid_cand),
        sample_by_cluster(major_cand)
    ], ignore_index=True)


    result_df = recs[["행 번호", "AISAC_URL"]].dropna().reset_index(drop=True)

    print("\n🎯 [추천 결과]")
    for _, row in result_df.iterrows():
        print(f"✅ 행 번호: {row['행 번호']}, URL: {row['AISAC_URL']}")

    return {
    "ads": result_df,                                    # 기존 추천
    "mid_scores": high_score_mid_df[["중업종 분류", "비율"]]  # 새 그래프용 데이터
    }
