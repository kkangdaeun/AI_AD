import os, sys, importlib
import pandas as pd
import numpy as np
import shutil
import random

# ===== 0. 경로‧파라미터 =====
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ADS_CSV        = "./data/AISAC 광고소재명별 광고 정보.csv"
ADSRECS_CSV    = "./recs/ads_recs.csv"
ADSRECS_OLD    = "./recs/ads_recs_old.csv"
WATCH_LOG_CSV  = "./recs/user_watch_log.csv"
MID_SCORE_CSV  = "./recs/mid_score.csv"

USER_ID        = "user_1"
WATCH_RATE     = 70
TOTAL_RECS     = 30

N_USER    = int(TOTAL_RECS * 0.6)
N_CLUSTER = int(TOTAL_RECS * 0.2)
N_RANDOM  = TOTAL_RECS - N_USER - N_CLUSTER

# ===== 1. 기존 추천 결과로 시청 로그 및 중업종 점수 누적 =====
adsrec_path = os.path.join(BASE_DIR, ADSRECS_CSV)
watch_log_path = os.path.join(BASE_DIR, WATCH_LOG_CSV)
mid_score_path = os.path.join(BASE_DIR, MID_SCORE_CSV)

if os.path.isfile(adsrec_path):
    prev_recs = pd.read_csv(adsrec_path)
    prev_recs.insert(0, "사용자 ID", USER_ID)
    prev_recs["시청 비율"] = WATCH_RATE

    cols_order = [
        "사용자 ID", "광고소재명", "광고소재등록일", "광고소재초수",
        "대업종 분류", "중업종 분류", "소업종 분류",
        "광고주명", "광고회사명", "광고제작사", "시청 비율", "추천 소스", "AISAC_URL", "실제재생초수"
    ]
    prev_recs = prev_recs.reindex(columns=cols_order)

    if os.path.isfile(watch_log_path):
        old_log = pd.read_csv(watch_log_path)
        combined_log = pd.concat([old_log, prev_recs], ignore_index=True)
        if len(combined_log) > 70:
            combined_log = combined_log.iloc[35:].reset_index(drop=True)
    else:
        combined_log = prev_recs

    combined_log.to_csv(watch_log_path, index=False, encoding="utf-8-sig")
    print(f"\n사용자 시청 로그 불러오기 완료 → '{WATCH_LOG_CSV}' ({len(combined_log)} rows)")

    user_pref = (
        combined_log.groupby(["사용자 ID", "중업종 분류"])["시청 비율"]
        .apply(lambda x: np.select([x >= 50, x < 50], [x / 100, -1 + x / 100]))
        .apply(np.sum)
        .reset_index(name="점수")
    )
    user_pref = user_pref[user_pref["점수"] > 0.1]
    mid_score_df = user_pref.groupby("중업종 분류", as_index=False)["점수"].sum()
    mid_score_df = mid_score_df[mid_score_df["점수"] > 0.1]

    if os.path.isfile(mid_score_path):
        prev_score = pd.read_csv(mid_score_path)
        combined_score = pd.concat([prev_score, mid_score_df], ignore_index=True)
        combined_score = combined_score.groupby("중업종 분류", as_index=False)["점수"].sum()
    else:
        combined_score = mid_score_df

    combined_score.to_csv(mid_score_path, index=False, encoding="utf-8-sig")
    print("\n중업종 점수 저장 완료 (\ub204적): ./recs/mid_score.csv")

# ===== 2. 기존 추천 결과 백업 =====
adsrec_old_path = os.path.join(BASE_DIR, ADSRECS_OLD)
if os.path.isfile(adsrec_path):
    shutil.copy(adsrec_path, adsrec_old_path)
    print(f"\n기존 추천 결과 백업 완료 → '{ADSRECS_OLD}'")

# ===== 3. 새로운 추천 수행 =====
ADS_PATH = os.path.join(BASE_DIR, ADS_CSV)
ads_df = pd.read_csv(ADS_PATH)

import recommend_ads
import AdRecommender
importlib.reload(recommend_ads)
importlib.reload(AdRecommender)

user_recs = recommend_ads.recommend_ads(USER_ID, top_n=TOTAL_RECS).head(N_USER)
user_recs["추천 소스"] = "사용자 기반"

cluster_recs = AdRecommender.recommend_by_cluster(USER_ID, top_n=N_CLUSTER)
cluster_recs["추천 소스"] = "군집 기반"

used_ads = pd.concat([user_recs, cluster_recs])["광고소재명"]
random_pool = ads_df[~ads_df["광고소재명"].isin(used_ads)].copy()
random_recs = random_pool.sample(n=N_RANDOM, random_state=42).copy()
random_recs["추천 소스"] = "랜덤 추천"

for col in ["광고 클러스터", "추천 유형"]:
    for df in [user_recs, cluster_recs, random_recs]:
        if col not in df.columns:
            df[col] = None

final_recs = pd.concat([user_recs, cluster_recs, random_recs], ignore_index=True)
final_recs.to_csv(adsrec_path, index=False, encoding="utf-8-sig")
print(f"\n새 추천 결과 저장 완료 → '{ADSRECS_CSV}' ({len(final_recs)} rows)")
