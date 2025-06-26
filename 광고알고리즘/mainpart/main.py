import os, sys, importlib
import pandas as pd
import shutil
import random

# ===== 0. 경로‧파라미터 =====
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ADS_CSV        = "./data/AISAC 광고소재명별 광고 정보.csv"
ADSRECS_CSV    = "./recs/ads_recs.csv"
ADSRECS_OLD    = "./recs/ads_recs_old.csv"
WATCH_LOG_CSV  = "./recs/user_watch_log.csv"
USER_ID        = "user_1"
WATCH_RATE     = 70
TOTAL_RECS     = 30  # 총 추천 수

# 비율
N_USER    = int(TOTAL_RECS * 0.6)  # 사용자 기반: 60%
N_CLUSTER = int(TOTAL_RECS * 0.2)  # 군집 기반: 20%
N_RANDOM  = TOTAL_RECS - N_USER - N_CLUSTER  # 나머지 랜덤

# ===== 1. 광고 메타 CSV 확인 =====
ads_path_try = [os.path.join(BASE_DIR, ADS_CSV)]
ADS_PATH = next((p for p in ads_path_try if os.path.isfile(p)), None)
if ADS_PATH is None:
    sys.exit(f"[ERROR] '{ADS_CSV}' 파일을 못 찾음. 위치를 확인해 주세요.")

ads_df = pd.read_csv(ADS_PATH)

# ===== 2. 기존 추천 결과 백업 =====
adsrec_path = os.path.join(BASE_DIR, ADSRECS_CSV)
adsrec_old_path = os.path.join(BASE_DIR, ADSRECS_OLD)
if os.path.isfile(adsrec_path):
    shutil.copy(adsrec_path, adsrec_old_path)
    print(f"\n기존 추천 결과 백업 완료 → '{ADSRECS_OLD}'")

# ===== 3. 사용자 기반 + 군집 기반 + 랜덤 추천 수행 =====
import recommend_ads
import AdRecommender
importlib.reload(recommend_ads)
importlib.reload(AdRecommender)

# 사용자 기반
user_recs = recommend_ads.recommend_ads(USER_ID, top_n=TOTAL_RECS).head(N_USER)
user_recs["추천 소스"] = "사용자 기반"

# 군집 기반
cluster_recs = AdRecommender.recommend_by_cluster(USER_ID, top_n=N_CLUSTER)
cluster_recs["추천 소스"] = "군집 기반"

# 랜덤 추천
used_ads = pd.concat([user_recs, cluster_recs])["광고소재명"]
random_pool = ads_df[~ads_df["광고소재명"].isin(used_ads)].copy()
random_recs = random_pool.sample(n=N_RANDOM, random_state=42).copy()
random_recs["추천 소스"] = "랜덤 추천"

# 컬럼 정리 (랜덤에는 없는 컬럼 추가)
for col in ["광고 클러스터", "추천 유형"]:
    if col not in user_recs.columns:
        user_recs[col] = None
    if col not in cluster_recs.columns:
        cluster_recs[col] = None
    if col not in random_recs.columns:
        random_recs[col] = None

# 최종 병합
final_recs = pd.concat([user_recs, cluster_recs, random_recs], ignore_index=True)

# 저장
final_recs.to_csv(adsrec_path, index=False, encoding="utf-8-sig")
print(f"\n새 추천 결과 저장 완료 → '{ADSRECS_CSV}' ({len(final_recs)} rows)")

# ===== 4. user_watch_log.csv 확장 저장 =====
watch_log = final_recs.copy()
watch_log.insert(0, "사용자 ID", USER_ID)
watch_log["시청 비율"] = WATCH_RATE

cols_order = [
    "사용자 ID", "광고소재명", "광고소재등록일", "광고소재초수",
    "대업종 분류", "중업종 분류", "소업종 분류",
    "광고주명", "광고회사명", "광고제작사", "시청 비율", "추천 소스"
]
watch_log = watch_log.reindex(columns=cols_order)

watch_log_path = os.path.join(BASE_DIR, WATCH_LOG_CSV)
if os.path.isfile(watch_log_path):
    old_log = pd.read_csv(watch_log_path)
    combined_log = pd.concat([old_log, watch_log], ignore_index=True)

    if len(combined_log) > 70:
        combined_log = combined_log.iloc[35:].reset_index(drop=True)

else:
    combined_log = watch_log

combined_log.to_csv(watch_log_path, index=False, encoding="utf-8-sig")
print(f"\n누적 시청 로그 저장 완료 → '{WATCH_LOG_CSV}' ({len(combined_log)} rows)")

# ===== 5. 결과 미리보기 =====
def print_csv_preview(file_path, title):
    if not os.path.isfile(file_path):
        print(f"\n[SKIP] '{file_path}' 파일이 없음")
        return
    df = pd.read_csv(file_path)
    print(f"\n============== {title} ({len(df)} rows) ==============")
    print(",".join(df.columns))
    for _, row in df.iterrows():
        print(",".join(map(str, row.values)))

print_csv_preview(adsrec_old_path, "기존 추천 결과 (ads_recs_old.csv)")
print_csv_preview(adsrec_path,     "새로운 추천 결과 (ads_recs.csv)")
