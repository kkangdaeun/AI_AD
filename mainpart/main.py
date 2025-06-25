import os, sys, importlib
import pandas as pd
import shutil

# ===== 0. 경로‧파라미터 =====
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ADS_CSV        = "./data/AISAC 광고소재명별 광고 정보.csv"
ADSRECS_CSV    = "./recs/ads_recs.csv"                 # 새 추천 결과
ADSRECS_OLD    = "./recs/ads_recs_old.csv"             # 기존 추천 결과 백업
WATCH_LOG_CSV  = "./recs/user_watch_log.csv"
USER_ID        = "user_1"
WATCH_RATE     = 70
TOP_N_FIN      = 18

# ===== 1. 광고 메타 CSV 확인 =====
ads_path_try = [os.path.join(BASE_DIR, ADS_CSV)]
ADS_PATH = next((p for p in ads_path_try if os.path.isfile(p)), None)
if ADS_PATH is None:
    sys.exit(f"[ERROR] '{ADS_CSV}' 파일을 못 찾음. 위치를 확인해 주세요.")

# ===== 2. 기존 ads_recs.csv → 백업 저장 (있을 경우) =====
adsrec_path = os.path.join(BASE_DIR, ADSRECS_CSV)
adsrec_old_path = os.path.join(BASE_DIR, ADSRECS_OLD)
if os.path.isfile(adsrec_path):
    shutil.copy(adsrec_path, adsrec_old_path)
    print(f"\n기존 추천 결과 백업 완료 → '{ADSRECS_OLD}'")

# ===== 3. recommend_ads.py로 새로운 ads_recs.csv 생성 =====
import recommend_ads
importlib.reload(recommend_ads)

final_recs = recommend_ads.recommend_ads(USER_ID, top_n=TOP_N_FIN)
final_recs.to_csv(adsrec_path, index=False, encoding="utf-8-sig")
print(f"\n새 추천 결과 저장 완료 → '{ADSRECS_CSV}' ({len(final_recs)} rows)")

# ===== 4. ads_recs.csv → user_watch_log.csv로 누적 저장 =====
watch_log = pd.read_csv(adsrec_path)

# 열 추가
watch_log.insert(0, "사용자 ID", USER_ID)
watch_log["시청 비율"] = WATCH_RATE

cols_order = [
    "사용자 ID", "광고소재명", "광고소재등록일", "광고소재초수",
    "대업종 분류", "중업종 분류", "소업종 분류",
    "광고주명", "광고회사명", "광고제작사", "시청 비율"
]
watch_log = watch_log.reindex(columns=cols_order)

# 누적 저장
watch_log_path = os.path.join(BASE_DIR, WATCH_LOG_CSV)
if os.path.isfile(watch_log_path):
    old_log = pd.read_csv(watch_log_path)
    combined_log = pd.concat([old_log, watch_log], ignore_index=True)
else:
    combined_log = watch_log

combined_log.to_csv(watch_log_path, index=False, encoding="utf-8-sig")
print(f"\n누적 시청 로그 저장 완료 → '{WATCH_LOG_CSV}' ({len(combined_log)} rows)")

# ===== 5. 기존 + 새로운 추천 결과 둘 다 출력 =====
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

