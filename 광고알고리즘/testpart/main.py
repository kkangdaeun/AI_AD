import os, sys, importlib
import pandas as pd

# 0. 경로‧파라미터 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

ADS_CSV        = "./data/AISAC 광고소재명별 광고 정보.csv"     # 광고 메타
ADRREC_CSV     = "./recs/initial_ads.csv"                    # ← ① 이미 만든 추천 CSV
WATCH_LOG_CSV  = "./recs/user_watch_log.csv"                  # ← ② 새로 만들 로그 CSV

SEGMENT    = "중장년_40-50대_남성"
USER_ID    = "user_1"
WATCH_RATE = 70
TOP_N_ADR  = 20
TOP_N_FIN  = 18

# 1. 광고 메타 CSV 위치 찾기
ads_path_try = [
    os.path.join(BASE_DIR, ADS_CSV)]
ADS_PATH = next((p for p in ads_path_try if os.path.isfile(p)), None)
if ADS_PATH is None:
    sys.exit(f"[ERROR] '{ADS_CSV}' 파일을 못 찾음. 위치를 확인해 주세요.")

# 2. AdRecommender 돌려서 initial_ads.csv 생성

from AdRecommender import AdRecommender
from user_clustering import pivot_rate, genre_cols

ad_df  = pd.read_csv(ADS_PATH)
engine = AdRecommender(ad_df); engine.fit()
recs = engine.recommend(
    segment        = SEGMENT,        # 추천할 타깃 세그먼트
    top_k          = TOP_N_ADR,      # 뽑을 광고 개수
    candidate_size = len(ad_df),     # 후보 풀 크기 (전체 광고)
    w_title        = 0.2,            # 제목 TF-IDF 가중치
    w_cat          = 0.8,            # 카테고리 TF-IDF 가중치
    max_per_cat    = 3,              # 같은 소업종 최대 3개
    top_n_per_cat  = 5,              # 점수가 높은 상위 N개만 샘플링 후보
    temperature    = 1.5,            # 샘플링 랜덤성 (낮으면 보수적)
    major_props    = engine.infer_major_props(SEGMENT, epsilon=0.2)
)


# 3. adrec_output.csv → user_watch_log.csv 변환
adrrec_path = os.path.join(BASE_DIR, ADRREC_CSV)
if not os.path.isfile(adrrec_path):
    sys.exit(f"[ERROR] '{ADRREC_CSV}' 파일이 없습니다. 먼저 만들어 주세요.")

watch_log = pd.read_csv(adrrec_path)

# 열 추가
watch_log.insert(0, "사용자 ID", USER_ID)   # 첫번째 열에 삽입
watch_log["시청 비율"] = WATCH_RATE        # 끝에 새 열

# 열 순서 재정렬 (필요할 때만)
cols_order = [
    "사용자 ID", "광고소재명", "광고소재등록일", "광고소재초수",
    "대업종 분류", "중업종 분류", "소업종 분류",
    "광고주명", "광고회사명", "광고제작사", "시청 비율"
]
watch_log = watch_log.reindex(columns=cols_order)

watch_log.to_csv(os.path.join(BASE_DIR, WATCH_LOG_CSV),
                 index=False, encoding="utf-8-sig")
print(f"\n'{WATCH_LOG_CSV}' 저장 완료 ({len(watch_log)} rows)")

#  4. recommend_ads.py 로 최종 개인화 추천
import recommend_ads
importlib.reload(recommend_ads)           # LOGS_PATH 다시 읽도록

final_recs = recommend_ads.recommend_ads(USER_ID, top_n=TOP_N_FIN)

print("\n================== 사용자 맞춤 광고 ==================")

# 열 제목 출력
print(",".join(final_recs.columns))

# 각 행을 CSV 형식으로 한 줄씩 출력
for _, row in final_recs.iterrows():
    print(",".join(map(str, row.values)))

output_path = os.path.join(BASE_DIR, "./recs/ads_recs.csv")
final_recs.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n사용자 맞춤 광고 저장 : {output_path}")
