import pandas as pd
import re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔 폰트나 맑은 고딕 중 하나 경로 지정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  

# 1. CSV 로드
df_kw = pd.read_csv("한국방송광고진흥공사_AiSAC시스템 AI 인식결과 및 광고소재명별 키워드_20240731_1.csv")
df_cl = pd.read_csv("AISAC_clustered_mid.csv")

# 2. 두 데이터 합치기 (광고소재명 + 등록일 기준)
merge_cols = ["광고소재명", "광고소재등록일"]
df = (
    pd.merge(df_cl, df_kw, on=merge_cols, how="inner")
      .dropna(subset=["cluster_id"])
)

# 3. 텍스트 전처리
text_cols = ["키워드", "AI_인식(사물)", "AI_인식(인물)", "AI_인식(장소)"]

def clean_text(s):
    if pd.isna(s):
        return ""
    s = re.sub(r'\d+(\.\d+)?"', '', s)                # 0", 1.5" 같은 타임스탬프 제거
    s = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", s)        # 한글·영문·숫자 외 제거
    return s

df["doc"] = (
    df[text_cols]
    .fillna("")
    .agg(" ".join, axis=1)
    .apply(clean_text)
)

# 4. TF-IDF 벡터화
vec = TfidfVectorizer(min_df=2, max_features=5000)
X = vec.fit_transform(df["doc"])

# 4.5 클러스터 ID 추출
cluster_ids = sorted(df["cluster_id"].astype(int).unique())

# 5. 클러스터별 centroid 계산
centroids = []
for cid in cluster_ids:
    rows = df["cluster_id"] == cid
    centroid_vec = np.asarray(X[rows].mean(axis=0)).ravel()
    centroids.append(centroid_vec)
centroids = np.vstack(centroids)


# 6. 클러스터 간 코사인 유사도 → 거리 행렬
sim  = cosine_similarity(centroids) 
dist = 1 - sim       # MDS용 거리

# 7. MDS 2차원 임베딩
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
coords = mds.fit_transform(dist)

# 8. 시각화
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1])

for i, cid in enumerate(cluster_ids):
    plt.text(coords[i, 0], coords[i, 1], str(cid), fontsize=8)

plt.title("중업종 클러스터 간 내용 기반 관계 (MDS 2D)")
plt.xlabel("MDS Dim 1")
plt.ylabel("MDS Dim 2")
plt.tight_layout()
plt.show()
