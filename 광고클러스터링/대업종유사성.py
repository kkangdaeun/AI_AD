import pandas as pd, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ── 한글 폰트 (윈도우: Malgun Gothic, macOS: AppleGothic 등) ──
plt.rcParams["font.family"] = "Malgun Gothic"   # 설치된 한글 폰트 이름으로 수정 가능
plt.rcParams["axes.unicode_minus"] = False

# 1. CSV 로드
df_kw = pd.read_csv("한국방송광고진흥공사_AiSAC시스템 AI 인식결과 및 광고소재명별 키워드_20240731_1.csv")
df_cl = pd.read_csv("AISAC_clustered_대업종.csv")

# 2. 두 파일 머지 (광고소재명 + 등록일)
merge_cols = ["광고소재명", "광고소재등록일"]
df = (
    pd.merge(df_cl, df_kw, on=merge_cols, how="inner")
      .dropna(subset=["cluster_id"])
)

# 3. 텍스트 전처리
text_cols = ["키워드", "AI_인식(사물)", "AI_인식(인물)", "AI_인식(장소)"]
def clean_text(s):
    if pd.isna(s): return ""
    s = re.sub(r'\d+(\.\d+)?"', "", s)                  # 타임스탬프 제거
    s = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", s)          # 특수문자 제거
    return s
df["doc"] = df[text_cols].fillna("").agg(" ".join, axis=1).apply(clean_text)

# 4. TF-IDF 벡터화
vec = TfidfVectorizer(min_df=2, max_features=5000)
X   = vec.fit_transform(df["doc"])

# 5. 클러스터별 centroid 계산
cluster_ids = sorted(df["cluster_id"].astype(int).unique())   # 0‒39
centroids = np.vstack([
    np.asarray(X[df["cluster_id"] == cid].mean(axis=0)).ravel()
    for cid in cluster_ids
])

# 6. cluster_id → 대업종 이름 매핑
cluster2major = (
    df_cl.drop_duplicates("cluster_id")[["cluster_id", "대업종 분류"]]
         .set_index("cluster_id")["대업종 분류"]
         .to_dict()
)

# 7. 코사인 유사도 → 거리 행렬
dist = 1 - cosine_similarity(centroids)

# 8. MDS 2D 임베딩
coords = MDS(n_components=2, dissimilarity="precomputed", random_state=1)\
         .fit_transform(dist)

# 9. 40색 팔레트 (tab20 + tab20b)
tab20, tab20b = plt.cm.get_cmap("tab20").colors, plt.cm.get_cmap("tab20b").colors
palette40 = list(tab20) + list(tab20b)
colors = [palette40[cid] for cid in cluster_ids]

# 10. 시각화
plt.figure(figsize=(12, 9))
plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=50, edgecolor="k")

# 범례 (두 열로 깔끔하게)
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=palette40[i], markersize=6,
                      label=f"{i}: {cluster2major.get(i, '')}")
           for i in cluster_ids]
plt.legend(handles=handles, title="대업종 Cluster", ncol=2,
           bbox_to_anchor=(1.02, 1), loc="upper left",
           fontsize=7, title_fontsize=8)

plt.title("대업종 클러스터 간 내용 기반 관계 (MDS 2D, 이름 포함)")
plt.xlabel("MDS Dim 1")
plt.ylabel("MDS Dim 2")
plt.tight_layout()
plt.show()
