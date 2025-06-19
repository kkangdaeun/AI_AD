import pandas as pd, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 1. CSV
df_kw = pd.read_csv("한국방송광고진흥공사_AiSAC시스템 AI 인식결과 및 광고소재명별 키워드_20240731_1.csv")
df_cl = pd.read_csv("AISAC_clustered_mid.csv")

# 2. 머지
merge_cols = ["광고소재명", "광고소재등록일"]
df = (
    pd.merge(df_cl, df_kw, on=merge_cols, how="inner")
      .dropna(subset=["cluster_id"])
)

# --- 대업종·클러스터 매핑은 원본에서 추출 ---
cluster2major = (
    df_cl.drop_duplicates("cluster_id")[["cluster_id", "대업종 분류"]]
         .set_index("cluster_id")["대업종 분류"]
         .to_dict()
)
major_list = sorted(df_cl["대업종 분류"].unique())   # 40개 다 잡힘

# 3. 텍스트 전처리
text_cols = ["키워드", "AI_인식(사물)", "AI_인식(인물)", "AI_인식(장소)"]
def clean_text(s):
    if pd.isna(s): return ""
    s = re.sub(r'\d+(\.\d+)?"', "", s)
    s = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", s)
    return s
df["doc"] = df[text_cols].fillna("").agg(" ".join, axis=1).apply(clean_text)

# 4. TF-IDF
vec, X = TfidfVectorizer(min_df=2, max_features=5000), None
X = vec.fit_transform(df["doc"])

# 5. centroid
cluster_ids  = sorted(df["cluster_id"].astype(int).unique())
centroids = np.vstack([
    np.asarray(X[df["cluster_id"] == cid].mean(axis=0)).ravel()
    for cid in cluster_ids
])

# 6. 거리
dist = 1 - cosine_similarity(centroids)

# 7. MDS
coords = MDS(n_components=2, dissimilarity="precomputed", random_state=1)\
         .fit_transform(dist)

# ----------- 40색 팔레트 준비 -----------
# tab20 20색 + tab20b 20색 ⇒ 총 40색
tab20  = plt.cm.get_cmap("tab20").colors
tab20b = plt.cm.get_cmap("tab20b").colors
palette40 = list(tab20) + list(tab20b)      # (40, 4)

major2color = {m: palette40[i] for i, m in enumerate(major_list)}
colors = [major2color[cluster2major[c]] for c in cluster_ids]
# ----------------------------------------

# 8. plot
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=40, edgecolor="k")

for i, cid in enumerate(cluster_ids):
    plt.text(coords[i, 0], coords[i, 1], str(cid), fontsize=7, va="center", ha="center")

handles = [plt.Line2D([0], [0], marker="o", color="w",
           markerfacecolor=major2color[m], markersize=6, label=m)
           for m in major_list]
plt.legend(handles=handles, title="대업종", bbox_to_anchor=(1.02, 1), loc="upper left",
           fontsize=7, title_fontsize=8, ncol=2)  # 두 줄로 보여주기
plt.title("중업종 클러스터 간 내용 기반 관계 (MDS 2D, 대업종 40색)")
plt.xlabel("MDS Dim 1"); plt.ylabel("MDS Dim 2")
plt.tight_layout()
plt.show()


