# ===== [0] 라이브러리 불러오기 =====
import os
os.system("pip install numpy pandas matplotlib scikit-learn scipy")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster      import KMeans
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch


# ===== [1] 데이터 불러오기 & 헤더 병합 =====
data_path = "./data/user/고정형_TV_실시간__장르별_시청시간_시청기록_연령대별__성별__20250618202618.csv"
df = pd.read_csv(data_path, encoding='cp949', header=[0,1,2])
df.columns = [
    f"{l0}_{l1}_{l2}" if 'Unnamed' not in str(l0) else l2
    for l0,l1,l2 in df.columns
]
df = df.rename(columns={
    '장르별(1)_장르별(1)_장르별(1)': '장르별',
    '연령대별(1)_연령대별(1)_연령대별(1)': '연령대별'
})


# ===== [2] 남자/여자 "시청률(%)" 평균 계산 =====
for gender in ['남자','여자']:
    rate_cols = [c for c in df.columns if f"{gender}_시청률" in c]
    df[f'{gender} 시청률(%)'] = df[rate_cols].astype(float).mean(axis=1)
    
    
# ===== [3] long 포맷으로 변환 =====
# 남자
m = df[['장르별','연령대별', '남자 시청률(%)']].copy()
m.columns = ['장르별','연령대별','시청률(%)']
m['성별'] = '남자'
# 여자
f = df[['장르별','연령대별', '여자 시청률(%)']].copy()
f.columns = ['장르별','연령대별','시청률(%)']
f['성별'] = '여자'

long = pd.concat([m,f], ignore_index=True).dropna()


# ===== [4] pivot → (연령대별, 성별)×장르별 시청률(%) 테이블 생성 =====
pivot_rate = long.pivot_table(
    index=['연령대별','성별'],
    columns='장르별',
    values='시청률(%)',
    aggfunc='mean',
    fill_value=0
)


# ===== [5] 스케일링 & KMeans =====
X = StandardScaler().fit_transform(pivot_rate.values)
k = 4
labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
pivot_rate['cluster'] = labels
genre_cols = [
    col for col in pivot_rate.columns
    if col not in ('cluster', 'segment')
]
# 세그먼트 매핑
cluster_to_segment = {
    0: '어린이_4-9세',
    1: '시니어_50-60대',
    2: '청장년_10-30대',
    3: '중장년_40-50대_남성'
}
pivot_rate['segment'] = pivot_rate['cluster'].map(cluster_to_segment)


# ===== [6] PCA 시각화 =====
# coords = PCA(n_components=2).fit_transform(X)
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(8,6))
# for i, lab in enumerate(labels):
#     age, gen = pivot_rate.index[i]
#     x,y = coords[i]
#     plt.scatter(x, y, 
#                 c=[plt.cm.tab10(lab)], s=100, alpha=0.8)
#     plt.annotate(f"{age}–{gen}", (x,y),
#                  textcoords="offset points", xytext=(5,3))
# plt.title('시청률 기반 클러스터링 (k=4)')
# plt.xlabel('PCA 1'); plt.ylabel('PCA 2')
# plt.grid(True)
# plt.legend(handles=[
#     plt.Line2D([],[],marker='o', color=plt.cm.tab10(c), linestyle='', label=f'Cluster {c}')
#     for c in range(k)
# ], title='Cluster', bbox_to_anchor=(1.05,1), loc='upper left')
# plt.tight_layout()
# plt.show()


# ===== [7] 덴드로그램 =====
# plt.figure(figsize=(10,4))
# sch.dendrogram(
#     sch.linkage(X, method='ward'),
#     labels=[f"{age}-{gen}" for age,gen in pivot_rate.index],
#     leaf_rotation=45
# )
# plt.title('시청률 기반 덴드로그램')
# plt.tight_layout()
# plt.show()