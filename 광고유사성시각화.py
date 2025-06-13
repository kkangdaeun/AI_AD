import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

# ========= 1. CSV 파일 불러오기 ========= #
file_path = '한국방송광고진흥공사_AISAC 광고소재명별 광고 정보_20240731.csv'
df = pd.read_csv(file_path)

# ========= 2. 중업종-대업종 매핑 생성 ========= #
major_mode_map = (
    df.groupby(['중업종 분류', '대업종 분류']).size()
    .reset_index(name='count')
    .sort_values(['중업종 분류', 'count'], ascending=[True, False])
    .drop_duplicates(subset=['중업종 분류'])
    .set_index('중업종 분류')['대업종 분류']
)

# ========= 3. 중업종 × 소업종 교차표 생성 ========= #
pivot = df.pivot_table(index='중업종 분류', columns='소업종 분류', aggfunc='size', fill_value=0)

# ========= 4. 벡터 정규화 및 거리 행렬 계산 ========= #
pivot_norm = normalize(pivot, norm='l2')  # 각 중업종을 벡터로 정규화
distance_matrix = cosine_distances(pivot_norm)  # 코사인 거리 계산

# ========= 5. MDS를 이용한 2D 좌표 생성 ========= #
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(distance_matrix)

# ========= 6. 시각화용 데이터프레임 정리 ========= #
mds_df = pd.DataFrame(coords, columns=['x', 'y'], index=pivot.index)
mds_df['대업종'] = mds_df.index.map(major_mode_map)

# ========= 7. 시각화 ========= #
plt.figure(figsize=(12, 10))
sns.set(font="Malgun Gothic")  # 한글 깨짐 방지
sns.scatterplot(data=mds_df, x='x', y='y', hue='대업종', s=100, palette='tab10')

# 중업종 라벨 표시
for idx, row in mds_df.iterrows():
    plt.text(row['x'] + 0.01, row['y'], idx, fontsize=9)

plt.title("중업종 간 소업종 분포 기반 관계 시각화 (MDS)")
plt.xlabel("MDS 1차 축")
plt.ylabel("MDS 2차 축")
plt.legend(title="대업종", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
