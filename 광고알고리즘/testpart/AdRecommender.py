import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import Dict, List, Optional

# pivot_rate 및 genre_cols 를 user_clustering 모듈에서 불러와 자동으로 segment_docs 생성
from user_clustering import pivot_rate, genre_cols

class AdRecommender:
    def __init__(
        self,
        ad_df: pd.DataFrame,
        title_col: str = '광고소재명',
        category_cols: List[str] = ['대업종 분류','중업종 분류','소업종 분류'],
        top_n_genres: int = 5
    ):
        
        self.ad_df = ad_df.copy().reset_index(drop=True)
        self.title_col = title_col
        self.category_cols = category_cols

        # segment_docs 자동 생성
        self.segment_docs = {}
        for seg, group in pivot_rate.groupby('segment'):
            genre_scores = group[genre_cols].iloc[0]
            top_genres = genre_scores.sort_values(ascending=False).head(top_n_genres).index.tolist()
            self.segment_docs[seg] = ' '.join(top_genres)

        self.vec_title = None
        self.vec_cat = None
        self.tfidf_title = None
        self.tfidf_cat = None
        self.n_ads = 0
        self.segment_names = list(self.segment_docs.keys())

    def fit(
        self,
        max_title_features: int = 1000,
        max_cat_features: int   = 3000,
        ngram_range: tuple      = (1,2)
    ):
        titles = self.ad_df[self.title_col].fillna('').tolist()
        cats = self.ad_df[self.category_cols].fillna('').agg(' '.join, axis=1).tolist()
        seg_docs = [self.segment_docs[s] for s in self.segment_names]

        self.vec_title = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_title_features,
            sublinear_tf=True,
            norm='l2'
        )
        self.vec_cat   = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_cat_features,
            sublinear_tf=True,
            norm='l2'
        )

        self.tfidf_title = self.vec_title.fit_transform(titles + seg_docs)
        self.tfidf_cat   = self.vec_cat.fit_transform(cats   + seg_docs)
        self.n_ads = len(titles)

    def recommend(
        self,
        segment: str,
        top_k: int = 10,
        candidate_size: Optional[int] = None,
        w_title: float = 0.2,
        w_cat: float   = 0.8,
        major_props: Dict[str,float] = None,
        max_per_cat: int = 2,
        top_n_per_cat: int = 3,
        temperature: float = 1.5
    ) -> pd.DataFrame:
        idx = self.segment_names.index(segment)
        q_idx = self.n_ads + idx
        sim_t = cosine_similarity(
            self.tfidf_title[q_idx], self.tfidf_title[:self.n_ads]
        ).flatten()
        sim_c = cosine_similarity(
            self.tfidf_cat[q_idx], self.tfidf_cat[:self.n_ads]
        ).flatten()
        sim = w_title * sim_t + w_cat * sim_c
        sim += np.random.normal(scale=1e-6, size=sim.shape)

        size = candidate_size or self.n_ads
        idxs = np.argsort(sim)[::-1][:size]
        df = self.ad_df.loc[idxs].copy().reset_index(drop=True)
        df['score'] = sim[idxs]

        scores = df['score']
        q_low, q_high = scores.quantile([0.33, 0.66])
        buckets = {
            'high': (q_high, scores.max()+1),
            'mid':  (q_low, q_high),
            'low':  (scores.min()-1, q_low)
        }
        ratios = {'high': 0.5, 'mid': 0.35, 'low': 0.15}

        picks = []
        seen = set()
        seen_title_group = set()
        for name, (lo, hi) in buckets.items():
            cnt = int(round(ratios[name] * top_k))
            bucket_df = df[(df['score'] >= lo) & (df['score'] < hi) & (~df[self.title_col].isin(seen))]
            if bucket_df.empty or cnt <= 0:
                continue
            bucket_df = (
                bucket_df
                .sort_values('score', ascending=False)
                .groupby(self.category_cols[2], group_keys=False)
                .head(top_n_per_cat)
                .groupby(self.category_cols[2], group_keys=False)
                .apply(lambda grp: grp.sample(n=min(len(grp), max_per_cat)))
                .reset_index(drop=True)
            )
            bucket_df = bucket_df[~bucket_df[self.title_col].isin(seen)]
            s = bucket_df['score'].values
            probs = np.exp(s / temperature)
            probs /= probs.sum()
            sample = bucket_df.sample(n=min(cnt, len(bucket_df)), weights=probs, replace=False)
            picks.append(sample)
            seen |= set(sample[self.title_col])

        result = pd.concat(picks) if picks else pd.DataFrame(columns=df.columns)
        if len(result) < top_k:
            rest = df[~df[self.title_col].isin(seen)]
            rest = (
                rest
                .sort_values('score', ascending=False)
                .groupby(self.category_cols[2], group_keys=False)
                .head(top_n_per_cat)
                .groupby(self.category_cols[2], group_keys=False)
                .apply(lambda grp: grp.sample(n=min(len(grp), max_per_cat)))
                .reset_index(drop=True)
            )
            rest = rest[~rest[self.title_col].isin(seen)]
            fill = rest.sample(n=top_k - len(result), replace=False)
            result = pd.concat([result, fill])

        return result.sample(frac=1).reset_index(drop=True)

    def infer_major_props(
        self, segment: str, top_n: int = 100, epsilon: float = 0.2
    ) -> Dict[str, float]:
        recs = self.recommend(segment=segment, top_k=top_n)
        counts = recs['대업종 분류'].value_counts(normalize=True)
        sm = counts + epsilon
        return (sm / sm.sum()).to_dict()

# ===== 사용 예시 =====
data_path = './data/AISAC 광고소재명별 광고 정보.csv'
ad_data = pd.read_csv(data_path)
engine = AdRecommender(ad_data)
engine.fit()

# 사용자 정보 받아오기
num_of_queue = 20
user_segment = '중장년_40-50대_남성'
props = engine.infer_major_props(user_segment, epsilon=0.2)
recs = engine.recommend(
    segment=user_segment,
    top_k=num_of_queue,
    candidate_size=len(ad_data),
    w_title=0.2,
    w_cat=0.8,
    major_props=props,
    max_per_cat=3,
    top_n_per_cat=5,
    temperature=1.5
)

print("\n================== 군집 맞춤 광고 ==================")
print(recs[['광고소재명','광고소재등록일','대업종 분류','중업종 분류','소업종 분류', '광고주명', '광고회사명', '광고제작사', "AISAC_URL", "실제재생초수"]])
recs.drop(columns='score').to_csv("./recs/initial_ads.csv",
                                 index=False,
                                 encoding="utf-8-sig")
