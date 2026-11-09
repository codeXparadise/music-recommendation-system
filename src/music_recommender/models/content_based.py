"""Content-based recommendation algorithm using TF-IDF and Cosine Similarity."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """Recommends songs based on metadata similarity (artist, genre, language, mood)."""

    def __init__(self, songs_df: pd.DataFrame):
        self.catalog = songs_df.copy()
        self.catalog["song_id"] = self.catalog["song_id"].astype(str)
        self._build_similarity_matrix()

    def _build_similarity_matrix(self):
        """Build TF-IDF matrix and cosine similarity matrix over metadata."""
        self.catalog["features"] = (
            self.catalog["artist"].fillna("") + " " +
            self.catalog["genre"].fillna("") + " " +
            self.catalog["language"].fillna("") + " " +
            self.catalog["mood"].fillna("")
        )
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.catalog["features"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self.index_by_id = {sid: i for i, sid in enumerate(self.catalog["song_id"])}

    def recommend(
        self,
        seed_song_ids: List[str],
        top_n: int = 10,
        genre: Optional[str] = None,
    ) -> pd.DataFrame:
        """Recommend songs whose metadata resembles the user's selected seed songs."""
        catalog = self.catalog.copy()
        seed_strings = [str(sid) for sid in seed_song_ids]
        valid_seeds = [self.index_by_id[sid] for sid in seed_strings if sid in self.index_by_id]

        if not valid_seeds:
            # Fallback if no valid seeds match
            candidates = catalog
            if genre:
                candidates = candidates[candidates["genre"].str.lower() == genre.lower()]
            if "popularity" in candidates.columns:
                candidates = candidates.sort_values("popularity", ascending=False)
            res = candidates.head(top_n).drop(columns=["features"], errors="ignore").copy()
            if "score" not in res.columns:
                res["score"] = 0.0
            return res

        # Average cosine similarity across all seed songs
        scores = self.similarity_matrix[valid_seeds].mean(axis=0)
        catalog["score"] = np.asarray(scores).ravel()

        # Exclude songs already in the seed list
        catalog = catalog[~catalog["song_id"].isin(seed_strings)]

        # Optional genre filter
        if genre:
            catalog = catalog[catalog["genre"].str.lower() == genre.lower()]

        # Return top N sorted by similarity score
        return (
            catalog.sort_values("score", ascending=False)
            .head(top_n)
            .drop(columns=["features"], errors="ignore")
        )

