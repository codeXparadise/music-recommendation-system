"""Collaborative filtering algorithm using weighted implicit feedback and item-item similarity."""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..config import DEFAULT_ACTION_WEIGHTS


class CollaborativeRecommender:
    """Item-based collaborative filtering from implicit feedback patterns."""

    def __init__(
        self,
        songs_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        action_weights: Optional[Dict[str, float]] = None,
    ):
        self.songs = songs_df.drop(columns=["score"], errors="ignore").copy()
        self.songs["song_id"] = self.songs["song_id"].astype(str)
        self.interactions = interactions_df.copy()
        self.interactions["song_id"] = self.interactions["song_id"].astype(str)
        self.interactions["user_id"] = self.interactions["user_id"].astype(str)
        self.action_weights = action_weights or DEFAULT_ACTION_WEIGHTS
        self.user_item_matrix = self._build_user_item_matrix()

    def _build_user_item_matrix(self) -> pd.DataFrame:
        """Aggregate implicit feedback into user-song preference weights."""
        data = self.interactions.copy()
        data["weight"] = data["action"].map(self.action_weights).fillna(0.0)
        matrix = data.pivot_table(
            index="user_id",
            columns="song_id",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        return matrix

    def recommend(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        """Recommend songs using item-neighbour collaborative filtering."""
        user_id = str(user_id)
        cols = list(self.songs.columns) + ["score"]

        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame(columns=cols)

        user_vector = self.user_item_matrix.loc[user_id].values.astype(float)
        if not np.any(user_vector > 0):
            return pd.DataFrame(columns=cols)

        # Transpose to get item vectors
        item_matrix = self.user_item_matrix.T
        item_similarity = cosine_similarity(item_matrix)
        item_ids = list(item_matrix.index)

        # Calculate scores as weighted sum of similarities to items user liked
        scores = np.zeros(len(item_ids), dtype=float)
        for i, preference in enumerate(user_vector):
            if preference != 0:
                scores += item_similarity[i] * preference

        # Exclude songs the user has already seen/interacted with
        seen = set(
            self.interactions.loc[
                self.interactions["user_id"] == user_id, "song_id"
            ].astype(str)
        )

        results = pd.DataFrame({"song_id": item_ids, "score": scores})
        results = results[~results["song_id"].isin(seen)]
        songs_clean = self.songs.drop(columns=["score"], errors="ignore")
        results = results.merge(songs_clean, on="song_id", how="inner")
        if "score" not in results.columns:
            results["score"] = 0.0
        return results.sort_values("score", ascending=False).head(top_n)

