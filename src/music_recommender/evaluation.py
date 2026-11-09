"""Evaluation metrics (Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K) for recommendation models."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import DEFAULT_ACTION_WEIGHTS
from .models.content_based import ContentRecommender


def compute_ndcg_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    recommended = recommended[:k]
    num_relevant = min(k, len(relevant))
    if num_relevant == 0:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum(1.0 / np.log2(j + 2) for j in range(num_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def compute_map_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """Mean Average Precision at rank K."""
    recommended = recommended[:k]
    num_relevant = min(k, len(relevant))
    if num_relevant == 0:
        return 0.0

    hits = 0
    score = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / num_relevant


def compute_hit_rate_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """Hit Rate at rank K (1 if at least one hit in top K, else 0)."""
    recommended = recommended[:k]
    return 1.0 if any(item in relevant for item in recommended) else 0.0


class ModelEvaluator:
    """Evaluates recommendation models using leave-positive-out holdout strategy."""

    def __init__(
        self,
        songs_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        action_weights: Optional[Dict[str, float]] = None,
    ):
        self.songs = songs_df.copy()
        self.songs["song_id"] = self.songs["song_id"].astype(str)
        self.interactions = interactions_df.copy()
        self.interactions["song_id"] = self.interactions["song_id"].astype(str)
        self.interactions["user_id"] = self.interactions["user_id"].astype(str)
        self.action_weights = action_weights or DEFAULT_ACTION_WEIGHTS
        self.content_model = ContentRecommender(self.songs)

    def _build_user_item_matrix(self, interactions: pd.DataFrame) -> pd.DataFrame:
        data = interactions.copy()
        data["weight"] = data["action"].map(self.action_weights).fillna(0.0)
        return data.pivot_table(
            index="user_id",
            columns="song_id",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )

    def evaluate(self, k: int = 5) -> pd.DataFrame:
        """Run leave-one/two-positive-interaction-out evaluation across users."""
        positive = self.interactions[
            self.interactions["action"].isin(["like", "replay", "complete"])
        ].copy()

        metric_cols = ["model", "user_id", "precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", "hit_rate_at_k"]
        if positive.empty:
            return pd.DataFrame(columns=metric_cols)

        rows = []
        base_matrix = self._build_user_item_matrix(self.interactions)

        for user_id, group in positive.groupby("user_id"):
            if len(group) < 2:
                continue

            num_held_out = max(1, min(2, len(group) - 1))
            relevant = set(group["song_id"].astype(str).tail(num_held_out))
            seen = (
                set(
                    self.interactions.loc[
                        self.interactions["user_id"] == str(user_id), "song_id"
                    ].astype(str)
                )
                - relevant
            )

            if str(user_id) not in base_matrix.index:
                continue

            # Content model: seeds are remaining positive interactions
            seed_ids = list(
                set(positive.loc[positive["user_id"] == user_id, "song_id"].astype(str))
                - relevant
            )
            if seed_ids:
                content_recs = self.content_model.recommend(seed_ids, top_n=k)
                content_ids = list(content_recs["song_id"].astype(str))
            else:
                content_ids = []

            # Collaborative model: zero-out held-out items
            matrix = base_matrix.copy()
            for sid in relevant:
                if sid in matrix.columns:
                    matrix.loc[str(user_id), sid] = 0.0

            item_similarity = cosine_similarity(matrix.T)
            ids = list(matrix.columns.astype(str))
            vector = matrix.loc[str(user_id)].values.astype(float)
            scores = item_similarity @ vector

            cf_df = pd.DataFrame({"song_id": ids, "score": scores})
            cf_df = cf_df[~cf_df["song_id"].isin(seen)].sort_values("score", ascending=False)
            cf_ids = list(cf_df["song_id"].astype(str).head(k))

            for name, recommended in [("Content-based", content_ids), ("Collaborative", cf_ids)]:
                recommended = recommended[:k]
                hits = len(set(recommended) & relevant)
                precision = hits / k if k > 0 else 0.0
                recall = hits / len(relevant) if relevant else 0.0
                ndcg = compute_ndcg_at_k(recommended, relevant, k)
                map_k = compute_map_at_k(recommended, relevant, k)
                hit_rate = compute_hit_rate_at_k(recommended, relevant, k)

                rows.append({
                    "model": name,
                    "user_id": str(user_id),
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "ndcg_at_k": ndcg,
                    "map_at_k": map_k,
                    "hit_rate_at_k": hit_rate,
                })

        return pd.DataFrame(rows)

    def get_summary(self, detailed_metrics: Optional[pd.DataFrame] = None, k: int = 5) -> pd.DataFrame:
        """Compute average evaluation metrics grouped by model."""
        if detailed_metrics is None or detailed_metrics.empty:
            detailed_metrics = self.evaluate(k=k)

        cols = ["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", "hit_rate_at_k"]
        if detailed_metrics.empty:
            return pd.DataFrame(columns=["model"] + cols)

        existing_cols = [c for c in cols if c in detailed_metrics.columns]
        summary = (
            detailed_metrics.groupby("model", as_index=False)[existing_cols]
            .mean()
        )
        return summary
