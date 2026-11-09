"""Hybrid recommendation and cold-start handling."""

from typing import Dict, List, Optional
import pandas as pd

from .content_based import ContentRecommender
from .collaborative import CollaborativeRecommender


class HybridRecommender:
    """Combines Content-based and Collaborative filtering with cold-start fallback."""

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

        self.content_model = ContentRecommender(self.songs)
        self.collaborative_model = CollaborativeRecommender(
            self.songs, self.interactions, action_weights
        )

    def cold_start_recommend(
        self,
        chosen_genres: Optional[List[str]] = None,
        chosen_artists: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Provide recommendations for new users with no prior listening history."""
        chosen_genres = chosen_genres or []
        chosen_artists = chosen_artists or []

        if chosen_artists or chosen_genres:
            mask = pd.Series(False, index=self.songs.index)
            if chosen_artists:
                mask |= self.songs["artist"].isin(chosen_artists)
            if chosen_genres:
                mask |= self.songs["genre"].isin(chosen_genres)

            matching = self.songs[mask]
            if not matching.empty:
                preferred = matching.sort_values("popularity", ascending=False)
                if len(preferred) >= top_n:
                    recs = preferred.head(top_n).copy()
                else:
                    seeds = matching["song_id"].astype(str).tolist()
                    fill_count = top_n - len(preferred)
                    content_fill = self.content_model.recommend(seeds, top_n=fill_count)
                    recs = pd.concat([preferred, content_fill], ignore_index=True).drop_duplicates("song_id").head(top_n)
            else:
                recs = self.songs.sort_values("popularity", ascending=False).head(top_n).copy()
        else:
            # Fallback to top tracks by popularity
            if "popularity" in self.songs.columns:
                recs = self.songs.sort_values("popularity", ascending=False).head(top_n).copy()
            else:
                recs = self.songs.head(top_n).copy()

        if "score" not in recs.columns:
            recs["score"] = 0.0
        return recs

    def recommend(
        self,
        user_id: Optional[str] = None,
        seed_song_ids: Optional[List[str]] = None,
        top_n: int = 5,
        genres: Optional[List[str]] = None,
        artists: Optional[List[str]] = None,
        method: str = "hybrid",
    ) -> pd.DataFrame:
        """Unified recommendation method supporting hybrid, content, collaborative, and cold-start."""
        method = method.lower()
        genres = genres or []
        artists = artists or []

        # Determine if this is a cold-start scenario
        known_users = set(self.interactions["user_id"])
        is_cold_start = (user_id is None) or (str(user_id) not in known_users)

        if is_cold_start:
            return self.cold_start_recommend(
                chosen_genres=genres, chosen_artists=artists, top_n=top_n
            )

        # Retrieve user positive seed songs if not explicitly provided
        if seed_song_ids is None:
            positive_actions = ["like", "replay", "complete"]
            seed_song_ids = self.interactions.loc[
                (self.interactions["user_id"] == str(user_id)) &
                (self.interactions["action"].isin(positive_actions)),
                "song_id",
            ].astype(str).tolist()

        genre_filter = genres[0] if len(genres) == 1 else None

        if method == "content":
            recs = self.content_model.recommend(
                seed_song_ids, top_n=top_n, genre=genre_filter
            )
        elif method == "collaborative":
            recs = self.collaborative_model.recommend(user_id, top_n=top_n)
            if recs.empty:
                # Fallback to content if collaborative has insufficient history
                recs = self.content_model.recommend(
                    seed_song_ids, top_n=top_n, genre=genre_filter
                )
        else:  # hybrid
            content_candidates = self.content_model.recommend(
                seed_song_ids, top_n=top_n * 2, genre=genre_filter
            )
            collab_candidates = self.collaborative_model.recommend(
                user_id, top_n=top_n * 2
            )
            combined = pd.concat([content_candidates, collab_candidates], ignore_index=True)

            if not combined.empty:
                combined["score"] = pd.to_numeric(
                    combined.get("score", 0), errors="coerce"
                ).fillna(0)
                recs = (
                    combined.sort_values("score", ascending=False)
                    .drop_duplicates("song_id")
                    .head(top_n)
                )
            else:
                recs = self.content_model.recommend(
                    seed_song_ids, top_n=top_n, genre=genre_filter
                )

        # Apply multi-genre filter if multiple genres specified
        if genres and not recs.empty:
            preferred = recs[recs["genre"].isin(genres)]
            if not preferred.empty:
                recs = preferred.head(top_n)

        # Exclude songs user already interacted with (e.g. skipped or seen)
        if user_id and not recs.empty:
            seen = set(
                self.interactions.loc[
                    self.interactions["user_id"] == str(user_id), "song_id"
                ].astype(str)
            )
            recs = recs[~recs["song_id"].isin(seen)]

        return recs.head(top_n)
