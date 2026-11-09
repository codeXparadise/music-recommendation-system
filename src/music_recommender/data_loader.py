"""Data loading and preprocessing utilities for TuneMatch."""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd

from .config import DEFAULT_SONGS_PATH, DEFAULT_INTERACTIONS_PATH


class DataLoader:
    """Handles loading, validation, and querying of songs and interaction data."""

    def __init__(
        self,
        songs_path: Optional[Path | str] = None,
        interactions_path: Optional[Path | str] = None,
    ):
        self.songs_path = Path(songs_path) if songs_path else DEFAULT_SONGS_PATH
        self.interactions_path = (
            Path(interactions_path) if interactions_path else DEFAULT_INTERACTIONS_PATH
        )
        self._songs_df: Optional[pd.DataFrame] = None
        self._interactions_df: Optional[pd.DataFrame] = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load songs and interaction datasets from CSV files."""
        if not self.songs_path.exists():
            raise FileNotFoundError(f"Songs data file not found at: {self.songs_path}")
        if not self.interactions_path.exists():
            raise FileNotFoundError(
                f"Interactions data file not found at: {self.interactions_path}"
            )

        songs = pd.read_csv(self.songs_path)
        interactions = pd.read_csv(self.interactions_path)

        # Standardize ID columns as strings
        songs["song_id"] = songs["song_id"].astype(str)
        interactions["song_id"] = interactions["song_id"].astype(str)
        interactions["user_id"] = interactions["user_id"].astype(str)

        self._songs_df = songs
        self._interactions_df = interactions
        return songs, interactions

    @property
    def songs(self) -> pd.DataFrame:
        if self._songs_df is None:
            self.load_data()
        return self._songs_df

    @property
    def interactions(self) -> pd.DataFrame:
        if self._interactions_df is None:
            self.load_data()
        return self._interactions_df

    def get_user_ids(self) -> List[str]:
        """Return a sorted list of unique known user IDs."""
        return sorted(self.interactions["user_id"].unique().tolist())

    def get_genres(self) -> List[str]:
        """Return a sorted list of unique genres in the catalog."""
        return sorted(self.songs["genre"].dropna().unique().tolist())

    def get_artists(self) -> List[str]:
        """Return a sorted list of unique artists in the catalog."""
        return sorted(self.songs["artist"].dropna().unique().tolist())

    def get_user_positive_seed_ids(self, user_id: str) -> List[str]:
        """Return list of song IDs that a user interacted with positively."""
        positive_actions = ["like", "replay", "complete"]
        mask = (self.interactions["user_id"] == str(user_id)) & (
            self.interactions["action"].isin(positive_actions)
        )
        return self.interactions.loc[mask, "song_id"].astype(str).tolist()

    def get_user_seen_song_ids(self, user_id: str) -> set:
        """Return set of all song IDs a user has already interacted with."""
        mask = self.interactions["user_id"] == str(user_id)
        return set(self.interactions.loc[mask, "song_id"].astype(str))

