"""Real-time interaction store for tracking user events (like, replay, complete, partial, skip)."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .config import DEFAULT_ACTION_WEIGHTS, DEFAULT_INTERACTIONS_PATH


class InteractionStore:
    """Manages real-time user interaction events and compiles them for the recommender."""

    def __init__(
        self,
        initial_df: Optional[pd.DataFrame] = None,
        action_weights: Optional[Dict[str, float]] = None,
    ):
        self.action_weights = dict(action_weights or DEFAULT_ACTION_WEIGHTS)
        self._events: List[Dict] = []

        if initial_df is not None and not initial_df.empty:
            self._ingest_dataframe(initial_df)

    def _ingest_dataframe(self, df: pd.DataFrame):
        """Load pre-existing interactions into store."""
        for _, row in df.iterrows():
            action = str(row["action"]).lower()
            weight = self.action_weights.get(action, 0.0)
            self._events.append({
                "user_id": str(row["user_id"]),
                "song_id": str(row["song_id"]),
                "action": action,
                "weight": weight,
                "timestamp": row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "song_title": row.get("song_title", ""),
                "artist": row.get("artist", ""),
            })

    def update_action_weights(self, new_weights: Dict[str, float]):
        """Dynamically update action weights in real time and re-score all stored events."""
        self.action_weights.update(new_weights)
        for e in self._events:
            e["weight"] = self.action_weights.get(e["action"], 0.0)

    def record(
        self,
        user_id: str,
        song_id: str,
        action: str,
        song_title: str = "",
        artist: str = "",
        auto_save: bool = True,
    ) -> Dict:
        """Record a live interaction event and automatically update the CSV file."""
        action = action.lower()
        weight = self.action_weights.get(action, 0.0)
        event = {
            "user_id": str(user_id),
            "song_id": str(song_id),
            "action": action,
            "weight": weight,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "song_title": song_title,
            "artist": artist,
        }
        self._events.append(event)
        if auto_save:
            try:
                self.save_to_csv()
            except Exception:
                pass
        return event

    def get_dataframe(self) -> pd.DataFrame:
        """Export all events as a standardized interactions DataFrame with current weights."""
        if not self._events:
            return pd.DataFrame(columns=["user_id", "song_id", "action", "weight", "timestamp", "song_title", "artist"])
        df = pd.DataFrame(self._events)
        df["user_id"] = df["user_id"].astype(str)
        df["song_id"] = df["song_id"].astype(str)
        # Ensure weight column matches current active weights
        df["weight"] = df["action"].map(self.action_weights).fillna(0.0)
        return df

    def get_user_events(self, user_id: str) -> List[Dict]:
        """Return list of events recorded for a specific user, most recent first."""
        user_id = str(user_id)
        return [e for e in reversed(self._events) if e["user_id"] == user_id]

    def get_action_counts(self, user_id: Optional[str] = None) -> Dict[str, int]:
        """Count frequencies of each action type (like, skip, etc.)."""
        counts = {k: 0 for k in self.action_weights.keys()}
        target_events = self._events if user_id is None else [e for e in self._events if e["user_id"] == str(user_id)]
        for e in target_events:
            action = e["action"]
            counts[action] = counts.get(action, 0) + 1
        return counts

    def get_user_seed_song_ids(self, user_id: str) -> List[str]:
        """Return song IDs with positive feedback for user profile."""
        user_id = str(user_id)
        positive_actions = {"like", "replay", "complete"}
        # Latest positive items first
        seen = set()
        seeds = []
        for e in reversed(self._events):
            if e["user_id"] == user_id and e["action"] in positive_actions:
                sid = e["song_id"]
                if sid not in seen:
                    seen.add(sid)
                    seeds.append(sid)
        return seeds

    def get_user_skipped_song_ids(self, user_id: str) -> set:
        """Return song IDs whose latest interaction was a skip."""
        user_id = str(user_id)
        latest_action = {}
        for e in self._events:
            if e["user_id"] == user_id:
                latest_action[e["song_id"]] = e["action"]
        return {sid for sid, act in latest_action.items() if act == "skip"}

    def get_user_interacted_song_ids(self, user_id: str) -> set:
        """Return set of all song IDs user has interacted with."""
        user_id = str(user_id)
        return {e["song_id"] for e in self._events if e["user_id"] == user_id}

    def get_song_status_for_user(self, user_id: str, song_id: str) -> Optional[Dict]:
        """Return the latest action, cumulative weight, and timestamp for a user on a given song."""
        user_id = str(user_id)
        song_id = str(song_id)
        total_weight = 0.0
        latest_action = None
        latest_timestamp = None
        for e in self._events:
            if e["user_id"] == user_id and e["song_id"] == song_id:
                latest_action = e["action"]
                latest_timestamp = e["timestamp"]
                total_weight += self.action_weights.get(e["action"], 0.0)
        if latest_action is None:
            return None
        return {
            "action": latest_action,
            "weight": total_weight,
            "timestamp": latest_timestamp,
        }

    def clear_user_history(self, user_id: str, auto_save: bool = True):
        """Remove all recorded events for a given user and update the CSV file."""
        user_id = str(user_id)
        self._events = [e for e in self._events if e["user_id"] != user_id]
        if auto_save:
            try:
                self.save_to_csv()
            except Exception:
                pass

    def save_to_csv(self, filepath: Optional[Path | str] = None) -> Path:
        """Persist current interactions to CSV file."""
        target_path = Path(filepath) if filepath else DEFAULT_INTERACTIONS_PATH
        df = self.get_dataframe()[["user_id", "song_id", "action"]]
        df.to_csv(target_path, index=False)
        return target_path
