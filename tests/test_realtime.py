"""Unit tests for InteractionStore and MusicAPIService."""

import unittest
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from music_recommender.data_loader import DataLoader
from music_recommender.interaction_store import InteractionStore
from music_recommender.services.music_api import MusicAPIService, infer_mood


class TestRealTimeComponents(unittest.TestCase):

    def setUp(self):
        loader = DataLoader()
        self.songs, self.interactions = loader.load_data()
        self.store = InteractionStore(self.interactions)

    def test_record_like_interaction(self):
        event = self.store.record("U_TEST", "S001", "like", "Midnight Drive", "Nova Lane", auto_save=False)
        self.assertEqual(event["action"], "like")
        self.assertEqual(event["weight"], 3.0)
        self.assertIn("S001", self.store.get_user_seed_song_ids("U_TEST"))

    def test_record_skip_interaction(self):
        event = self.store.record("U_TEST", "S002", "skip", "Paper Skies", "Nova Lane", auto_save=False)
        self.assertEqual(event["action"], "skip")
        self.assertEqual(event["weight"], -1.0)
        self.assertIn("S002", self.store.get_user_skipped_song_ids("U_TEST"))

    def test_action_counts(self):
        self.store.record("U_TEST2", "S001", "like", auto_save=False)
        self.store.record("U_TEST2", "S002", "skip", auto_save=False)
        self.store.record("U_TEST2", "S003", "replay", auto_save=False)
        counts = self.store.get_action_counts("U_TEST2")
        self.assertEqual(counts["like"], 1)
        self.assertEqual(counts["skip"], 1)
        self.assertEqual(counts["replay"], 1)

    def test_save_to_csv_persistence(self):
        import tempfile
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self.store.record("U_PERSIST", "S001", "like", auto_save=False)
            self.store.save_to_csv(tmp_path)
            loaded = pd.read_csv(tmp_path)
            self.assertIn("U_PERSIST", loaded["user_id"].values)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_mood_inference(self):
        self.assertEqual(infer_mood("Sad Tears In The Rain", "Pop", "Artist"), "Melancholic")
        self.assertEqual(infer_mood("Party Night Bass", "Electronic", "Artist"), "Energetic")
        self.assertEqual(infer_mood("Sunday Morning Coffee", "Acoustic", "Artist"), "Chill")

    def test_api_service_formatting(self):
        api = MusicAPIService()
        mock_item = {
            "trackId": 12345,
            "trackName": "Test Track",
            "artistName": "Test Artist",
            "primaryGenreName": "Pop",
            "artworkUrl100": "http://example.com/100x100bb.jpg",
            "previewUrl": "http://example.com/preview.m4a",
            "country": "USA",
        }
        formatted = api._format_track(mock_item)
        self.assertEqual(formatted["song_id"], "API_12345")
        self.assertEqual(formatted["song_title"], "Test Track")
        self.assertIn("600x600bb.jpg", formatted["artwork_url"])


if __name__ == "__main__":
    unittest.main()

