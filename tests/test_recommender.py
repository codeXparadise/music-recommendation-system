"""Unit tests for Recommender models (Content-based, Collaborative, Hybrid)."""

import unittest
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from music_recommender.data_loader import DataLoader
from music_recommender.models import ContentRecommender, CollaborativeRecommender, HybridRecommender


class TestRecommenders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        loader = DataLoader()
        cls.songs, cls.interactions = loader.load_data()
        cls.content_model = ContentRecommender(cls.songs)
        cls.collab_model = CollaborativeRecommender(cls.songs, cls.interactions)
        cls.hybrid_model = HybridRecommender(cls.songs, cls.interactions)

    def test_content_recommender_returns_recommendations(self):
        recs = self.content_model.recommend(seed_song_ids=["S001"], top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertNotIn("S001", recs["song_id"].tolist())
        self.assertIn("score", recs.columns)

    def test_content_recommender_genre_filter(self):
        recs = self.content_model.recommend(seed_song_ids=["S001"], top_n=3, genre="Bollywood")
        self.assertTrue(all(g.lower() == "bollywood" for g in recs["genre"]))

    def test_collaborative_recommender_known_user(self):
        recs = self.collab_model.recommend(user_id="U001", top_n=5)
        self.assertFalse(recs.empty)
        seen = set(self.interactions.loc[self.interactions["user_id"] == "U001", "song_id"])
        self.assertTrue(set(recs["song_id"]).isdisjoint(seen))

    def test_hybrid_recommender_known_user(self):
        recs = self.hybrid_model.recommend(user_id="U001", top_n=5, method="hybrid")
        self.assertLessEqual(len(recs), 5)
        self.assertFalse(recs.empty)

    def test_cold_start_recommendations(self):
        recs = self.hybrid_model.cold_start_recommend(chosen_genres=["Bollywood"], top_n=4)
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(g.lower() == "bollywood" for g in recs["genre"]))

        recs_pop = self.hybrid_model.cold_start_recommend(top_n=5)
        self.assertEqual(len(recs_pop), 5)


if __name__ == "__main__":
    unittest.main()

