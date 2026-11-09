"""Unit tests for DataLoader."""

import unittest
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from music_recommender.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader()
        self.songs, self.interactions = self.loader.load_data()

    def test_load_data_not_empty(self):
        self.assertFalse(self.songs.empty)
        self.assertFalse(self.interactions.empty)

    def test_id_types_are_strings(self):
        self.assertTrue(all(isinstance(x, str) for x in self.songs["song_id"]))
        self.assertTrue(all(isinstance(x, str) for x in self.interactions["song_id"]))
        self.assertTrue(all(isinstance(x, str) for x in self.interactions["user_id"]))

    def test_get_user_ids(self):
        users = self.loader.get_user_ids()
        self.assertIsInstance(users, list)
        self.assertIn("U001", users)

    def test_get_genres(self):
        genres = self.loader.get_genres()
        self.assertIsInstance(genres, list)
        self.assertIn("Pop", genres)

    def test_get_artists(self):
        artists = self.loader.get_artists()
        self.assertIsInstance(artists, list)
        self.assertIn("Dua Lipa", artists)


if __name__ == "__main__":
    unittest.main()

