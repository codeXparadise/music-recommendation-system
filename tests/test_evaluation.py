"""Unit tests for ModelEvaluator including Precision, Recall, NDCG, MAP, and Hit Rate."""

import unittest
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from music_recommender.data_loader import DataLoader
from music_recommender.evaluation import (
    ModelEvaluator,
    compute_ndcg_at_k,
    compute_map_at_k,
    compute_hit_rate_at_k,
)


class TestEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        loader = DataLoader()
        cls.songs, cls.interactions = loader.load_data()
        cls.evaluator = ModelEvaluator(cls.songs, cls.interactions)

    def test_ranking_metric_helpers(self):
        # Case 1: First item is relevant
        rec1 = ["S1", "S2", "S3"]
        rel1 = {"S1"}
        self.assertEqual(compute_hit_rate_at_k(rec1, rel1, k=3), 1.0)
        self.assertEqual(compute_map_at_k(rec1, rel1, k=3), 1.0)
        self.assertEqual(compute_ndcg_at_k(rec1, rel1, k=3), 1.0)

        # Case 2: Second item is relevant (should have lower NDCG than rank 1)
        rec2 = ["S2", "S1", "S3"]
        self.assertEqual(compute_hit_rate_at_k(rec2, rel1, k=3), 1.0)
        self.assertLess(compute_ndcg_at_k(rec2, rel1, k=3), 1.0)
        self.assertLess(compute_map_at_k(rec2, rel1, k=3), 1.0)

        # Case 3: No relevant items in top-k
        rec3 = ["S4", "S5", "S6"]
        self.assertEqual(compute_hit_rate_at_k(rec3, rel1, k=3), 0.0)
        self.assertEqual(compute_map_at_k(rec3, rel1, k=3), 0.0)
        self.assertEqual(compute_ndcg_at_k(rec3, rel1, k=3), 0.0)

    def test_evaluate_returns_metrics(self):
        metrics = self.evaluator.evaluate(k=5)
        self.assertFalse(metrics.empty)
        expected_cols = ["model", "precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", "hit_rate_at_k"]
        for col in expected_cols:
            self.assertIn(col, metrics.columns)

    def test_summary_aggregates(self):
        metrics = self.evaluator.evaluate(k=5)
        summary = self.evaluator.get_summary(metrics, k=5)
        self.assertFalse(summary.empty)
        models = summary["model"].tolist()
        self.assertIn("Content-based", models)
        self.assertIn("Collaborative", models)
        for col in ["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", "hit_rate_at_k"]:
            self.assertTrue((summary[col] >= 0).all() and (summary[col] <= 1).all())


if __name__ == "__main__":
    unittest.main()
