"""TuneMatch Music Recommendation System package."""

from .data_loader import DataLoader
from .models.content_based import ContentRecommender
from .models.collaborative import CollaborativeRecommender
from .models.hybrid import HybridRecommender
from .evaluation import ModelEvaluator
from .interaction_store import InteractionStore
from .services.music_api import MusicAPIService

__version__ = "1.1.0"

__all__ = [
    "DataLoader",
    "ContentRecommender",
    "CollaborativeRecommender",
    "HybridRecommender",
    "ModelEvaluator",
    "InteractionStore",
    "MusicAPIService",
    "__version__",
]
