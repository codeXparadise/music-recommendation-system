"""Configuration settings and constants for TuneMatch Music Recommendation System."""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Default data file paths
DEFAULT_SONGS_PATH = DATA_DIR / "songs.csv"
DEFAULT_INTERACTIONS_PATH = DATA_DIR / "interactions.csv"

# Implicit feedback weights
# Used to convert actions (like, replay, complete, partial, skip) into numeric interaction scores
DEFAULT_ACTION_WEIGHTS = {
    "like": 3.0,
    "replay": 2.5,
    "complete": 2.0,
    "partial": 0.5,
    "skip": -1.0,
}

# Recommendation settings
DEFAULT_TOP_K = 5
DEFAULT_METHOD = "hybrid"

