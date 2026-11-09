#!/usr/bin/env python
"""TuneMatch Music Recommendation System - Main Entry Point.

Run commands with arguments:
    python main.py recommend --user U001 --method hybrid --top-k 5
    python main.py recommend --genre Pop --artist "Nova Lane" --top-k 3
    python main.py evaluate --k 5
    python main.py evaluate --k 5 --detailed
    python main.py web --port 8501
"""

import sys
from pathlib import Path

# Add src to sys.path so music_recommender is importable without pip install -e .
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from music_recommender.cli import main

if __name__ == "__main__":
    sys.exit(main())

