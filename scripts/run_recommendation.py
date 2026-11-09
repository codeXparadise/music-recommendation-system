#!/usr/bin/env python
"""Script to run music recommendations with command-line arguments.

Usage examples:
    python scripts/run_recommendation.py --user U001 --method hybrid --top-k 5
    python scripts/run_recommendation.py --user U002 --method collaborative --top-k 5
    python scripts/run_recommendation.py --user U003 --method content --genre Indie --top-k 3
    python scripts/run_recommendation.py --genre Bollywood --artist "Arjun Mehta" --top-k 5
"""

import sys
from pathlib import Path

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from music_recommender.cli import build_parser, handle_recommend


def main():
    parser = build_parser()
    # Find the recommend subparser and reuse its arguments
    rec_parser = [
        action for action in parser._actions if action.dest == "command"
    ][0].choices["recommend"]
    
    args = rec_parser.parse_args()
    return handle_recommend(args)


if __name__ == "__main__":
    sys.exit(main())

