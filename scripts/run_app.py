#!/usr/bin/env python
"""Script to launch the Streamlit web application with command-line arguments.

Usage examples:
    python scripts/run_app.py
    python scripts/run_app.py --port 8502
    python scripts/run_app.py --port 8501 --host 0.0.0.0 --headless
"""

import sys
from pathlib import Path

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from music_recommender.cli import build_parser, handle_web


def main():
    parser = build_parser()
    web_parser = [
        action for action in parser._actions if action.dest == "command"
    ][0].choices["web"]
    
    args = web_parser.parse_args()
    return handle_web(args)


if __name__ == "__main__":
    sys.exit(main())

