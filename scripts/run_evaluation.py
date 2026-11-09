#!/usr/bin/env python
"""Script to evaluate recommendation models with command-line arguments.

Usage examples:
    python scripts/run_evaluation.py --k 5
    python scripts/run_evaluation.py --k 10 --detailed
    python scripts/run_evaluation.py --k 5 --output evaluation_results.csv
"""

import sys
from pathlib import Path

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from music_recommender.cli import build_parser, handle_evaluate


def main():
    parser = build_parser()
    eval_parser = [
        action for action in parser._actions if action.dest == "command"
    ][0].choices["evaluate"]
    
    args = eval_parser.parse_args()
    return handle_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())

