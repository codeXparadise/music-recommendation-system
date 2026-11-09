"""Command-line interface (CLI) for TuneMatch Music Recommendation System."""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd

from .config import (
    DEFAULT_SONGS_PATH,
    DEFAULT_INTERACTIONS_PATH,
    DEFAULT_TOP_K,
    DEFAULT_METHOD,
    PROJECT_ROOT,
)
from .data_loader import DataLoader
from .models.hybrid import HybridRecommender
from .evaluation import ModelEvaluator


def format_dataframe(df: pd.DataFrame, fmt: str = "table") -> str:
    """Format dataframe into requested display format."""
    if fmt == "csv":
        return df.to_csv(index=False)
    elif fmt == "json":
        return df.to_json(orient="records", indent=2)
    else:
        # Default ASCII table format
        display_cols = [
            c for c in ["song_id", "song_title", "artist", "genre", "language", "mood", "score", "popularity"]
            if c in df.columns
        ]
        return df[display_cols].to_string(index=False)


def handle_recommend(args: argparse.Namespace) -> int:
    """Handle the 'recommend' CLI subcommand."""
    loader = DataLoader(args.songs_path, args.interactions_path)
    songs, interactions = loader.load_data()
    recommender = HybridRecommender(songs, interactions)

    genres = [args.genre] if args.genre else []
    artists = [args.artist] if args.artist else []

    user_label = args.user if args.user else "Cold Start (New User)"
    print(f"\n=======================================================")
    print(f"  TuneMatch Recommendation Engine")
    print(f"  User:   {user_label}")
    print(f"  Method: {args.method.capitalize()}")
    print(f"  Top-K:  {args.top_k}")
    if args.genre:
        print(f"  Genre:  {args.genre}")
    if args.artist:
        print(f"  Artist: {args.artist}")
    print(f"=======================================================\n")

    recs = recommender.recommend(
        user_id=args.user,
        top_n=args.top_k,
        genres=genres,
        artists=artists,
        method=args.method,
    )

    if recs.empty:
        print("No recommendations found matching the criteria.")
        return 0

    print(format_dataframe(recs, fmt=args.format))

    if args.output:
        out_path = Path(args.output)
        recs.to_csv(out_path, index=False)
        print(f"\nRecommendations successfully saved to: {out_path.resolve()}")

    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    """Handle the 'evaluate' CLI subcommand."""
    loader = DataLoader(args.songs_path, args.interactions_path)
    songs, interactions = loader.load_data()
    evaluator = ModelEvaluator(songs, interactions)

    print(f"\n=======================================================")
    print(f"  TuneMatch Model Evaluation (Precision@{args.k} & Recall@{args.k})")
    print(f"  Dataset: {len(songs)} songs, {len(interactions)} interactions")
    print(f"=======================================================\n")

    detailed = evaluator.evaluate(k=args.k)

    if detailed.empty:
        print("Not enough interaction history per user to compute evaluation metrics.")
        return 0

    if args.detailed:
        print("--- Detailed Per-User Evaluation ---")
        print(detailed.to_string(index=False))
        print()

    summary = evaluator.get_summary(detailed, k=args.k)
    print("--- Model Summary ---")
    print(summary.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        target_df = detailed if args.detailed else summary
        target_df.to_csv(out_path, index=False)
        print(f"\nEvaluation metrics successfully saved to: {out_path.resolve()}")

    return 0


def handle_web(args: argparse.Namespace) -> int:
    """Launch the Streamlit web application."""
    import subprocess

    app_path = PROJECT_ROOT / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])

    print(f"Starting TuneMatch Streamlit Web App on http://{args.host}:{args.port}...")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nWeb app stopped.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for TuneMatch CLI."""
    parser = argparse.ArgumentParser(
        prog="tunematch",
        description="TuneMatch Music Recommendation System - CLI & Web App",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: recommend
    rec_parser = subparsers.add_parser(
        "recommend",
        help="Generate music recommendations for a user or cold-start profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rec_parser.add_argument(
        "-u", "--user",
        type=str,
        default=None,
        help="User ID to generate recommendations for (e.g. U001). Leave unset for cold-start.",
    )
    rec_parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["hybrid", "content", "collaborative"],
        default=DEFAULT_METHOD,
        help="Recommendation algorithm to use.",
    )
    rec_parser.add_argument(
        "-k", "--k", "--top-k",
        dest="top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of recommendations to return.",
    )
    rec_parser.add_argument(
        "-g", "--genre",
        type=str,
        default=None,
        help="Filter or seed by genre (e.g. 'Pop', 'Bollywood', 'Rock').",
    )
    rec_parser.add_argument(
        "-a", "--artist",
        type=str,
        default=None,
        help="Filter or seed by artist (e.g. 'Nova Lane', 'Arjun Mehta').",
    )
    rec_parser.add_argument(
        "-f", "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output display format.",
    )
    rec_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path to save recommendations CSV file.",
    )
    rec_parser.add_argument(
        "--songs-path",
        type=str,
        default=str(DEFAULT_SONGS_PATH),
        help="Path to songs dataset CSV.",
    )
    rec_parser.add_argument(
        "--interactions-path",
        type=str,
        default=str(DEFAULT_INTERACTIONS_PATH),
        help="Path to interactions dataset CSV.",
    )

    # Subcommand: evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate recommendation algorithms (Precision@K, Recall@K)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "-k", "--k", "--top-k",
        dest="k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Cutoff rank K for Precision@K and Recall@K.",
    )
    eval_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show per-user breakdown in addition to model averages.",
    )
    eval_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path to save evaluation results as CSV.",
    )
    eval_parser.add_argument(
        "--songs-path",
        type=str,
        default=str(DEFAULT_SONGS_PATH),
        help="Path to songs dataset CSV.",
    )
    eval_parser.add_argument(
        "--interactions-path",
        type=str,
        default=str(DEFAULT_INTERACTIONS_PATH),
        help="Path to interactions dataset CSV.",
    )

    # Subcommand: web
    web_parser = subparsers.add_parser(
        "web",
        help="Launch the interactive Streamlit dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    web_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8501,
        help="Port to serve the Streamlit app on.",
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Network interface to bind Streamlit server to.",
    )
    web_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless mode (no auto-opening browser).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "recommend":
        return handle_recommend(args)
    elif args.command == "evaluate":
        return handle_evaluate(args)
    elif args.command == "web":
        return handle_web(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
