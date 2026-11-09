"""TuneMatch - Spotify-style Real-Time Music Streaming & Recommendation System.

Features:
- Light Mode (Default) & Dark Mode toggle.
- Real-Time Live Session Stats tracking Likes, Skips, Replays, and Listens.
- Spotify Playlist view for the Music Catalog Explorer with real listener data.
- Live Real-Time Recommendation Engine powered by Apple Music API.
- Dynamic Action Weights Tuner (Like: +3.0, Skip: -1.0, etc.) re-scoring models on the fly.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from music_recommender.config import DEFAULT_SONGS_PATH, DEFAULT_INTERACTIONS_PATH
from music_recommender.data_loader import DataLoader
from music_recommender.models import HybridRecommender
from music_recommender.evaluation import ModelEvaluator
from music_recommender.interaction_store import InteractionStore
from music_recommender.services.music_api import MusicAPIService


# ==============================================================================
# CLI Argument Handling (For direct terminal usage)
# ==============================================================================

def parse_cli_args():
    parser = argparse.ArgumentParser(description="TuneMatch Music CLI", add_help=False)
    parser.add_argument("--user", "-u", type=str, default=None)
    parser.add_argument("--method", "-m", type=str, choices=["hybrid", "content", "collaborative"], default="hybrid")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--genre", "-g", type=str, default=None)
    parser.add_argument("--artist", "-a", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--help-cli", action="store_true")
    args, _ = parser.parse_known_args()
    return args


is_streamlit = False
try:
    if hasattr(st, "runtime") and st.runtime.exists():
        is_streamlit = True
except Exception:
    is_streamlit = False


def run_cli_mode(args):
    from music_recommender.cli import handle_recommend, handle_evaluate

    if args.help_cli or ("--help" in sys.argv or "-h" in sys.argv):
        print("""
TuneMatch Spotify-grade CLI Runner:
  python app.py --user U001 --method hybrid --top-k 5
  python app.py --genre Bollywood --top-k 3
  python app.py --eval --top-k 5

To launch the Spotify WebUI:
  streamlit run app.py
""")
        return 0

    if args.eval:
        eval_args = argparse.Namespace(
            songs_path=str(DEFAULT_SONGS_PATH),
            interactions_path=str(DEFAULT_INTERACTIONS_PATH),
            k=args.top_k,
            detailed=False,
            output=None,
        )
        return handle_evaluate(eval_args)
    else:
        rec_args = argparse.Namespace(
            songs_path=str(DEFAULT_SONGS_PATH),
            interactions_path=str(DEFAULT_INTERACTIONS_PATH),
            user=args.user,
            method=args.method,
            top_k=args.top_k,
            genre=args.genre,
            artist=args.artist,
            format="table",
            output=None,
        )
        return handle_recommend(rec_args)


cli_args = parse_cli_args()

if not is_streamlit:
    sys.exit(run_cli_mode(cli_args))


# ==============================================================================
# Streamlit Spotify Web Application
# ==============================================================================

st.set_page_config(
    page_title="TuneMatch | Spotify Experience",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# Session State & Services Initialization (Persisted across reruns)
# ==============================================================================

@st.cache_resource
def get_api_service():
    return MusicAPIService()

api_service = get_api_service()

if "initialized" not in st.session_state:
    loader = DataLoader()
    songs_df, interactions_df = loader.load_data()
    st.session_state["catalog_df"] = songs_df
    st.session_state["interaction_store"] = InteractionStore(interactions_df)
    st.session_state["active_user"] = "U001"
    st.session_state["search_results"] = []
    st.session_state["toast_msg"] = None
    st.session_state["session_stats"] = {
        "like": 0,
        "skip": 0,
        "replay": 0,
        "complete": 0,
    }
    if not songs_df.empty:
        st.session_state["now_playing"] = songs_df.iloc[0].to_dict()
    else:
        st.session_state["now_playing"] = None
    st.session_state["initialized"] = True

if "session_stats" not in st.session_state:
    st.session_state["session_stats"] = {"like": 0, "skip": 0, "replay": 0, "complete": 0}

store: InteractionStore = st.session_state["interaction_store"]
catalog: pd.DataFrame = st.session_state["catalog_df"]


# ==============================================================================
# Sidebar - Theme Toggle & Settings
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg' width='38'>
        <span style='font-size:1.5rem; font-weight:800; color:#1DB954;'>TuneMatch</span>
    </div>
    """, unsafe_allow_html=True)

    # Theme Toggle Button (Default is Light Mode)
    dark_mode = st.toggle("🌙 Dark Mode", value=False, help="Toggle between Light Mode (Default) and Spotify Dark Theme")

    st.markdown("---")
    st.markdown("### 👤 Listener Account")
    known_users = sorted(list(set(store.get_dataframe()["user_id"].unique())))
    user_options = ["New User (Cold Start)"] + known_users

    cur_idx = 0
    if st.session_state["active_user"] in known_users:
        cur_idx = known_users.index(st.session_state["active_user"]) + 1

    sel_user = st.selectbox(
        "Active Listener Profile",
        user_options,
        index=cur_idx,
    )
    if sel_user == "New User (Cold Start)":
        st.session_state["active_user"] = "New User"
    else:
        st.session_state["active_user"] = sel_user

    st.markdown("---")
    st.markdown("### 🎛️ AI Recommendation Settings")
    rec_mode = st.radio(
        "Algorithm Mode",
        ["Hybrid (Smart Mix)", "Content-Based (TF-IDF)", "Collaborative (Item CF)"],
        index=0,
    )
    mode_key = "hybrid" if "Hybrid" in rec_mode else ("content" if "Content" in rec_mode else "collaborative")

    top_k = st.slider("Queue Size (Top-K)", min_value=3, max_value=12, value=6)

    genres_avail = sorted(st.session_state["catalog_df"]["genre"].dropna().unique().tolist())
    selected_genres = st.multiselect("Filter by Genre", genres_avail, default=[])

    with st.expander("⚖️ Real-Time Interaction Weights", expanded=False):
        st.caption("Adjust feedback weights in real time to influence AI recommendation scoring:")
        w_like = st.slider("Like Weight 💚", min_value=1.0, max_value=5.0, value=float(store.action_weights.get("like", 3.0)), step=0.5)
        w_replay = st.slider("Replay Weight 🔁", min_value=1.0, max_value=5.0, value=float(store.action_weights.get("replay", 2.5)), step=0.5)
        w_complete = st.slider("Complete Listen ▶️", min_value=0.5, max_value=4.0, value=float(store.action_weights.get("complete", 2.0)), step=0.5)
        w_partial = st.slider("Partial Listen ⏳", min_value=0.0, max_value=2.0, value=float(store.action_weights.get("partial", 0.5)), step=0.25)
        w_skip = st.slider("Skip Penalty ⏭️", min_value=-5.0, max_value=0.0, value=float(store.action_weights.get("skip", -1.0)), step=0.5)

        active_weights = {
            "like": w_like,
            "replay": w_replay,
            "complete": w_complete,
            "partial": w_partial,
            "skip": w_skip,
        }
        store.update_action_weights(active_weights)

    st.markdown("---")
    st.markdown("### 📊 Live Session Stats")
    active_user = st.session_state["active_user"]
    user_counts = store.get_action_counts(active_user)
    total_actions = sum(user_counts.values())

    c_s1, c_s2 = st.columns(2)
    c_s1.metric("Liked 💚", user_counts.get("like", 0))
    c_s2.metric("Skipped ⏭️", user_counts.get("skip", 0))
    c_s3, c_s4 = st.columns(2)
    c_s3.metric("Replayed 🔁", user_counts.get("replay", 0))
    c_s4.metric("Listened ▶️", user_counts.get("complete", 0))

    st.caption(f"Fetched & Auto-Synced with `interactions.csv`: **{total_actions}** events for **{active_user}**")

    if st.button("🔄 Reset Listener Taste"):
        store.clear_user_history(active_user, auto_save=True)
        st.success(f"Tastes cleared and updated in interactions.csv for {active_user}!")
        st.rerun()


# ==============================================================================
# Theme CSS Generator (Dynamic Light vs Dark Mode)
# ==============================================================================

if dark_mode:
    # Spotify Dark Theme
    st.markdown("""
    <style>
        .stApp {
            background-color: #121212 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid #222222 !important;
        }
        .spotify-hero {
            background: linear-gradient(180deg, #1e3a24 0%, #181818 100%);
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            border: 1px solid #282828;
        }
        .spotify-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
        }
        .spotify-subtitle {
            color: #b3b3b3;
            font-size: 1rem;
            margin-top: 4px;
        }
        .spotify-card {
            background-color: #181818;
            border-radius: 8px;
            padding: 16px;
            transition: background-color 0.25s ease, transform 0.2s ease;
            border: 1px solid #222222;
            margin-bottom: 16px;
        }
        .spotify-card:hover {
            background-color: #242424;
            transform: translateY(-4px);
        }
        .card-title {
            font-weight: 700;
            font-size: 1rem;
            color: #ffffff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 2px;
        }
        .card-artist {
            font-size: 0.85rem;
            color: #b3b3b3;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 8px;
        }
        .card-badge {
            display: inline-block;
            padding: 2px 8px;
            font-size: 0.72rem;
            font-weight: 600;
            border-radius: 12px;
            background: #2a2a2a;
            color: #b3b3b3;
            margin-right: 4px;
        }
        .now-playing-box {
            background-color: #181818;
            border: 1.5px solid #1db954;
            border-radius: 12px;
            padding: 18px 24px;
            margin-bottom: 28px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.7), 0 0 15px rgba(29, 185, 84, 0.2);
        }
        div.stButton > button {
            background-color: #242424;
            color: #ffffff;
            border: 1px solid #383838;
            border-radius: 20px;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #383838;
            color: #1DB954;
            border-color: #1DB954;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Modern Clean Light Theme (Default)
    st.markdown("""
    <style>
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        .spotify-hero {
            background: linear-gradient(180deg, #E8F5E9 0%, #FFFFFF 100%);
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        }
        .spotify-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0F172A;
            margin: 0;
        }
        .spotify-subtitle {
            color: #475569;
            font-size: 1rem;
            margin-top: 4px;
        }
        .spotify-card {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 16px;
            transition: background-color 0.25s ease, transform 0.2s ease, box-shadow 0.2s ease;
            border: 1px solid #E2E8F0;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .spotify-card:hover {
            background-color: #F8FAFC;
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        }
        .card-title {
            font-weight: 700;
            font-size: 1rem;
            color: #0F172A;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 2px;
        }
        .card-artist {
            font-size: 0.85rem;
            color: #475569;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 8px;
        }
        .card-badge {
            display: inline-block;
            padding: 2px 8px;
            font-size: 0.72rem;
            font-weight: 600;
            border-radius: 12px;
            background: #F1F5F9;
            color: #334155;
            margin-right: 4px;
            border: 1px solid #E2E8F0;
        }
        .now-playing-box {
            background-color: #FFFFFF;
            border: 2px solid #1DB954;
            border-radius: 12px;
            padding: 18px 24px;
            margin-bottom: 28px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06), 0 0 12px rgba(29, 185, 84, 0.15);
        }
        div.stButton > button {
            background-color: #FFFFFF;
            color: #0F172A;
            border: 1px solid #CBD5E1;
            border-radius: 20px;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #F1F5F9;
            color: #1DB954;
            border-color: #1DB954;
        }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# Interaction Recording & Autoplay Callbacks
# ==============================================================================

def record_action(song_dict: dict, action: str):
    """Log an interaction event, update live session counters, and trigger real-time updates."""
    user = st.session_state["active_user"]
    sid = str(song_dict.get("song_id", ""))
    title = str(song_dict.get("song_title", "Track"))
    artist = str(song_dict.get("artist", ""))

    act_key = action.lower()
    # Update live session counter
    if act_key in st.session_state["session_stats"]:
        st.session_state["session_stats"][act_key] += 1

    event = store.record(
        user_id=user,
        song_id=sid,
        action=action,
        song_title=title,
        artist=artist,
    )
    icons = {
        "like": "💚 Liked",
        "skip": "⏭️ Skipped",
        "replay": "🔁 Replayed",
        "complete": "▶️ Completed",
        "partial": "⏳ Partial",
    }
    sign = "+" if event["weight"] > 0 else ""
    st.session_state["toast_msg"] = f"{icons.get(action, action)} '{title}' ({sign}{event['weight']}) — AI Model Re-scored!"


def play_track(song_dict: dict):
    """Set song as now playing and record a play action."""
    st.session_state["now_playing"] = song_dict
    record_action(song_dict, "complete")


def skip_to_next_track():
    """Skip the current track, record skip penalty, and auto-advance to next recommendation."""
    current = st.session_state.get("now_playing")
    if current:
        record_action(current, "skip")

    recommender = HybridRecommender(
        songs_df=st.session_state["catalog_df"],
        interactions_df=store.get_dataframe(),
        action_weights=store.action_weights,
    )
    skipped_ids = store.get_user_skipped_song_ids(st.session_state["active_user"])
    recs = recommender.recommend(user_id=st.session_state["active_user"], top_n=6)
    recs = recs[~recs["song_id"].isin(skipped_ids)]

    if not recs.empty:
        next_track = recs.iloc[0].to_dict()
        st.session_state["now_playing"] = next_track
    st.rerun()


if st.session_state.get("toast_msg"):
    st.toast(st.session_state["toast_msg"])
    st.session_state["toast_msg"] = None


# ==============================================================================
# Persistent Spotify "Now Playing" Bar
# ==============================================================================

np_track = st.session_state.get("now_playing")

if np_track:
    np_title = np_track.get("song_title", "Unknown Track")
    np_artist = np_track.get("artist", "Unknown Artist")
    np_art = np_track.get("artwork_url", "")
    np_preview = np_track.get("preview_url", "")
    np_genre = np_track.get("genre", "Music")

    title_color = "#FFFFFF" if dark_mode else "#0F172A"
    sub_color = "#888888" if dark_mode else "#475569"

    st.markdown("<div class='now-playing-box'>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns([1.2, 4.5, 3])

    with p_col1:
        if np_art and str(np_art).startswith("http"):
            st.image(np_art, width=80)
        else:
            st.markdown("<div style='height:80px; width:80px; background:#e2e8f0; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:32px;'>🎵</div>", unsafe_allow_html=True)

    with p_col2:
        st.markdown(f"<span style='color:#1DB954; font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;'>● NOW PLAYING ON SPOTIFY</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.3rem; font-weight:800; color:{title_color};'>{np_title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#1DB954; font-weight:600; font-size:0.95rem; margin-bottom:8px;'>{np_artist} &nbsp;•&nbsp; <span style='color:{sub_color};'>{np_genre}</span></div>", unsafe_allow_html=True)

        if np_preview and str(np_preview).startswith("http"):
            st.audio(np_preview, format="audio/mp4", autoplay=True)
        else:
            st.caption("*(Preview audio buffer unavailable for this track)*")

    with p_col3:
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns(3)
        with ctrl_c1:
            if st.button("💚 Like", key="np_like", help="Thumbs up! (+3.0 implicit weight)"):
                record_action(np_track, "like")
                st.rerun()
        with ctrl_c2:
            if st.button("🔁 Replay", key="np_replay", help="Listen again (+2.5 implicit weight)"):
                record_action(np_track, "replay")
                st.rerun()
        with ctrl_c3:
            if st.button("⏭️ Next / Skip", key="np_skip", help="Skip track (-1.0 penalty & auto-advance)"):
                skip_to_next_track()

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# Main Spotify Tabs
# ==============================================================================

tab_discover, tab_search, tab_feed, tab_eval, tab_library = st.tabs([
    "🟢 Made For You",
    "🔍 Search & Browse",
    "📜 History & Skips",
    "📊 Recommender Benchmark",
    "📚 Full Library & Playlist",
])


# ------------------------------------------------------------------------------
# TAB 1: Made For You (Real-Time Recommendations Feed)
# ------------------------------------------------------------------------------
with tab_discover:
    st.markdown("""
    <div class='spotify-hero'>
        <div class='spotify-title'>Discover Weekly</div>
        <div class='spotify-subtitle'>Your real-time personalized playlist generated with machine learning based on what you like and skip.</div>
    </div>
    """, unsafe_allow_html=True)

    cur_interactions = store.get_dataframe()
    cur_catalog = st.session_state["catalog_df"]

    recommender = HybridRecommender(
        songs_df=cur_catalog,
        interactions_df=cur_interactions,
        action_weights=store.action_weights,
    )

    is_cold = (st.session_state["active_user"] == "New User") or (
        st.session_state["active_user"] not in cur_interactions["user_id"].values
    )

    if is_cold:
        st.info("💡 **Welcome to Spotify TuneMatch!** Hit **▶ Play** on any track to start listening, or **💚 Like** / **⏭️ Skip** to teach the algorithm your music taste in real time!")
        recommendations = recommender.cold_start_recommend(
            chosen_genres=selected_genres,
            top_n=top_k,
        )
    else:
        skipped_ids = store.get_user_skipped_song_ids(st.session_state["active_user"])
        recommendations = recommender.recommend(
            user_id=st.session_state["active_user"],
            top_n=top_k,
            genres=selected_genres,
            method=mode_key,
        )
        if not recommendations.empty and skipped_ids:
            recommendations = recommendations[~recommendations["song_id"].isin(skipped_ids)]

    # Filter to ensure 100% of recommendations have verified live preview audio from API
    if not recommendations.empty and "preview_url" in recommendations.columns:
        recommendations = recommendations[recommendations["preview_url"].fillna("").str.startswith("http")]

    # If more playable songs needed, fetch fresh live tracks directly from Apple Music API matching taste
    if len(recommendations) < top_k:
        liked_seeds = store.get_user_seed_song_ids(st.session_state["active_user"])
        liked_artists = cur_catalog.loc[cur_catalog["song_id"].isin(liked_seeds), "artist"].dropna().unique().tolist()
        fresh_api_tracks = api_service.discover_live_recommendations(
            liked_artists=liked_artists,
            liked_genres=selected_genres,
            limit=(top_k - len(recommendations) + 3),
        )
        if fresh_api_tracks:
            fresh_df = api_service.convert_to_dataframe(fresh_api_tracks)
            st.session_state["catalog_df"] = pd.concat([st.session_state["catalog_df"].drop(columns=["score"], errors="ignore"), fresh_df.drop(columns=["score"], errors="ignore")], ignore_index=True).drop_duplicates("song_id")
            fresh_df["score"] = 4.50
            recommendations = pd.concat([recommendations, fresh_df], ignore_index=True).drop_duplicates("song_id").head(top_k)

    c_btn1, c_btn2 = st.columns([2, 5])
    with c_btn1:
        if st.button("⚡ Discover Fresh API Music", help="Fetch brand-new playable songs from Apple Music API matching your taste"):
            liked_seeds = store.get_user_seed_song_ids(st.session_state["active_user"])
            liked_artists = cur_catalog.loc[cur_catalog["song_id"].isin(liked_seeds), "artist"].dropna().unique().tolist()
            api_tracks = api_service.discover_live_recommendations(liked_artists=liked_artists, liked_genres=selected_genres, limit=top_k)
            if api_tracks:
                api_df = api_service.convert_to_dataframe(api_tracks)
                st.session_state["catalog_df"] = pd.concat([st.session_state["catalog_df"].drop(columns=["score"], errors="ignore"), api_df.drop(columns=["score"], errors="ignore")], ignore_index=True).drop_duplicates("song_id")
                api_df["score"] = 4.80
                recommendations = api_df.head(top_k)
                st.success("Fetched fresh live tracks from Apple Music API!")
                st.rerun()

    if recommendations.empty:
        st.warning("No recommendations available for this criteria. Try clearing genre filters or searching for new tracks!")
    else:
        cols_per_row = 3
        rows = [recommendations.iloc[i:i + cols_per_row] for i in range(0, len(recommendations), cols_per_row)]

        for r_idx, row_batch in enumerate(rows):
            cols = st.columns(cols_per_row)
            for c_idx, (_, song) in enumerate(row_batch.iterrows()):
                with cols[c_idx]:
                    sid = str(song["song_id"])
                    title = str(song.get("song_title", "Untitled"))
                    artist = str(song.get("artist", "Artist"))
                    genre = str(song.get("genre", "Music"))
                    mood = str(song.get("mood", "Chill"))
                    art_url = song.get("artwork_url", "")
                    preview_url = song.get("preview_url", "")
                    score = song.get("score", None)

                    st.markdown("<div class='spotify-card'>", unsafe_allow_html=True)
                    if art_url and str(art_url).startswith("http"):
                        st.image(art_url, width=280)
                    else:
                        st.markdown("<div style='height:180px; background:#e2e8f0; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:48px;'>🎵</div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='card-title' title='{title}'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card-artist' title='{artist}'>{artist}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span class='card-badge'>{genre}</span><span class='card-badge'>{mood}</span>", unsafe_allow_html=True)
                    
                    if score is not None:
                        st.markdown(f"<div style='font-size:0.75rem; color:#1DB954; font-weight:700; margin:4px 0;'>Match: {float(score):.2f} pts</div>", unsafe_allow_html=True)

                    if preview_url and str(preview_url).startswith("http"):
                        st.audio(preview_url, format="audio/mp4")

                    c_play, c_like, c_skip = st.columns([1.5, 1, 1])
                    with c_play:
                        if st.button("▶ Play", key=f"rec_play_{sid}_{r_idx}_{c_idx}", help="Stream in main player"):
                            play_track(song.to_dict())
                            st.rerun()
                    with c_like:
                        if st.button("💚", key=f"rec_like_{sid}_{r_idx}_{c_idx}", help="Like track (+3.0)"):
                            record_action(song.to_dict(), "like")
                            st.rerun()
                    with c_skip:
                        if st.button("⏭️", key=f"rec_skip_{sid}_{r_idx}_{c_idx}", help="Skip track (-1.0)"):
                            record_action(song.to_dict(), "skip")
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: Search & Browse (Live Apple Music API Integration)
# ------------------------------------------------------------------------------
with tab_search:
    st.subheader("🔍 Search Any Song, Artist, or Genre")
    st.caption("Access millions of real songs through the Apple Music API. Every result is streamable with real 30s previews!")

    st.markdown("**Popular Music Categories:**")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    quick_search = None
    if g1.button("🔥 Global Pop"):
        quick_search = "Dua Lipa Harry Styles Pop Hits"
    if g2.button("🎵 Bollywood"):
        quick_search = "Arijit Singh Pritam Bollywood Hits"
    if g3.button("⚡ Rock / Alt"):
        quick_search = "Coldplay Imagine Dragons Rock"
    if g4.button("🎧 EDM / Dance"):
        quick_search = "Martin Garrix Avicii EDM"
    if g5.button("🎤 Hip-Hop"):
        quick_search = "Drake Post Malone Hip-Hop"
    if g6.button("☕ Lo-Fi Beats"):
        quick_search = "Lofi Chill Beats Relax"

    s_box, s_btn = st.columns([4.5, 1])
    with s_box:
        search_query = st.text_input(
            "Search Spotify Library:",
            value=quick_search or "",
            placeholder="Type a song title, artist, or album...",
        )
    with s_btn:
        st.write("")
        st.write("")
        do_search = st.button("🔎 Search", use_container_width=True)

    active_q = quick_search if quick_search else (search_query if do_search else None)

    if active_q:
        with st.spinner(f"Fetching real tracks for '{active_q}'..."):
            found_tracks = api_service.search_tracks(active_q, limit=9)
            st.session_state["search_results"] = found_tracks

            if found_tracks:
                new_tracks_df = api_service.convert_to_dataframe(found_tracks)
                updated_cat = pd.concat([st.session_state["catalog_df"], new_tracks_df], ignore_index=True)
                st.session_state["catalog_df"] = updated_cat.drop_duplicates("song_id")

    search_items = st.session_state.get("search_results", [])
    if search_items:
        st.success(f"Found {len(search_items)} real songs. Click '▶ Play' to stream instantly and personalize your AI recommendations!")
        
        cols_per_row = 3
        rows = [search_items[i:i + cols_per_row] for i in range(0, len(search_items), cols_per_row)]
        for r_idx, row_batch in enumerate(rows):
            cols = st.columns(cols_per_row)
            for c_idx, item in enumerate(row_batch):
                with cols[c_idx]:
                    sid = str(item["song_id"])
                    title = item["song_title"]
                    artist = item["artist"]
                    art_url = item["artwork_url"]
                    genre = item["genre"]
                    mood = item["mood"]
                    s_prev = item.get("preview_url", "")

                    st.markdown("<div class='spotify-card'>", unsafe_allow_html=True)
                    if art_url and str(art_url).startswith("http"):
                        st.image(art_url, width=280)
                    else:
                        st.markdown("<div style='height:180px; background:#e2e8f0; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:48px;'>🎵</div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='card-title' title='{title}'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card-artist' title='{artist}'>{artist}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span class='card-badge'>{genre}</span><span class='card-badge'>{mood}</span>", unsafe_allow_html=True)

                    if s_prev and str(s_prev).startswith("http"):
                        st.audio(s_prev, format="audio/mp4")

                    c_play, c_like, c_skip = st.columns([1.5, 1, 1])
                    with c_play:
                        if st.button("▶ Play", key=f"search_play_{sid}_{r_idx}_{c_idx}"):
                            play_track(item)
                            st.rerun()
                    with c_like:
                        if st.button("💚", key=f"search_like_{sid}_{r_idx}_{c_idx}", help="Like track"):
                            record_action(item, "like")
                            st.rerun()
                    with c_skip:
                        if st.button("⏭️", key=f"search_skip_{sid}_{r_idx}_{c_idx}", help="Skip track"):
                            record_action(item, "skip")
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 3: History & Skips
# ------------------------------------------------------------------------------
with tab_feed:
    st.subheader(f"📜 Listener Timeline: {st.session_state['active_user']}")
    st.caption("Live streaming log tracking every Like, Replay, Complete, and Skip event.")

    user_events = store.get_user_events(st.session_state["active_user"])
    counts = store.get_action_counts(st.session_state["active_user"])

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Likes 💚", counts.get("like", 0))
    c_m2.metric("Skips ⏭️", counts.get("skip", 0))
    c_m3.metric("Completes ▶️", counts.get("complete", 0))
    c_m4.metric("Replays 🔁", counts.get("replay", 0))

    if not user_events:
        st.info("No interactions recorded yet. Play a song or click Like/Skip to see events stream here in real time!")
    else:
        c_left, c_right = st.columns([1.5, 2.5])
        with c_left:
            st.markdown("**Action Distribution Chart:**")
            active_counts = {k.capitalize(): v for k, v in counts.items() if v > 0}
            if active_counts:
                chart_bg = "#121212" if dark_mode else "#FFFFFF"
                fig, ax = plt.subplots(figsize=(4, 4), facecolor=chart_bg)
                ax.set_facecolor(chart_bg)
                wedges, texts, autotexts = ax.pie(
                    list(active_counts.values()),
                    labels=list(active_counts.keys()),
                    autopct="%1.1f%%",
                    colors=["#1DB954", "#60a5fa", "#facc15", "#c084fc", "#f87171"],
                    startangle=90,
                )
                t_color = "#FFFFFF" if dark_mode else "#0F172A"
                for t in texts:
                    t.set_color(t_color)
                for at in autotexts:
                    at.set_color("#000000")
                    at.set_weight("bold")
                st.pyplot(fig)
                plt.close(fig)

        with c_right:
            st.markdown("**Live Event Stream:**")
            evt_df = pd.DataFrame(user_events)[["timestamp", "action", "song_title", "artist", "weight", "song_id"]]
            st.dataframe(evt_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.download_button(
                "📥 Export History as CSV",
                evt_df.to_csv(index=False).encode("utf-8"),
                file_name=f"spotify_history_{st.session_state['active_user']}.csv",
                mime="text/csv",
            )
        with d_c2:
            if st.button("💾 Persist History to Disk (`data/interactions.csv`)"):
                save_loc = store.save_to_csv()
                st.success(f"Saved {len(store.get_dataframe())} events to {save_loc}")


# ------------------------------------------------------------------------------
# TAB 4: Model Evaluation (Precision, Recall, NDCG, MAP)
# ------------------------------------------------------------------------------
with tab_eval:
    st.subheader("📊 Recommendation Quality Benchmark")
    st.caption("Holdout evaluation on actual user interactions measuring Precision@K, Recall@K, NDCG@K, and MAP@K.")

    k_eval = st.slider("Cutoff Rank (K)", min_value=3, max_value=10, value=5)

    if st.button("🚀 Calculate Benchmark Metrics"):
        evaluator = ModelEvaluator(st.session_state["catalog_df"], store.get_dataframe())
        with st.spinner("Running holdout cross-validation..."):
            metrics_df = evaluator.evaluate(k=k_eval)
            if metrics_df.empty:
                st.warning("Not enough user interactions to compute benchmark metrics. Interact with more songs to generate evaluation data.")
            else:
                summary = evaluator.get_summary(metrics_df, k=k_eval)
                chart_bg = "#121212" if dark_mode else "#FFFFFF"
                text_col = "#FFFFFF" if dark_mode else "#0F172A"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Precision@{k_eval} vs Recall@{k_eval}**")
                    fig1, ax1 = plt.subplots(facecolor=chart_bg)
                    ax1.set_facecolor(chart_bg)
                    ax1.bar(summary["model"], summary["precision_at_k"], color="#1DB954", label="Precision@K", width=0.35, align="center")
                    ax1.bar(summary["model"], summary["recall_at_k"], color="#60a5fa", label="Recall@K", width=0.35, align="edge")
                    ax1.set_ylim(0, 1)
                    ax1.tick_params(colors=text_col)
                    ax1.legend(facecolor=chart_bg, labelcolor=text_col)
                    st.pyplot(fig1)
                    plt.close(fig1)

                with col2:
                    st.markdown(f"**NDCG@{k_eval} (Ranking Quality)**")
                    fig2, ax2 = plt.subplots(facecolor=chart_bg)
                    ax2.set_facecolor(chart_bg)
                    ax2.bar(summary["model"], summary["ndcg_at_k"], color="#facc15")
                    ax2.set_ylim(0, 1)
                    ax2.tick_params(colors=text_col)
                    ax2.set_ylabel("NDCG", color=text_col)
                    st.pyplot(fig2)
                    plt.close(fig2)

                st.dataframe(summary, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 5: Spotify Playlist View (Music Catalog Explorer with Real User Data)
# ------------------------------------------------------------------------------
with tab_library:
    cat_df = st.session_state["catalog_df"]
    active_user = st.session_state["active_user"]
    user_counts = store.get_action_counts(active_user)
    likes_cnt = user_counts.get("like", 0)
    skips_cnt = user_counts.get("skip", 0)

    # Spotify Playlist Hero Banner
    banner_bg = "linear-gradient(180deg, #1e3a24 0%, #161616 100%)" if dark_mode else "linear-gradient(180deg, #E8F5E9 0%, #FFFFFF 100%)"
    banner_border = "#282828" if dark_mode else "#E2E8F0"
    banner_title_col = "#FFFFFF" if dark_mode else "#0F172A"
    banner_sub_col = "#B3B3B3" if dark_mode else "#475569"

    st.markdown(f"""
    <div style='display:flex; gap:24px; align-items:flex-end; padding:24px; background:{banner_bg}; border-radius:12px; margin-bottom:20px; border:1px solid {banner_border};'>
        <img src='https://cdn-icons-png.flaticon.com/512/3844/3844724.png' style='width:120px; height:120px; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.4);'>
        <div>
            <div style='font-size:0.8rem; font-weight:700; color:#1DB954; text-transform:uppercase; letter-spacing:1px;'>PUBLIC PLAYLIST</div>
            <div style='font-size:2.2rem; font-weight:800; color:{banner_title_col}; margin:4px 0;'>Global Catalog & Live Playlist</div>
            <div style='color:{banner_sub_col}; font-size:0.95rem; margin-bottom:8px;'>Every track in the library enriched with your real-time listener feedback, live AI weights, and playable streams.</div>
            <div style='font-size:0.85rem; color:#888;'>
                <span style='color:{banner_title_col}; font-weight:600;'>Listener: {active_user}</span> &nbsp;•&nbsp; {len(cat_df)} songs &nbsp;•&nbsp; <span style='color:#1DB954; font-weight:700;'>{likes_cnt} Liked</span> &nbsp;•&nbsp; <span style='color:#f87171; font-weight:700;'>{skips_cnt} Skipped</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Playlist Filters
    p_f1, p_f2 = st.columns([1.5, 2.5])
    with p_f1:
        playlist_filter = st.selectbox(
            "Filter Playlist View",
            ["All Tracks", "💚 Liked by You", "⏭️ Skipped by You", "▶️ Listened Tracks", "⚪ Unplayed Tracks"],
            index=0,
            key="playlist_filter_select",
        )
    with p_f2:
        search_lib = st.text_input(
            "Search within playlist:",
            placeholder="Search by track name, artist, or album...",
            key="playlist_search_input",
        )

    filtered_cat = cat_df.copy()

    if playlist_filter == "💚 Liked by You":
        user_liked_ids = store.get_user_seed_song_ids(active_user)
        filtered_cat = filtered_cat[filtered_cat["song_id"].isin(user_liked_ids)]
    elif playlist_filter == "⏭️ Skipped by You":
        user_skipped_ids = store.get_user_skipped_song_ids(active_user)
        filtered_cat = filtered_cat[filtered_cat["song_id"].isin(user_skipped_ids)]
    elif playlist_filter == "▶️ Listened Tracks":
        interacted_ids = store.get_user_interacted_song_ids(active_user)
        filtered_cat = filtered_cat[filtered_cat["song_id"].isin(interacted_ids)]
    elif playlist_filter == "⚪ Unplayed Tracks":
        interacted_ids = store.get_user_interacted_song_ids(active_user)
        filtered_cat = filtered_cat[~filtered_cat["song_id"].isin(interacted_ids)]

    if search_lib.strip():
        s_mask = (
            filtered_cat["song_title"].str.contains(search_lib, case=False, na=False) |
            filtered_cat["artist"].str.contains(search_lib, case=False, na=False) |
            filtered_cat["album"].str.contains(search_lib, case=False, na=False)
        )
        filtered_cat = filtered_cat[s_mask]

    st.caption(f"Showing **{len(filtered_cat)}** tracks in playlist")

    # Spotify Playlist Table Header
    header_border = "#282828" if dark_mode else "#E2E8F0"
    header_col = "#B3B3B3" if dark_mode else "#64748B"

    st.markdown(f"""
    <div style='display:flex; align-items:center; color:{header_col}; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid {header_border}; padding:10px 14px; margin-bottom:12px;'>
        <div style='flex:0.4;'>#</div>
        <div style='flex:3.5;'>TITLE & ARTIST</div>
        <div style='flex:2;'>ALBUM</div>
        <div style='flex:1.8;'>GENRE / MOOD</div>
        <div style='flex:2.2;'>YOUR STATUS & WEIGHT</div>
        <div style='flex:2.1; text-align:right;'>CONTROLS</div>
    </div>
    """, unsafe_allow_html=True)

    if filtered_cat.empty:
        st.info("No tracks found matching your filter.")
    else:
        for idx, (_, song) in enumerate(filtered_cat.reset_index(drop=True).iterrows(), 1):
            sid = str(song["song_id"])
            title = str(song.get("song_title", "Untitled"))
            artist = str(song.get("artist", "Unknown Artist"))
            album = str(song.get("album", "Single"))
            genre = str(song.get("genre", "Music"))
            mood = str(song.get("mood", "Chill"))
            art_url = song.get("artwork_url", "")
            preview_url = song.get("preview_url", "")

            user_status = store.get_song_status_for_user(active_user, sid)
            
            if user_status:
                act = user_status["action"]
                act_weight = user_status["weight"]
                if act == "like":
                    status_badge = f"<span style='color:#1DB954; font-weight:700;'>💚 Liked (+{act_weight:.1f})</span>"
                elif act == "skip":
                    status_badge = f"<span style='color:#f87171; font-weight:700;'>⏭️ Skipped ({act_weight:.1f})</span>"
                elif act == "replay":
                    status_badge = f"<span style='color:#60a5fa; font-weight:700;'>🔁 Replayed (+{act_weight:.1f})</span>"
                elif act == "complete":
                    status_badge = f"<span style='color:#facc15; font-weight:700;'>▶️ Listened (+{act_weight:.1f})</span>"
                else:
                    status_badge = f"<span style='color:#c084fc; font-weight:700;'>⏳ Partial (+{act_weight:.1f})</span>"
            else:
                status_badge = "<span style='color:#666666;'>⚪ Unplayed</span>"

            row_text_col = "#FFFFFF" if dark_mode else "#0F172A"
            row_sub_col = "#B3B3B3" if dark_mode else "#475569"
            divider_col = "#202020" if dark_mode else "#E2E8F0"

            with st.container():
                r_c1, r_c2, r_c3, r_c4, r_c5, r_c6 = st.columns([0.4, 3.5, 2, 1.8, 2.2, 2.1])

                with r_c1:
                    st.markdown(f"<div style='color:#888; font-size:0.9rem; padding-top:14px;'>{idx}</div>", unsafe_allow_html=True)

                with r_c2:
                    t_sub1, t_sub2 = st.columns([1, 4])
                    with t_sub1:
                        if art_url and str(art_url).startswith("http"):
                            st.image(art_url, width=44)
                        else:
                            st.markdown("<div style='height:44px; width:44px; background:#e2e8f0; border-radius:4px; display:flex; align-items:center; justify-content:center;'>🎵</div>", unsafe_allow_html=True)
                    with t_sub2:
                        st.markdown(f"<div style='font-size:0.95rem; font-weight:700; color:{row_text_col}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:0.8rem; color:{row_sub_col}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{artist}</div>", unsafe_allow_html=True)

                with r_c3:
                    st.markdown(f"<div style='font-size:0.85rem; color:{row_sub_col}; padding-top:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{album}</div>", unsafe_allow_html=True)

                with r_c4:
                    st.markdown(f"<div style='padding-top:10px;'><span class='card-badge'>{genre}</span><span class='card-badge'>{mood}</span></div>", unsafe_allow_html=True)

                with r_c5:
                    st.markdown(f"<div style='font-size:0.85rem; padding-top:12px;'>{status_badge}</div>", unsafe_allow_html=True)

                with r_c6:
                    btn_c1, btn_c2, btn_c3 = st.columns([1.2, 1, 1])
                    with btn_c1:
                        if st.button("▶", key=f"pl_play_{sid}_{idx}", help="Stream track"):
                            play_track(song.to_dict())
                            st.rerun()
                    with btn_c2:
                        if st.button("💚", key=f"pl_like_{sid}_{idx}", help="Like track"):
                            record_action(song.to_dict(), "like")
                            st.rerun()
                    with btn_c3:
                        if st.button("⏭️", key=f"pl_skip_{sid}_{idx}", help="Skip track"):
                            record_action(song.to_dict(), "skip")
                            st.rerun()

                st.markdown(f"<hr style='margin:6px 0; border-color:{divider_col};'>", unsafe_allow_html=True)
