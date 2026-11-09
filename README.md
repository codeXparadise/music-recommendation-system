# 🎵 TuneMatch — Real-Time Spotify-Style Music Recommendation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/Unit%20Tests-19%2F19%20Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, real-time music recommendation system built with Python, Streamlit, and Scikit-Learn. Inspired by Spotify, **TuneMatch** seamlessly blends Content-Based, Collaborative, and Hybrid recommendation engines with live streaming audio previews from the Apple Music / iTunes API, immediate implicit feedback adaptation, dynamic weight tuning, and dual Light/Dark styling.

---

## 🌟 Key Highlights

- **🎧 Spotify-Style Sticky Top Player**: Persistent audio player bar with instant 30-second playable audio previews, autoplay, volume controls, and next/skip actions.
- **⚡ Real Music Library API**: Direct, zero-config integration with the Apple Music / iTunes Search API — access millions of tracks, high-resolution 600×600 album artwork, and real audio streams.
- **🔄 Instant Dynamic Re-scoring**: The recommendation engine recomputes candidate rankings on every user action without requiring batch re-training.
- **📊 Real-Time Live Session Stats & CSV Persistence**:
  - Live interaction counters (Liked 💚, Skipped ⏭️, Replayed 🔁, Listened ▶️) fetch directly from `data/interactions.csv` for selected accounts (`U001`–`U008`).
  - Every new reaction is automatically written and persisted to disk in real time.
- **⚖️ Live Interaction Weight Sliders**: Fine-tune algorithmic weights on the fly (Like: `+3.0`, Replay: `+2.5`, Listen: `+2.0`, Partial: `+0.5`, Skip penalty: `-1.0`).
- **🗂️ Spotify Public Playlist Explorer**: Replaces raw tables with a rich Spotify-like playlist table displaying user reaction status badges (`💚 Liked`, `⏭️ Skipped`, etc.) and instant 1-click play/react controls.
- **🌓 Light & Dark Theme Support**: Features a modern, accessible **Light Mode by default** with a single-click toggle to a high-contrast **Spotify Dark Mode**.
- **📈 Comprehensive Holdout Evaluation**: Benchmarks ranking quality with Precision@K, Recall@K, NDCG@K, MAP@K, and Hit Rate@K.

---

## 🏗️ Project Architecture (`src`-layout)

The project follows the standard Python packaging folder structure (`src`-layout):

```
music_recommendation_system/
├── src/                                   # Application source packages
│   └── music_recommender/                 # Core modular package
│       ├── __init__.py                    # Public API exports & package version
│       ├── config.py                      # File paths, constants, and action weights
│       ├── data_loader.py                 # CSV ingestion, data validation, and queries
│       ├── evaluation.py                  # Precision@K, Recall@K, NDCG, MAP metrics
│       ├── interaction_store.py           # Real-time event capture, CSV persistence & stats
│       ├── cli.py                         # Unified argparse CLI handler
│       ├── services/                      # External services
│       │   ├── __init__.py
│       │   └── music_api.py               # Apple iTunes API client (search, artwork, previews)
│       └── models/                        # Core recommendation algorithms
│           ├── __init__.py
│           ├── content_based.py           # TF-IDF + Cosine similarity model
│           ├── collaborative.py           # Implicit item-item collaborative filter
│           └── hybrid.py                  # Weighted hybrid engine & cold-start logic
├── scripts/                               # Standalone CLI execution scripts
│   ├── run_recommendation.py              # CLI recommendation runner with arguments
│   ├── run_evaluation.py                  # Standalone model evaluator script
│   └── run_app.py                         # Web application launcher
├── tests/                                 # Automated test suite (unittest)
│   ├── __init__.py
│   ├── test_data_loader.py                # Data loading and schema tests
│   ├── test_recommender.py                # Algorithmic recommendation tests
│   ├── test_evaluation.py                 # Metric evaluation tests
│   └── test_realtime.py                   # Real-time store, persistence & API tests
├── data/                                  # Persistent data storage
│   ├── songs.csv                          # Catalog enriched with audio previews & artwork
│   └── interactions.csv                   # Historical & live user interaction events
├── reports/                               # Academic & engineering documentation
│   └── project_report.md                  # Comprehensive technical report
├── main.py                                # Root CLI entrypoint (`recommend`, `evaluate`, `web`)
├── app.py                                 # Streamlit Spotify-styled WebUI application
├── pyproject.toml                         # Modern package build configuration (PEP 517/621)
├── setup.py                               # Setuptools installer
├── requirements.txt                       # Project runtime dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/codeXparadise/music-recommendation-system.git
cd music-recommendation-system
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Or install in editable development mode:
pip install -e .
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🖥️ WebUI Overview

| View / Feature | Description |
|---|---|
| **🎧 Real-Time Recommendations** | Live personalized cards showing cover art, playback controls, and 1-click reaction triggers (Like, Skip, Replay, Complete, Partial). |
| **⚡ Discover Fresh Music** | On-demand live fetch from Apple Music API tailored to current listener favorites. |
| **🌍 Real Music Library (API Search)** | Search millions of real tracks, audition 30s previews, and bookmark them into the catalog. |
| **📜 Interaction Feed & Live Analytics** | Real-time audit log of all events, action breakdown donut charts, and CSV downloads. |
| **📊 Model Evaluation Dashboard** | Interactive Precision@K vs. Recall@K charts comparing Content vs. Collaborative filtering. |
| **🗂️ Global Catalog & Live Playlist** | Spotify-like playlist table showing song metadata, user status badges, and 1-click play buttons. |
| **⚖️ Dynamic Weight Sliders** | Sidebar sliders allowing live adjustment of implicit feedback penalties and bonuses. |

---

## 💻 CLI Usage (Running with Arguments)

You can run recommendations and evaluations directly from the command line:

### Unified CLI Entrypoint (`main.py`)
```bash
# 1. Hybrid recommendations for a specific listener
python main.py recommend --user U001 --method hybrid --top-k 5

# 2. Collaborative filtering
python main.py recommend --user U002 --method collaborative --top-k 5

# 3. Content-based filtering with genre constraints
python main.py recommend --user U003 --method content --genre Pop --top-k 5

# 4. Cold-start recommendations for a new listener
python main.py recommend --genre Bollywood --artist "Arijit Singh" --top-k 4

# 5. Model evaluation
python main.py evaluate --k 5 --detailed

# 6. Launch WebUI on custom port
python main.py web --port 8501
```

### Standalone Scripts (`scripts/`)
```bash
# Run recommendations script
python scripts/run_recommendation.py --user U001 --method hybrid --top-k 5

# Run evaluation script
python scripts/run_evaluation.py --k 5 --detailed

# Run web app script
python scripts/run_app.py --port 8501
```

---

## 🧪 Automated Unit Tests

The test suite covers data loading, algorithmic correctness, cold-start handling, metric computations, real-time store persistence, and API formatting:

```bash
python -m unittest discover -s tests -v
```

**Results:**
```
Ran 19 tests in 0.735s
OK
```

---

## 🧠 Recommendation Algorithms

### 1. Content-Based Filtering
- Extracts multidimensional song features: `genre`, `mood`, `artist`, `language`, and `album`.
- Generates a TF-IDF feature space and computes pairwise **Cosine Similarity**.
- Computes aggregate user preference profiles by averaging the vectors of positively engaged tracks.

### 2. Collaborative Filtering (Item-Based Implicit Feedback)
- Constructs a sparse User-Item interaction matrix where cell values represent cumulative interaction weights:
  $$\text{weight} = \sum_{\text{events}} w(\text{action})$$
- Computes item-item similarity vectors.
- Predicts preference scores by taking the dot product of the listener's interaction history vector with candidate item similarities.

### 3. Hybrid Strategy
- Merges candidate sets from both Content-Based and Collaborative Filtering models.
- Gracefully handles cold-start users (listeners with no prior history) by switching to popularity-ranked genre and artist filtering.
- Implements immediate negative penalty filtering: skipped tracks are eliminated from active candidate recommendations.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
