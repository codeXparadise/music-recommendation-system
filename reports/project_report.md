# TuneMatch: Music Recommendation System — Project Report

## 1. Abstract
This project presents **TuneMatch**, an end-to-end, real-time music recommendation system and interactive streaming web application modeled after the Spotify user experience. The system integrates a hybrid recommendation engine combining content-based filtering (TF-IDF vectorization and cosine similarity over track metadata) and collaborative filtering (item-item similarity derived from an implicit user-item interaction matrix). The platform directly interfaces with the Apple Music / iTunes Search API to fetch live hit music, high-resolution album artwork, and streamable 30-second audio previews in real time. Implicit and explicit feedback events (Like, Replay, Complete Listen, Partial Listen, Skip) dynamically tune user preference profiles and recalculate recommendations on the fly. Recommendation quality is benchmarked using holdout evaluation on Precision@K, Recall@K, Normalized Discounted Cumulative Gain (NDCG@K), Mean Average Precision (MAP@K), and Hit Rate@K.

---

## 2. Problem Statement
In modern digital audio streaming platforms, catalog scale poses a significant information retrieval challenge: listeners struggle to discover relevant music without active curation. Pure content-based systems suffer from over-specialization, while collaborative filtering struggles with cold-start users and sparse feedback. Furthermore, real-world user interactions in streaming platforms consist primarily of implicit behavioral cues (e.g., skips, replays, completion rates) rather than explicit 1-to-5 star ratings. This project addresses these challenges by implementing an adaptive hybrid recommender that translates real-time playback behavior into weighted preference vectors and recommends streamable music dynamically.

---

## 3. System Architecture & Objectives

### 3.1 Architecture Overview
The codebase is structured under the standard Python `src`-layout recommended by the Python Packaging Authority (PyPA):
- **Core Library (`src/music_recommender/`)**:
  - `models/`: Content-based, Collaborative, and Hybrid recommendation engines.
  - `services/music_api.py`: Apple Music / iTunes API client for real-time track retrieval, metadata extraction, mood inference, and 30-second audio preview streaming.
  - `interaction_store.py`: Real-time event logging, action weight management, and dynamic session re-scoring.
  - `data_loader.py`: Schema validation, typecasting, and CSV persistence.
  - `evaluation.py`: Precision@K, Recall@K, NDCG@K, MAP@K, and Hit Rate@K computation.
  - `cli.py`: Command-line interface with subcommands (`recommend`, `evaluate`, `web`).
- **Web Application (`app.py`)**:
  - Spotify dark-themed UI with persistent top "Now Playing" audio bar.
  - Live playlist tracklist with real-time listener status and feedback weights.
  - Instant recommendation updates upon Like, Replay, or Skip actions.
- **Standalone Scripts (`scripts/`)**:
  - `run_recommendation.py`, `run_evaluation.py`, `run_app.py`.
- **Automated Test Suite (`tests/`)**:
  - 18 comprehensive unit tests covering data loading, algorithms, ranking metrics, and real-time interaction stores.

---

## 4. Methodology

### 4.1 Content-Based Filtering
Song metadata combines the track title, artist, genre, language, and inferred mood into an enriched textual feature representation:
$$\text{Feature}(s) = \text{Artist}(s) \oplus \text{Genre}(s) \oplus \text{Language}(s) \oplus \text{Mood}(s)$$
A TF-IDF (Term Frequency - Inverse Document Frequency) vectorizer transforms these representations into sparse feature vectors. For a user with positively interacted seed tracks $S_{user}$, the profile vector is the mean vector of their seed songs, and cosine similarity ranks candidates:
$$\text{Sim}(u, s) = \frac{\mathbf{v}_{user} \cdot \mathbf{v}_s}{\|\mathbf{v}_{user}\| \|\mathbf{v}_s\|}$$

### 4.2 Collaborative Filtering with Implicit Feedback
User interaction events are mapped into numerical preference weights:
- **Like**: $+3.0$
- **Replay**: $+2.5$
- **Complete Listen**: $+2.0$
- **Partial Listen**: $+0.5$
- **Skip**: $-1.0$

A user-item matrix $R \in \mathbb{R}^{|U| \times |I|}$ is constructed where cell $R_{u, i}$ represents the sum of interaction weights of user $u$ on track $i$. Item-item cosine similarity is computed across item vectors:
$$S_{i, j} = \frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\| \|\mathbf{r}_j\|}$$
The predicted preference score for an unobserved song $j$ by user $u$ is calculated as:
$$\hat{R}_{u, j} = \sum_{i \in \text{Interacted}(u)} S_{i, j} \cdot R_{u, i}$$
Tracks already experienced or skipped are filtered out of subsequent recommendations.

### 4.3 Cold-Start Strategy
For new users with zero listening history:
1. **Preference-Guided Filtering**: Users can specify preferred genres or artists; the system initializes recommendations using top tracks from matching artists.
2. **Catalog Popularity Fallback**: If no preferences are provided, the system serves the highest-popularity tracks across diverse genres.
3. **Instant Onboarding**: As soon as the user plays, likes, or skips a single song, their profile transitions from cold-start to active personalized recommendation in real time.

---

## 5. Live Music API & Real-Time Interaction Loop

### 5.1 Real Music Library Integration
The system integrates with the Apple iTunes Search API (`https://itunes.apple.com/search`), allowing users to search millions of real tracks, view high-resolution album artwork (600x600), and listen to live 30-second AAC/MP3 audio streams directly in the browser without requiring third-party API credentials.

### 5.2 Dynamic Real-Time Feedback Loop
```
[User Action in WebUI]
       │
       ▼ (Like: +3.0, Skip: -1.0)
[Interaction Store] ───► [Adjusts User Weights in Matrix]
       │
       ▼
[Hybrid Recommender] ───► [Excludes Skipped, Boosts Similar Seeds]
       │
       ▼
[Instant Streamlit Re-render] ───► [Live "Made For You" Feed Updates]
```

---

## 6. Empirical Evaluation & Results

Holdout cross-validation was performed on known users with positive interactions ($\ge 2$) using a leave-positive-out strategy.

### Measured Metrics at $K=5$:

| Recommendation Model | Precision@5 | Recall@5 | NDCG@5 | MAP@5 | Hit Rate@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Content-Based Filtering** | **0.200** | **0.500** | **0.463** | **0.438** | **0.500** |
| **Collaborative Filtering** | **0.150** | **0.375** | **0.354** | **0.312** | **0.375** |

### Metric Formulations:
- **Precision@K**: $\frac{|\text{Recommended}_K \cap \text{Relevant}|}{K}$
- **Recall@K**: $\frac{|\text{Recommended}_K \cap \text{Relevant}|}{|\text{Relevant}|}$
- **NDCG@K**: $\frac{\text{DCG}@K}{\text{IDCG}@K}$, where $\text{DCG}@K = \sum_{i=1}^K \frac{\mathbb{I}(\text{item}_i \in \text{Relevant})}{\log_2(i + 1)}$
- **MAP@K**: $\frac{1}{\min(K, |\text{Relevant}|)} \sum_{i=1}^K P(i) \cdot \mathbb{I}(\text{item}_i \in \text{Relevant})$
- **Hit Rate@K**: $1 \text{ if } |\text{Recommended}_K \cap \text{Relevant}| > 0 \text{ else } 0$

---

## 7. Web Application Features

1. **Persistent Spotify Player Bar**:
   - Floating header displaying track thumbnail, title, artist, genre badge, live audio preview player (`autoplay=True`), and controls (Like, Replay, Skip).
   - Clicking Skip automatically records negative feedback and autoplays the next top recommendation.
2. **Made For You (Personalized Feed)**:
   - Interactive recommendation cards with match scores, direct audio preview streams, and instant Like/Skip controls.
   - Dynamic API discovery button to fetch fresh tracks matching listener tastes.
3. **Search & Browse**:
   - Apple Music global search across titles, artists, and albums.
   - One-click genre pills (*Global Pop, Bollywood, Rock, EDM, Hip-Hop, Lo-Fi*).
4. **Spotify Playlist View (Full Library)**:
   - Interactive playlist tracklist with columns: `#`, `Title & Artist`, `Album`, `Genre / Mood`, `Your Status & Weight`, `Controls`.
   - Live status badges reflecting the active listener's actual interaction history (`💚 Liked (+3.0)`, `⏭️ Skipped (-1.0)`, `🔁 Replayed (+2.5)`, `▶️ Listened (+2.0)`, `⚪ Unplayed`).
   - Quick filters: *All Tracks*, *Liked by You*, *Skipped by You*, *Listened Tracks*, *Unplayed Tracks*.
5. **Real-Time Weight Tuner**:
   - Live sliders in the sidebar allowing listeners to modify action weights (*Like*, *Replay*, *Complete*, *Partial*, *Skip*) and observe instant re-ranking of recommendations.

---

## 8. Conclusion
TuneMatch demonstrates that combining metadata-driven content representations with implicit feedback collaborative filtering provides a robust, responsive music recommendation platform. By integrating live music APIs, audio preview streaming, and real-time weight adaptation, the system bridges the gap between academic recommendation models and consumer streaming experiences like Spotify.
