"""Real Music Library API Service using Apple iTunes Search API.

Provides real-time search, streaming 30s preview audio URLs, high-resolution
album artwork, and catalog expansion without requiring API keys or authentication.
"""

import json
import logging
import urllib.parse
import urllib.request
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def infer_mood(title: str, genre: str, artist: str) -> str:
    """Infer an illustrative mood tag from track metadata."""
    text = f" {title} {genre} {artist} ".lower()

    if any(k in text for k in ["party", "dance", "rock", "fever", "fire", "club", "bass", "energy", "run", "fast", "workout"]):
        return "Energetic"
    if any(k in text for k in ["sad", "heartbreak", "lonely", "rain", "tears", "cry", "sorrow", "grief"]):
        return "Melancholic"
    if any(k in text for k in ["love", "dil", "romantic", "romance", "forever", "sweetheart", "kiss"]):
        return "Romantic"
    if any(k in text for k in ["chill", "coffee", "breeze", "relax", "lofi", "acoustic", "peace"]):
        return "Chill"
    if any(k in text for k in ["happy", "sun", "shine", "smile", "uplift", "good vibes", "alive", "celebrate"]):
        return "Uplifting"
    if any(k in text for k in ["space", "echo", "ocean", "stars", "moon", "cloud", "ambient", "dream"]):
        return "Dreamy"

    genre_mood_map = {
        "pop": "Uplifting",
        "rock": "Energetic",
        "electronic": "Energetic",
        "dance": "Energetic",
        "hip-hop/rap": "Energetic",
        "r&b/soul": "Romantic",
        "jazz": "Chill",
        "classical": "Chill",
        "folk": "Chill",
        "indie pop": "Dreamy",
        "bollywood": "Romantic",
    }
    return genre_mood_map.get(genre.lower(), "Uplifting")


class MusicAPIService:
    """Integrates with Apple iTunes Search API to fetch live music records."""

    BASE_URL = "https://itunes.apple.com/search"

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self._cache: Dict[str, List[Dict]] = {}

    def search_tracks(self, query: str, limit: int = 15) -> List[Dict]:
        """Search tracks by keyword, song title, or artist name."""
        query = query.strip()
        if not query:
            return []

        cache_key = f"{query.lower()}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "term": query,
            "entity": "song",
            "media": "music",
            "limit": min(limit, 50),
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TuneMatch-MusicRecommender/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                raw_items = payload.get("results", [])
                standardized = [self._format_track(item) for item in raw_items if self._is_valid(item)]
                self._cache[cache_key] = standardized
                return standardized
        except Exception as e:
            logger.warning(f"Error fetching from iTunes API for query '{query}': {e}")
            return []

    def get_genre_preset_tracks(self, genre_keyword: str, limit: int = 10) -> List[Dict]:
        """Fetch popular tracks for a given genre preset."""
        return self.search_tracks(f"{genre_keyword} hits", limit=limit)

    def discover_live_recommendations(
        self,
        liked_artists: Optional[List[str]] = None,
        liked_genres: Optional[List[str]] = None,
        limit: int = 8,
    ) -> List[Dict]:
        """Fetch fresh playable music from Apple Music API matching listener tastes."""
        liked_artists = liked_artists or []
        liked_genres = liked_genres or []
        results = []
        seen_ids = set()

        for artist in liked_artists[:3]:
            tracks = self.search_tracks(f"{artist} top songs", limit=4)
            for t in tracks:
                if t["song_id"] not in seen_ids and t.get("preview_url"):
                    seen_ids.add(t["song_id"])
                    results.append(t)

        for genre in liked_genres[:2]:
            tracks = self.search_tracks(f"{genre} hit songs", limit=4)
            for t in tracks:
                if t["song_id"] not in seen_ids and t.get("preview_url"):
                    seen_ids.add(t["song_id"])
                    results.append(t)

        if not results:
            tracks = self.search_tracks("global top hits", limit=limit)
            for t in tracks:
                if t["song_id"] not in seen_ids and t.get("preview_url"):
                    seen_ids.add(t["song_id"])
                    results.append(t)

        return results[:limit]

    def _is_valid(self, item: Dict) -> bool:
        """Filter out incomplete or non-song items."""
        return bool(
            item.get("trackId")
            and item.get("trackName")
            and item.get("artistName")
        )

    def _format_track(self, item: Dict) -> Dict:
        """Map iTunes response object into TuneMatch standard song format."""
        raw_art = item.get("artworkUrl100", "")
        # Upgrade artwork to 600x600 for sharp UI rendering
        hd_art = raw_art.replace("100x100bb.jpg", "600x600bb.jpg") if raw_art else ""

        genre = item.get("primaryGenreName", "Pop")
        title = item.get("trackName", "Unknown Title")
        artist = item.get("artistName", "Unknown Artist")
        country = item.get("country", "USA")
        language = "Hindi" if "bollywood" in genre.lower() or "india" in country.lower() else "English"
        mood = infer_mood(title, genre, artist)

        return {
            "song_id": f"API_{item['trackId']}",
            "song_title": title,
            "artist": artist,
            "genre": genre,
            "language": language,
            "mood": mood,
            "popularity": 85,  # Default popularity score for API hits
            "artwork_url": hd_art or raw_art,
            "preview_url": item.get("previewUrl", ""),
            "track_url": item.get("trackViewUrl", ""),
            "album": item.get("collectionName", "Single"),
            "release_date": (item.get("releaseDate") or "")[:10],
        }

    def convert_to_dataframe(self, tracks: List[Dict]) -> pd.DataFrame:
        """Convert a list of track dictionaries to a pandas DataFrame."""
        if not tracks:
            return pd.DataFrame(
                columns=[
                    "song_id", "song_title", "artist", "genre",
                    "language", "mood", "popularity", "artwork_url",
                    "preview_url", "track_url", "album", "release_date"
                ]
            )
        df = pd.DataFrame(tracks)
        df["song_id"] = df["song_id"].astype(str)
        return df
