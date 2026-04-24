import os
import logging
import requests
from typing import Dict, List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("creatorrocket")

API_KEY = os.getenv("YOUTUBE_API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3"

def _request(endpoint: str, params: Dict) -> Optional[Dict]:
    if not API_KEY or API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        logger.error("YouTube API Key missing or invalid in .env")
        return None

    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params={**params, "key": API_KEY}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error("YouTube API request failed for %s: %s", endpoint, e)
        return None

def normalize_creator_input(raw_input: str) -> str:
    value = (raw_input or "").strip()
    if not value:
        return ""
    if value.startswith("@"):
        return value
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        path = parsed.path.strip("/")
        if not path:
            return value
        parts = path.split("/")
        if parts[0] in {"channel", "user", "c"} and len(parts) > 1:
            return parts[1]
        if parts[0].startswith("@"):
            return parts[0]
        return parts[-1]
    return value

def _extract_channel(item: Dict) -> Dict:
    stats = item.get("statistics", {})
    snippet = item.get("snippet", {})
    branding = item.get("brandingSettings", {}).get("channel", {})
    return {
        "channel_id": item.get("id", ""),
        "name": snippet.get("title", "Unknown"),
        "subscriber_count": int(stats.get("subscriberCount", 0)),
        "view_count": int(stats.get("viewCount", 0)),
        "video_count": int(stats.get("videoCount", 0)),
        "published_at": snippet.get("publishedAt"),
        "handle": snippet.get("customUrl") or branding.get("unsubscribedTrailer", ""),
        "description": snippet.get("description", ""),
        "country": snippet.get("country", "Global"),
        "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", "") or snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
    }

def get_channel_by_id(channel_id: str) -> Optional[Dict]:
    data = _request("channels", {"part": "snippet,statistics,brandingSettings", "id": channel_id})
    if not data or not data.get("items"):
        return None
    return _extract_channel(data["items"][0])

def search_channel(query: str) -> Optional[Dict]:
    data = _request("search", {
        "part": "snippet",
        "q": query,
        "type": "channel",
        "maxResults": 5
    })
    if not data:
        return None

    candidate_ids = [item["snippet"].get("channelId") for item in data.get("items", []) if item.get("snippet", {}).get("channelId")]
    if not candidate_ids:
        return None

    details = _request("channels", {
        "part": "snippet,statistics,brandingSettings",
        "id": ",".join(candidate_ids)
    })
    if not details:
        return None

    normalized_query = query.lower().lstrip("@")
    channels = [_extract_channel(item) for item in details.get("items", [])]

    def score_channel(channel: Dict) -> tuple:
        handle = (channel.get("handle") or "").lower().lstrip("@")
        name = (channel.get("name") or "").lower()
        exact_handle = handle == normalized_query
        exact_name = name == normalized_query
        starts = handle.startswith(normalized_query) or name.startswith(normalized_query)
        return (
            1 if exact_handle else 0,
            1 if exact_name else 0,
            1 if starts else 0,
            channel.get("subscriber_count", 0)
        )

    channels.sort(key=score_channel, reverse=True)
    return channels[0] if channels else None

def search_channels(query: str, max_results: int = 5) -> List[Dict]:
    data = _request("search", {
        "part": "snippet",
        "q": query,
        "type": "channel",
        "maxResults": max_results
    })
    if not data:
        return []

    results = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        channel_id = snippet.get("channelId") or item.get("id", {}).get("channelId")
        if not channel_id:
            continue
        results.append({
            "channel_id": channel_id,
            "name": snippet.get("title", ""),
            "description": (snippet.get("description", "") or "")[:80],
            "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", "")
        })
    return results

def resolve_channel(raw_input: str) -> Optional[Dict]:
    normalized = normalize_creator_input(raw_input)
    if not normalized:
        return None

    if normalized.startswith("UC") and len(normalized) >= 20:
        return get_channel_by_id(normalized)

    data = _request("channels", {"part": "snippet,statistics,brandingSettings", "forUsername": normalized.lstrip("@")})
    if data and data.get("items"):
        return _extract_channel(data["items"][0])

    return search_channel(normalized)

def get_channel_stats(channel_id: str) -> Optional[Dict]:
    """Fetch channel statistics and snippet using YouTube Data API v3."""
    return get_channel_by_id(channel_id)

def get_recent_videos(channel_id: str, max_results: int = 5) -> List[Dict]:
    """Fetch recent videos and their stats for engagement proxy calculation."""
    if not API_KEY:
        return []

    # Step 1: Search for recent video IDs
    search_data = _request("search", {
        "part": "id,snippet",
        "channelId": channel_id,
        "maxResults": max_results,
        "order": "date",
        "type": "video",
    })
    if not search_data:
        return []

    video_lookup = {
        item["id"]["videoId"]: item.get("snippet", {})
        for item in search_data.get("items", [])
        if "videoId" in item.get("id", {})
    }
    video_ids = list(video_lookup.keys())
    if not video_ids:
        return []

    vids_data = _request("videos", {"part": "statistics,snippet", "id": ",".join(video_ids)})
    if not vids_data:
        return []

    return [
        {
            "video_id": v["id"],
            "title": v.get("snippet", {}).get("title", ""),
            "published_at": v.get("snippet", {}).get("publishedAt") or video_lookup.get(v["id"], {}).get("publishedAt"),
            "views": int(v.get("statistics", {}).get("viewCount", 0)),
            "likes": int(v.get("statistics", {}).get("likeCount", 0)),
            "comments": int(v.get("statistics", {}).get("commentCount", 0))
        }
        for v in vids_data.get("items", [])
    ]
