import os
import logging
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("creatorrocket")

API_KEY = os.getenv("YOUTUBE_API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3"

def get_channel_stats(channel_id: str) -> Optional[Dict]:
    """Fetch channel statistics and snippet using YouTube Data API v3."""
    if not API_KEY or API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        logger.error("YouTube API Key missing or invalid in .env")
        return None

    url = f"{BASE_URL}/channels"
    params = {
        "part": "snippet,statistics",
        "id": channel_id,
        "key": API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("items"):
            logger.warning(f"No channel found for ID: {channel_id}")
            return None
            
        item = data["items"][0]
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})
        
        return {
            "channel_id": channel_id,
            "name": snippet.get("title", "Unknown"),
            "subscriber_count": int(stats.get("subscriberCount", 0)),
            "view_count": int(stats.get("viewCount", 0)),
            "video_count": int(stats.get("videoCount", 0)),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", "")  # ADD THIS
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"YouTube API request failed for channel {channel_id}: {e}")
        return None



def search_channels(query: str, max_results: int = 5) -> list:
    """Search YouTube channels by name, return id + name + thumbnail."""
    if not API_KEY:
        return []
    url = f"{BASE_URL}/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "channel",
        "maxResults": max_results,
        "key": API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            results.append({
                "channel_id": item["id"]["channelId"],
                "name": snippet.get("title", ""),
                "description": snippet.get("description", "")[:80],
                "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", "")
            })
        return results
    except requests.exceptions.RequestException as e:
        logger.error(f"Channel search failed: {e}")
        return []


def get_recent_videos(channel_id: str, max_results: int = 5) -> List[Dict]:
    """Fetch recent videos and their stats for engagement proxy calculation."""
    if not API_KEY:
        return []

    # Step 1: Search for recent video IDs
    search_url = f"{BASE_URL}/search"
    search_params = {
        "part": "id",
        "channelId": channel_id,
        "maxResults": max_results,
        "order": "date",
        "type": "video",
        "key": API_KEY
    }
    
    try:
        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        
        video_ids = [item["id"]["videoId"] for item in search_data.get("items", []) if "videoId" in item.get("id", {})]
        if not video_ids:
            return []
            
        # Step 2: Get stats for those videos
        videos_url = f"{BASE_URL}/videos"
        videos_params = {
            "part": "statistics",
            "id": ",".join(video_ids),
            "key": API_KEY
        }
        
        vids_resp = requests.get(videos_url, params=videos_params, timeout=10)
        vids_resp.raise_for_status()
        vids_data = vids_resp.json()
        
        return [
            {
                "video_id": v["id"],
                "views": int(v["statistics"].get("viewCount", 0)),
                "likes": int(v["statistics"].get("likeCount", 0)),
                "comments": int(v["statistics"].get("commentCount", 0))
            }
            for v in vids_data.get("items", [])
        ]
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch recent videos for {channel_id}: {e}")
        return []
