import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List

import requests

logger = logging.getLogger("creatorrocket")

ANALYTICS_BASE_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
ACCESS_TOKEN = os.getenv("YOUTUBE_ANALYTICS_ACCESS_TOKEN")
AUTHORIZED_CHANNEL_ID = (os.getenv("YOUTUBE_ANALYTICS_CHANNEL_ID") or "mine").strip()


def analytics_available_for(channel_id: str) -> bool:
    if not ACCESS_TOKEN:
        return False
    if not AUTHORIZED_CHANNEL_ID or AUTHORIZED_CHANNEL_ID.lower() == "mine":
        return True
    return channel_id == AUTHORIZED_CHANNEL_ID


def _query_report(params: Dict) -> Dict:
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    try:
        response = requests.get(ANALYTICS_BASE_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("YouTube Analytics request failed: %s", exc)
        return {}


def _single_row_to_dict(payload: Dict) -> Dict:
    headers = [header["name"] for header in payload.get("columnHeaders", [])]
    rows = payload.get("rows", [])
    if not headers or not rows:
        return {}
    return dict(zip(headers, rows[0]))


def get_recent_video_analytics(channel_id: str, videos: List[Dict]) -> Dict[str, Dict]:
    """Fetch richer owner-only analytics when OAuth is available for the channel."""
    if not analytics_available_for(channel_id):
        return {}

    metrics = ",".join([
        "views",
        "likes",
        "comments",
        "shares",
        "estimatedMinutesWatched",
        "averageViewDuration",
        "averageViewPercentage",
        "subscribersGained",
        "subscribersLost"
    ])

    analytics_by_video: Dict[str, Dict] = {}
    end_date = (datetime.utcnow() - timedelta(days=2)).date()

    for video in videos[:12]:
        video_id = video.get("video_id")
        if not video_id:
            continue

        published_at = video.get("published_at")
        if published_at:
            try:
                start_date = datetime.fromisoformat(published_at.replace("Z", "+00:00")).date()
            except ValueError:
                start_date = end_date - timedelta(days=28)
        else:
            start_date = end_date - timedelta(days=28)

        if start_date > end_date:
            start_date = end_date

        payload = _query_report({
            "ids": "channel==MINE",
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "metrics": metrics,
            "filters": f"video=={video_id}"
        })
        row = _single_row_to_dict(payload)
        if row:
            analytics_by_video[video_id] = row

    return analytics_by_video
