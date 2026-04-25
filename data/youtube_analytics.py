import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List

import requests

from db.database import SessionLocal
from data.youtube_oauth import get_valid_access_token

logger = logging.getLogger("creatorrocket")

ANALYTICS_BASE_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
ENV_ACCESS_TOKEN = os.getenv("YOUTUBE_ANALYTICS_ACCESS_TOKEN")
ENV_CHANNEL_ID = (os.getenv("YOUTUBE_ANALYTICS_CHANNEL_ID") or "").strip()


def _query_report(access_token: str, params: Dict) -> Dict:
    headers = {"Authorization": f"Bearer {access_token}"}
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


def _resolve_access_token(channel_id: str) -> str:
    if ENV_ACCESS_TOKEN and (not ENV_CHANNEL_ID or ENV_CHANNEL_ID == channel_id):
        return ENV_ACCESS_TOKEN

    db = SessionLocal()
    try:
        return get_valid_access_token(db, channel_id=channel_id) or ""
    finally:
        db.close()


def analytics_available_for(channel_id: str) -> bool:
    return bool(_resolve_access_token(channel_id))


def get_recent_video_analytics(channel_id: str, videos: List[Dict]) -> Dict[str, Dict]:
    """Fetch richer owner-only analytics when OAuth is available for the channel."""
    access_token = _resolve_access_token(channel_id)
    if not access_token:
        return {}

    metrics = ",".join(
        [
            "views",
            "likes",
            "comments",
            "shares",
            "estimatedMinutesWatched",
            "averageViewDuration",
            "averageViewPercentage",
            "subscribersGained",
            "subscribersLost",
        ]
    )

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

        payload = _query_report(
            access_token,
            {
                "ids": "channel==MINE",
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "metrics": metrics,
                "filters": f"video=={video_id}",
            },
        )
        row = _single_row_to_dict(payload)
        if row:
            analytics_by_video[video_id] = row

    return analytics_by_video
