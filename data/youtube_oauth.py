import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
from urllib.parse import urlencode

import requests
from sqlalchemy.orm import Session

from db.models import AnalyticsConnection, OAuthState

logger = logging.getLogger("creatorrocket")

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
]


def oauth_configured() -> bool:
    return bool(
        os.getenv("YOUTUBE_OAUTH_CLIENT_ID")
        and os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET")
        and os.getenv("YOUTUBE_OAUTH_REDIRECT_URI")
    )


def _client_config() -> Dict[str, str]:
    return {
        "client_id": os.getenv("YOUTUBE_OAUTH_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET", "").strip(),
        "redirect_uri": os.getenv("YOUTUBE_OAUTH_REDIRECT_URI", "").strip(),
    }


def cleanup_expired_states(db: Session) -> None:
    db.query(OAuthState).filter(OAuthState.expires_at < datetime.utcnow()).delete()
    db.commit()


def create_authorization_url(db: Session, redirect_path: Optional[str] = None) -> str:
    if not oauth_configured():
        raise RuntimeError("YouTube OAuth is not configured.")

    cleanup_expired_states(db)
    state_value = secrets.token_urlsafe(24)
    state = OAuthState(
        provider="youtube",
        state=state_value,
        redirect_path=redirect_path or "/",
        expires_at=datetime.utcnow() + timedelta(minutes=15),
    )
    db.add(state)
    db.commit()

    config = _client_config()
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state_value,
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def _fetch_channel_identity(access_token: str) -> Optional[Dict[str, str]]:
    try:
        response = requests.get(
            CHANNELS_URL,
            params={"part": "id,snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch connected YouTube channel identity: %s", exc)
        return None

    items = data.get("items") or []
    if not items:
        return None

    item = items[0]
    snippet = item.get("snippet", {})
    thumbnails = snippet.get("thumbnails", {})
    return {
        "channel_id": item.get("id", ""),
        "channel_name": snippet.get("title", "Connected channel"),
        "profile_picture_url": thumbnails.get("high", {}).get("url") or thumbnails.get("default", {}).get("url", ""),
    }


def _exchange_code_for_tokens(code: str) -> Dict:
    config = _client_config()
    response = requests.post(
        TOKEN_URL,
        data={
            "code": code,
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "redirect_uri": config["redirect_uri"],
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _refresh_access_token(connection: AnalyticsConnection) -> Optional[Dict]:
    config = _client_config()
    if not connection.refresh_token:
        return None

    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "refresh_token": connection.refresh_token,
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "grant_type": "refresh_token",
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.warning("Failed to refresh YouTube OAuth token: %s", exc)
        return None


def get_active_connection(db: Session, channel_id: Optional[str] = None) -> Optional[AnalyticsConnection]:
    query = db.query(AnalyticsConnection).filter(AnalyticsConnection.is_active.is_(True))
    if channel_id:
        connection = query.filter(AnalyticsConnection.channel_id == channel_id).first()
        if connection:
            return connection
    return query.order_by(AnalyticsConnection.connected_at.desc()).first()


def get_valid_access_token(db: Session, channel_id: Optional[str] = None) -> Optional[str]:
    connection = get_active_connection(db, channel_id=channel_id)
    if not connection:
        return None

    expires_at = connection.token_expires_at
    if expires_at and expires_at > datetime.utcnow() + timedelta(minutes=5):
        return connection.access_token

    refreshed = _refresh_access_token(connection)
    if not refreshed:
        return connection.access_token if not expires_at else None

    connection.access_token = refreshed["access_token"]
    expires_in = int(refreshed.get("expires_in", 3600))
    connection.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
    if refreshed.get("refresh_token"):
        connection.refresh_token = refreshed["refresh_token"]
    db.commit()
    return connection.access_token


def get_oauth_status(db: Session) -> Dict:
    connection = get_active_connection(db)
    return {
        "configured": oauth_configured(),
        "connected": connection is not None,
        "channel_id": connection.channel_id if connection else None,
        "channel_name": connection.channel_name if connection else None,
        "profile_picture_url": connection.profile_picture_url if connection else None,
        "connected_at": connection.connected_at.isoformat() if connection else None,
        "expires_at": connection.token_expires_at.isoformat() if connection and connection.token_expires_at else None,
        "scope": connection.scope if connection else None,
    }


def disconnect_oauth(db: Session) -> Dict:
    connection = get_active_connection(db)
    if connection:
        connection.is_active = False
        db.commit()
    return get_oauth_status(db)


def handle_oauth_callback(db: Session, code: str, state_value: str) -> Dict:
    cleanup_expired_states(db)
    state = db.query(OAuthState).filter(OAuthState.state == state_value).first()
    if not state:
        raise RuntimeError("This YouTube connect link has expired. Please try again.")

    token_data = _exchange_code_for_tokens(code)
    identity = _fetch_channel_identity(token_data["access_token"])
    if not identity or not identity.get("channel_id"):
        raise RuntimeError("Connected successfully, but the authorized YouTube channel could not be identified.")

    existing = db.query(AnalyticsConnection).filter(AnalyticsConnection.channel_id == identity["channel_id"]).first()
    expires_in = int(token_data.get("expires_in", 3600))
    expiry = datetime.utcnow() + timedelta(seconds=expires_in)

    if existing:
        connection = existing
    else:
        connection = AnalyticsConnection(channel_id=identity["channel_id"], provider="youtube")
        db.add(connection)

    connection.channel_name = identity.get("channel_name")
    connection.profile_picture_url = identity.get("profile_picture_url")
    connection.access_token = token_data["access_token"]
    if token_data.get("refresh_token"):
        connection.refresh_token = token_data["refresh_token"]
    connection.scope = token_data.get("scope", " ".join(SCOPES))
    connection.token_expires_at = expiry
    connection.is_active = True

    db.delete(state)
    db.commit()

    return {
        "redirect_path": state.redirect_path or "/",
        "channel_id": connection.channel_id,
        "channel_name": connection.channel_name,
    }
