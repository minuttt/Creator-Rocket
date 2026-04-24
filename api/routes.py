import os
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional, Tuple, List, Dict
from db.database import get_db
from db.models import Creator, Snapshot
from data.youtube_client import get_channel_stats, get_recent_videos, resolve_channel, search_channels
from services import feature_engineer, predictor, explanation_engine, simulation

logger = logging.getLogger("creatorrocket")
router = APIRouter()

@router.get("/api/search-channels")
async def search_channels_endpoint(q: str):
    """Search YouTube channels by name for frontend autocomplete."""
    if not q or len(q.strip()) < 2:
        return []
    return search_channels(q.strip(), max_results=5)

def _find_creator(db: Session, creator_input: str) -> Optional[Creator]:
    normalized = (creator_input or "").strip()
    if not normalized:
        return None
    return (
        db.query(Creator)
        .filter((Creator.channel_id == normalized) | (Creator.name == normalized))
        .first()
    )

def _upsert_creator_with_snapshot(db: Session, creator_input: str, force_snapshot: bool = False) -> Tuple[Optional[Creator], Optional[Dict], List[Snapshot]]:
    resolved = resolve_channel(creator_input)
    if not resolved:
        return None, None, []

    creator = db.query(Creator).filter(Creator.channel_id == resolved["channel_id"]).first()
    if creator is None:
        creator = Creator(
            channel_id=resolved["channel_id"],
            name=resolved["name"],
            platform="youtube",
            profile_picture_url=resolved.get("thumbnail", "")
        )
        db.add(creator)
        db.commit()
        db.refresh(creator)
    else:
        creator.name = resolved["name"]
        creator.profile_picture_url = resolved.get("thumbnail", creator.profile_picture_url or "")
        db.commit()

    snapshots = (
        db.query(Snapshot)
        .filter(Snapshot.creator_id == creator.id)
        .order_by(Snapshot.timestamp.desc())
        .limit(30)
        .all()
    )

    should_write_snapshot = force_snapshot or not snapshots
    if snapshots and not force_snapshot:
        latest = snapshots[0]
        changed = (
            latest.subscriber_count != resolved["subscriber_count"] or
            latest.view_count != resolved["view_count"] or
            latest.video_count != resolved["video_count"]
        )
        stale = datetime.utcnow() - latest.timestamp > timedelta(hours=6)
        should_write_snapshot = changed or stale

    if should_write_snapshot:
        new_snapshot = Snapshot(
            creator_id=creator.id,
            subscriber_count=resolved["subscriber_count"],
            view_count=resolved["view_count"],
            video_count=resolved["video_count"]
        )
        db.add(new_snapshot)
        db.commit()
        snapshots = (
            db.query(Snapshot)
            .filter(Snapshot.creator_id == creator.id)
            .order_by(Snapshot.timestamp.desc())
            .limit(30)
            .all()
        )

    return creator, resolved, snapshots

def _build_analysis_payload(creator: Creator, resolved: Dict, snapshots: List[Snapshot], latest_vids: List[Dict], platform: str = "youtube") -> Dict:
    features = feature_engineer.compute_features_from_snapshots(snapshots, latest_vids)
    prediction = predictor.predict_trend(features)
    raw_data = {
        "name": creator.name,
        "handle": f"@{resolved.get('handle') or creator.channel_id}",
        "niche": "YouTube",
        "location": resolved.get("country") or "Global",
        "followers": features["followers"],
        "followersDisplay": features["followersDisplay"],
        "cadence": feature_engineer.estimate_cadence_label(latest_vids, snapshots),
        "thumbnail": creator.profile_picture_url or resolved.get("thumbnail", ""),
        **features
    }
    expl = explanation_engine.generate_explanation(raw_data, {"prob_score": prediction["prob_score"]})
    hist = [s.subscriber_count for s in reversed(snapshots)] if snapshots else [resolved["subscriber_count"]]
    engagement_history = feature_engineer.build_engagement_history(latest_vids, features["engagement_rate"])
    forecasts = feature_engineer.build_follower_forecast(hist[-1] if hist else resolved["subscriber_count"], features, prediction["confidence"])

    return {
        "username": creator.channel_id,
        "platform": platform,
        "name": creator.name,
        "handle": raw_data["handle"],
        "niche": raw_data["niche"],
        "location": raw_data["location"],
        "followers": features["followers"],
        "followersDisplay": features["followersDisplay"],
        "cadence": raw_data["cadence"],
        "thumbnail": raw_data["thumbnail"],
        "probScore": prediction["prob_score"],
        "prob12": min(99, max(1, prediction["prob_score"] + 6)),
        "prob24": min(99, max(1, prediction["prob_score"] + 10)),
        "velocity": features["velocity_7d"],
        "engagementRate": features["engagement_rate"],
        "viralityIndex": features["virality_score"],
        "consistencyScore": features["consistency_score"],
        "drivers": expl["drivers"],
        "risks": expl["risks"],
        "collab": expl["collab"],
        "explanation": expl["explanation"],
        "trajectoryHistorical": hist,
        "trajectoryForecast6": forecasts["6m"],
        "trajectoryForecast12": forecasts["12m"],
        "trajectoryForecast24": forecasts["24m"],
        "engagementData": engagement_history
    }

@router.post("/creators/track")
async def track_creator(channel_id: str, db: Session = Depends(get_db)):
    """Track a new YouTube creator by channel ID."""
    existing = _find_creator(db, channel_id)
    creator, resolved, snapshots = _upsert_creator_with_snapshot(db, channel_id, force_snapshot=existing is None)
    if not creator or not resolved:
        raise HTTPException(status_code=404, detail="Channel not found or API error. Check your YOUTUBE_API_KEY and input.")

    status = "already_tracked" if existing is not None else "tracked"
    return {
        "status": status,
        "creator_id": creator.id,
        "name": creator.name,
        "channel_id": creator.channel_id,
        "followers": resolved["subscriber_count"]
    }

@router.get("/trending")
async def get_trending(db: Session = Depends(get_db)):
    """Return list of tracked creators ranked by real trend score."""
    creators = db.query(Creator).all()
    results = []

    for creator in creators:
        snapshots = db.query(Snapshot).filter(Snapshot.creator_id == creator.id).order_by(Snapshot.timestamp.desc()).limit(30).all()
        if not snapshots:
            continue
            
        latest_vids = get_recent_videos(creator.channel_id)
        features = feature_engineer.compute_features_from_snapshots(snapshots, latest_vids)
        prediction = predictor.predict_trend(features)
        latest_snap = snapshots[0]
        
        results.append({
            "channel_id": creator.channel_id,
            "name": creator.name,
            "thumbnail": creator.profile_picture_url or "",
            "subscribers": latest_snap.subscriber_count,
            "trend_score": prediction["trend_score"],
            "confidence": prediction["confidence"],
            "breakout_probability": prediction["prob_score"]
        })

    results.sort(key=lambda x: x["trend_score"], reverse=True)
    return results

@router.post("/api/analyze")
async def analyze_creator(username: str, platform: str = "youtube", db: Session = Depends(get_db)):
    """Bridge to existing frontend. Uses live YouTube data when possible, simulation only as a last fallback."""
    if platform != "youtube":
        return simulation.generate_simulated_analysis(username, platform)

    creator, resolved, snapshots = _upsert_creator_with_snapshot(db, username)
    if creator and resolved:
        latest_vids = get_recent_videos(creator.channel_id, max_results=12)
        return _build_analysis_payload(creator, resolved, snapshots, latest_vids, platform="youtube")

    existing_creator = _find_creator(db, username)
    if existing_creator:
        refreshed = get_channel_stats(existing_creator.channel_id)
        if refreshed:
            snapshots = db.query(Snapshot).filter(Snapshot.creator_id == existing_creator.id).order_by(Snapshot.timestamp.desc()).limit(30).all()
            latest_vids = get_recent_videos(existing_creator.channel_id, max_results=12)
            return _build_analysis_payload(existing_creator, refreshed, snapshots, latest_vids, platform="youtube")

    return simulation.generate_simulated_analysis(username, platform)

@router.get("/api/health")
async def health():
    model_loaded = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "xgboost_model.pkl"))
    return {"status": "ok", "model_loaded": model_loaded, "realtime_youtube_enabled": True}
