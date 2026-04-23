import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import Creator, Snapshot
from data.youtube_client import get_channel_stats, get_recent_videos
from services import feature_engineer, predictor, explanation_engine, simulation

logger = logging.getLogger("creatorrocket")
router = APIRouter()


@router.get("/api/search-channels")
async def search_channels_endpoint(q: str):
    """Search YouTube channels by name for the frontend autocomplete."""
    if not q or len(q) < 2:
        return []
    from data.youtube_client import search_channels
    return search_channels(q, max_results=5)

@router.post("/creators/track")
async def track_creator(channel_id: str, db: Session = Depends(get_db)):
    """Track a new YouTube creator by channel ID."""
    existing = db.query(Creator).filter(Creator.channel_id == channel_id).first()
    if existing:
        return {"status": "already_tracked", "creator_id": existing.id, "name": existing.name}

    stats = get_channel_stats(channel_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Channel not found or API error. Check your YOUTUBE_API_KEY and channel_id.")

    new_creator = Creator(
        channel_id=channel_id,
        name=stats["name"],
        platform="youtube",
        profile_picture_url=stats.get("thumbnail", "")
    )
    db.add(new_creator)
    db.commit()
    db.refresh(new_creator)

    initial_snap = Snapshot(
        creator_id=new_creator.id,
        subscriber_count=stats["subscriber_count"],
        view_count=stats["view_count"],
        video_count=stats["video_count"]
    )
    db.add(initial_snap)
    db.commit()

    return {"status": "tracked", "creator_id": new_creator.id, "name": new_creator.name}

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
            "subscribers": latest_snap.subscriber_count,
            "trend_score": prediction["trend_score"],
            "confidence": prediction["confidence"],
            "breakout_probability": prediction["prob_score"]
        })

    results.sort(key=lambda x: x["trend_score"], reverse=True)
    return results

@router.post("/api/analyze")
async def analyze_creator(username: str, platform: str = "youtube", db: Session = Depends(get_db)):
    """Bridge to existing frontend. Uses real data if tracked, else fallback simulation."""
    # Check if channel is tracked in DB
    creator = db.query(Creator).filter(Creator.channel_id == username).first()
    
    if creator:
        snapshots = db.query(Snapshot).filter(Snapshot.creator_id == creator.id).order_by(Snapshot.timestamp.desc()).limit(30).all()
        
        # AUTO-REFRESH: If we have less than 2 snapshots OR the latest is older than 10 mins, fetch fresh data!
        needs_update = False
        if len(snapshots) < 2:
            needs_update = True
        elif snapshots:
            from datetime import datetime, timedelta
            if datetime.utcnow() - snapshots[0].timestamp > timedelta(minutes=10):
                needs_update = True
                
        if needs_update:
            stats = get_channel_stats(creator.channel_id)
            if stats:
                new_snap = Snapshot(
                    creator_id=creator.id,
                    subscriber_count=stats["subscriber_count"],
                    view_count=stats["view_count"],
                    video_count=stats["video_count"]
                )
                db.add(new_snap)
                db.commit()
                # Refresh the snapshots list to include the brand new one
                snapshots = db.query(Snapshot).filter(Snapshot.creator_id == creator.id).order_by(Snapshot.timestamp.desc()).limit(30).all()
                logger.info(f"Auto-refreshed data for {creator.name}")
                
        latest_vids = get_recent_videos(creator.channel_id)
        features = feature_engineer.compute_features_from_snapshots(snapshots, latest_vids)
        prediction = predictor.predict_trend(features)
        
        raw_data = {
            "name": creator.name, "handle": f"@{creator.channel_id}", "niche": "YouTube",
            "location": "Global", "followers": features["followers"], "followersDisplay": features["followersDisplay"],
            "cadence": f"{snapshots[0].video_count} videos" if snapshots else "0 videos",
            "thumbnail": creator.profile_picture_url or "",  # ADD THIS
            **features
        }
        expl = explanation_engine.generate_explanation(raw_data, {"prob_score": prediction["prob_score"]})
        
        hist = [s.subscriber_count for s in reversed(snapshots)] if snapshots else [0]
        
        return {
            "username": username, "platform": "youtube", "name": creator.name, 
            "handle": f"@{creator.channel_id}", "niche": "YouTube", "location": "Global",
            "followers": features["followers"], "followersDisplay": features["followersDisplay"], "cadence": raw_data["cadence"],
            "probScore": prediction["prob_score"], "prob12": max(1, prediction["prob_score"] - 5), "prob24": min(99, prediction["prob_score"] + 8),
            "velocity": features["velocity_7d"], "engagementRate": features["engagement_rate"], "viralityIndex": features["virality_score"], "consistencyScore": features["consistency_score"],
            "drivers": expl["drivers"], "risks": expl["risks"], "collab": expl["collab"], "explanation": expl["explanation"],
            "trajectoryHistorical": hist,
            "trajectoryForecast6": [int(features["followers"] * (1.05**i)) for i in range(1,7)],
            "trajectoryForecast12": [int(features["followers"] * (1.05**i)) for i in range(1,13)],
            "trajectoryForecast24": [int(features["followers"] * (1.05**i)) for i in range(1,25)],
            "engagementData": [features["engagement_rate"]] * 13,
            "thumbnail": creator.profile_picture_url or ""
        }
    # WITH this:
    else:
        # Channel not yet tracked — auto-track it on first analyze request
        logger.info(f"Channel {username} not in DB. Auto-tracking now...")
        
        stats = get_channel_stats(username)
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"Channel '{username}' not found. Check the Channel ID and ensure your YOUTUBE_API_KEY is set."
            )
        
        # Create creator record
        new_creator = Creator(
            channel_id=username,
            name=stats["name"],
            platform="youtube",
            profile_picture_url=stats.get("thumbnail", "")
        )
        db.add(new_creator)
        db.commit()
        db.refresh(new_creator)
        
        # Save first snapshot
        initial_snap = Snapshot(
            creator_id=new_creator.id,
            subscriber_count=stats["subscriber_count"],
            view_count=stats["view_count"],
            video_count=stats["video_count"]
        )
        db.add(initial_snap)
        db.commit()
        
        logger.info(f"Auto-tracked {stats['name']}. Fetching first analysis...")
        
        # Now run the real analysis with the data we just fetched
        snapshots = db.query(Snapshot).filter(Snapshot.creator_id == new_creator.id).order_by(Snapshot.timestamp.desc()).limit(30).all()
        latest_vids = get_recent_videos(username)
        features = feature_engineer.compute_features_from_snapshots(snapshots, latest_vids)
        prediction = predictor.predict_trend(features)
        
        raw_data = {
            "name": stats["name"], "handle": f"@{username}", "niche": "YouTube",
            "location": "Global", "followers": features["followers"], "followersDisplay": features["followersDisplay"],
            "cadence": f"{snapshots[0].video_count} videos" if snapshots else "0 videos",
            **features
        }
        expl = explanation_engine.generate_explanation(raw_data, {"prob_score": prediction["prob_score"]})
        hist = [s.subscriber_count for s in reversed(snapshots)]
        
        return {
            "username": username, "platform": "youtube", "name": stats["name"],
            "handle": f"@{username}", "niche": "YouTube", "location": "Global",
            "followers": features["followers"], "followersDisplay": features["followersDisplay"],
            "cadence": raw_data["cadence"],
            "probScore": prediction["prob_score"], "prob12": max(1, prediction["prob_score"] - 5), "prob24": min(99, prediction["prob_score"] + 8),
            "velocity": features["velocity_7d"], "engagementRate": features["engagement_rate"],
            "viralityIndex": features["virality_score"], "consistencyScore": features["consistency_score"],
            "drivers": expl["drivers"], "risks": expl["risks"], "collab": expl["collab"], "explanation": expl["explanation"],
            "trajectoryHistorical": hist,
            "trajectoryForecast6": [int(features["followers"] * (1.05**i)) for i in range(1, 7)],
            "trajectoryForecast12": [int(features["followers"] * (1.05**i)) for i in range(1, 13)],
            "trajectoryForecast24": [int(features["followers"] * (1.05**i)) for i in range(1, 25)],
            "engagementData": [features["engagement_rate"]] * 13,
            "thumbnail": stats.get("thumbnail", ""),
            "first_analysis": True
        }

@router.get("/api/health")
async def health():
    model_loaded = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "xgboost_model.pkl"))
    return {"status": "ok", "model_loaded": model_loaded}
