import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from data.youtube_analytics import get_recent_video_analytics
from data.youtube_client import get_channel_stats, get_recent_videos, resolve_channel, search_channels
from data.youtube_oauth import create_authorization_url, disconnect_oauth, get_oauth_status, handle_oauth_callback
from db.database import get_db
from db.models import Creator, Snapshot
from services import explanation_engine, feature_engineer, predictor, simulation
from services.training_pipeline import get_training_status, record_feature_observation, refresh_training_pipeline

logger = logging.getLogger("creatorrocket")
router = APIRouter()


@router.get("/api/search-channels")
async def search_channels_endpoint(q: str):
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


def _build_data_quality(features: Dict, snapshots: List[Snapshot], latest_videos: List[Dict], training_status: Dict) -> Dict:
    snapshot_count = len(snapshots)
    recent_video_count = len(latest_videos)
    analytics_coverage = float(features.get("analytics_coverage", 0.0) or 0.0)
    observed_model_available = bool(training_status.get("model_available"))
    score = min(
        100,
        round(
            22
            + min(snapshot_count, 30) * 1.6
            + min(recent_video_count, 12) * 1.8
            + analytics_coverage * 24
            + (12 if observed_model_available else 0)
        ),
    )
    if score >= 80:
        label = "High"
    elif score >= 60:
        label = "Medium"
    else:
        label = "Developing"

    return {
        "score": score,
        "label": label,
        "snapshot_count": snapshot_count,
        "recent_video_count": recent_video_count,
        "analytics_coverage": round(analytics_coverage, 2),
        "owner_analytics_used": analytics_coverage > 0,
        "observed_model_available": observed_model_available,
        "training_example_count": training_status.get("example_count", 0),
    }


def _upsert_creator_with_snapshot(
    db: Session,
    creator_input: str,
    force_snapshot: bool = False,
) -> Tuple[Optional[Creator], Optional[Dict], List[Snapshot]]:
    resolved = resolve_channel(creator_input)
    if not resolved:
        return None, None, []

    creator = db.query(Creator).filter(Creator.channel_id == resolved["channel_id"]).first()
    if creator is None:
        creator = Creator(
            channel_id=resolved["channel_id"],
            name=resolved["name"],
            platform="youtube",
            profile_picture_url=resolved.get("thumbnail", ""),
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
            latest.subscriber_count != resolved["subscriber_count"]
            or latest.view_count != resolved["view_count"]
            or latest.video_count != resolved["video_count"]
        )
        stale = datetime.utcnow() - latest.timestamp > timedelta(hours=6)
        should_write_snapshot = changed or stale

    if should_write_snapshot:
        new_snapshot = Snapshot(
            creator_id=creator.id,
            subscriber_count=resolved["subscriber_count"],
            view_count=resolved["view_count"],
            video_count=resolved["video_count"],
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


def _build_analysis_payload(
    db: Session,
    creator: Creator,
    resolved: Dict,
    snapshots: List[Snapshot],
    latest_vids: List[Dict],
    platform: str = "youtube",
    source: str = "analysis",
) -> Dict:
    analytics_by_video = get_recent_video_analytics(creator.channel_id, latest_vids)
    features = feature_engineer.compute_features_from_snapshots(snapshots, latest_vids, analytics_by_video)
    prediction = predictor.predict_trend(features)
    record_feature_observation(db, creator, features, predictor_mode=prediction["predictor_mode"], source=source)
    training_status = get_training_status(db)
    data_quality = _build_data_quality(features, snapshots, latest_vids, training_status)

    raw_data = {
        "name": creator.name,
        "handle": f"@{resolved.get('handle') or creator.channel_id}",
        "niche": "YouTube",
        "location": resolved.get("country") or "Global",
        "followers": features["followers"],
        "followersDisplay": features["followersDisplay"],
        "cadence": feature_engineer.estimate_cadence_label(latest_vids, snapshots),
        "thumbnail": creator.profile_picture_url or resolved.get("thumbnail", ""),
        **features,
    }
    expl = explanation_engine.generate_explanation(raw_data, {"prob_score": prediction["prob_score"]})
    hist = [s.subscriber_count for s in reversed(snapshots)] if snapshots else [resolved["subscriber_count"]]
    engagement_history = feature_engineer.build_engagement_history(latest_vids, features["engagement_rate"])
    forecasts = feature_engineer.build_follower_forecast(
        hist[-1] if hist else resolved["subscriber_count"], features, prediction["confidence"]
    )

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
        "predictorMode": prediction.get("predictor_mode", "evidence_based_v2"),
        "predictionConfidence": prediction["confidence"],
        "modelBlend": prediction.get("model_blend", 0.0),
        "analyticsCoverage": features.get("analytics_coverage", 0.0),
        "dataQuality": data_quality,
        "trainingStatus": training_status,
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
        "engagementData": engagement_history,
    }


@router.get("/api/youtube/oauth/status")
async def youtube_oauth_status(db: Session = Depends(get_db)):
    return get_oauth_status(db)


@router.get("/api/youtube/oauth/start")
async def youtube_oauth_start(
    redirect_path: str = Query("/", description="Path to return to after connecting"),
    db: Session = Depends(get_db),
):
    try:
        url = create_authorization_url(db, redirect_path=redirect_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(url=url)


@router.get("/api/youtube/oauth/callback", response_class=HTMLResponse)
async def youtube_oauth_callback(code: str, state: str, db: Session = Depends(get_db)):
    try:
        result = handle_oauth_callback(db, code=code, state_value=state)
    except Exception as exc:
        return HTMLResponse(
            f"""
            <html><body style="font-family:Arial;padding:24px;background:#07070d;color:#f8fafc">
            <h2>Connection failed</h2>
            <p>{str(exc)}</p>
            <script>if(window.opener){{window.opener.postMessage({{"type":"creatorrocket-oauth","status":"error","message":{repr(str(exc))}}},"*");}}</script>
            </body></html>
            """,
            status_code=400,
        )

    return HTMLResponse(
        f"""
        <html>
        <body style="font-family:Arial;padding:24px;background:#07070d;color:#f8fafc">
          <h2>YouTube Analytics connected</h2>
          <p>{result["channel_name"]} is now linked to CreatorRocket.</p>
          <script>
            if (window.opener) {{
              window.opener.postMessage({{"type":"creatorrocket-oauth","status":"connected","channelName":{result["channel_name"]!r}}}, "*");
              window.close();
            }} else {{
              window.location.href = {result["redirect_path"]!r};
            }}
          </script>
        </body>
        </html>
        """
    )


@router.post("/api/youtube/oauth/disconnect")
async def youtube_oauth_disconnect(db: Session = Depends(get_db)):
    return disconnect_oauth(db)


@router.get("/api/model/status")
async def model_status(db: Session = Depends(get_db)):
    return get_training_status(db)


@router.post("/api/model/retrain")
async def retrain_model(db: Session = Depends(get_db)):
    return refresh_training_pipeline(db)


@router.post("/creators/track")
async def track_creator(channel_id: str, db: Session = Depends(get_db)):
    existing = _find_creator(db, channel_id)
    creator, resolved, _ = _upsert_creator_with_snapshot(db, channel_id, force_snapshot=existing is None)
    if not creator or not resolved:
        raise HTTPException(status_code=404, detail="Channel not found or API error. Check your YOUTUBE_API_KEY and input.")

    status = "already_tracked" if existing is not None else "tracked"
    return {
        "status": status,
        "creator_id": creator.id,
        "name": creator.name,
        "channel_id": creator.channel_id,
        "followers": resolved["subscriber_count"],
    }


@router.get("/trending")
async def get_trending(db: Session = Depends(get_db)):
    creators = db.query(Creator).all()
    results = []

    for creator in creators:
        snapshots = (
            db.query(Snapshot)
            .filter(Snapshot.creator_id == creator.id)
            .order_by(Snapshot.timestamp.desc())
            .limit(30)
            .all()
        )
        if not snapshots:
            continue

        latest_vids = get_recent_videos(creator.channel_id, max_results=12)
        analytics_by_video = get_recent_video_analytics(creator.channel_id, latest_vids)
        features = feature_engineer.compute_features_from_snapshots(snapshots, latest_vids, analytics_by_video)
        prediction = predictor.predict_trend(features)
        latest_snap = snapshots[0]
        record_feature_observation(db, creator, features, predictor_mode=prediction["predictor_mode"], source="trending")

        results.append(
            {
                "channel_id": creator.channel_id,
                "name": creator.name,
                "thumbnail": creator.profile_picture_url or "",
                "subscribers": latest_snap.subscriber_count,
                "trend_score": prediction["trend_score"],
                "confidence": prediction["confidence"],
                "breakout_probability": prediction["prob_score"],
            }
        )

    results.sort(key=lambda x: x["trend_score"], reverse=True)
    return results


@router.post("/api/analyze")
async def analyze_creator(username: str, platform: str = "youtube", db: Session = Depends(get_db)):
    if platform != "youtube":
        return simulation.generate_simulated_analysis(username, platform)

    creator, resolved, snapshots = _upsert_creator_with_snapshot(db, username)
    if creator and resolved:
        latest_vids = get_recent_videos(creator.channel_id, max_results=12)
        return _build_analysis_payload(db, creator, resolved, snapshots, latest_vids, platform="youtube", source="analysis")

    existing_creator = _find_creator(db, username)
    if existing_creator:
        refreshed = get_channel_stats(existing_creator.channel_id)
        if refreshed:
            snapshots = (
                db.query(Snapshot)
                .filter(Snapshot.creator_id == existing_creator.id)
                .order_by(Snapshot.timestamp.desc())
                .limit(30)
                .all()
            )
            latest_vids = get_recent_videos(existing_creator.channel_id, max_results=12)
            return _build_analysis_payload(
                db,
                existing_creator,
                refreshed,
                snapshots,
                latest_vids,
                platform="youtube",
                source="analysis",
            )

    return simulation.generate_simulated_analysis(username, platform)


@router.get("/api/health")
async def health(db: Session = Depends(get_db)):
    training_status = get_training_status(db)
    oauth_status = get_oauth_status(db)
    return {
        "status": "ok",
        "model_loaded": True,
        "realtime_youtube_enabled": True,
        "predictor_mode": "hybrid_observed_v1" if training_status.get("model_available") else "evidence_based_v2",
        "youtube_oauth_configured": oauth_status["configured"],
        "youtube_oauth_connected": oauth_status["connected"],
        "observed_model_available": training_status["model_available"],
        "training_examples": training_status["example_count"],
    }
