import logging
import time
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from data.youtube_analytics import get_recent_video_analytics
from data.youtube_client import get_channel_stats, get_recent_videos
from db.database import SessionLocal
from db.models import Creator, Snapshot
from services import feature_engineer, predictor
from services.training_pipeline import record_feature_observation, refresh_training_pipeline

logger = logging.getLogger("creatorrocket")


def fetch_scheduled_stats():
    """Background task to fetch stats, collect feature observations, and refresh the model."""
    logger.info("Scheduler running: Fetching YouTube stats...")
    db: Session = SessionLocal()

    try:
        creators = db.query(Creator).all()
        for creator in creators:
            logger.info("Fetching updates for %s (%s)", creator.name, creator.channel_id)
            stats = get_channel_stats(creator.channel_id)

            if stats:
                latest = (
                    db.query(Snapshot)
                    .filter(Snapshot.creator_id == creator.id)
                    .order_by(Snapshot.timestamp.desc())
                    .first()
                )
                changed = latest is None or (
                    latest.subscriber_count != stats["subscriber_count"]
                    or latest.view_count != stats["view_count"]
                    or latest.video_count != stats["video_count"]
                )
                stale = latest is None or (datetime.utcnow() - latest.timestamp > timedelta(hours=12))

                if changed or stale:
                    new_snapshot = Snapshot(
                        creator_id=creator.id,
                        subscriber_count=stats["subscriber_count"],
                        view_count=stats["view_count"],
                        video_count=stats["video_count"],
                    )
                    db.add(new_snapshot)
                    db.commit()
                    logger.info("Added new snapshot for %s", creator.name)
                else:
                    logger.info("No material change for %s; skipped snapshot write", creator.name)

                snapshots = (
                    db.query(Snapshot)
                    .filter(Snapshot.creator_id == creator.id)
                    .order_by(Snapshot.timestamp.desc())
                    .limit(30)
                    .all()
                )
                latest_videos = get_recent_videos(creator.channel_id, max_results=12)
                analytics_by_video = get_recent_video_analytics(creator.channel_id, latest_videos)
                features = feature_engineer.compute_features_from_snapshots(snapshots, latest_videos, analytics_by_video)
                prediction = predictor.predict_trend(features)
                record_feature_observation(
                    db,
                    creator,
                    features,
                    predictor_mode=prediction["predictor_mode"],
                    source="scheduler",
                )
            else:
                logger.warning("Skipped %s due to API error or missing data.", creator.name)

            time.sleep(1)

        training_status = refresh_training_pipeline(db)
        logger.info(
            "Training pipeline refreshed: %s observations, %s examples, model=%s",
            training_status.get("observation_count"),
            training_status.get("example_count"),
            training_status.get("model_available"),
        )
    except Exception as exc:
        logger.error("Scheduler error: %s", exc)
    finally:
        db.close()
