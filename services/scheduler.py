import logging
import time
from sqlalchemy.orm import Session
from db.database import SessionLocal
from db.models import Creator, Snapshot
from data.youtube_client import get_channel_stats

logger = logging.getLogger("creatorrocket")

def fetch_scheduled_stats():
    """Background task to fetch stats for all tracked creators."""
    logger.info("Scheduler running: Fetching YouTube stats...")
    db: Session = SessionLocal()
    
    try:
        creators = db.query(Creator).all()
        for creator in creators:
            logger.info(f"Fetching updates for {creator.name} ({creator.channel_id})")
            stats = get_channel_stats(creator.channel_id)
            
            if stats:
                # Append new snapshot (do NOT overwrite old data)
                new_snapshot = Snapshot(
                    creator_id=creator.id,
                    subscriber_count=stats["subscriber_count"],
                    view_count=stats["view_count"],
                    video_count=stats["video_count"]
                )
                db.add(new_snapshot)
                db.commit()
                logger.info(f"Added new snapshot for {creator.name}")
            else:
                logger.warning(f"Skipped {creator.name} due to API error or missing data.")
            
            # Basic rate limit handling for YouTube API (1 second pause between channels)
            time.sleep(1) 
            
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
    finally:
        db.close()