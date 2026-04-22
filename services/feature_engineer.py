import logging
from typing import Dict, Any, List
from db.models import Snapshot

logger = logging.getLogger("creatorrocket")

def compute_features_from_snapshots(snapshots: List[Snapshot], latest_videos: List[Dict] = None) -> Dict[str, Any]:
    """Compute real ML features from historical snapshot data."""
    
    if not snapshots or len(snapshots) < 2:
        return get_default_features()
        
    # Sort snapshots chronologically (oldest first)
    sorted_snaps = sorted(snapshots, key=lambda x: x.timestamp)
    latest = sorted_snaps[-1]
    previous = sorted_snaps[-2]
    
    # Time delta in days
    time_delta_days = max(1, (latest.timestamp - previous.timestamp).total_seconds() / 86400)
    
    # Velocity (growth per day)
    sub_velocity = (latest.subscriber_count - previous.subscriber_count) / time_delta_days
    
    # Velocity 7d and 30d proxy (normalize to weekly/monthly)
    velocity_7d = int(sub_velocity * 7)
    velocity_30d = int(sub_velocity * 30)
    
    # Acceleration (comparing recent growth to older growth if available)
    acceleration = 50.0 # Default neutral
    if len(sorted_snaps) >= 3:
        older = sorted_snaps[-3]
        older_time_delta = max(1, (previous.timestamp - older.timestamp).total_seconds() / 86400)
        older_velocity = (previous.subscriber_count - older.subscriber_count) / older_time_delta
        if older_velocity > 0:
            accel_change = (sub_velocity - older_velocity) / older_velocity
            acceleration = min(100, max(0, 50 + (accel_change * 50)))
    
    # Engagement Proxy (Average Views per Subscriber)
    engagement_rate = 5.0
    if latest.subscriber_count > 0 and latest_videos:
        avg_views = sum(v.get("views", 0) for v in latest_videos) / len(latest_videos)
        engagement_rate = round((avg_views / latest.subscriber_count) * 100, 1)
        
    # Virality Score (Max views on recent video vs subs)
    virality_score = 5.0
    if latest.subscriber_count > 0 and latest_videos:
        max_views = max(v.get("views", 0) for v in latest_videos)
        virality_score = round((max_views / latest.subscriber_count) * 10, 1)
        
    # Consistency Score (Based on snapshot frequency and growth stability)
    consistency_score = 70 # Base score for having snapshots
    if len(sorted_snaps) >= 3:
        growths = [(sorted_snaps[i].subscriber_count - sorted_snaps[i-1].subscriber_count) for i in range(1, len(sorted_snaps))]
        avg_growth = sum(growths) / len(growths)
        if avg_growth > 0:
            variance = sum((g - avg_growth)**2 for g in growths) / len(growths)
            consistency_score = int(max(30, min(100, 100 - (variance / avg_growth) * 10)))
    
    # Confidence Score (How reliable is this data?)
    confidence = min(1.0, len(sorted_snaps) / 10.0) # Maxes out at 10 snapshots
    
    return {
        "velocity_7d": velocity_7d,
        "velocity_30d": velocity_30d,
        "acceleration": round(acceleration, 1),
        "engagement_rate": engagement_rate,
        "virality_score": virality_score,
        "consistency_score": consistency_score,
        "niche_momentum": 60.0, # Requires cross-channel analysis to calculate accurately
        "audience_quality": 75.0, # Requires bot audit to calculate accurately
        "confidence": confidence,
        "followers": latest.subscriber_count,
        "followersDisplay": format_followers(latest.subscriber_count)
    }

def get_default_features() -> Dict[str, Any]:
    """Fallback to neutral features if data is insufficient."""
    return {
        "velocity_7d": 0, "velocity_30d": 0, "acceleration": 50.0,
        "engagement_rate": 5.0, "virality_score": 5.0, "consistency_score": 50,
        "niche_momentum": 50.0, "audience_quality": 75.0,
        "confidence": 0.1,
        "followers": 0, "followersDisplay": "0"
    }

def format_followers(count: int) -> str:
    if count >= 1_000_000: return f"{count / 1_000_000:.1f}M"
    if count >= 1_000: return f"{count / 1_000:.1f}K"
    return str(count)