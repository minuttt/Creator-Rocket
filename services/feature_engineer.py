import logging
import math
from datetime import datetime
from typing import Dict, Any, List, Optional
from db.models import Snapshot

logger = logging.getLogger("creatorrocket")

def _parse_published_at(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None

def compute_features_from_snapshots(snapshots: List[Snapshot], latest_videos: List[Dict] = None) -> Dict[str, Any]:
    """Compute real ML features from historical snapshot data."""

    if not snapshots:
        return get_default_features()

    if len(snapshots) < 2:
        latest = snapshots[0]
        engagement_rate = 5.0
        virality_score = 5.0
        if latest.subscriber_count > 0 and latest_videos:
            avg_views = sum(v.get("views", 0) for v in latest_videos) / len(latest_videos)
            max_views = max(v.get("views", 0) for v in latest_videos)
            engagement_rate = round(max(2.0, min(20.0, (avg_views / latest.subscriber_count) * 100)), 1)
            virality_score = round(max(1.0, min(10.0, (max_views / latest.subscriber_count) * 10)), 1)

        return {
            "velocity_7d": estimate_velocity_from_videos(latest, latest_videos),
            "velocity_30d": estimate_velocity_from_videos(latest, latest_videos, window_days=30),
            "acceleration": 50.0,
            "engagement_rate": engagement_rate,
            "virality_score": virality_score,
            "consistency_score": estimate_consistency_from_videos(latest_videos),
            "niche_momentum": 55.0,
            "audience_quality": round(min(100.0, 58.0 + (engagement_rate * 1.3)), 1),
            "confidence": 0.35,
            "followers": latest.subscriber_count,
            "followersDisplay": format_followers(latest.subscriber_count)
        }
        
    # Sort snapshots chronologically (oldest first)
    sorted_snaps = sorted(snapshots, key=lambda x: x.timestamp)
    latest = sorted_snaps[-1]
    previous = sorted_snaps[-2]
    follower_base = max(latest.subscriber_count, previous.subscriber_count, 1)
    
    # Time delta in days
    time_delta_days = max(1, (latest.timestamp - previous.timestamp).total_seconds() / 86400)
    
    # Velocity (growth per day)
    sub_velocity = (latest.subscriber_count - previous.subscriber_count) / time_delta_days
    growth_ratio = max(0.0, (latest.subscriber_count - previous.subscriber_count) / max(previous.subscriber_count, 1))
    
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
    
    # Engagement Proxy:
    # Blend subscriber-relative engagement with a size-aware absolute view score
    # so larger creators are not unfairly crushed by denominator effects alone.
    engagement_rate = 5.0
    if latest.subscriber_count > 0 and latest_videos:
        avg_views = sum(v.get("views", 0) for v in latest_videos) / len(latest_videos)
        relative_engagement = (avg_views / latest.subscriber_count) * 100
        absolute_view_signal = min(20.0, math.log10(max(avg_views, 1)) * 3.2)
        growth_signal = min(20.0, growth_ratio * 400)
        engagement_rate = round(
            max(2.0, min(20.0, (relative_engagement * 0.6) + (absolute_view_signal * 0.25) + (growth_signal * 0.15))),
            1
        )
        
    # Virality Score:
    # Account for both relative breakout and the absolute size of recent reach.
    virality_score = 5.0
    if latest.subscriber_count > 0 and latest_videos:
        max_views = max(v.get("views", 0) for v in latest_videos)
        relative_virality = (max_views / latest.subscriber_count) * 10
        absolute_virality = min(10.0, math.log10(max(max_views, 1)) * 1.5)
        virality_score = round(max(1.0, min(10.0, (relative_virality * 0.7) + (absolute_virality * 0.3))), 1)
        
    # Consistency Score (based on snapshot frequency and normalized growth stability)
    consistency_score = 70
    if len(sorted_snaps) >= 3:
        growths = [
            max(0, sorted_snaps[i].subscriber_count - sorted_snaps[i - 1].subscriber_count)
            for i in range(1, len(sorted_snaps))
        ]
        avg_growth = sum(growths) / len(growths)
        if avg_growth > 0:
            std_dev = math.sqrt(sum((g - avg_growth) ** 2 for g in growths) / len(growths))
            coeff_var = std_dev / avg_growth
            consistency_score = int(max(35, min(100, 92 - (coeff_var * 18))))

    size_norm = min(100.0, max(0.0, math.log10(follower_base) * 20.0))
    velocity_norm = min(100.0, max(0.0, math.log10(max(abs(velocity_7d), 1)) * 28.0))
    growth_norm = min(100.0, max(0.0, growth_ratio * 1500))
    acceleration_norm = min(100.0, max(0.0, acceleration))
    niche_momentum = round(min(100.0, max(35.0, 45.0 + (growth_norm * 0.25) + (velocity_norm * 0.2))), 1)
    audience_quality = round(min(100.0, max(45.0, 55.0 + (engagement_rate * 1.2) + (consistency_score * 0.15) - (size_norm * 0.08))), 1)
    
    # Confidence Score (How reliable is this data?)
    confidence = min(1.0, 0.2 + (len(sorted_snaps) / 12.0))
    
    return {
        "velocity_7d": velocity_7d,
        "velocity_30d": velocity_30d,
        "acceleration": round(acceleration, 1),
        "engagement_rate": engagement_rate,
        "virality_score": virality_score,
        "consistency_score": consistency_score,
        "niche_momentum": niche_momentum,
        "audience_quality": audience_quality,
        "confidence": confidence,
        "followers": latest.subscriber_count,
        "followersDisplay": format_followers(latest.subscriber_count)
    }

def estimate_velocity_from_videos(snapshot: Snapshot, latest_videos: List[Dict] = None, window_days: int = 7) -> int:
    if not latest_videos or snapshot.subscriber_count <= 0:
        return 0
    now = datetime.utcnow()
    cutoff_days = max(window_days, 1)
    recent = []
    for video in latest_videos:
        published_at = _parse_published_at(video.get("published_at"))
        if published_at is None:
            continue
        if (now - published_at).days <= cutoff_days:
            recent.append(video)
    if not recent:
        recent = latest_videos[: min(len(latest_videos), 4)]
    total_views = sum(v.get("views", 0) for v in recent)
    engagement_factor = min(0.12, total_views / max(snapshot.subscriber_count * 40, 1))
    estimated = snapshot.subscriber_count * engagement_factor
    return int(max(0, estimated if window_days == 30 else estimated / 4))

def estimate_consistency_from_videos(latest_videos: List[Dict] = None) -> int:
    if not latest_videos or len(latest_videos) < 2:
        return 55
    dates = sorted([d for d in (_parse_published_at(v.get("published_at")) for v in latest_videos) if d], reverse=True)
    if len(dates) < 2:
        return 55
    gaps = [(dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)]
    avg_gap = sum(gaps) / len(gaps)
    variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
    gap_penalty = min(35.0, avg_gap * 1.5)
    variance_penalty = min(25.0, math.sqrt(variance) * 2.5)
    return int(max(35, min(95, 92 - gap_penalty - variance_penalty)))

def estimate_cadence_label(latest_videos: List[Dict] = None, snapshots: List[Snapshot] = None) -> str:
    if latest_videos:
        dates = sorted([d for d in (_parse_published_at(v.get("published_at")) for v in latest_videos) if d], reverse=True)
        if len(dates) >= 2:
            gaps = [(dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)]
            avg_gap = max(1, round(sum(gaps) / len(gaps)))
            per_month = max(1, round(30 / avg_gap))
            return f"~{per_month} videos/month"
        return f"{len(latest_videos)} recent videos"
    if snapshots:
        return f"{snapshots[0].video_count} total videos"
    return "Unknown cadence"

def build_engagement_history(latest_videos: List[Dict] = None, fallback_rate: float = 5.0) -> List[float]:
    if latest_videos:
        ordered = sorted(
            latest_videos,
            key=lambda v: _parse_published_at(v.get("published_at")) or datetime.utcnow()
        )
        history = []
        for video in ordered[-12:]:
            views = max(video.get("views", 0), 0)
            likes = max(video.get("likes", 0), 0)
            comments = max(video.get("comments", 0), 0)
            quality_rate = min(20.0, (likes * 0.8 + comments * 4.0) / max(views, 1) * 100)
            history.append(round(max(0.5, quality_rate), 1))
        if history:
            while len(history) < 12:
                history.insert(0, history[0])
            return history
    return [round(fallback_rate, 1)] * 12

def build_follower_forecast(current_followers: int, features: Dict[str, Any], confidence: float) -> Dict[str, List[int]]:
    current = max(int(current_followers), 0)
    weekly_growth = max(float(features.get("velocity_7d", 0)), 0.0)
    acceleration = ((float(features.get("acceleration", 50.0)) - 50.0) / 50.0) * 0.08
    confidence_factor = max(0.25, min(1.0, confidence))
    engagement_factor = max(0.4, min(1.2, float(features.get("engagement_rate", 5.0)) / 10.0))
    monthly_growth = max(0.0, (weekly_growth * 4.0 * confidence_factor * engagement_factor) / max(current, 1))
    monthly_growth = min(0.18, monthly_growth)

    def project(months: int) -> List[int]:
        values = []
        followers = float(current)
        growth = monthly_growth
        for _ in range(months):
            followers = followers * (1.0 + growth)
            values.append(int(round(followers)))
            growth = max(0.0, min(0.2, growth * (1.0 + acceleration * 0.35)))
        return values

    return {
        "6m": project(6),
        "12m": project(12),
        "24m": project(24)
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
