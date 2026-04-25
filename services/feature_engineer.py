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


def _median(values: List[float]) -> float:
    cleaned = sorted(v for v in values if v is not None)
    if not cleaned:
        return 0.0
    mid = len(cleaned) // 2
    if len(cleaned) % 2:
        return float(cleaned[mid])
    return float((cleaned[mid - 1] + cleaned[mid]) / 2.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_ratio(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return _clamp(value / target, 0.0, 1.5)


def compute_features_from_snapshots(
    snapshots: List[Snapshot],
    latest_videos: List[Dict] = None,
    analytics_by_video: Optional[Dict[str, Dict]] = None
) -> Dict[str, Any]:
    """Compute creator features using public data first, with optional owner analytics."""

    if not snapshots:
        return get_default_features()

    latest_videos = latest_videos or []
    analytics_by_video = analytics_by_video or {}

    if len(snapshots) < 2:
        latest = snapshots[0]
        video_metrics = summarize_video_signals(latest_videos, latest.subscriber_count, analytics_by_video)
        return {
            "velocity_7d": estimate_velocity_from_videos(latest, latest_videos),
            "velocity_30d": estimate_velocity_from_videos(latest, latest_videos, window_days=30),
            "acceleration": 50.0,
            "engagement_rate": video_metrics["engagement_rate"],
            "virality_score": video_metrics["virality_score"],
            "consistency_score": video_metrics["consistency_score"],
            "niche_momentum": video_metrics["niche_momentum"],
            "audience_quality": video_metrics["audience_quality"],
            "confidence": video_metrics["confidence"],
            "followers": latest.subscriber_count,
            "followersDisplay": format_followers(latest.subscriber_count),
            **video_metrics["diagnostics"]
        }

    sorted_snaps = sorted(snapshots, key=lambda x: x.timestamp)
    latest = sorted_snaps[-1]
    previous = sorted_snaps[-2]
    follower_base = max(latest.subscriber_count, previous.subscriber_count, 1)

    time_delta_days = max(1, (latest.timestamp - previous.timestamp).total_seconds() / 86400)
    sub_velocity = (latest.subscriber_count - previous.subscriber_count) / time_delta_days
    growth_ratio = max(0.0, (latest.subscriber_count - previous.subscriber_count) / max(previous.subscriber_count, 1))

    velocity_7d = int(sub_velocity * 7)
    velocity_30d = int(sub_velocity * 30)

    acceleration = 50.0
    if len(sorted_snaps) >= 3:
        older = sorted_snaps[-3]
        older_time_delta = max(1, (previous.timestamp - older.timestamp).total_seconds() / 86400)
        older_velocity = (previous.subscriber_count - older.subscriber_count) / older_time_delta
        if older_velocity > 0:
            accel_change = (sub_velocity - older_velocity) / older_velocity
            acceleration = min(100.0, max(0.0, 50.0 + (accel_change * 50.0)))

    video_metrics = summarize_video_signals(latest_videos, latest.subscriber_count, analytics_by_video)
    engagement_rate = video_metrics["engagement_rate"]
    virality_score = video_metrics["virality_score"]

    consistency_score = max(70.0, video_metrics["consistency_score"])
    if len(sorted_snaps) >= 3:
        growths = [
            max(0, sorted_snaps[i].subscriber_count - sorted_snaps[i - 1].subscriber_count)
            for i in range(1, len(sorted_snaps))
        ]
        avg_growth = sum(growths) / len(growths)
        if avg_growth > 0:
            std_dev = math.sqrt(sum((g - avg_growth) ** 2 for g in growths) / len(growths))
            coeff_var = std_dev / avg_growth
            consistency_score = max(35.0, min(100.0, 92.0 - (coeff_var * 18.0), consistency_score))

    size_norm = min(100.0, max(0.0, math.log10(follower_base) * 20.0))
    velocity_norm = min(100.0, max(0.0, math.log10(max(abs(velocity_7d), 1)) * 28.0))
    growth_norm = min(100.0, max(0.0, growth_ratio * 1500))

    niche_momentum = round(
        min(100.0, max(35.0, 35.0 + (growth_norm * 0.18) + (velocity_norm * 0.15) + (video_metrics["niche_momentum"] * 0.45))),
        1
    )
    audience_quality = round(
        min(
            100.0,
            max(
                45.0,
                35.0 + (video_metrics["audience_quality"] * 0.55) + (engagement_rate * 1.1) + (consistency_score * 0.12) - (size_norm * 0.06)
            )
        ),
        1
    )

    confidence = min(1.0, max(video_metrics["confidence"], 0.2 + (len(sorted_snaps) / 12.0)))

    return {
        "velocity_7d": velocity_7d,
        "velocity_30d": velocity_30d,
        "acceleration": round(acceleration, 1),
        "engagement_rate": engagement_rate,
        "virality_score": virality_score,
        "consistency_score": round(consistency_score, 1),
        "niche_momentum": niche_momentum,
        "audience_quality": audience_quality,
        "confidence": round(confidence, 2),
        "followers": latest.subscriber_count,
        "followersDisplay": format_followers(latest.subscriber_count),
        **video_metrics["diagnostics"]
    }


def summarize_video_signals(
    latest_videos: List[Dict] = None,
    followers: int = 0,
    analytics_by_video: Optional[Dict[str, Dict]] = None
) -> Dict[str, Any]:
    latest_videos = latest_videos or []
    analytics_by_video = analytics_by_video or {}

    if not latest_videos:
        return {
            "engagement_rate": 5.0,
            "virality_score": 5.0,
            "consistency_score": 55.0,
            "niche_momentum": 50.0,
            "audience_quality": 60.0,
            "confidence": 0.25,
            "diagnostics": {
                "comment_rate_per_1k": 0.0,
                "like_rate_per_1k": 0.0,
                "share_rate_per_1k": 0.0,
                "avg_view_percentage": 0.0,
                "subscriber_conversion_per_1k": 0.0,
                "recent_view_velocity": 0.0,
                "breakout_ratio": 1.0,
                "analytics_coverage": 0.0,
                "retention_score": 50.0
            }
        }

    now = datetime.utcnow()
    merged = []
    analytics_hits = 0

    for video in latest_videos:
        analytics = analytics_by_video.get(video.get("video_id", ""), {})
        if analytics:
            analytics_hits += 1

        published_at = _parse_published_at(video.get("published_at")) or now
        age_days = max(1.0, (now - published_at).total_seconds() / 86400.0)
        views = float(analytics.get("views", video.get("views", 0)) or 0)
        likes = float(analytics.get("likes", video.get("likes", 0)) or 0)
        comments = float(analytics.get("comments", video.get("comments", 0)) or 0)
        shares = float(analytics.get("shares", 0) or 0)
        duration_seconds = max(float(video.get("duration_seconds", 0) or 0), 1.0)
        avg_view_duration = float(analytics.get("averageViewDuration", 0) or 0)
        avg_view_percentage = float(analytics.get("averageViewPercentage", 0) or 0)
        subscribers_gained = float(analytics.get("subscribersGained", 0) or 0)
        subscribers_lost = float(analytics.get("subscribersLost", 0) or 0)

        merged.append({
            "views_per_day": views / age_days,
            "view_ratio": views / max(float(followers), 1.0),
            "like_rate_per_1k": (likes / max(views, 1.0)) * 1000.0,
            "comment_rate_per_1k": (comments / max(views, 1.0)) * 1000.0,
            "share_rate_per_1k": (shares / max(views, 1.0)) * 1000.0,
            "avg_view_percentage": avg_view_percentage,
            "avg_view_duration_ratio": avg_view_duration / duration_seconds if avg_view_duration else 0.0,
            "subscriber_conversion_per_1k": ((subscribers_gained - subscribers_lost) / max(views, 1.0)) * 1000.0
        })

    view_velocity = [item["views_per_day"] for item in merged]
    view_ratio = [item["view_ratio"] for item in merged]
    like_rates = [item["like_rate_per_1k"] for item in merged]
    comment_rates = [item["comment_rate_per_1k"] for item in merged]
    share_rates = [item["share_rate_per_1k"] for item in merged]
    avg_view_percentages = [item["avg_view_percentage"] for item in merged if item["avg_view_percentage"] > 0]
    duration_ratios = [item["avg_view_duration_ratio"] for item in merged if item["avg_view_duration_ratio"] > 0]
    subscriber_conversions = [item["subscriber_conversion_per_1k"] for item in merged if item["subscriber_conversion_per_1k"] != 0]

    recent_median = _median(view_velocity[:3])
    older_median = max(_median(view_velocity[3:]) if len(view_velocity) > 3 else recent_median, 1.0)
    breakout_ratio = recent_median / older_median
    cadence_consistency = estimate_consistency_from_videos(latest_videos)

    engagement_score = (
        (_normalize_ratio(_median(comment_rates), 8.0) * 32.0) +
        (_normalize_ratio(_median(like_rates), 45.0) * 24.0) +
        (_normalize_ratio(_median(share_rates), 2.0) * 14.0) +
        (_normalize_ratio(_median(view_ratio), 0.35) * 20.0) +
        (_normalize_ratio(_median(avg_view_percentages), 45.0) * 10.0)
    )
    retention_score = (
        (_normalize_ratio(_median(avg_view_percentages), 50.0) * 60.0) +
        (_normalize_ratio(_median(duration_ratios), 0.45) * 40.0)
    ) if avg_view_percentages or duration_ratios else 50.0
    virality_score = _clamp(
        (_normalize_ratio(max(view_ratio) if view_ratio else 0.0, 0.75) * 7.0) +
        (_normalize_ratio(breakout_ratio, 1.6) * 3.0),
        1.0,
        10.0
    )
    niche_momentum = _clamp(
        35.0 +
        (_normalize_ratio(recent_median, max(float(followers) / 30.0, 1.0)) * 25.0) +
        (_normalize_ratio(breakout_ratio, 1.5) * 20.0) +
        (_normalize_ratio(_median(view_ratio), 0.4) * 20.0),
        35.0,
        100.0
    )
    audience_quality = _clamp(
        40.0 +
        (_normalize_ratio(_median(comment_rates), 8.0) * 18.0) +
        (_normalize_ratio(_median(share_rates), 2.0) * 10.0) +
        (_normalize_ratio(_median(subscriber_conversions), 2.0) * 18.0) +
        (_normalize_ratio(retention_score, 60.0) * 14.0),
        40.0,
        100.0
    )
    confidence = _clamp(
        0.28 +
        (min(len(latest_videos), 12) / 24.0) +
        ((analytics_hits / max(len(latest_videos), 1)) * 0.25),
        0.25,
        0.95
    )

    return {
        "engagement_rate": round(_clamp(engagement_score / 5.0, 2.0, 20.0), 1),
        "virality_score": round(virality_score, 1),
        "consistency_score": round(cadence_consistency, 1),
        "niche_momentum": round(niche_momentum, 1),
        "audience_quality": round(audience_quality, 1),
        "confidence": round(confidence, 2),
        "diagnostics": {
            "comment_rate_per_1k": round(_median(comment_rates), 2),
            "like_rate_per_1k": round(_median(like_rates), 2),
            "share_rate_per_1k": round(_median(share_rates), 2),
            "avg_view_percentage": round(_median(avg_view_percentages), 2),
            "subscriber_conversion_per_1k": round(_median(subscriber_conversions), 2),
            "recent_view_velocity": round(recent_median, 2),
            "breakout_ratio": round(breakout_ratio, 2),
            "analytics_coverage": round(analytics_hits / max(len(latest_videos), 1), 2),
            "retention_score": round(retention_score, 2)
        }
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
    engagement_factor = max(0.4, min(1.25, float(features.get("engagement_rate", 5.0)) / 10.0))
    momentum_factor = max(0.6, min(1.5, float(features.get("breakout_ratio", 1.0))))
    monthly_growth = max(0.0, (weekly_growth * 4.0 * confidence_factor * engagement_factor * momentum_factor) / max(current, 1))
    monthly_growth = min(0.22, monthly_growth)

    def project(months: int) -> List[int]:
        values = []
        followers = float(current)
        growth = monthly_growth
        for _ in range(months):
            followers = followers * (1.0 + growth)
            values.append(int(round(followers)))
            growth = max(0.0, min(0.22, growth * (1.0 + acceleration * 0.35)))
        return values

    return {
        "6m": project(6),
        "12m": project(12),
        "24m": project(24)
    }


def get_default_features() -> Dict[str, Any]:
    return {
        "velocity_7d": 0,
        "velocity_30d": 0,
        "acceleration": 50.0,
        "engagement_rate": 5.0,
        "virality_score": 5.0,
        "consistency_score": 50.0,
        "niche_momentum": 50.0,
        "audience_quality": 75.0,
        "confidence": 0.1,
        "followers": 0,
        "followersDisplay": "0",
        "comment_rate_per_1k": 0.0,
        "like_rate_per_1k": 0.0,
        "share_rate_per_1k": 0.0,
        "avg_view_percentage": 0.0,
        "subscriber_conversion_per_1k": 0.0,
        "recent_view_velocity": 0.0,
        "breakout_ratio": 1.0,
        "analytics_coverage": 0.0,
        "retention_score": 50.0
    }


def format_followers(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)
