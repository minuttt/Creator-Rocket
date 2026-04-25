import math
from typing import Dict, Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _score_components(features: Dict[str, Any]) -> Dict[str, float]:
    followers = max(float(features.get("followers", 0)), 1.0)
    velocity_7d = max(float(features.get("velocity_7d", 0)), 0.0)
    velocity_30d = max(float(features.get("velocity_30d", 0)), 0.0)
    acceleration = _clamp(float(features.get("acceleration", 50.0)) / 100.0, 0.0, 1.0)

    relative_growth = _clamp((velocity_7d / followers) * 18.0, 0.0, 1.0)
    absolute_growth = _clamp(math.log10(max(velocity_30d, 1.0)) / 4.6, 0.0, 1.0)
    engagement = _clamp(float(features.get("engagement_rate", 0.0)) / 20.0, 0.0, 1.0)
    virality = _clamp(float(features.get("virality_score", 0.0)) / 10.0, 0.0, 1.0)
    consistency = _clamp(float(features.get("consistency_score", 0.0)) / 100.0, 0.0, 1.0)
    niche_momentum = _clamp(float(features.get("niche_momentum", 0.0)) / 100.0, 0.0, 1.0)
    audience_quality = _clamp(float(features.get("audience_quality", 0.0)) / 100.0, 0.0, 1.0)

    comment_rate = _clamp(float(features.get("comment_rate_per_1k", 0.0)) / 8.0, 0.0, 1.2)
    like_rate = _clamp(float(features.get("like_rate_per_1k", 0.0)) / 45.0, 0.0, 1.2)
    share_rate = _clamp(float(features.get("share_rate_per_1k", 0.0)) / 2.0, 0.0, 1.2)
    retention = _clamp(float(features.get("retention_score", 50.0)) / 100.0, 0.0, 1.0)
    breakout_ratio = _clamp(float(features.get("breakout_ratio", 1.0)) / 1.8, 0.0, 1.3)
    subscriber_conversion = _clamp(float(features.get("subscriber_conversion_per_1k", 0.0)) / 2.0, 0.0, 1.2)
    analytics_coverage = _clamp(float(features.get("analytics_coverage", 0.0)), 0.0, 1.0)

    return {
        "growth": (relative_growth * 0.55) + (absolute_growth * 0.25) + (acceleration * 0.20),
        "distribution": (virality * 0.40) + (breakout_ratio * 0.40) + (niche_momentum * 0.20),
        "engagement": (comment_rate * 0.40) + (like_rate * 0.20) + (share_rate * 0.20) + (engagement * 0.20),
        "retention": (retention * 0.75) + (analytics_coverage * 0.25),
        "audience": (subscriber_conversion * 0.40) + (audience_quality * 0.40) + (consistency * 0.20),
        "confidence": _clamp(float(features.get("confidence", 0.1)), 0.1, 1.0),
    }


def predict_trend(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evidence-based breakout predictor.

    This intentionally avoids relying on the synthetic sample XGBoost model as the
    primary source of truth. The score is driven by signals we can justify from
    either public YouTube data or owner analytics when available.
    """
    components = _score_components(features)

    evidence_signal = (
        (components["growth"] * 0.26) +
        (components["distribution"] * 0.24) +
        (components["engagement"] * 0.20) +
        (components["retention"] * 0.15) +
        (components["audience"] * 0.15)
    )

    calibrated = _sigmoid((evidence_signal - 0.58) * 5.6)
    confidence = components["confidence"]
    confidence_shrink = 0.52 + (confidence * 0.48)
    prob_score = int(round(_clamp(calibrated * confidence_shrink, 0.02, 0.98) * 100))

    trend_score = (
        (components["growth"] * 35.0) +
        (components["distribution"] * 25.0) +
        (components["engagement"] * 18.0) +
        (components["retention"] * 10.0) +
        (components["audience"] * 12.0)
    ) * confidence

    return {
        "prob_score": max(1, min(99, prob_score)),
        "trend_score": round(_clamp(trend_score, 0.0, 100.0), 2),
        "confidence": round(confidence, 2),
        "predictor_mode": "evidence_based_v2"
    }
