import math
import os
from typing import Any, Dict, Optional

import joblib

from services.training_pipeline import FEATURE_COLUMNS, MODEL_PATH

_MODEL_CACHE = {"mtime": None, "artifact": None}


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


def _load_observed_model() -> Optional[Dict]:
    if not os.path.exists(MODEL_PATH):
        _MODEL_CACHE["artifact"] = None
        _MODEL_CACHE["mtime"] = None
        return None

    mtime = os.path.getmtime(MODEL_PATH)
    if _MODEL_CACHE["artifact"] is not None and _MODEL_CACHE["mtime"] == mtime:
        return _MODEL_CACHE["artifact"]

    artifact = joblib.load(MODEL_PATH)
    _MODEL_CACHE["artifact"] = artifact
    _MODEL_CACHE["mtime"] = mtime
    return artifact


def _predict_observed_model(features: Dict[str, Any]) -> Optional[Dict[str, float]]:
    artifact = _load_observed_model()
    if not artifact:
        return None

    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns") or FEATURE_COLUMNS
    row = [[float(features.get(column, 0.0) or 0.0) for column in feature_columns]]
    probability = float(model.predict_proba(row)[0][1])
    return {
        "probability": probability,
        "example_count": int(artifact.get("example_count", 0)),
        "holdout_accuracy": float(artifact.get("holdout_accuracy") or 0.0),
        "holdout_auc": float(artifact.get("holdout_auc") or 0.0),
    }


def predict_trend(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hybrid predictor.

    The heuristic evidence stack remains the fallback when there is not enough
    tracked data. Once the app has observed enough real creator outcomes, a
    trained logistic model is blended in and becomes the primary probability
    signal.
    """
    components = _score_components(features)

    evidence_signal = (
        (components["growth"] * 0.26)
        + (components["distribution"] * 0.24)
        + (components["engagement"] * 0.20)
        + (components["retention"] * 0.15)
        + (components["audience"] * 0.15)
    )

    heuristic_probability = _sigmoid((evidence_signal - 0.58) * 5.6)
    confidence = components["confidence"]
    observed_prediction = _predict_observed_model(features)

    predictor_mode = "evidence_based_v2"
    blended_probability = heuristic_probability
    training_strength = 0.0
    if observed_prediction:
        example_count = observed_prediction["example_count"]
        training_strength = _clamp(example_count / 120.0, 0.18, 0.6)
        quality_bonus = _clamp(observed_prediction["holdout_accuracy"], 0.45, 0.85) - 0.45
        training_strength = _clamp(training_strength + quality_bonus * 0.4, 0.2, 0.75)
        blended_probability = (
            (heuristic_probability * (1.0 - training_strength))
            + (observed_prediction["probability"] * training_strength)
        )
        predictor_mode = "hybrid_observed_v1"

    confidence_shrink = 0.52 + (confidence * 0.48)
    prob_score = int(round(_clamp(blended_probability * confidence_shrink, 0.02, 0.98) * 100))

    trend_score = (
        (components["growth"] * 35.0)
        + (components["distribution"] * 25.0)
        + (components["engagement"] * 18.0)
        + (components["retention"] * 10.0)
        + (components["audience"] * 12.0)
    ) * confidence

    return {
        "prob_score": max(1, min(99, prob_score)),
        "trend_score": round(_clamp(trend_score, 0.0, 100.0), 2),
        "confidence": round(confidence, 2),
        "predictor_mode": predictor_mode,
        "model_blend": round(training_strength, 2),
        "model_probability": round(observed_prediction["probability"], 4) if observed_prediction else None,
        "heuristic_probability": round(heuristic_probability, 4),
    }
