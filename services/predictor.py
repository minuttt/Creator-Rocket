import os
import logging
import math
from typing import Dict, Any
import numpy as np
import joblib

logger = logging.getLogger("creatorrocket")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
FEATURE_NAMES = [
    "velocity_7d", "velocity_30d", "acceleration", "engagement_rate",
    "virality_score", "consistency_score", "niche_momentum", "audience_quality"
]

model = None

def load_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                logger.info("XGBoost model loaded from disk")
            except Exception as exc:
                logger.warning("Unable to load model from disk, falling back to heuristic calibration: %s", exc)
                model = False
        else:
            logger.error("Model file not found! Run application startup to generate base model.")
            model = False
    return None if model is False else model

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _heuristic_probability(features: Dict[str, Any]) -> int:
    followers = max(float(features.get("followers", 0)), 1.0)
    follower_scale = _clamp(math.log10(followers) / 5.0, 0.0, 1.0)
    velocity_7d = max(float(features.get("velocity_7d", 0)), 0.0)
    velocity_30d = max(float(features.get("velocity_30d", 0)), 0.0)
    acceleration = _clamp(float(features.get("acceleration", 50.0)) / 100.0, 0.0, 1.0)
    engagement = _clamp(float(features.get("engagement_rate", 0.0)) / 20.0, 0.0, 1.0)
    virality = _clamp(float(features.get("virality_score", 0.0)) / 10.0, 0.0, 1.0)
    consistency = _clamp(float(features.get("consistency_score", 0.0)) / 100.0, 0.0, 1.0)
    niche = _clamp(float(features.get("niche_momentum", 50.0)) / 100.0, 0.0, 1.0)
    audience = _clamp(float(features.get("audience_quality", 60.0)) / 100.0, 0.0, 1.0)
    confidence = _clamp(float(features.get("confidence", 0.1)), 0.1, 1.0)

    relative_growth = _clamp(velocity_7d / max(followers, 1.0) * 20.0, 0.0, 1.0)
    absolute_growth = _clamp(math.log10(max(velocity_30d, 1.0)) / 4.0, 0.0, 1.0)

    core_signal = (
        (relative_growth * 0.22) +
        (absolute_growth * 0.18) +
        (engagement * 0.20) +
        (virality * 0.12) +
        (consistency * 0.10) +
        (acceleration * 0.08) +
        (niche * 0.06) +
        (audience * 0.04)
    )

    # Larger creators should need stronger evidence, but they should not be zeroed out
    # just because subscriber-relative ratios compress at scale.
    scale_bonus = (follower_scale - 0.45) * 0.10
    confidence_shrink = 0.55 + (confidence * 0.45)
    prob = (core_signal + scale_bonus) * confidence_shrink
    return int(round(_clamp(prob, 0.02, 0.98) * 100))

def predict_trend(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict breakout probability and calculate composite trend score."""
    m = load_model()
    confidence = _clamp(float(features.get("confidence", 0.1)), 0.1, 1.0)
    heuristic_prob = _heuristic_probability(features)

    if m is None:
        model_prob_score = heuristic_prob
    else:
        vector = [float(features.get(name, 0)) for name in FEATURE_NAMES]
        X = np.array([vector], dtype=np.float32)

        prob = float(m.predict_proba(X)[0][1])
        model_prob_score = max(1, min(99, int(round(prob * 100))))

    # Blend model output with a calibrated heuristic so the score stays stable
    # when live features drift away from the training distribution.
    prob_score = int(round((model_prob_score * 0.45) + (heuristic_prob * 0.55)))
    
    # Real Trend Score Calculation
    followers = max(float(features.get("followers", 0)), 1.0)
    relative_velocity = _clamp(float(features.get("velocity_7d", 0)) / followers * 1500, 0.0, 100.0)
    absolute_velocity = _clamp(math.log10(max(float(features.get("velocity_30d", 0)), 1.0)) * 24.0, 0.0, 100.0)
    sub_growth_norm = (relative_velocity * 0.55) + (absolute_velocity * 0.45)
    engagement_norm = _clamp((float(features.get("engagement_rate", 0)) / 20.0) * 100, 0.0, 100.0)
    velocity_norm = _clamp(float(features.get("acceleration", 50.0)), 0.0, 100.0)
    
    # Formula: 0.5 * subscriber_growth + 0.3 * engagement + 0.2 * velocity
    trend_score = (0.5 * sub_growth_norm) + (0.3 * engagement_norm) + (0.2 * velocity_norm)
    
    # Adjust trend score by data confidence
    adjusted_trend_score = trend_score * confidence
    
    return {
        "prob_score": max(1, min(99, prob_score)),
        "trend_score": round(adjusted_trend_score, 2),
        "confidence": round(confidence, 2)
    }
