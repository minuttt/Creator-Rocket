import os
import logging
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
            model = joblib.load(MODEL_PATH)
            logger.info("XGBoost model loaded from disk")
        else:
            logger.error("Model file not found! Run application startup to generate base model.")
    return model

def predict_trend(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict breakout probability and calculate composite trend score."""
    m = load_model()
    if m is None:
        return {"prob_score": 0, "trend_score": 0.0, "confidence": features.get("confidence", 0.1)}

    vector = [float(features.get(name, 0)) for name in FEATURE_NAMES]
    X = np.array([vector], dtype=np.float32)

    prob = float(m.predict_proba(X)[0][1])
    prob_score = max(1, min(99, int(round(prob * 100))))
    
    # Real Trend Score Calculation
    sub_growth_norm = min(100, max(0, (features["velocity_7d"] / 2000) * 100))
    engagement_norm = min(100, max(0, (features["engagement_rate"] / 15) * 100))
    velocity_norm = min(100, max(0, features["acceleration"]))
    
    # Formula: 0.5 * subscriber_growth + 0.3 * engagement + 0.2 * velocity
    trend_score = (0.5 * sub_growth_norm) + (0.3 * engagement_norm) + (0.2 * velocity_norm)
    
    # Adjust trend score by data confidence
    confidence = features.get("confidence", 0.1)
    adjusted_trend_score = trend_score * confidence
    
    return {
        "prob_score": prob_score,
        "trend_score": round(adjusted_trend_score, 2),
        "confidence": round(confidence, 2)
    }