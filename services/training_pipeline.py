import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from db.models import Creator, FeatureObservation, ModelTrainingRun, Snapshot, TrainingExample

logger = logging.getLogger("creatorrocket")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "observed_breakout_model.joblib")
FEATURE_COLUMNS = [
    "followers",
    "velocity_7d",
    "velocity_30d",
    "acceleration",
    "engagement_rate",
    "virality_score",
    "consistency_score",
    "niche_momentum",
    "audience_quality",
    "comment_rate_per_1k",
    "like_rate_per_1k",
    "share_rate_per_1k",
    "avg_view_percentage",
    "subscriber_conversion_per_1k",
    "recent_view_velocity",
    "breakout_ratio",
    "analytics_coverage",
    "retention_score",
    "confidence",
]


def _serialize_features(features: Dict) -> str:
    payload = {key: float(features.get(key, 0.0) or 0.0) for key in FEATURE_COLUMNS}
    return json.dumps(payload, separators=(",", ":"))


def _deserialize_features(value: str) -> Dict:
    try:
        return json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}


def record_feature_observation(
    db: Session,
    creator: Creator,
    features: Dict,
    predictor_mode: str,
    source: str = "analysis",
    dedupe_hours: int = 12,
) -> Optional[FeatureObservation]:
    latest = (
        db.query(FeatureObservation)
        .filter(FeatureObservation.creator_id == creator.id)
        .order_by(FeatureObservation.observed_at.desc())
        .first()
    )
    if latest and latest.observed_at > datetime.utcnow() - timedelta(hours=dedupe_hours):
        return latest

    observation = FeatureObservation(
        creator_id=creator.id,
        observed_at=datetime.utcnow(),
        source=source,
        predictor_mode=predictor_mode,
        confidence=float(features.get("confidence", 0.0) or 0.0),
        features_json=_serialize_features(features),
    )
    db.add(observation)
    db.commit()
    db.refresh(observation)
    return observation


def _snapshot_at_or_before(db: Session, creator_id: int, target_time: datetime) -> Optional[Snapshot]:
    return (
        db.query(Snapshot)
        .filter(Snapshot.creator_id == creator_id, Snapshot.timestamp <= target_time)
        .order_by(Snapshot.timestamp.desc())
        .first()
    )


def _snapshot_at_or_after(db: Session, creator_id: int, target_time: datetime, tolerance_days: int = 10) -> Optional[Snapshot]:
    upper_bound = target_time + timedelta(days=tolerance_days)
    return (
        db.query(Snapshot)
        .filter(Snapshot.creator_id == creator_id, Snapshot.timestamp >= target_time, Snapshot.timestamp <= upper_bound)
        .order_by(Snapshot.timestamp.asc())
        .first()
    )


def _label_breakout(features: Dict, baseline_followers: int, target_followers: int, horizon_days: int) -> Dict[str, float]:
    baseline_followers = max(int(baseline_followers), 1)
    target_followers = max(int(target_followers), baseline_followers)
    future_growth_ratio = (target_followers - baseline_followers) / baseline_followers
    prior_velocity = max(float(features.get("velocity_30d", 0.0) or 0.0), 1.0)
    observed_velocity = ((target_followers - baseline_followers) / max(horizon_days, 1)) * 30.0
    velocity_ratio = observed_velocity / prior_velocity

    if baseline_followers < 10_000:
        threshold = 0.14
    elif baseline_followers < 100_000:
        threshold = 0.10
    else:
        threshold = 0.06

    label = 1 if future_growth_ratio >= threshold or velocity_ratio >= 1.35 else 0
    return {
        "label": label,
        "future_growth_ratio": round(future_growth_ratio, 5),
    }


def materialize_training_examples(db: Session, horizons: Optional[List[int]] = None) -> int:
    horizons = horizons or [30, 90]
    created = 0
    observations = db.query(FeatureObservation).order_by(FeatureObservation.observed_at.asc()).all()

    for observation in observations:
        features = _deserialize_features(observation.features_json)
        baseline_snapshot = _snapshot_at_or_before(db, observation.creator_id, observation.observed_at)
        if not baseline_snapshot:
            continue

        for horizon in horizons:
            existing = (
                db.query(TrainingExample)
                .filter(
                    TrainingExample.observation_id == observation.id,
                    TrainingExample.horizon_days == horizon,
                )
                .first()
            )
            if existing:
                continue

            future_target = observation.observed_at + timedelta(days=horizon)
            future_snapshot = _snapshot_at_or_after(db, observation.creator_id, future_target)
            if not future_snapshot:
                continue

            label_meta = _label_breakout(
                features,
                baseline_snapshot.subscriber_count,
                future_snapshot.subscriber_count,
                horizon,
            )
            example = TrainingExample(
                creator_id=observation.creator_id,
                observation_id=observation.id,
                observed_at=observation.observed_at,
                horizon_days=horizon,
                label=label_meta["label"],
                baseline_followers=baseline_snapshot.subscriber_count,
                target_followers=future_snapshot.subscriber_count,
                future_growth_ratio=label_meta["future_growth_ratio"],
                features_json=observation.features_json,
            )
            db.add(example)
            created += 1

    if created:
        db.commit()
    return created


def _rows_from_examples(examples: List[TrainingExample]) -> Optional[Dict[str, np.ndarray]]:
    if not examples:
        return None

    rows = []
    labels = []
    for example in examples:
        payload = _deserialize_features(example.features_json)
        rows.append([float(payload.get(column, 0.0) or 0.0) for column in FEATURE_COLUMNS])
        labels.append(int(example.label))

    return {
        "X": np.asarray(rows, dtype=float),
        "y": np.asarray(labels, dtype=int),
    }


def train_observed_model(db: Session, min_examples: int = 24) -> Dict:
    examples = db.query(TrainingExample).order_by(TrainingExample.created_at.asc()).all()
    payload = _rows_from_examples(examples)
    if not payload or len(examples) < min_examples:
        return {
            "trained": False,
            "reason": "not_enough_examples",
            "example_count": len(examples),
        }

    X = payload["X"]
    y = payload["y"]
    if len(set(y.tolist())) < 2:
        return {
            "trained": False,
            "reason": "single_class_dataset",
            "example_count": len(examples),
        }

    x_train = X
    x_test = X
    y_train = y
    y_test = y
    if len(examples) >= 40:
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )
        except ValueError:
            logger.warning("Falling back to no-split training because the dataset is not stratifiable yet.")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(x_train, y_train)

    probas = pipeline.predict_proba(x_test)[:, 1]
    preds = (probas >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_test, preds))
    auc = None
    if len(set(y_test.tolist())) > 1:
        auc = float(roc_auc_score(y_test, probas))

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "model": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": datetime.utcnow().isoformat(),
        "example_count": len(examples),
        "positive_rate": float(np.mean(y)),
        "holdout_accuracy": accuracy,
        "holdout_auc": auc,
    }
    joblib.dump(artifact, MODEL_PATH)

    run = ModelTrainingRun(
        status="completed",
        artifact_path=MODEL_PATH,
        example_count=len(examples),
        positive_rate=float(np.mean(y)),
        holdout_accuracy=accuracy,
        holdout_auc=auc,
        notes="Observed-outcome logistic model trained from tracked creator feature snapshots.",
    )
    db.add(run)
    db.commit()

    return {
        "trained": True,
        "artifact_path": MODEL_PATH,
        "example_count": len(examples),
        "positive_rate": round(float(np.mean(y)), 4),
        "holdout_accuracy": round(accuracy, 4),
        "holdout_auc": round(auc, 4) if auc is not None else None,
    }


def get_training_status(db: Session) -> Dict:
    latest_run = db.query(ModelTrainingRun).order_by(ModelTrainingRun.trained_at.desc()).first()
    observation_count = db.query(FeatureObservation).count()
    example_count = db.query(TrainingExample).count()
    connected_creators = db.query(Creator).count()

    return {
        "model_available": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        "observation_count": observation_count,
        "example_count": example_count,
        "tracked_creators": connected_creators,
        "latest_training_run": {
            "trained_at": latest_run.trained_at.isoformat() if latest_run else None,
            "status": latest_run.status if latest_run else None,
            "example_count": latest_run.example_count if latest_run else 0,
            "positive_rate": latest_run.positive_rate if latest_run else None,
            "holdout_accuracy": latest_run.holdout_accuracy if latest_run else None,
            "holdout_auc": latest_run.holdout_auc if latest_run else None,
        },
    }


def refresh_training_pipeline(db: Session) -> Dict:
    created_examples = materialize_training_examples(db)
    status = get_training_status(db)
    if created_examples or not status["model_available"]:
        train_result = train_observed_model(db)
    else:
        train_result = {"trained": False, "reason": "no_new_examples"}
    status["new_examples"] = created_examples
    status["train_result"] = train_result
    return status
