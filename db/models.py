from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from db.database import Base


class Creator(Base):
    __tablename__ = "creators"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    platform = Column(String, default="youtube")
    profile_picture_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    snapshots = relationship("Snapshot", back_populates="creator", cascade="all, delete-orphan")
    observations = relationship("FeatureObservation", back_populates="creator", cascade="all, delete-orphan")
    training_examples = relationship("TrainingExample", back_populates="creator", cascade="all, delete-orphan")


class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer, ForeignKey("creators.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    subscriber_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    video_count = Column(Integer, default=0)

    creator = relationship("Creator", back_populates="snapshots")


class AnalyticsConnection(Base):
    __tablename__ = "analytics_connections"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String, default="youtube")
    channel_id = Column(String, unique=True, index=True)
    channel_name = Column(String, nullable=True)
    profile_picture_url = Column(String, nullable=True)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    scope = Column(Text, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    connected_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OAuthState(Base):
    __tablename__ = "oauth_states"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String, default="youtube")
    state = Column(String, unique=True, index=True)
    redirect_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


class FeatureObservation(Base):
    __tablename__ = "feature_observations"

    id = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer, ForeignKey("creators.id"), index=True)
    observed_at = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String, default="analysis")
    predictor_mode = Column(String, nullable=True)
    confidence = Column(Float, default=0.0)
    features_json = Column(Text, nullable=False)

    creator = relationship("Creator", back_populates="observations")
    training_examples = relationship("TrainingExample", back_populates="observation", cascade="all, delete-orphan")


class TrainingExample(Base):
    __tablename__ = "training_examples"
    __table_args__ = (UniqueConstraint("observation_id", "horizon_days", name="uq_training_observation_horizon"),)

    id = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer, ForeignKey("creators.id"), index=True)
    observation_id = Column(Integer, ForeignKey("feature_observations.id"), index=True)
    observed_at = Column(DateTime, nullable=False, index=True)
    horizon_days = Column(Integer, nullable=False, default=30)
    label = Column(Integer, nullable=False)
    baseline_followers = Column(Integer, nullable=False)
    target_followers = Column(Integer, nullable=False)
    future_growth_ratio = Column(Float, nullable=False, default=0.0)
    features_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    creator = relationship("Creator", back_populates="training_examples")
    observation = relationship("FeatureObservation", back_populates="training_examples")


class ModelTrainingRun(Base):
    __tablename__ = "model_training_runs"

    id = Column(Integer, primary_key=True, index=True)
    trained_at = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String, default="completed")
    artifact_path = Column(String, nullable=True)
    example_count = Column(Integer, default=0)
    positive_rate = Column(Float, default=0.0)
    holdout_accuracy = Column(Float, nullable=True)
    holdout_auc = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
