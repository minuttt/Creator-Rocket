from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

class Creator(Base):
    __tablename__ = "creators"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    platform = Column(String, default="youtube")
    created_at = Column(DateTime, default=datetime.utcnow)

    snapshots = relationship("Snapshot", back_populates="creator", cascade="all, delete-orphan")

class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer, ForeignKey("creators.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    subscriber_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    video_count = Column(Integer, default=0)

    creator = relationship("Creator", back_populates="snapshots")