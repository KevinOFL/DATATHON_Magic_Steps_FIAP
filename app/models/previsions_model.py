from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.config.database_config import Base


class PredicitonRequest(Base):
    __tablename__ = 'prediction_requests'
    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    user = relationship("User", backref="predictions")
