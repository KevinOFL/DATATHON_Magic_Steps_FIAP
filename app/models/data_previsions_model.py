from sqlalchemy import Column, Integer, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.config.database_config import Base


class DataPrevisions(Base):
    __tablename__ = 'data_prevision'
    id = id = Column(Integer, primary_key=True)
    request_id = Column(Integer, ForeignKey("prediction_requests.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    idade = Column(Integer)
    genero = Column(Integer)
    ano_ingresso = Column(Integer)
    inde = Column(Float)
    iaa = Column(Float)
    ieg = Column(Float)
    ips = Column(Float)
    ida = Column(Float)
    mat = Column(Float)
    por = Column(Float)
    ing = Column(Float)
    ipv = Column(Float)
    ian = Column(Float)
    estudou_ingles = Column(Integer)
    diferenca_mat = Column(Float)
    diferenca_por = Column(Float)
    diferenca_ing = Column(Float)
    diferenca_ieg = Column(Float)
    diferenca_inde = Column(Float)
    # Output do modelo
    probabilidade_risco_percentual = Column(Float)
    alerta_evasao = Column(Integer)

    request = relationship("PredicitonRequest")
