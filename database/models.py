from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    model = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    features = Column(String)
    fraud_prob = Column(Float)
    explanation = Column(String)
