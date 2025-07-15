from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class TrafficSigns(Base):
    __tablename__ = 'traffic_signs'

    filename = Column(String, primary_key=True)
    width = Column(String, nullable=False)
    height = Column(String, nullable=False)
    roi_x1 = Column(Integer, nullable=False)
    roi_y1 = Column(Integer, nullable=False)
    roi_x2 = Column(Integer, nullable=False)
    roi_y2 = Column(Integer, nullable=False)
    class_id = Column(Integer, nullable=False)

class Predictions(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    class_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserUploads(Base):
    __tablename__ = 'user_uploads'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    mode = Column(String, nullable=False)  # 'train' or 'test'
    upload_time = Column(DateTime, default=datetime.utcnow)

    
