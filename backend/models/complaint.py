from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey
from database import Base


class Complaint(Base):
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    image_path = Column(String, nullable=True)
    damage_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    locality = Column(String, nullable=True)   # human-readable location name

    description = Column(Text, nullable=True)
    status = Column(String, default="Submitted")