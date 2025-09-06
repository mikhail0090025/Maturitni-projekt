from sqlalchemy import Column, Integer, String, Date, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    surname = Column(String(50), nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    born_date = Column(Date, nullable=False)
    bio = Column(String(100), default="")
    registration_date = Column(DateTime(timezone=True), server_default=func.now())
