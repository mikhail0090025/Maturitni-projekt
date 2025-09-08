from enum import Enum
from sqlalchemy import Column, Integer, String, Date, DateTime, func, ForeignKey, Enum as EnumType
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DataType(Enum):
    NOISE = "noise"
    IMAGE = "image"
    VECTOR = "vector"
    BINARY = "binary"

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

    projects = relationship("Project", back_populates="owner")

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255), default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner_username = Column(String(50), ForeignKey("users.username"), nullable=False)
    input_type = Column(EnumType(DataType), nullable=False)
    output_type = Column(EnumType(DataType), nullable=False)

    owner = relationship("User", back_populates="projects")