from enum import Enum
from sqlalchemy import Column, Integer, String, Date, DateTime, func, ForeignKey, Enum as EnumType, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DataType(Enum):
    IMAGE = "image"
    VECTOR = "vector"

class LossType(Enum):
    CROSS_ENTROPY = "CrossEntropyLoss"
    MSE = "MSELoss"
    L1 = "L1Loss"
    SMOOTH_L1 = "SmoothL1Loss"

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
    architecture_json = Column(Text(), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    dataset_preprocess_json = Column(Text(), nullable=True)

    optimizer_json = Column(Text(), nullable=True)
    scheduler_json = Column(Text(), nullable=True)
    loss_function = Column(EnumType(LossType), nullable=True, default=LossType.MSE)

    dataset = relationship("Dataset")
    owner = relationship("User", back_populates="projects")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255), default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    storage_id = Column(String(128), unique=True, nullable=False)
    dataset_type = Column(String(50), nullable=False, default="unknown")

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)