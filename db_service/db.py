from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
import models

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://nn_manager:secret@postgres:5432/nn_manager_db")
engine = create_engine(DATABASE_URL, future=True)

models.Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def insert_user(name, surname, username, password_hash, born_date, bio=""):
    with SessionLocal() as session:
        user = models.User(
            name=name,
            surname=surname,
            username=username,
            password_hash=password_hash,
            born_date=born_date,
            bio=bio
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

def get_user(username: str):
    with SessionLocal() as session:
        user = session.query(models.User).filter_by(username=username).first()
        if user:
            return {
                "name": user.name,
                "surname": user.surname,
                "username": user.username,
                "born_date": str(user.born_date),
                "password_hash": user.password_hash,
                "bio": user.bio
            }
        return None

def update_user(username, new_name=None, new_surname=None, new_username=None,
                new_password_hash=None, new_born_date=None, new_bio=None):
    with SessionLocal() as session:
        user = session.query(models.User).filter_by(username=username).first()
        if not user:
            return None

        if new_name is not None:
            user.name = new_name
        if new_surname is not None:
            user.surname = new_surname
        if new_username is not None:
            user.username = new_username
        if new_password_hash is not None:
            user.password_hash = new_password_hash
        if new_born_date is not None:
            user.born_date = new_born_date
        if new_bio is not None:
            user.bio = new_bio

        session.commit()
        session.refresh(user)
        return user

def delete_user(username: str):
    with SessionLocal() as session:
        user = session.query(models.User).filter_by(username=username).first()
        if not user:
            return False
        session.delete(user)
        session.commit()
        return True

def get_all_users():
    with SessionLocal() as session:
        users = session.query(models.User).all()
        return [
            {
                "name": u.name,
                "surname": u.surname,
                "username": u.username,
                "born_date": str(u.born_date),
                "bio": u.bio
            } for u in users
        ]
