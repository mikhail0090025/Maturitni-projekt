from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
import models

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://nn_manager:secret@postgres:5432/nn_manager_db")
engine = create_engine(DATABASE_URL, future=True)

models.Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

"""CRUD operations for User model"""

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

"""CRUD operations for Project model"""

def insert_project(name, description, owner_username, input_type, output_type):
    with SessionLocal() as session:
        project = models.Project(
            name=name,
            description=description,
            owner_username=owner_username,
            input_type=input_type,
            output_type=output_type
        )
        session.add(project)
        session.commit()
        session.refresh(project)
        return project

def get_project(id):
    with SessionLocal() as session:
        project = session.query(models.Project).filter_by(id=id).first()
        if project:
            return {
                "name": project.name,
                "description": project.description,
                "owner_username": project.owner_username,
                "input_type": project.input_type.value,
                "output_type": project.output_type.value,
                "created_at": str(project.created_at)
            }
        return None

def update_project(id, new_name=None, new_description=None, new_owner_username=None,
                   new_input_type=None, new_output_type=None):
    with SessionLocal() as session:
        project = session.query(models.Project).filter_by(id=id).first()
        if not project:
            return None

        if new_name is not None:
            project.name = new_name
        if new_description is not None:
            project.description = new_description
        if new_owner_username is not None:
            project.owner_username = new_owner_username
        if new_input_type is not None:
            project.input_type = new_input_type
        if new_output_type is not None:
            project.output_type = new_output_type

        session.commit()
        session.refresh(project)
        return project

def delete_project(id):
    with SessionLocal() as session:
        project = session.query(models.Project).filter_by(id=id).first()
        if not project:
            return False
        session.delete(project)
        session.commit()
        return True

def get_projects_of_user(owner_username: str):
    with SessionLocal() as session:
        projects = session.query(models.Project).filter_by(owner_username=owner_username).all()
        return [
            {
                "name": p.name,
                "description": p.description,
                "owner_username": p.owner_username,
                "input_type": p.input_type.value,
                "output_type": p.output_type.value,
                "created_at": str(p.created_at)
            } for p in projects
        ]

def get_all_projects():
    with SessionLocal() as session:
        projects = session.query(models.Project).all()
        return [
            {
                "name": p.name,
                "description": p.description,
                "owner_username": p.owner_username,
                "input_type": p.input_type.value,
                "output_type": p.output_type.value,
                "created_at": str(p.created_at)
            } for p in projects
        ]
