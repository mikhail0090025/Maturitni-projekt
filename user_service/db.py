from sqlalchemy import create_engine, text
import os
from contextlib import contextmanager
import pytest

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://nn_manager:secret@postgres:5432/nn_manager_db")

@contextmanager
def db_connection():
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()

'''
CRUD - Create, Read, Update, Delete operations for users
'''

def insert_user(db, name, surname, username, password_hash, born_date):
    db.execute(
        text(
            "INSERT INTO users (name, surname, username, password_hash, born_date) "
            "VALUES (:n, :s, :u, :p, :b)"
        ),
        {"n": name, "s": surname, "u": username, "p": password_hash, "b": born_date}
    )
    db.commit()

def get_user(db, username):
    result = db.execute(
        text("SELECT name, surname, username, born_date, password_hash FROM users WHERE username=:u"),
        {"u": username}
    ).fetchone()
    if result:
        return {
            "name": result[0],
            "surname": result[1],
            "username": result[2],
            "born_date": str(result[3]),
            "password_hash": result[4]
        }
    return None

def update_user(db, username, new_name, new_surname, new_username, new_password_hash, new_born_date):
    db.execute(
        text("""
            UPDATE users
            SET name = :n,
                surname = :s,
                username = :u,
                password_hash = :p,
                born_date = :b
            WHERE username = :old_u
        """),
        {
            "n": new_name,
            "s": new_surname,
            "u": new_username,
            "p": new_password_hash,
            "b": new_born_date,
            "old_u": username
        }
    )
    db.commit()

def delete_user(db, username):
    db.execute(
        text("DELETE FROM users WHERE username=:u"),
        {"u": username}
    )
    db.commit()

''' For testing purposes only '''
def get_all_users(db):
    result = db.execute(
        text("SELECT name, surname, username, born_date FROM users")
    ).fetchall()
    if result:
        return result
    return []