from fastapi import FastAPI, Request, Response
import db
import bcrypt
from fastapi.responses import JSONResponse
import secrets
import redis
import uuid
import json

app = FastAPI()

@app.get("/")
def root():
    return JSONResponse(content={"message": "User service is up!"}, status_code=200)

@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.post("/register")
async def register(request: Request):
    data = await request.json()
    required_fields = ["name", "surname", "username", "password", "born_date"]
    if not all(field in data for field in required_fields):
        return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

    name = data["name"]
    surname = data["surname"]
    username = data["username"]
    password = data["password"]
    born_date = data["born_date"]

    with db.db_connection() as db_conn:
        if db.get_user(db_conn, username):
            return JSONResponse(content={"error": "Username already exists"}, status_code=400)

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db.insert_user(db_conn, name, surname, username, password_hash, born_date)
        return JSONResponse(content={"message": "User registered successfully"}, status_code=201)

@app.get("/get_user/{username}")
async def get_user(username: str):
    with db.db_connection() as db_conn:
        user = db.get_user(db_conn, username)
        if not user:
            return JSONResponse(content={"error": "User not found"}, status_code=404)
        user.pop("password_hash", None)
        return JSONResponse(content=user, status_code=200)

@app.post("/edit_user")
async def edit_user(request: Request):
    with db.db_connection() as db_conn:
        data = await request.json()
        required_fields = ["username", "new_name", "new_surname", "new_username", "new_password", "new_born_date"]
        if not all(field in data for field in required_fields):
            return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

        username = data["username"]
        new_name = data["new_name"]
        new_surname = data["new_surname"]
        new_username = data["new_username"]
        new_password = data["new_password"]
        new_born_date = data["new_born_date"]

        user = db.get_user(db_conn, username)
        if not user:
            return JSONResponse(content={"error": "User not found"}, status_code=404)

        new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db.update_user(db_conn, username, new_name, new_surname, new_username, new_password_hash, new_born_date)
        return JSONResponse(content={"message": "User updated successfully"}, status_code=200)

@app.post("/delete_user")
async def delete_user(request: Request):
    with db.db_connection() as db_conn:
        data = await request.json()
        if "username" not in data:
            return JSONResponse(content={"error": "Missing username field"}, status_code=400)

        username = data["username"]
        user = db.get_user(db_conn, username)
        if not user:
            return JSONResponse(content={"error": "User not found"}, status_code=404)

        db.delete_user(db_conn, username)
        return JSONResponse(content={"message": "User deleted successfully"}, status_code=200)