from fastapi import FastAPI, Request, Response
import db
import bcrypt
from fastapi.responses import JSONResponse
import secrets
import redis
import uuid
import json
import requests
r = redis.Redis(host="redis", port=6379, db=0)

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
        required_fields = ["username", "new_name", "new_surname", "new_username", "new_born_date", "new_bio"]
        if not all(field in data for field in required_fields):
            return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

        username = data["username"]
        new_name = data["new_name"]
        new_surname = data["new_surname"]
        new_username = data["new_username"]
        new_born_date = data["new_born_date"]
        new_bio = data["new_bio"]

        user = db.get_user(db_conn, username)
        if not user:
            return JSONResponse(content={"error": "User not found"}, status_code=404)

        # new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db.update_user(db_conn, username, new_name, new_surname, new_username, user['password_hash'], new_born_date, new_bio)
        return JSONResponse(content={"message": "User updated successfully"}, status_code=200)

@app.delete("/delete_user")
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

@app.delete("/delete_me")
async def delete_user(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return JSONResponse(content={"error": "Not authenticated"}, status_code=401)

    username = r.get(session_id)
    if not username:
        return JSONResponse(content={"error": "Session expired or invalid"}, status_code=401)

    username = username.decode("utf-8")
    with db.db_connection() as db_conn:
        user = db.get_user(db_conn, username)
        if not user:
            return JSONResponse(content={"error": "User not found"}, status_code=404)

        db.delete_user(db_conn, username)
        return JSONResponse(content={"message": "User deleted successfully"}, status_code=200)

@app.post("/login")
async def login_user(request: Request):
    data = await request.json()
    if "username" not in data or "password" not in data:
        return JSONResponse(content={"error": "Username or password are null"}, status_code=400)

    with db.db_connection() as db_conn:
        user = db.get_user(db_conn, data["username"])
        if not user:
            return JSONResponse(content={"error": "User not found"}, status_code=404)
        
        if bcrypt.checkpw(data["password"].encode("utf-8"), user["password_hash"].encode("utf-8")):
            session_id = secrets.token_hex(16)
            r.set(session_id, user["username"], ex=3600)
            response = JSONResponse(content=user, status_code=200)
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
                secure=False,
                max_age=3600,
                path="/"
            )
            return response
        
        return JSONResponse(content={"error": "Wrong password"}, status_code=401)

@app.post("/logout")
async def logout_user(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id:
        r.delete(session_id)
        response = JSONResponse(content={"message": "Logged out successfully"}, status_code=200)
        response.delete_cookie(key="session_id", path="/")
        return response
    return JSONResponse(content={"error": "No active session"}, status_code=400)

@app.get("/me")
async def get_current_user(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return JSONResponse(content={"error": "Not authenticated"}, status_code=401)

    username = r.get(session_id)
    if not username:
        return JSONResponse(content={"error": "Session expired or invalid"}, status_code=401)

    username = username.decode("utf-8")
    with db.db_connection() as db_conn:
        user = db.get_user(db_conn, username)
        if not user:
            return JSONResponse(content={"error": "User not found"}, status_code=404)
        user.pop("password_hash", None)
        return JSONResponse(content=user, status_code=200)