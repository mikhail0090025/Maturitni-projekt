# main.py (user_service, обращается к db_service)
import os
import requests
import bcrypt
import secrets
import redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DB_SERVICE = os.getenv("DB_SERVICE_URL", "http://db_service:8000")
REQUEST_TIMEOUT = 5

REDIS_HOST = "redis"
REDIS_PORT = 6379
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)

app = FastAPI(title="User Service (gateway to db_service)")

def _db_get_user(username: str):
    try:
        resp = requests.get(f"{DB_SERVICE}/get_user/{username}", timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return {"error": "db_unreachable", "detail": str(e)}, 503

    if resp.status_code == 200:
        return resp.json(), 200
    elif resp.status_code == 404:
        return None, 404
    else:
        # propagate error body if any
        try:
            return {"error": "db_error", "detail": resp.json()}, resp.status_code
        except Exception:
            return {"error": "db_error", "detail": resp.text}, resp.status_code

def _db_create_user(payload: dict):
    try:
        resp = requests.post(f"{DB_SERVICE}/create_user", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return {"error": "db_unreachable", "detail": str(e)}, 503

    try:
        body = resp.json()
    except Exception:
        body = {"text": resp.text}
    return body, resp.status_code

def _db_update_user(username: str, payload: dict):
    try:
        resp = requests.put(f"{DB_SERVICE}/update_user/{username}", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return {"error": "db_unreachable", "detail": str(e)}, 503

    try:
        body = resp.json()
    except Exception:
        body = {"text": resp.text}
    return body, resp.status_code

def _db_delete_user(username: str):
    try:
        resp = requests.delete(f"{DB_SERVICE}/delete_user/{username}", timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return {"error": "db_unreachable", "detail": str(e)}, 503

    try:
        body = resp.json()
    except Exception:
        body = {"text": resp.text}
    return body, resp.status_code

@app.get("/")
def root():
    return JSONResponse(content={"message": "User service (gateway) is up!"}, status_code=200)

@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.post("/register")
async def register(request: Request):
    data = await request.json()
    required_fields = ["name", "surname", "username", "password", "born_date"]
    if not all(field in data for field in required_fields):
        return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

    username = data["username"]

    # check existing user via db_service
    user_res, code = _db_get_user(username)
    if code == 503:
        return JSONResponse(content=user_res, status_code=503)
    if code == 200:
        return JSONResponse(content={"error": "Username already exists"}, status_code=400)

    # hash password locally, then create via db_service
    password_hash = bcrypt.hashpw(data["password"].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    payload = {
        "name": data["name"],
        "surname": data["surname"],
        "username": username,
        "password_hash": password_hash,
        "born_date": data["born_date"],
        "bio": data.get("bio", "")
    }

    body, status = _db_create_user(payload)
    return JSONResponse(content=body, status_code=status)

@app.get("/get_user/{username}")
async def get_user(username: str):
    user_res, code = _db_get_user(username)
    if code == 503:
        return JSONResponse(content=user_res, status_code=503)
    if code == 404 or user_res is None:
        return JSONResponse(content={"error": "User not found"}, status_code=404)

    # remove password_hash before returning
    user_res.pop("password_hash", None)
    return JSONResponse(content=user_res, status_code=200)

@app.post("/edit_user")
async def edit_user(request: Request):
    data = await request.json()
    required_fields = ["username", "new_name", "new_surname", "new_username", "new_born_date", "new_bio"]
    if not all(field in data for field in required_fields):
        return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

    username = data["username"]

    # fetch existing user to ensure exists and to get old password_hash if password not changed
    user_res, code = _db_get_user(username)
    if code == 503:
        return JSONResponse(content=user_res, status_code=503)
    if code == 404 or user_res is None:
        return JSONResponse(content={"error": "User not found"}, status_code=404)

    # If client wants to change password, expect new_password field (optional)
    new_password_hash = None
    if "new_password" in data and data["new_password"]:
        new_password_hash = bcrypt.hashpw(data["new_password"].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    else:
        new_password_hash = user_res.get("password_hash")

    payload = {
        "new_name": data["new_name"],
        "new_surname": data["new_surname"],
        "new_username": data["new_username"],
        "new_password_hash": new_password_hash,
        "new_born_date": data["new_born_date"],
        "new_bio": data["new_bio"]
    }

    body, status = _db_update_user(username, payload)
    return JSONResponse(content=body, status_code=status)

@app.delete("/delete_user")
async def delete_user(request: Request):
    data = await request.json()
    if "username" not in data:
        return JSONResponse(content={"error": "Missing username field"}, status_code=400)
    username = data["username"]

    # check existence
    user_res, code = _db_get_user(username)
    if code == 503:
        return JSONResponse(content=user_res, status_code=503)
    if code == 404 or user_res is None:
        return JSONResponse(content={"error": "User not found"}, status_code=404)

    body, status = _db_delete_user(username)
    return JSONResponse(content=body, status_code=status)

@app.delete("/delete_me")
async def delete_me(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return JSONResponse(content={"error": "Not authenticated"}, status_code=401)

    username_b = r.get(session_id)
    if not username_b:
        return JSONResponse(content={"error": "Session expired or invalid"}, status_code=401)
    username = username_b.decode("utf-8")

    # double-check user exists
    user_res, code = _db_get_user(username)
    if code == 503:
        return JSONResponse(content=user_res, status_code=503)
    if code == 404 or user_res is None:
        return JSONResponse(content={"error": "User not found"}, status_code=404)

    body, status = _db_delete_user(username)
    if status == 200:
        # remove session
        r.delete(session_id)
    return JSONResponse(content=body, status_code=status)

@app.post("/login")
async def login_user(request: Request):
    data = await request.json()
    if "username" not in data or "password" not in data:
        return JSONResponse(content={"error": "Username or password are null"}, status_code=400)

    username = data["username"]
    password = data["password"]

    user_res, code = _db_get_user(username)
    if code == 503:
        return JSONResponse(content=user_res, status_code=503)
    if code == 404 or user_res is None:
        return JSONResponse(content={"error": "User not found"}, status_code=404)

    password_hash = user_res.get("password_hash", "")
    if not password_hash:
        return JSONResponse(content={"error": "User has no password hash"}, status_code=500)

    if bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
        session_id = secrets.token_hex(16)
        r.set(session_id, username, ex=3600)
        # hide password hash in response
        safe_user = dict(user_res)
        safe_user.pop("password_hash", None)
        response = JSONResponse(content=safe_user, status_code=200)
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            secure=(os.getenv("ENV") == "production"),
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

    username_b = r.get(session_id)
    if not username_b:
        return JSONResponse(content={"error": "Session expired or invalid"}, status_code=401)
    username = username_b.decode("utf-8")

    user_res, code = _db_get_user(username)
    if code == 503:
        return JSONResponse(content=user_res, status_code=503)
    if code == 404 or user_res is None:
        return JSONResponse(content={"error": "User not found"}, status_code=404)

    user_res.pop("password_hash", None)
    return JSONResponse(content=user_res, status_code=200)
