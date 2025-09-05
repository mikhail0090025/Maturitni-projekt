from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import bcrypt
import secrets
import uuid
import json
import requests

app = FastAPI()

app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/styles", StaticFiles(directory="templates/styles"), name="styles")
app.mount("/js", StaticFiles(directory="templates/js"), name="js")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def root():
    return JSONResponse(content={"message": "Frontend service is up!"}, status_code=200)

@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)

''' Pages '''

@app.get("/registration_page")
def registration_page(request: Request):
    return templates.TemplateResponse("registration.html", {"request": request})

@app.get("/login_page")
def login_page_endpoint_get(request: Request):
    response_to_user = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if response_to_user.json().get("username"):
        return RedirectResponse(url="/profile_page")

    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/profile_page")
def profile_page_endpoint_get(request: Request):
    response_to_user = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if response_to_user.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = response_to_user.json()
    return templates.TemplateResponse("mainpage.html", {"request": request, "bio": user_data["bio"], "username": user_data["username"], "name": user_data["name"], "surname": user_data["surname"]})

@app.get("/settings_page")
def profile_page_endpoint_get(request: Request):
    response_to_user = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if response_to_user.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = response_to_user.json()
    print("User data:", user_data)
    return templates.TemplateResponse("settings.html", {"request": request, "bio": user_data["bio"], "username": user_data["username"], "name": user_data["name"], "surname": user_data["surname"], "born_date": user_data["born_date"]})

''' User service endpoints '''

@app.post("/register")
async def register(request: Request):
    data = await request.json()
    required_fields = ["name", "surname", "username", "password", "born_date"]
    if not all(field in data for field in required_fields):
        return JSONResponse(content={"error": "Missing required fields"}, status_code=400)
    request = requests.post("http://user_service:8000/register", json=data)
    return JSONResponse(content=request.json(), status_code=request.status_code)

@app.get("/get_user/{username}")
async def get_user(username: str):
    request = requests.get(f"http://user_service:8000/get_user/{username}")
    return JSONResponse(content=request.json(), status_code=request.status_code)

@app.post("/edit_user")
async def edit_user(request: Request):
    data = await request.json()
    required_fields = ["username", "new_name", "new_surname", "new_username", "new_born_date", "new_bio"]
    if not all(field in data for field in required_fields):
        return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

    request = requests.post(f"http://user_service:8000/edit_user", json=data)
    return JSONResponse(content=request.json(), status_code=request.status_code)

@app.delete("/delete_user")
async def delete_user(request: Request):
    data = await request.json()
    if "username" not in data:
        return JSONResponse(content={"error": "Missing username field"}, status_code=400)

    response = requests.post(f"http://user_service:8000/delete_user", json=data)
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.delete("/delete_me")
async def delete_user(request: Request):
    response = requests.delete(f"http://user_service:8000/delete_me", cookies=request.cookies)
    return JSONResponse(content=response.json(), status_code=response.status_code)

''' Functional endpoints '''

@app.post("/login")
async def login_user(request: Request):
    data = await request.json()
    if "username" not in data or "password" not in data:
        return JSONResponse(content={"error": "Username or password are null"}, status_code=400)

    response = requests.post(
        "http://user_service:8000/login",
        cookies=request.cookies,
        json=data
    )

    if response.status_code >= 400:
        return JSONResponse(content={"error": "Error while login has happened."}, status_code=500)

    user_data = response.json()
    cookie_header = response.headers.get("set-cookie")

    resp = JSONResponse(content=user_data, status_code=response.status_code)
    if cookie_header:
        resp.headers["set-cookie"] = cookie_header

    return resp

@app.post("/logout")
async def logout_user(request: Request):
    response = requests.post(
        "http://user_service:8000/logout",
        cookies=request.cookies
    )

    if response.status_code >= 400:
        return JSONResponse(content={"error": "Error while logout has happened."}, status_code=500)

    resp = JSONResponse(content=response.json(), status_code=response.status_code)
    resp.delete_cookie("session_id", path="/")
    return resp