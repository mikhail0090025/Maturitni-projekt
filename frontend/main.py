from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
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

@app.get("/registration_page")
def registration_page(request: Request):
    return templates.TemplateResponse("registration.html", {"request": request})

@app.get("/login_page")
def login_page_endpoint_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

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
    required_fields = ["username", "new_name", "new_surname", "new_username", "new_password", "new_born_date"]
    if not all(field in data for field in required_fields):
        return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

    request = requests.post(f"http://user_service:8000/edit_user", json=data)
    return JSONResponse(content=request.json(), status_code=request.status_code)

@app.post("/delete_user")
async def delete_user(request: Request):
    data = await request.json()
    if "username" not in data:
        return JSONResponse(content={"error": "Missing username field"}, status_code=400)

    response = requests.post(f"http://user_service:8000/delete_user", json=data)
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/login")
async def login_user(request: Request):
    data = await request.json()
    if "username" not in data or "password" not in data:
        return JSONResponse(content={"error": "Username or password are null"}, status_code=400)
    
    response = requests.post("https://user_service:8000/login", cookies=request.cookies)
    if response.status_code >= 400:
        return JSONResponse(content={"error": "Error while login has happened."}, status_code=500)
    return JSONResponse(content=response.json(), status_code=response.status_code)