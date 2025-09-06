from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import db
from fastapi.responses import JSONResponse

app = FastAPI(title="DB Service")

class UserCreate(BaseModel):
    name: str
    surname: str
    username: str
    password_hash: str
    born_date: str
    bio: Optional[str] = ""

class UserUpdate(BaseModel):
    new_name: Optional[str] = None
    new_surname: Optional[str] = None
    new_username: Optional[str] = None
    new_password_hash: Optional[str] = None
    new_born_date: Optional[str] = None
    new_bio: Optional[str] = None

@app.post("/create_user")
def create_user(user: UserCreate):
    try:
        new_user = db.insert_user(
            name=user.name,
            surname=user.surname,
            username=user.username,
            password_hash=user.password_hash,
            born_date=user.born_date,
            bio=user.bio or ""
        )
        return {"status": "success", "user": {"username": new_user.username}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_user/{username}")
def read_user(username: str):
    user = db.get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/update_user/{username}")
def update_user(username: str, user_update: UserUpdate):
    updated_user = db.update_user(
        username,
        new_name=user_update.new_name,
        new_surname=user_update.new_surname,
        new_username=user_update.new_username,
        new_password_hash=user_update.new_password_hash,
        new_born_date=user_update.new_born_date,
        new_bio=user_update.new_bio
    )
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "user": {"username": updated_user.username}}

@app.delete("/delete_user/{username}")
def delete_user(username: str):
    success = db.delete_user(username)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success"}

@app.get("/users", response_model=List[dict])
def list_users():
    return db.get_all_users()

##################
@app.get("/")
def root():
    return JSONResponse(content={"message": "DB service (gateway) is up!"}, status_code=200)

@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)