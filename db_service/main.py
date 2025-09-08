from fastapi import FastAPI, HTTPException, Request, APIRouter
from pydantic import BaseModel
from typing import Optional, List
import db
from models import DataType
from fastapi.responses import JSONResponse

app = FastAPI(title="DB Service")

''' CRUD endpoints for users'''

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

''' CRUD endpoints for projects '''

projects_router = APIRouter(prefix="/projects", tags=["projects"])

# --- Schemas ---
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    owner_username: str
    input_type: DataType
    output_type: DataType

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    owner_username: Optional[str] = None
    input_type: Optional[DataType] = None
    output_type: Optional[DataType] = None


# --- Endpoints ---
@projects_router.post("/", response_model=dict)
def create_project(project: ProjectCreate):
    new_project = db.insert_project(
        name=project.name,
        description=project.description,
        owner_username=project.owner_username,
        input_type=project.input_type,
        output_type=project.output_type,
    )
    return {
        "id": new_project.id,
        "name": new_project.name,
        "description": new_project.description,
        "owner_username": new_project.owner_username,
        "input_type": new_project.input_type.value,
        "output_type": new_project.output_type.value,
        "created_at": str(new_project.created_at),
    }

@projects_router.get("/{project_id}", response_model=dict)
def read_project(project_id: int):
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@projects_router.put("/{project_id}", response_model=dict)
def modify_project(project_id: int, updates: ProjectUpdate):
    updated = db.update_project(
        id=project_id,
        new_name=updates.name,
        new_description=updates.description,
        new_owner_username=updates.owner_username,
        new_input_type=updates.input_type,
        new_output_type=updates.output_type,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "id": updated.id,
        "name": updated.name,
        "description": updated.description,
        "owner_username": updated.owner_username,
        "input_type": updated.input_type.value,
        "output_type": updated.output_type.value,
        "created_at": str(updated.created_at),
    }

@projects_router.delete("/{project_id}", response_model=dict)
def remove_project(project_id: int):
    deleted = db.delete_project(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"detail": "Project deleted successfully"}

@projects_router.get("/")
def list_projects():
    all_projects = db.get_all_projects()
    print(all_projects)
    return JSONResponse(content=all_projects, status_code=200)

''' OTHER '''

enum_router = APIRouter(prefix="/enums", tags=["enums"])

@enum_router.get("/datatypes")
def list_data_types():
    return [dt.value for dt in DataType]

##################
@app.get("/")
def root():
    return JSONResponse(content={"message": "DB service (gateway) is up!"}, status_code=200)

@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)

app.include_router(projects_router)
app.include_router(enum_router)