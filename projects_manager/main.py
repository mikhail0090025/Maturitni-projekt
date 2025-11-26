from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import json
import requests
from typing import Optional, List
from data import IndexToDataType, DataTypeToIndex
import neural_net_manager as nn_manager
import os

app = FastAPI()

@app.get("/")
def root():
    return JSONResponse(content={"message": "Projects service is up!"}, status_code=200)

@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)

# --- Schemas ---
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    owner_username: str
    input_type: str
    output_type: str

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    owner_username: Optional[str] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    project_json: Optional[str] = None

# --- Endpoints ---
@app.post("/", response_model=dict)
def create_project(project: ProjectCreate):
    request_ = requests.post("http://db_service:8002/projects/", json=project.dict())
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to create project"}, status_code=500)
    return request_.json()

@app.get("/{project_id}", response_model=dict)
def read_project(project_id: int):
    request_ = requests.get(f"http://db_service:8002/projects/{project_id}")
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to fetch project"}, status_code=500)
    return request_.json()

@app.get("/json/{project_id}", response_model=dict)
def read_project_json(project_id: int):
    resp = requests.get(f"http://db_service:8002/projects/{project_id}")

    if resp.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if resp.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to fetch project"}, status_code=500)

    print("resp.json()", resp.json(), type(resp.json()))

    # Просто берем поле project_json
    project_json = resp.json()['project_json']
    if project_json is None:
        return JSONResponse(content={"detail": "Project JSON not found"}, status_code=500)

    return Response(project_json, 200)

@app.put("/{project_id}", response_model=dict)
def modify_project(project_id: int, updates: ProjectUpdate):
    request_ = requests.put(f"http://db_service:8002/projects/{project_id}", json=updates.dict(exclude_unset=True))
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to update project"}, status_code=500)
    return request_.json()

@app.delete("/{project_id}", response_model=dict)
def remove_project(project_id: int):
    request_ = requests.delete(f"http://db_service:8002/projects/{project_id}")
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to delete project"}, status_code=500)
    return request_.json()

@app.get("/user/{owner_username}", response_model=List[dict])
def list_user_projects(owner_username: str):
    request_ = requests.get(f"http://db_service:8002/projects/user/{owner_username}")
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to fetch user's projects"}, status_code=500)
    return request_.json()

@app.get("/projects/list")
# @app.get("/list_projects", response_model=list[dict])
def list_projects():
    request_ = requests.get("http://db_service:8002/projects/")
    print(request_.json())
    print(request_.status_code)
    print(request_)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to fetch projects"}, status_code=500)
    return request_.json()

''' MODELS ENDPOINTS '''
@app.post("/create_model/{project_id}")
def create_model_endpoint(project_id: int):
    request_ = requests.get(f"http://db_service:8002/projects/{project_id}")
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to fetch project"}, status_code=500)
    project = request_.json()
    model = nn_manager.create_model(project['project_json'])
    full_model = nn_manager.FullModel(model)
    print("Creating model for project:", project)
    full_model.save(os.path.join("projects", f"project_{project_id}_model.pth"))
    return JSONResponse(content={"detail": "Model creation triggered", "project": project}, status_code=200)

@app.get("/get_model_file/{project_id}")
def get_model_file(project_id: int):
    model_path = os.path.join("projects", f"project_{project_id}_model.pth")
    if not os.path.exists(model_path):
        return JSONResponse(content={"detail": "Model file not found"}, status_code=404)
    with open(model_path, "rb") as f:
        model_data = f.read()
    return Response(content=model_data, media_type="application/octet-stream")

''' OTHER ENDPOINTS '''

@app.get("/data_type_to_index/{data_type}")
def data_type_to_index(data_type: str):
    print("Data type:", data_type)
    try:
        index = DataTypeToIndex(data_type)
        return JSONResponse(content={"index": index}, status_code=200)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": "Failed to fetch data type index"}, status_code=500)