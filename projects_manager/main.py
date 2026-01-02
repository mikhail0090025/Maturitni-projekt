from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import json
import requests
from typing import Optional, List
from data import IndexToDataType, DataTypeToIndex
import neural_net_manager as nn_manager
import os
from datasets import get_dataset
import datasets_templates as ds_templates
from torch.utils.data import Dataset, DataLoader

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
    project_json: Optional[str] = None
    dataset_id: Optional[int] = None
    dataset_preprocess_json: Optional[str] = None
    optimizer_json: Optional[str] = None
    scheduler_json: Optional[str] = None
    loss_function: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    owner_username: Optional[str] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    project_json: Optional[str] = None
    dataset_id: Optional[int] = None
    dataset_preprocess_json: Optional[str] = None
    optimizer_json: Optional[str] = None
    scheduler_json: Optional[str] = None
    loss_function: Optional[str] = None

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

''' DATASET ENDPOINTS '''

@app.get("/datasets/prepare_dataset/{dataset_id}/for_project/{project_id}")
def prepare_dataset_for_project(request: Request, dataset_id: int, project_id: int):
    request_ = requests.get(f"http://localhost:8003/{project_id}", cookies=request.cookies)
    if request_.status_code == 404:
        return JSONResponse(content={"detail": f"Project not found. ({request_.json()})"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": f"Failed to fetch project. {request_.json()}"}, status_code=500)
    project = request_.json()

    try:
        dataset = get_dataset(
            dataset_id=dataset_id,
            project_id=project_id,
            preprocess_json_text=project.get('dataset_preprocess_json', ''),
            dataset_type=ds_templates.input_output_type_to_dataset_type(
                project['input_type'], project['output_type'])
        )
        return JSONResponse(content={"detail": "Dataset prepared successfully", "num_samples": len(dataset)}, status_code=200)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": "Failed to prepare dataset"}, status_code=500)

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

@app.put("/set_training_config/")
async def set_training_config(request: Request):
    config = await request.json()
    print("Received optimizer config:", config.get("optimizer_json"))
    print("Received scheduler config:", config.get("scheduler_json"))
    print("For project ID:", config.get("projectId"))
    body_dict = {
        "optimizer_json": json.dumps(config.get("optimizer_json")),
        "scheduler_json": json.dumps(config.get("scheduler_json")) if config.get("scheduler_json") else None,
    }
    request_ = requests.put(
        f"http://db_service:8002/projects/{config.get('projectId')}",
        json=body_dict
    )
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": f"Failed to update project with training config: {request_.json()}"}, status_code=500)
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.post("/initialize_training/{project_id}")
def initialize_training(project_id: int, request: Request):
   
    request_get_config = requests.get(f"http://db_service:8002/projects/{project_id}")
    if request_get_config.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to fetch project for training config"}, status_code=500)
    project = request_get_config.json()
    print("Project training config:", project)
    optimizer_json = project.get("optimizer_json")
    scheduler_json = project.get("scheduler_json")
    architecture_json = project.get("project_json")
    loss_function_str = project.get("loss_function")
    print("Optimizer JSON for training:", optimizer_json)
    print("Scheduler JSON for training:", scheduler_json)
    print("Architecture JSON for training:", architecture_json)
    print("Loss function for training:", loss_function_str)
    full_model = nn_manager.create_full_model(
        architecture_json,
        optimizer_json,
        scheduler_json,
        criterion_name=loss_function_str if loss_function_str else "MSELoss"
    )
    full_model.save(os.path.join("projects", f"project_{project_id}_training_model.pth"))
    return JSONResponse(content={"detail": "Training initialization triggered"}, status_code=200)

@app.post("/start_training/{project_id}")
async def start_training(project_id: int, request: Request):
    try:
        request_body = await request.json()
        print("Received training start request with body:", request_body)
        training_data = request_body.get("training", {})
        print("Training data received:", training_data)
        model_path = os.path.join("projects", f"project_{project_id}_training_model.pth")
        if not os.path.exists(model_path):
            return JSONResponse(content={"detail": "Training model file not found. Initialize training first."}, status_code=404)

        request_get_config = requests.get(f"http://db_service:8002/projects/{project_id}")
        if request_get_config.status_code >= 400:
            return JSONResponse(content={"detail": "Failed to fetch project for training config"}, status_code=500)
        project = request_get_config.json()
        print("Project training config:", project)
        optimizer_json = project.get("optimizer_json")
        scheduler_json = project.get("scheduler_json")
        architecture_json = project.get("project_json")
        loss_function_str = project.get("loss_function")
        print("Optimizer JSON for training:", optimizer_json)
        print("Scheduler JSON for training:", scheduler_json)
        print("Architecture JSON for training:", architecture_json)
        print("Loss function for training:", loss_function_str)

        full_model = nn_manager.create_full_model(
            architecture_json,
            optimizer_json,
            scheduler_json,
            criterion_name=loss_function_str if loss_function_str else "MSELoss"
        )
        full_model.load(path=model_path)
        print("Loaded training model from:", model_path)
        print(project['dataset_id'])
        print(project_id)
        print(project.get('dataset_preprocess_json', ''))
        print(ds_templates.input_output_type_to_dataset_type(
                project['input_type'], project['output_type']))
        print(request.cookies)
        
        dataset = get_dataset(
            dataset_id=project['dataset_id'],
            project_id=project_id,
            preprocess_json_text=project.get('dataset_preprocess_json', ''),
            dataset_type=ds_templates.input_output_type_to_dataset_type(
                project['input_type'], project['output_type']),
            cookies=request.cookies
        )
        print(f"Dataset for training loaded. Number of samples: {len(dataset)}")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"Starting training for project {project_id} with model: {full_model.model}")
        return JSONResponse(content={"detail": "Training started"}, status_code=200)
    except Exception as e:
        print("Error starting training:", str(e))
        return JSONResponse(content={"detail": f"Failed to start training: {str(e)}"}, status_code=500)

@app.get("/model_size/{project_id}")
def model_size(project_id: int):
    model_path = os.path.join("projects", f"project_{project_id}_training_model.pth")
    if not os.path.exists(model_path):
        return JSONResponse(content={"detail": "Training model file not found. Initialize training first."}, status_code=404)

    request_get_config = requests.get(f"http://db_service:8002/projects/{project_id}")
    if request_get_config.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to fetch project for training config"}, status_code=500)
    project = request_get_config.json()
    print("Project training config:", project)
    optimizer_json = project.get("optimizer_json")
    scheduler_json = project.get("scheduler_json")
    architecture_json = project.get("project_json")
    loss_function_str = project.get("loss_function")
    print("Optimizer JSON for training:", optimizer_json)
    print("Scheduler JSON for training:", scheduler_json)
    print("Architecture JSON for training:", architecture_json)
    print("Loss function for training:", loss_function_str)

    full_model = nn_manager.create_full_model(
        architecture_json,
        optimizer_json,
        scheduler_json,
        criterion_name=loss_function_str if loss_function_str else "MSELoss"
    )

    parameters_count = 0
    for param in full_model.model.parameters():
        parameters_count += param.numel()

    if not os.path.exists(model_path):
        return JSONResponse(content={"detail": "Model file not found"}, status_code=404)
    size_bytes = os.path.getsize(model_path)
    return JSONResponse(content={"model_size_bytes": size_bytes, "parameters_count": parameters_count}, status_code=200)