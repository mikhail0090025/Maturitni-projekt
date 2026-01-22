from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, APIRouter, Form
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import bcrypt
import secrets
import uuid
import json
import requests
import os
import httpx
from typing import Optional, Dict, Any
from pydantic import BaseModel
import io
import torch
import numpy as np
from PIL import Image

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
    return templates.TemplateResponse("settings.html", {"request": request, "bio": user_data["bio"], "username": user_data["username"], "name": user_data["name"], "surname": user_data["surname"], "born_date": user_data["born_date"]})

@app.get("/new_project_page")
def profile_page_endpoint_get(request: Request):
    response_to_user = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if response_to_user.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = response_to_user.json()

    available_data_types = requests.get("http://db_service:8002/enums/datatypes").json()
    print(available_data_types)
    if not available_data_types:
        return JSONResponse(content={"error": "No data types available"}, status_code=500)
    return templates.TemplateResponse("new_project.html", {"request": request, "available_data_types": available_data_types, "bio": user_data["bio"], "username": user_data["username"], "name": user_data["name"], "surname": user_data["surname"], "born_date": user_data["born_date"]})

@app.get("/all_datasets_page")
def all_datasets_page(request: Request):
    response_to_user = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if response_to_user.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = response_to_user.json()

    datasets_response = requests.get("http://db_service:8002/datasets")
    if datasets_response.status_code >= 400:
        return JSONResponse(content={"error": "Failed to fetch datasets"}, status_code=500)
    datasets = datasets_response.json()
    print("Datasets:")
    print(datasets)
    for dataset in datasets:
        dataset_type_info = requests.get(f"http://datasets_manager:8004/datasets/dataset_type_name/{dataset['dataset_type']}")
        if dataset_type_info.status_code == 200:
            dataset['dataset_type_description'] = dataset_type_info.json().get('description', 'No description available')
        else:
            dataset['dataset_type_description'] = 'No description available'

    from datetime import datetime
    for dataset in datasets:
        dataset['created_at'] = datetime.fromisoformat(dataset['created_at']).strftime('%Y-%m-%d %H:%M:%S')

    return templates.TemplateResponse("all_datasets.html", {"request": request, "datasets": datasets, "bio": user_data["bio"], "username": user_data["username"], "name": user_data["name"], "surname": user_data["surname"], "born_date": user_data["born_date"]})

@app.get("/new_dataset_page")
def new_dataset_page(request: Request):
    response_to_user = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if response_to_user.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = response_to_user.json()
    return templates.TemplateResponse("new_dataset.html", {"request": request, "bio": user_data["bio"], "username": user_data["username"], "name": user_data["name"], "surname": user_data["surname"], "born_date": user_data["born_date"]})

@app.get("/detr_page")
def detr_page(request: Request):
    return templates.TemplateResponse("detr.html", {"request": request})

@app.get("/cats_page")
def cats_page(request: Request):
    return templates.TemplateResponse("cats.html", {"request": request})

@app.get("/gif_generate")
async def gif_endpoint():
    req_response = requests.get("http://my_models:8005/gif_generate", stream=True)
    if req_response.status_code != 200:
        return Response(content=req_response.content, status_code=req_response.status_code)
    return StreamingResponse(req_response.iter_content(chunk_size=8192), media_type="image/gif")

@app.get("/generate")
async def image_endpoint():
    req_response = requests.get("http://my_models:8005/generate", stream=True)
    if req_response.status_code != 200:
        return Response(content=req_response.content, status_code=req_response.status_code)
    return StreamingResponse(req_response.iter_content(chunk_size=8192), media_type="image/png")

@app.get("/mainpage")
def mainpage(request: Request):
    return RedirectResponse(url="/profile_page")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((240, 240), Image.BILINEAR)
    image_np = np.array(image)
    image = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    image = image / 255.0
    tensor_bytes = io.BytesIO()
    torch.save(image, tensor_bytes)
    tensor_bytes.seek(0)
    response = requests.post(
        "http://my_models:8005/predict",
        files={"tensor_file": tensor_bytes}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Model service error")
    print(response.json())
    return JSONResponse(content=response.json())

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

''' Projects manager endpoints '''

@app.post("/new_project")
async def new_project(request: Request):
    data = await request.json()
    print("Data:", data)
    required_fields = ["name", "owner_username", "input_type", "output_type"]
    if not all(field in data for field in required_fields):
        return JSONResponse(content={"error": "Missing required fields"}, status_code=400)

    response = requests.post("http://projects_manager:8003/", json=data)
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.get("/data_type_to_index/{data_type}")
def data_type_to_index(data_type: str):
    print("Data type:", data_type)
    request = requests.get(f"http://projects_manager:8003/data_type_to_index/{data_type}")
    if request.status_code >= 400:
        return JSONResponse(content={"error": "Failed to fetch data type index"}, status_code=500)
    index = request.json().get("index")
    if index is None:
        return JSONResponse(content={"error": "Data type not found"}, status_code=404)
    return JSONResponse(content={"index": index}, status_code=200)

@app.get("/my_projects")
def get_my_projects(request: Request):
    response_to_user = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if response_to_user.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = response_to_user.json()
    username = user_data['username']
    projects_response = requests.get(f'http://projects_manager:8003/user/{username}')
    if projects_response.status_code >= 400:
        return JSONResponse(content={'error': f'Couldnt get projects of user {username}'})
    data = projects_response.json()
    
    return data

@app.get("/projects/{project_id}")
def get_project(project_id: int, request: Request):
    project_response = requests.get(f'http://projects_manager:8003/{project_id}')
    if project_response.status_code >= 400:
        return JSONResponse(content={'error': f'Couldnt get project with id {project_id}'})
    user_response = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    print(user_response.status_code)
    print(user_response.json())
    if user_response.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = user_response.json()
    if user_data['username'] != project_response.json().get('owner_username'):
        return templates.TemplateResponse("error_page.html", {"request": request, "error_message": "You are not the owner of this project!"})
    data = project_response.json()

    datasets_response = requests.get("http://db_service:8002/datasets")
    if datasets_response.status_code >= 400:
        return templates.TemplateResponse("error_page.html", {"request": request, "error_message": f"Failed to fetch datasets: {datasets_response.text}"})
    datasets = datasets_response.json()
    print("Datasets:")
    print(datasets)
    for dataset in datasets:
        dataset_type_info = requests.get(f"http://datasets_manager:8004/datasets/dataset_type_name/{dataset['dataset_type']}")
        if dataset_type_info.status_code == 200:
            dataset['dataset_type_description'] = dataset_type_info.json().get('description', 'No description available')
        else:
            dataset['dataset_type_description'] = 'No description available'

    from datetime import datetime
    for dataset in datasets:
        dataset['created_at'] = datetime.fromisoformat(dataset['created_at']).strftime('%Y-%m-%d %H:%M:%S')

    return templates.TemplateResponse("project_page.html", {"request": request, "project_data": data, "datasets": datasets})

@app.post("/projects/save")
async def update_project(request: Request):

    body = await request.json()
    project_id = body["project_id"]
    project_json = body["project_json"]

    project_response = requests.get(f'http://projects_manager:8003/{project_id}')
    if project_response.status_code >= 400:
        return JSONResponse(content={'error': f'Couldnt get project with id {project_id}'})
    user_response = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if user_response.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = user_response.json()
    if user_data['username'] != project_response.json().get('owner_username'):
        return templates.TemplateResponse("error_page.html", {"request": request, "error_message": "You are not the owner of this project!"})
    data = project_response.json()

    project_update_response = requests.put(
        f'http://projects_manager:8003/{project_id}',
        json={
            "name": data['name'],
            "description": data.get('description', ''),
            "owner_username": data['owner_username'],
            "input_type": data['input_type'],
            "output_type": data['output_type'],
            "project_json": project_json
        }
    )
    if project_update_response.status_code >= 400:
        return templates.TemplateResponse("error_page.html", {"request": request, "error_message": "Failed to save project data!"})

    return JSONResponse(content={"detail": "Project saved successfully"}, status_code=200)

@app.put("/loss")
async def update_project_loss(request: Request):
    try:
        body = await request.json()
        project_id = body["project_id"]
        loss_function = body["loss_function"]
        print("Updating project", project_id, "with loss function", loss_function)

        project_response = requests.get(f'http://projects_manager:8003/{project_id}')
        if project_response.status_code >= 400:
            return JSONResponse(content={'error': f'Couldnt get project with id {project_id}'})
        user_response = requests.get(
            "http://user_service:8000/me",
            cookies=request.cookies
        )
        if user_response.status_code >= 400:
            return RedirectResponse(url="/login_page")
        user_data = user_response.json()
        if user_data['username'] != project_response.json().get('owner_username'):
            return templates.TemplateResponse("error_page.html", {"request": request, "error_message": "You are not the owner of this project!"})
        data = project_response.json()

        project_update_response = requests.put(
            f'http://projects_manager:8003/{project_id}',
            json={
                "name": data['name'],
                "description": data.get('description', ''),
                "owner_username": data['owner_username'],
                "input_type": data['input_type'],
                "output_type": data['output_type'],
                "loss_function": loss_function
            }
        )
        if project_update_response.status_code >= 400:
            return templates.TemplateResponse("error_page.html", {"request": request, "error_message": "Failed to save project loss function!"})

        return JSONResponse(content={"detail": "Project loss function saved successfully"}, status_code=200)
    
    except Exception as e:
        print("Error in update_project_loss:", str(e))
        return JSONResponse(content={"error": f"An error occurred while updating the loss function: {str(e)}"}, status_code=500)

@app.post("/delete_project/{project_id}")
def delete_project_endpoint(project_id: int, request: Request):
    project_response = requests.get(f'http://projects_manager:8003/{project_id}')
    if project_response.status_code >= 400:
        return JSONResponse(content={'error': f'Couldnt get project with id {project_id}'})
    user_response = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if user_response.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = user_response.json()
    if user_data['username'] != project_response.json().get('owner_username'):
        return templates.TemplateResponse("error_page.html", {"request": request, "error_message": "You are not the owner of this project!"})

    delete_response = requests.delete(f'http://projects_manager:8003/{project_id}')
    if delete_response.status_code >= 400:
        return templates.TemplateResponse("error_page.html", {"request": request, "error_message": "Failed to delete project!"})

    return JSONResponse(content={"detail": "Project deleted successfully"}, status_code=200)

''' Datasets manager endpoints '''

DATASETS_MANAGER_URL = "http://datasets_manager:8004"

dataset_router = APIRouter(prefix="/datasets", tags=["datasets"])
upload_router = APIRouter(prefix="/upload", tags=["upload"])

# --- Proxy Endpoints для Datasets ---

@dataset_router.post("/")
async def create_dataset(dataset: dict):
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{DATASETS_MANAGER_URL}/datasets/", json=dataset)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()

@dataset_router.get("/")
async def list_datasets():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{DATASETS_MANAGER_URL}/datasets/")
        return JSONResponse(content=r.json(), status_code=r.status_code)

@dataset_router.get("/{dataset_id}")
async def read_dataset(dataset_id: int):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{DATASETS_MANAGER_URL}/datasets/{dataset_id}")
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return r.json()

@dataset_router.put("/{dataset_id}")
async def modify_dataset(dataset_id: int, updates: dict):
    async with httpx.AsyncClient() as client:
        r = await client.put(f"{DATASETS_MANAGER_URL}/datasets/{dataset_id}", json=updates)
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return r.json()

@dataset_router.delete("/{dataset_id}")
async def remove_dataset(dataset_id: int):
    async with httpx.AsyncClient() as client:
        r = await client.delete(f"{DATASETS_MANAGER_URL}/datasets/{dataset_id}")
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"detail": "Dataset deleted successfully"}

@dataset_router.delete("/by_storage/{storage_id}")
def proxy_delete_dataset(storage_id: str, request: Request):
    print(f"[FRONTEND DELETE] Request to delete dataset with StorageID: {storage_id}")
    cookies = request.cookies

    resp = requests.delete(
        f"http://datasets_manager:8004/datasets/by_storage/{storage_id}",
        cookies=cookies
    )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("Content-Type", "application/json"),
    )

@dataset_router.get("/download/{storage_id}")
def proxy_download(storage_id: str, request: Request):
    resp = requests.get(
        f"http://datasets_manager:8004/datasets/download/{storage_id}",
        cookies=request.cookies,
        stream=True
    )

    if resp.status_code != 200:
        return Response(content=resp.content, status_code=resp.status_code)

    return StreamingResponse(
        resp.raw,
        media_type=resp.headers["Content-Type"],
        headers={"Content-Disposition": resp.headers["Content-Disposition"]},
    )

# --- Proxy Endpoints для Upload ---

@upload_router.post("/zip")
async def upload_zip(
        file: UploadFile = File(...),
        name: str = Form(...),
        description: str = Form(""),
        request: Request = None
    ):
    print("Received upload request for file:", file.filename)

    files = {
        "file": (file.filename, await file.read(), file.content_type)
    }

    data = {
        "name": name,
        "description": description
    }

    url = f"{DATASETS_MANAGER_URL}/upload/zip"
    print(f"Forwarding upload to {url}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(
            url,
            files=files,
            data=data,
            cookies=request.cookies
        )
    print("Response from Datasets manager:", r.content)

    return JSONResponse(content=r.json(), status_code=r.status_code)

# Other endpoints

@app.post("/save_dataset_settings")
async def save_dataset_settings(request: Request):

    body = await request.json()
    dataset_id = body["dataset_id"]
    project_id = body["project_id"]
    dataset_preprocess_json = body["preprocessing_config"]

    project_response = requests.get(f'http://projects_manager:8003/{project_id}')
    if project_response.status_code >= 400:
        return JSONResponse(content={'error': f'Couldnt get project with id {project_id}'})
    user_response = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if user_response.status_code >= 400:
        return RedirectResponse(url="/login_page")
    user_data = user_response.json()
    if user_data['username'] != project_response.json().get('owner_username'):
        return templates.TemplateResponse("error_page.html", {"request": request, "error_message": "You are not the owner of this project!"})
    data = project_response.json()

    project_update_response = requests.put(
        f'http://projects_manager:8003/{project_id}',
        json={
            "name": data['name'],
            "description": data.get('description', ''),
            "owner_username": data['owner_username'],
            "input_type": data['input_type'],
            "output_type": data['output_type'],
            "dataset_id": dataset_id,
            "dataset_preprocess_json": dataset_preprocess_json,
        }
    )
    sentData = {
        "name": data['name'],
        "description": data.get('description', ''),
        "owner_username": data['owner_username'],
        "input_type": data['input_type'],
        "output_type": data['output_type'],
        "dataset_id": dataset_id,
        "dataset_preprocess_json": dataset_preprocess_json,
    }
    print("Sent data for project update:", sentData)
    if project_update_response.status_code >= 400:
        return JSONResponse(content={"error_message": f"Failed to save project data!: {project_update_response.text}"}, status_code=project_update_response.status_code)

    return JSONResponse(content={"detail": "Project saved successfully"}, status_code=200)

@app.get("/load_dataset_settings/{project_id}")
async def load_dataset_settings(project_id: int, request: Request):

    # 1. Проверка пользователя
    user_response = requests.get(
        "http://user_service:8000/me",
        cookies=request.cookies
    )
    if user_response.status_code >= 400:
        return RedirectResponse(url="/login_page")

    user_data = user_response.json()

    # 2. Получаем проект
    project_response = requests.get(
        f"http://projects_manager:8003/{project_id}"
    )
    if project_response.status_code >= 400:
        return JSONResponse(
            content={"error": f"Could not get project with id {project_id}"},
            status_code=404
        )

    project_data = project_response.json()

    # 3. Проверка владельца
    if project_data.get("owner_username") != user_data["username"]:
        return JSONResponse(
            content={"error": "You are not the owner of this project"},
            status_code=403
        )

    # 4. Возвращаем ТОЛЬКО нужное
    return JSONResponse(
        content={
            "dataset_id": project_data.get("dataset_id"),
            "dataset_preprocess_json": project_data.get("dataset_preprocess_json", "")
        },
        status_code=200
    )

@app.get("/prepare_dataset/{dataset_id}/for_project/{project_id}")
def prepare_dataset_for_project(request: Request, dataset_id: int, project_id: int):
    request_ = requests.get(f"http://projects_manager:8003/datasets/prepare_dataset/{dataset_id}/for_project/{project_id}", cookies=request.cookies)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to prepare dataset for project"}, status_code=500)
    return request_.json()

# Training endpoints

class OptimizerConfig(BaseModel):
    type: str
    lr: float
    weight_decay: float
    betas: list[float]

class SchedulerConfig(BaseModel):
    type: str
    total_steps: Optional[int] = None
    min_lr: Optional[float] = None
    warmup_steps: Optional[int] = None
    mode: Optional[str] = None
    factor: Optional[float] = None
    patience: Optional[int] = None

class TrainingConfig(BaseModel):
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    projectId: int

@app.post("/set_training_config/")
async def set_training_config(config: TrainingConfig):
    print("Received optimizer config:", config.optimizer)
    print("Received scheduler config:", config.scheduler)
    print("For project ID:", config.projectId)
    body_dict = {
        "optimizer_json": json.dumps(config.optimizer.dict()),
        "scheduler_json": json.dumps(config.scheduler.dict()) if config.scheduler else None,
        "projectId": config.projectId
    }
    print("Body dict to send:", body_dict)
    request_ = requests.put(
        f"http://projects_manager:8003/set_training_config/",
        json=body_dict
    )
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": f"Failed to update project with training config: {request_.json()}"}, status_code=500)
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.post("/initialize_training/{project_id}")
async def initialize_training(project_id: int, request: Request):
    request_body = await request.json()
    print("Initialize training request body:", request_body)
    request_ = requests.post(f"http://projects_manager:8003/initialize_training/{project_id}", json=request_body, cookies=request.cookies)
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to initialize training"}, status_code=500)
    return request_.json()

@app.post("/start_training/{project_id}")
async def start_training(project_id: int, request: Request):
    try:
        request_body = await request.json()
        print("Start training request body:", request_body)
        request_ = requests.post(f"http://projects_manager:8003/start_training/{project_id}", cookies=request.cookies, json=request_body)
        if request_.status_code == 404:
            return JSONResponse(content={"detail": "Project not found"}, status_code=404)
        if request_.status_code >= 400:

            return JSONResponse(content={"detail": f"Failed to start training: {request_.json()}"}, status_code=500)
        return request_.json()
    except Exception as e:
        print("Error during start_training:", str(e))
        return JSONResponse(content={"detail": f"Internal server error: {str(e)}"}, status_code=500)

@app.get("/model_size/{project_id}")
async def model_size(project_id: int):
    model_path = os.path.join("projects", f"project_{project_id}_model.pth")
    if not os.path.exists(model_path):
        return JSONResponse(content={"detail": "Model file not found"}, status_code=404)
    size_bytes = os.path.getsize(model_path)
    return JSONResponse(content={"model_size_bytes": size_bytes}, status_code=200)

@app.get("/get_train_status/{project_id}")
def get_train_status(project_id: int, request: Request):
    try:
        request_ = requests.get(f"http://projects_manager:8003/get_train_status/{project_id}", cookies=request.cookies)

        if request_.status_code == 404:
            return JSONResponse(content={"detail": "Project not found"}, status_code=404)
        if request_.status_code >= 400:
            return JSONResponse(content={"detail": f"Failed to start training: {request_.json()}"}, status_code=500)

        return request_.json()
    except Exception as e:
        print("Error during getting status:", str(e))
        return JSONResponse(content={"detail": f"Internal server error: {str(e)}"}, status_code=500)

@app.get("/projects/{project_id}/loss-plot")
async def get_loss_plot(project_id: int):
    request_ = requests.get(f"http://projects_manager:8003/projects/{project_id}/loss-plot")
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to get loss plot"}, status_code=500)
    return JSONResponse(content=request_.json(), status_code=200)

@app.post("/projects/{project_id}/reset")
async def get_loss_plot(project_id: int):
    request_ = requests.post(f"http://projects_manager:8003/projects/{project_id}/reset")
    if request_.status_code == 404:
        return JSONResponse(content={"detail": "Project not found"}, status_code=404)
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to get loss plot"}, status_code=500)
    return JSONResponse(content=request_.json(), status_code=200)

@app.get("/losstypes")
def get_loss_types():
    request_ = requests.get("http://db_service:8002/enums/losstypes")
    if request_.status_code >= 400:
        return JSONResponse(content={"detail": "Failed to fetch loss types"}, status_code=500)
    return JSONResponse(content=request_.json(), status_code=200)

# --- Health Check ---

@app.get("/health", tags=["health"])
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)

app.include_router(dataset_router)
app.include_router(upload_router)