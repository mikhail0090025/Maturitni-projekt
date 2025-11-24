from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File, Form, Request
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse, Response, FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
import httpx
import shutil
import uuid
import os
from functions import classify_dataset
import zipfile
import shutil
import tempfile
import requests
import secrets

app = FastAPI(title="Datasets Manager Service")

datasets_router = APIRouter(prefix="/datasets", tags=["datasets"])
upload_router = APIRouter(prefix="/upload", tags=["upload"])

DATASETS_DIR = "./datasets"

DB_SERVICE_URL = "http://db_service:8002/datasets"

# --- Schemas ---
class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    storage_id: str = secrets.token_hex(64)
    owner_id: int

class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    storage_id: Optional[str] = None
    owner_id: Optional[int] = None


# --- Endpoints-прокси ---

@datasets_router.post("/", response_model=dict)
async def create_dataset(dataset: DatasetCreate):
    async with httpx.AsyncClient() as client:
        r = await client.post(DB_SERVICE_URL + "/", json=dataset.dict())
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()


@datasets_router.get("/", response_model=dict)
async def list_datasets():
    async with httpx.AsyncClient() as client:
        r = await client.get(DB_SERVICE_URL + "/")
        return JSONResponse(content=r.json(), status_code=r.status_code)


@datasets_router.get("/{dataset_id}", response_model=dict)
async def read_dataset(dataset_id: int):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{DB_SERVICE_URL}/{dataset_id}")
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return r.json()


@datasets_router.put("/{dataset_id}", response_model=dict)
async def modify_dataset(dataset_id: int, updates: DatasetUpdate):
    async with httpx.AsyncClient() as client:
        r = await client.put(f"{DB_SERVICE_URL}/{dataset_id}", json=updates.dict())
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return r.json()


@datasets_router.delete("/{dataset_id}", response_model=dict)
async def remove_dataset(dataset_id: int):
    async with httpx.AsyncClient() as client:
        r = await client.delete(f"{DB_SERVICE_URL}/{dataset_id}")
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"detail": "Dataset deleted successfully"}

@upload_router.post("/zip")
async def upload_zip(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    request: Request = None
):
    print(f"[UPLOAD] Received: {file.filename}, type: {file.content_type}")

    # ----- 1. Проверяем тип файла, но мягко -----
    allowed_types = {
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
        "multipart/form-data"
    }

    if file.content_type not in allowed_types:
        print(f"[UPLOAD] Weird content type: {file.content_type}, but continuing anyway")

    # ----- 2. Генерируем storage_id -----
    dataset_id = str(uuid.uuid4())
    save_path = os.path.join(DATASETS_DIR, f"{dataset_id}.zip")

    print(f"[UPLOAD] Saving ZIP to: {save_path}")

    # ----- 3. Сохраняем ZIP как есть -----
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return Response(f"Failed to save file: {e}", 500)

    # ----- 4. Распаковка -----
    tmp_dir = tempfile.mkdtemp()
    print(f"[UPLOAD] Extracting to temp: {tmp_dir}")

    try:
        with zipfile.ZipFile(save_path, 'r') as z:
            z.extractall(tmp_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(tmp_dir)
        return Response("Uploaded file is not a valid ZIP archive", 400)
    except Exception as e:
        shutil.rmtree(tmp_dir)
        return Response(f"ZIP extraction failed: {e}", 500)

    # ----- 5. Классификация -----
    print("[UPLOAD] Classifying dataset...")
    try:
        dataset_type = classify_dataset(tmp_dir)
    except Exception as e:
        shutil.rmtree(tmp_dir)
        return Response(f"Classification failed: {e}", 500)

    print(f"[UPLOAD] Classified as: {dataset_type}")

    # ----- 6. Получаем user_id -----
    print("[UPLOAD] Fetching user ID…")

    try:
        user_data_response = requests.get(
            "http://user_service:8000/me",
            cookies=request.cookies
        )
    except Exception as e:
        shutil.rmtree(tmp_dir)
        return Response(f"User service unavailable: {e}", 500)

    if user_data_response.status_code != 200:
        shutil.rmtree(tmp_dir)
        return Response(
            f"Unable to fetch user data: {user_data_response.text}",
            400
        )

    user_id = user_data_response.json()["id"]
    print(f"[UPLOAD] User ID: {user_id}")

    # ----- 7. Создаём запись в БД -----
    print("[UPLOAD] Creating DB record…")

    try:
        # ЕСЛИ в Docker → замени localhost на имя сервиса
        db_response = requests.post(
            "http://datasets_manager:8004/datasets/",
            json={
                "name": name,
                "description": description,
                "storage_id": dataset_id,
                "owner_id": user_id
            }
        )
    except Exception as e:
        shutil.rmtree(tmp_dir)
        return Response(f"DB service unavailable: {e}", 500)

    if db_response.status_code != 200:
        shutil.rmtree(tmp_dir)
        return Response(f"Unable to create dataset: {db_response.text}", 500)

    print("[UPLOAD] DB record created successfully")

    # ----- 8. Чистим временную директорию -----
    shutil.rmtree(tmp_dir)

    # ----- 9. Возвращаем нормальный ответ -----
    return {
        "message": "OK",
        "dataset_type": dataset_type,
        "storage_id": dataset_id,
        "filename": file.filename,
        "saved_as": f"{dataset_id}.zip"
    }

@upload_router.get("/zip")
def upload_zip_get(request: Request = None):
    return {
        "message": "This is upload router GET endpoint. It is health"
    }

app.include_router(datasets_router)
app.include_router(upload_router)

@app.get("/health", tags=["health"])
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)