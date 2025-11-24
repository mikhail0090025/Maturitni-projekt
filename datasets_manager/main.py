from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File, HTTPException, Form, Request
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
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
    print(f"Received upload request for file: {file.filename}")
    # 1. Format check
    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    print("File format verified as ZIP")
    # 2. storage_id
    dataset_id = str(uuid.uuid4())
    save_path = os.path.join(DATASETS_DIR, f"{dataset_id}.zip")

    print(f"Generated dataset ID: {dataset_id}, saving to {save_path}")
    # 3. сохраняем ZIP как есть
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tmp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory at {tmp_dir} for extraction")

    try:
        print("Extracting ZIP contents...")
        # Extract
        with zipfile.ZipFile(save_path, 'r') as z:
            z.extractall(tmp_dir)

        print("Classifying dataset...")

        dataset_type = classify_dataset(tmp_dir)

        print(f"Dataset classified as type: {dataset_type}")

        user_data_response = requests.get("http://user_service:8000/me", cookies=request.cookies)
        if user_data_response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Unable to fetch user data {user_data_response.text}")
        user_data = user_data_response.json()
        user_id = user_data["id"]

        print(f"Fetched user ID: {user_id}")

        response = requests.post("http://localhost:8004/datasets/", json={
            "name": name,
            "description": description,
            "storage_id": dataset_id,
            "owner_id": user_id
        })
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Unable to create dataset record in DB {response.text}")
        
        print("Dataset record created in DB")

        dataset_record = response.json()
        print("Created dataset record:", dataset_record)

        return {
            "message": "OK",
            "dataset_type": dataset_type,
            "storage_id": dataset_id,
            "filename": file.filename,
            "saved_as": f"{dataset_id}.zip"
        }
    
    except zipfile.BadZipFile:
        print("Uploaded file is not a valid ZIP archive")
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive")
    
    except Exception as e:
        print(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir)

app.include_router(datasets_router)
app.include_router(upload_router)

@app.get("/health", tags=["health"])
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)