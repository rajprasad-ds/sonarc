import os
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from search.query_handler import search as run_search
from ingestion.indexer import index_folder

app = FastAPI(title="Sonarc API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class ScanRequest(BaseModel):
    folder_path: str


class OpenFolderRequest(BaseModel):
    file_path: str


@app.post("/search")
def search_endpoint(request: SearchRequest):
    results = run_search(request.query)
    return {"results": results}


@app.post("/scan")
def scan_endpoint(request: ScanRequest):
    index_folder(request.folder_path)
    return {"status": "done"}


@app.get("/audio")
def get_audio_file(path: str):
    if not os.path.exists(path):
        return {"error": "file not found"}
    return FileResponse(path)


@app.post("/open-folder")
def open_folder(request: OpenFolderRequest):
    subprocess.run(f'explorer /select,"{request.file_path}"')
    return {"status": "opened"}