from pydantic import BaseModel
from typing import List
import uuid

class TemporalBoundary(BaseModel):
    start_time: float   # in seconds, e.g. 74.0
    end_time: float     # in seconds, e.g. 77.0

class FileLocation(BaseModel):
    path: str           # e.g. "C:/Audio/epic_action.wav"

class SemanticEmbedding(BaseModel):
    vector: List[float] # 512 floats from CLAP model

class AudioTrack(BaseModel):
    id: str = ""
    name: str           # e.g. "epic_action.wav"
    file_location: FileLocation

    def __init__(self, **data):
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid.uuid4())
        super().__init__(**data)

class AudioSegment(BaseModel):
    id: str = ""
    parent_track_id: str
    temporal_boundary: TemporalBoundary
    embedding: SemanticEmbedding

    def __init__(self, **data):
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid.uuid4())
        super().__init__(**data)