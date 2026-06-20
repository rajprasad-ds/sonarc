# 🎧 Semantic Audio Search Engine — Master Context Document
> Paste this at the start of ANY new LLM conversation to restore full project context instantly.

---

## What This Project Is
A **local desktop app** for video editors. They type a natural language query like *"water bubble smashing"* or *"intense excitement"* and the app returns the **exact audio file + timestamp** (e.g., `epic_action.wav → 01:14 to 01:17`) from their local audio library.

No cloud. No subscription. Runs entirely on the user's machine.

---

## Tech Stack (Don't change these)
| Layer | Technology | Why |
|---|---|---|
| AI Model | `laion/clap-htsat-unfused` via HuggingFace | Maps audio AND text into same vector space |
| Audio Processing | Python + `librosa` | Slices .wav/.mp3 into 3-second chunks |
| Vector Database | `ChromaDB` (local) | Stores embeddings + metadata |
| Backend | `FastAPI` (Python) | Local server handling inference + search |
| Frontend | `React` + `Tauri` | Desktop app UI (Windows, Mac, Linux) |

---

## Project Folder Structure
```
semantic-audio-search/
├── backend/
│   ├── ingestion/
│   │   ├── audio_chunker.py       ← Slices audio into 3s segments
│   │   └── embedding_generator.py ← Runs CLAP model, stores to ChromaDB
│   ├── search/
│   │   └── query_handler.py       ← Takes text query, returns top 5 matches
│   ├── db/
│   │   └── chroma_client.py       ← ChromaDB setup and operations
│   ├── models/
│   │   └── domain_models.py       ← Pydantic data models (see below)
│   └── main.py                    ← FastAPI app entry point
└── frontend/
    └── src/
        └── App.jsx                ← React UI
```

---

## Domain Data Models (Pydantic — already designed)
```python
# backend/models/domain_models.py

from pydantic import BaseModel
from typing import List

class TemporalBoundary(BaseModel):
    start_time: float   # e.g., 74.0 (seconds)
    end_time: float     # e.g., 77.0 (seconds)

class FileLocation(BaseModel):
    path: str           # e.g., "C:/Audio/epic_action.wav"

class SemanticEmbedding(BaseModel):
    vector: List[float] # 512-dimensional float list from CLAP

class AudioTrack(BaseModel):
    id: str             # Unique ID, e.g., UUID
    file_location: FileLocation
    name: str           # e.g., "epic_action.wav"

class AudioSegment(BaseModel):
    id: str             # Unique ID for this chunk
    parent_track_id: str
    temporal_boundary: TemporalBoundary
    embedding: SemanticEmbedding
```

---

## How the Core Logic Works (Don't change this design)

### Step 1 — Indexing (runs once when user clicks "Scan Folder")
```
Audio File (.wav/.mp3)
  → librosa loads the file
  → Split into 3-second chunks with 1-second sliding step overlap
      [0s-3s], [1s-4s], [2s-5s], [3s-6s] ...
  → Each chunk → CLAP audio encoder → 512-float vector
  → Store vector + metadata (file path, start_time, end_time) in ChromaDB
```

### Step 2 — Searching (runs every time user types a query)
```
Text Query ("water bubble smashing")
  → CLAP text encoder → 512-float vector
  → ChromaDB cosine similarity search
  → Return top 5 matching AudioSegment records
  → Display: filename + timestamp to user
```

---

## Key API Endpoint
```
POST /search
Body: { "query": "intense excitement" }
Returns: [
  {
    "track_name": "epic_action.wav",
    "file_path": "C:/Audio/epic_action.wav",
    "start_time": 74.0,
    "end_time": 77.0,
    "score": 0.91
  },
  ... (top 5 results)
]
```

---

## Rules / Constraints (Never violate these)
1. **No cloud, no auth, no AWS** — everything runs locally
2. **Output is a pointer, not a clip** — return file path + timestamp only, never modify the audio file
3. **CLAP model is zero-shot** — no fine-tuning, use it directly from HuggingFace
4. **Sliding window chunking** — always use 3s window with 1s step to avoid missing sounds at boundaries
5. **Batch indexing only** — user manually triggers scan, never auto-index in background
6. **ChromaDB is local** — no Qdrant cloud, no remote DB

---

## Current Build Status
| Component | Status |
|---|---|
| Domain models | ⬜ Not started |
| Audio chunker | ⬜ Not started |
| Embedding generator | ⬜ Not started |
| ChromaDB client | ⬜ Not started |
| FastAPI /search endpoint | ⬜ Not started |
| React + Tauri frontend | ⬜ Not started |

*(Update this table as you complete each piece)*

---

## How to Ask an LLM for Help (Use This Template)
```
Context: I'm building a local desktop semantic audio search app.
Stack: Python FastAPI backend, CLAP model (laion/clap-htsat-unfused), ChromaDB, React+Tauri frontend.

Current task: [describe the single function you need]

Here is the relevant existing code:
[paste only the 1-2 files relevant to your question]

Write ONLY the function that does [specific task].
Do not rewrite other files. Do not add cloud or auth logic.
```

---

## Python Dependencies (backend)
```
fastapi
uvicorn
librosa
soundfile
transformers
torch
chromadb
pydantic
numpy
python-multipart
```

Install with:
```bash
pip install fastapi uvicorn librosa soundfile transformers torch chromadb pydantic numpy python-multipart
```

---
*This document is the single source of truth for the project. Update the "Build Status" table as you go.*
