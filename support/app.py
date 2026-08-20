"""
main.py
=======
FastAPI backend for the Object Tracking Intelligence dashboard.

Wraps data_engine.py (ported Streamlit data pipeline) and chatbot_engine.py
(unchanged Gemini function-calling backend) behind a REST API consumed by
the Next.js frontend.

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import data_engine as engine

try:
    import chatbot_engine as chatbot

    CHATBOT_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # noqa: BLE001 - degrade, don't crash the API
    chatbot = None
    CHATBOT_IMPORT_ERROR = str(e)

# --------------------------------------------------------------------------
# CONFIG — same two folders app.py used to scan, and the same env vars
# chatbot_engine.py already reads, so a CSV dropped in DATA_DIR is
# immediately visible to both the analytics endpoints and the chatbot tools.
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
print("Gemini API key loaded:", bool(os.getenv("GEMINI_API_KEY")))
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = BASE_DIR / "videos"
DATA_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

if chatbot is not None:
    chatbot.OUTPUT_FOLDER = DATA_DIR.resolve()
    chatbot.INPUT_VIDEO_FOLDER = VIDEO_DIR.resolve()
    chatbot.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    chatbot.INPUT_VIDEO_FOLDER.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Object Tracking Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_gemini_api_key(header_key: Optional[str]) -> Optional[str]:
    """Resolve the Gemini API key from (in order): a key the frontend sends
    with the request (equivalent of the old sidebar text_input), or the
    GEMINI_API_KEY environment variable on the server."""
    if header_key:
        return header_key
    return os.getenv("GEMINI_API_KEY") or None


def _resolve_dataset_path(file: str) -> Path:
    """Prevent path traversal — only filenames inside DATA_DIR are valid."""
    candidate = (DATA_DIR / file).resolve()
    if DATA_DIR.resolve() not in candidate.parents and candidate != DATA_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file name.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset '{file}' was not found.")
    return candidate


def _load_or_400(file: str):
    path = _resolve_dataset_path(file)
    df, errors = engine.load_tracking_csv(str(path), file)
    if df is None:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return df


# --------------------------------------------------------------------------
# GET /api/datasets — list available CSVs + videos
# --------------------------------------------------------------------------

@app.get("/api/datasets")
def list_datasets():
    csv_files = engine.discover_local_csv_files(str(DATA_DIR))
    video_map = engine.discover_local_videos(str(VIDEO_DIR))
    return {
        "csv_files": csv_files,
        "videos": video_map,  # { "<base name>": "<filename.mp4>" }
    }


# --------------------------------------------------------------------------
# POST /api/dataset/upload — CSV and/or MP4 upload
# --------------------------------------------------------------------------

@app.post("/api/dataset/upload")
async def upload_dataset(
    csv_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
):
    if csv_file is None and video_file is None:
        raise HTTPException(status_code=400, detail="Provide at least a csv_file or a video_file.")

    saved = {}

    if csv_file is not None:
        contents = await csv_file.read()
        dest = DATA_DIR / csv_file.filename
        dest.write_bytes(contents)

        # Validate immediately so the frontend gets fast feedback instead of
        # discovering a bad file only when it later requests analytics.
        df, errors = engine.load_tracking_csv(str(dest), csv_file.filename)
        if df is None:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail={"errors": errors})
        saved["csv_file"] = csv_file.filename

    if video_file is not None:
        contents = await video_file.read()
        dest = VIDEO_DIR / video_file.filename
        dest.write_bytes(contents)
        saved["video_file"] = video_file.filename

    return {"success": True, "saved": saved}


# --------------------------------------------------------------------------
# GET /api/dataset/analytics — KPIs + chart data for the Dashboard view
# --------------------------------------------------------------------------

@app.get("/api/dataset/analytics")
def dataset_analytics(
    file: str = Query(..., description="CSV filename inside data/"),
    classes: Optional[str] = Query(None, description="Comma-separated class filter, e.g. car,truck"),
):
    df = _load_or_400(file)

    all_classes = sorted(df["Class"].dropna().unique().tolist())
    selected = [c.strip().lower() for c in classes.split(",")] if classes else all_classes
    filtered = df[df["Class"].isin(selected)] if selected else df

    base_name = Path(file).stem
    video_map = engine.discover_local_videos(str(VIDEO_DIR))
    video_file = video_map.get(base_name)

    return {
        "file": file,
        "row_count": int(len(df)),
        "available_classes": all_classes,
        "selected_classes": selected,
        "video_file": video_file,
        "kpis": engine.compute_kpis(filtered),
        "class_distribution": engine.class_distribution_data(filtered),
        "objects_over_time": engine.objects_over_time_data(filtered),
    }


# --------------------------------------------------------------------------
# GET /api/dataset/explorer — raw rows + per-track summary, paginated
# --------------------------------------------------------------------------

@app.get("/api/dataset/explorer")
def dataset_explorer(
    file: str = Query(...),
    classes: Optional[str] = Query(None),
    tracking_id: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    df = _load_or_400(file)

    class_list = [c.strip().lower() for c in classes.split(",")] if classes else None
    rows = engine.explorer_rows(df, classes=class_list, tracking_id=tracking_id, page=page, page_size=page_size)

    return {
        "file": file,
        "available_classes": sorted(df["Class"].dropna().unique().tolist()),
        "available_tracking_ids": sorted(df["Tracking_ID"].dropna().unique().tolist(), key=str),
        **rows,
        "track_summary": engine.track_summary(df),
    }


# --------------------------------------------------------------------------
# GET /api/diagnostics — rule-based quick audit (generate_traffic_report)
# --------------------------------------------------------------------------

@app.get("/api/diagnostics")
def diagnostics(file: str = Query(...)):
    df = _load_or_400(file)
    return {"file": file, "log": engine.generate_traffic_report(df)}


# --------------------------------------------------------------------------
# GET /api/videos/{filename} — stream a tracking video for <video> playback
# --------------------------------------------------------------------------

@app.get("/api/videos/{filename}")
def get_video(filename: str):
    candidate = (VIDEO_DIR / filename).resolve()
    if VIDEO_DIR.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(candidate, media_type="video/mp4")


# --------------------------------------------------------------------------
# POST /api/chat — forwards to chatbot_engine.process_query()
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str
    source_label: Optional[str] = None
    previous_interaction_id: Optional[str] = None
    api_key: Optional[str] = None  # equivalent of the old sidebar manual key


class ChatResponse(BaseModel):
    answer: str
    interaction_id: Optional[str] = None


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail=f"The AI backend (chatbot_engine.py) could not be loaded: {CHATBOT_IMPORT_ERROR}",
        )

    api_key = _get_gemini_api_key(payload.api_key)

    contextual_prompt = (
        f"(Dataset currently open in the dashboard: {payload.source_label}.)\n{payload.prompt}"
        if payload.source_label
        else payload.prompt
    )

    answer, interaction_id = chatbot.process_query(
        contextual_prompt,
        api_key,
        payload.previous_interaction_id,
    )
    return ChatResponse(answer=answer, interaction_id=interaction_id)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "chatbot_available": chatbot is not None,
        "chatbot_import_error": CHATBOT_IMPORT_ERROR,
    }
