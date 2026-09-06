from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.cv_parser import extract_cv_text
from app.career_agent import run_career_agent


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="JobScout AI",
    description="Autonomous AI Career Agent for intelligent job discovery and matching.",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PROJECT PATHS
# =========================================================

# main.py is inside:
# backend/app/main.py
#
# parents[0] = backend/app
# parents[1] = backend

BASE_DIR = Path(__file__).resolve().parents[1]

# Frontend will be inside:
# backend/frontend

FRONTEND_DIR = BASE_DIR / "frontend"


# =========================================================
# HEALTH / HOME
# =========================================================

@app.get("/")
def root():
    return {
        "name": "JobScout AI",
        "status": "running",
        "message": "Autonomous Career Agent API is running.",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "JobScout AI",
    }


# =========================================================
# CV UPLOAD
# =========================================================

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    file_bytes = await file.read()

    cv_text = extract_cv_text(
        file_bytes,
        file.filename or "",
    )

    return {
        "filename": file.filename,
        "characters_extracted": len(cv_text),
        "text_preview": cv_text[:1000],
    }


# =========================================================
# AUTONOMOUS JOB SEARCH
# =========================================================

@app.post("/find-jobs")
async def find_jobs(file: UploadFile = File(...)):
    file_bytes = await file.read()

    result = run_career_agent(
        file_bytes=file_bytes,
        filename=file.filename or "",
    )

    return result


# =========================================================
# FRONTEND DASHBOARD
# =========================================================

# Serve the frontend from:
# backend/frontend/index.html

if FRONTEND_DIR.exists():
    app.mount(
        "/dashboard",
        StaticFiles(
            directory=str(FRONTEND_DIR),
            html=True,
        ),
        name="dashboard",
    )