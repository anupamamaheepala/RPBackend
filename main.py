# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.dyslexia_routes import router as dyslexia_router
from routes.dysgraphia_routes import router as dysgraphia_router
from routes.adhd_routes import router as adhd_router

from routes.learning_plan_routes import router as learning_plan_router
from routes.learning_task_routes import router as learning_task_router
from routes.adhd.g4_adhd_routes import router as g4_adhd_router
from routes.adhd.g4_learning_plan_routes import router as g4_lplan_router
from routes.adhd.g4_learning_task_routes import router as g4_ltask_router
from routes.adhd.g5_adhd_routes import router as g5_adhd_router
from routes.adhd.g5_learning_plan_routes import router as g5_lplan_router
from routes.adhd.g5_learning_task_routes import router as g5_ltask_router
from routes.adhd.g7_adhd_routes import router as g7_adhd_router
from routes.adhd.g7_learning_plan_routes import router as g7_lplan_router
from routes.adhd.g7_learning_task_routes import router as g7_ltask_router
from routes.adhd.g6_adhd_routes import router as g6_adhd_router
from routes.adhd.g6_learning_plan_routes import router as g6_lplan_router
from routes.adhd.g6_learning_task_routes import router as g6_ltask_router

from fastapi.responses import StreamingResponse
from bson import ObjectId
from services.db_service import get_db
import io


from routes.dyslexia_routes import router as dyslexia_sentence_router
from routes.dyslexia_api import router as dyslexia_api_router
from routes.dyscalculia_routes import router as dyscalculia_router
from routes.auth_routes import router as auth_router
from routes.dyslexia_progress_api import router as learning_router
from routes.dysgraphia_improvement_routes import router as dysgraphia_improvement_router

from pydantic import BaseModel
from typing import Optional
from jiwer import wer
from datetime import datetime
from bson import Binary
from openai import OpenAI
import tempfile
import os
import json
from services.db_service import get_db
from config.settings import settings
import re
#from difflib import SequenceMatcher
import re
from jiwer import wer as jiwer_wer
# -----------------------------

app = FastAPI(
    title="Reading Proficiency (RP) Backend",
    description="API for Dyslexia (Reading) and Dyscalculia (Math) assessment",
    version="1.0.0"
)

# Routers
app.include_router(dyslexia_sentence_router)   # /get-random  
app.include_router(dyslexia_api_router)
app.include_router(adhd_router)
app.include_router(learning_plan_router)
app.include_router(learning_task_router)
app.include_router(g4_adhd_router)
app.include_router(g4_lplan_router)
app.include_router(g4_ltask_router)
app.include_router(g5_adhd_router)
app.include_router(g5_lplan_router)
app.include_router(g5_ltask_router)
app.include_router(g7_adhd_router)
app.include_router(g7_lplan_router)
app.include_router(g7_ltask_router)
app.include_router(g6_adhd_router)
app.include_router(g6_lplan_router)
app.include_router(g6_ltask_router)

app.include_router(dysgraphia_router)
app.include_router(dyscalculia_router)
app.include_router(auth_router)
app.include_router(learning_router)
app.include_router(dysgraphia_improvement_router)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB
db = get_db()

# Audio streaming
@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    audio_doc = db["audio_files"].find_one({"_id": ObjectId(audio_id)})
    if not audio_doc:
        return {"ok": False, "error": "Audio not found"}

    return StreamingResponse(
        io.BytesIO(audio_doc["data"]),
        media_type=audio_doc.get("content_type", "audio/wav"),
        headers={"Content-Disposition": f"inline; filename={audio_doc['filename']}"}
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to RP Backend!", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
