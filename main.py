# # from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from routes.dyslexia_routes import router as dyslexia_router
# from docx import Document
# #from routes import data
# from pydantic import BaseModel
# from typing import Optional
# from jiwer import wer
# import tempfile, os
# #from routes.dyslexia_routes import router as dyslexia_router


# app = FastAPI()
# app.include_router(dyslexia_router)
# #app.include_router(dyslexia_router)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins - you can specify domains later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ----- Reference Texts -----
# TEXTS = [
#     {"id": "t1", "lang": "si", "content": "අද අහස පැහැදිලි ය. ළමයා පොතක් කියවයි."},
#     {"id": "t2", "lang": "en", "content": "The sky is clear today. The child reads a book."}
# ]
 
# def read_docx(file_path: str):
#     doc = Document(file_path)
#     lines = []

#     for para in doc.paragraphs:
#         text = para.text.strip()
#         if text:
#             lines.append(text)

#     return lines


# def get_text(text_id: str):
#     for t in TEXTS:
#         if t["id"] == text_id:
#             return t
#     return None

# # ----- Utility -----
# def compute_metrics(reference: str, transcript: str, duration: Optional[float]):
#     ref_words = reference.split()
#     hyp_words = transcript.split()
#     total_ref = len(ref_words)
#     total_spoken = len(hyp_words)
#     error_rate = wer(reference.lower(), transcript.lower())
#     accuracy = max(0.0, 1 - error_rate) * 100
#     correct_words = round(accuracy/100 * total_ref, 2)
#     words_per_sec = None
#     if duration:
#         words_per_sec = round(total_spoken / duration, 2)
#     return {
#         "reference": reference,
#         "transcript": transcript,
#         "total_words": total_spoken,
#         "correct_words": correct_words,
#         "accuracy_percent": round(accuracy, 2),
#         "wer": round(error_rate, 4),
#         "words_per_second": words_per_sec
#     }


# # app.include_router(data.router)

# # @app.get("/")
# # def read_root():
# #     return {"message": "Welcome to RP Backend!"}

# # @app.get("/health")
# # def health_check():
# #     return {"status": "healthy", "service": "FastAPI Backend"}

# # ----- API -----
# @app.get("/texts")
# def list_texts():
#     return TEXTS

# class CompareBody(BaseModel):
#     text_id: str
#     transcript: str
#     duration: Optional[float] = None

# @app.post("/dyslexia/compare")
# def compare_text(body: CompareBody):
#     text = get_text(body.text_id)
#     if not text:
#         return {"error": "Invalid text id"}
#     metrics = compute_metrics(text["content"], body.transcript, body.duration)
#     return {"ok": True, "metrics": metrics}

# @app.post("/dyslexia/submit-audio")
# async def submit_audio(
#     text_id: str = Form(...),
#     duration: Optional[float] = Form(None),
#     file: UploadFile = File(...)
# ):
#     """Placeholder if you later enable transcription"""
#     text = get_text(text_id)
#     if not text:
#         return {"error": "Invalid text id"}

#     # Just save temporarily (no transcription yet)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     # For now, return mock result
#     try:
#         metrics = compute_metrics(text["content"], text["content"], duration)
#         return {"ok": True, "metrics": metrics, "note": "Audio received (transcription disabled)"}
#     finally:
#         os.remove(tmp_path)



from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from routes.dyslexia_routes import router as dyslexia_router
from services.eye_tracking_service import analyze_eye_movements
from pydantic import BaseModel
from typing import Optional
from jiwer import wer
from datetime import datetime
from bson import Binary
from openai import OpenAI
import tempfile
import os

from services.db_service import get_db
from config.settings import settings

# -----------------------------
app = FastAPI(
    title="Reading Proficiency (RP) Backend",
    description="API for dyslexia/reading assessment",
    version="1.0.0"
)

# Routers
app.include_router(dyslexia_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # change to your Flutter origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DB + OpenAI clients
db = get_db()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# -----------------------------
def compute_metrics(reference: str, transcript: str, duration: Optional[float] = None):
    reference = reference.strip()
    transcript = transcript.strip()

    error_rate = wer(reference.lower(), transcript.lower())
    accuracy = max(0.0, 1 - error_rate) * 100

    ref_words = reference.split()
    correct_words = round((accuracy / 100) * len(ref_words), 2)

    words_per_sec = None
    if duration and duration > 0:
        words_per_sec = round(len(transcript.split()) / duration, 2)

    return {
        "reference": reference,
        "transcript": transcript,
        "total_words": len(transcript.split()),
        "correct_words": correct_words,
        "accuracy_percent": round(accuracy, 2),
        "wer": round(error_rate, 4),
        "words_per_second": words_per_sec,
    }

# -----------------------------
# Basic root endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to Reading Proficiency Backend!", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# -----------------------------
# Compare endpoint (manual transcript)
class CompareBody(BaseModel):
    reference_text: str
    transcript: str
    duration: Optional[float] = None

@app.post("/dyslexia/compare")
def compare_text(body: CompareBody):
    metrics = compute_metrics(body.reference_text, body.transcript, body.duration)
    return {"ok": True, "metrics": metrics}

# -----------------------------
# MAIN endpoint used by Flutter

@app.post("/dyslexia/submit-session")
async def submit_session(
    reference_text: str = Form(...),
    duration: Optional[float] = Form(None),
    grade: Optional[int] = Form(None),
    level: Optional[int] = Form(None),
    audio: UploadFile = File(...),
    video: UploadFile = File(...)
):
    audio_bytes = await audio.read()
    video_bytes = await video.read()

    audio_id = db["audio_files"].insert_one({
        "filename": audio.filename,
        "data": Binary(audio_bytes),
        "created_at": datetime.utcnow()
    }).inserted_id

    video_id = db["video_files"].insert_one({
        "filename": video.filename,
        "data": Binary(video_bytes),
        "created_at": datetime.utcnow()
    }).inserted_id

    # save temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as af:
        af.write(audio_bytes)
        audio_path = af.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf:
        vf.write(video_bytes)
        video_path = vf.name

    try:
        # TRANSCRIPTION
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )
        transcript = transcription.text.strip()

        # READING METRICS
        reading_metrics = compute_metrics(reference_text, transcript, duration)

        # EYE METRICS
        eye_metrics = analyze_eye_movements(video_path)

        session = {
            "audio_id": str(audio_id),
            "video_id": str(video_id),
            "reference_text": reference_text,
            "transcript": transcript,
            "duration": duration,
            "grade": grade,
            "level": level,
            "reading_metrics": reading_metrics,
            "eye_metrics": eye_metrics,
            "created_at": datetime.utcnow(),
        }

        sid = db["reading_sessions"].insert_one(session).inserted_id

        return {
            "ok": True,
            "session_id": str(sid),
            "metrics": {
                **reading_metrics,
                **eye_metrics
            }
        }

    finally:
        if os.path.exists(audio_path): os.unlink(audio_path)
        if os.path.exists(video_path): os.unlink(video_path)
