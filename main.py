from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from routes.dyslexia_routes import router as dyslexia_router
from routes.dyscalculia_routes import router as dyscalculia_router
from routes.auth_routes import router as auth_router

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
    description="API for Dyslexia (Reading) and Dyscalculia (Math) assessment",
    version="1.0.0"
)

# --- REGISTER ROUTERS ---
app.include_router(dyslexia_router)
app.include_router(dyscalculia_router)
app.include_router(auth_router)

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (update for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DB + OpenAI clients
db = get_db()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# -----------------------------
# SINHALA NORMALIZATION LOGIC
SINHALA_NORMALIZATION_MAP = {
    "ණ": "න",
    "ළ": "ල",
    "ශ": "ෂ",
    "ඤ": "ඥ",
}

def normalize_sinhala_word(word: str) -> str:
    for src, tgt in SINHALA_NORMALIZATION_MAP.items():
        word = word.replace(src, tgt)
    return word

def extract_word_errors(reference: str, transcript: str):
    ref_words = reference.strip().split()
    hyp_words = transcript.strip().split()

    correct = []
    incorrect = []

    for i, ref_word in enumerate(ref_words):
        ref_norm = normalize_sinhala_word(ref_word)

        if i < len(hyp_words):
            hyp_norm = normalize_sinhala_word(hyp_words[i])

            if hyp_norm == ref_norm:
                correct.append(ref_word)
            else:
                incorrect.append(ref_word)
        else:
            incorrect.append(ref_word)

    return correct, incorrect

def compute_metrics(reference: str, transcript: str, duration: Optional[float] = None):
    reference = reference.strip()
    transcript = transcript.strip()

    correct_words_list, incorrect_words_list = extract_word_errors(
        reference, transcript
    )

    ref_words = reference.split()
    hyp_words = transcript.split()
    total_words = len(ref_words)
    correct_words = len(correct_words_list)
    
    # Calculate Accuracy
    accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0.0
    error_rate = 100 - accuracy
    WER = error_rate
  
    speed = None
    if duration and duration > 0:
        speed = round(len(hyp_words) / duration, 2)

    return {
        "reference": reference,
        "transcript": transcript,
        "total_words": total_words,
        "correct_words": correct_words,
        "accuracy_percent": round(accuracy, 2),
        "wer": round(WER, 2),
        "words_per_second": speed,
        "incorrect_words": incorrect_words_list,
    }

# -----------------------------
# ROOT ENDPOINTS
@app.get("/")
def read_root():
    return {"message": "Welcome to RP Backend! Endpoints available for Dyslexia and Dyscalculia.", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# -----------------------------
# DYSLEXIA: Manual Compare Endpoint
class CompareBody(BaseModel):
    reference_text: str
    transcript: str
    duration: Optional[float] = None

@app.post("/dyslexia/compare")
def compare_text(body: CompareBody):
    metrics = compute_metrics(body.reference_text, body.transcript, body.duration)
    return {"ok": True, "metrics": metrics}

# -----------------------------
# DYSLEXIA: Audio Submission Endpoint
@app.post("/dyslexia/submit-audio")
async def submit_audio(
    reference_text: str = Form(...),
    duration: Optional[float] = Form(None),
    grade: Optional[int] = Form(None),
    level: Optional[int] = Form(None),
    file: UploadFile = File(...)
):
    """
    Handles Dyslexia Audio:
    1. Uploads audio
    2. Transcribes via OpenAI Whisper
    3. Computes accuracy metrics
    4. Saves to MongoDB
    """
    # 1) Read bytes
    audio_bytes = await file.read()

    # 2) Store audio in MongoDB (binary)
    audio_doc = {
        "filename": file.filename,
        "content_type": file.content_type,
        "data": Binary(audio_bytes),
        "grade": grade,
        "level": level,
        "duration": duration,
        "created_at": datetime.utcnow(),
    }
    audio_result = db["audio_files"].insert_one(audio_doc)
    audio_id = audio_result.inserted_id

    # 3) Save temporarily for Whisper API
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        # 4) Transcribe using Whisper (Sinhala supported via OpenAI)
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe", # Ensure this model name is correct for your OpenAI access
                file=audio_file
            )

        transcript_text = transcription.text.strip()

        # 5) Compute metrics
        metrics = compute_metrics(reference_text, transcript_text, duration)

        # 6) Store reading result in MongoDB
        reading_doc = {
            "audio_file_id": audio_id,
            "reference_text": reference_text,
            "transcript": transcript_text,
            "grade": grade,
            "level": level,
            "duration": duration,
            "metrics": metrics,
            "created_at": datetime.utcnow(),
        }
        reading_result = db["readings"].insert_one(reading_doc)

        return {
            "ok": True,
            "reading_id": str(reading_result.inserted_id),
            "metrics": metrics,
        }

    except Exception as e:
        return {"ok": False, "error": f"Transcription failed: {e}"}

    finally:
        # Clean temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)