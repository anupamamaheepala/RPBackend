from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from routes.dyslexia_routes import router as dyslexia_router
from routes.dysgraphia_routes import router as dysgraphia_router
from fastapi.responses import StreamingResponse
from bson import ObjectId
import io
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
import json
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

app.include_router(dysgraphia_router)
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


# def extract_word_errors(reference: str, transcript: str):
#     ref_words = reference.strip().split()
#     hyp_words = transcript.strip().split()

#     correct = []
#     incorrect = []

#     for i, ref_word in enumerate(ref_words):
#         if i < len(hyp_words) and hyp_words[i] == ref_word:
#             correct.append(ref_word)
#         else:
#             incorrect.append(ref_word)

#     return correct, incorrect

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

def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(value, max_value))


def compute_dyslexia_risk(audio_metrics: dict, eye_metrics: dict):
    # ---------------- PHONOLOGICAL ----------------
    accuracy = audio_metrics.get("accuracy_percent", 0)
    wer = audio_metrics.get("wer", 100)

    accuracy_risk = 1 - (accuracy / 100)
    wer_risk = wer / 100
    phonological_risk = (accuracy_risk + wer_risk) / 2

    # ---------------- FLUENCY ----------------
    wps = audio_metrics.get("words_per_second", 0) or 0
    fluency_risk = clamp((2.5 - wps) / 2.5)

    # ---------------- EYE TRACKING ----------------
    avg_fixation = eye_metrics.get("avg_fixation_ms", 0)
    regression_count = eye_metrics.get("regression_count", 0)

    fixation_risk = clamp((avg_fixation - 300) / 1200)
    regression_risk = clamp(regression_count / 5)

    eye_risk = (0.7 * fixation_risk) + (0.3 * regression_risk)

    # ---------------- FINAL SCORE ----------------
    final_risk = (
        0.35 * phonological_risk
        + 0.25 * fluency_risk
        + 0.40 * eye_risk
    )

    # ---------------- LEVEL ----------------
    if final_risk <= 0.30:
        level = "LOW"
    elif final_risk <= 0.60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "phonological_risk": round(phonological_risk, 3),
        "fluency_risk": round(fluency_risk, 3),
        "eye_risk": round(eye_risk, 3),
        "risk_score": round(final_risk, 3),
        "risk_level": level,
    }

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
        "incorrect_words": ", ".join(incorrect_words_list),

    }

#GET THE AUDIO LINK
@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    audio_doc = db["audio_files"].find_one(
        {"_id": ObjectId(audio_id)}
    )

    if not audio_doc:
        return {"ok": False, "error": "Audio not found"}

    return StreamingResponse(
        io.BytesIO(audio_doc["data"]),
        media_type=audio_doc.get("content_type", "audio/wav"),
        headers={
            "Content-Disposition": f"inline; filename={audio_doc['filename']}"
        }
    )


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
    #user=Depends(get_current_user()),
    #username: str = Form(...),
    reference_text: str = Form(...),
    duration: Optional[float] = Form(None),
    grade: Optional[int] = Form(None),
    level: Optional[int] = Form(None),
    eye_metrics: Optional[str] = Form(None), 
    file: UploadFile = File(...),
    #username: str = Depends(get_current_user)
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

        eye_data = {}

        if eye_metrics:
           try:
               eye_data = json.loads(eye_metrics)
           except Exception:
               eye_data = {}

        dyslexia_risk = compute_dyslexia_risk(metrics, eye_data)

        # 6) Store reading result in MongoDB
        reading_doc = {
           # "user_id": user["user_id"],      
           # "username": user["username"],
            #"username": username,
            "audio_file_id": audio_id,
            # "reference_text": reference_text,
            # "transcript": transcript_text,
            "grade": grade,
            "level": level,
            "duration": duration,
            #"metrics": metrics,
            # ---------------- AUDIO METRICS ----------------
            "audio_id": str(audio_id),   
            "audio_url": f"http://localhost:8000/audio/{audio_id}",  
            "audio_metrics": metrics,
            # ---------------- EYE TRACKING METRICS ----------------
            "eye_tracking": {
            "fixation_count": eye_data.get("fixation_count", 0),
            "avg_fixation_ms": eye_data.get("avg_fixation_ms", 0),
            "regression_count": eye_data.get("regression_count", 0),
            "saccade_count": eye_data.get("saccade_count", 0),
            "blink_rate_per_min": eye_data.get("blink_rate_per_min", 0),
        },
         "dyslexia_assessment": dyslexia_risk,
            "created_at": datetime.utcnow(),
        }
        reading_result = db["readings"].insert_one(reading_doc)

        return {
            "ok": True,
            "reading_id": str(reading_result.inserted_id),
            "metrics": metrics,
            "eye_tracking": eye_data,
            "dyslexia_assessment": dyslexia_risk,
        }

    except Exception as e:
        return {"ok": False, "error": f"Transcription failed: {e}"}

    finally:
        # Clean temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)