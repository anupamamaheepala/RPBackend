# routes/dyslexia_api.py
import os
import json
import tempfile
from datetime import datetime
from typing import Optional, Any, Dict, List

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from bson import Binary

from services.db_service import get_db
from config.settings import settings
from openai import OpenAI

from services.dyslexia_metrics import compute_metrics, compute_dyslexia_risk
#from models.dyslexia.predict import predict_dyslexia_risk_ml
from services.dyslexia_metrics import calculate_session_metrics
from models.dyslexia.predict import predict_dyslexia_risk_ml

router = APIRouter(prefix="/dyslexia", tags=["Dyslexia"])

db = get_db()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

db["reading_session_stats"].create_index([("user_id", 1), ("created_at", -1)])

# ---------- 1) PER SENTENCE ANALYZE ----------
@router.post("/analyze-audio")
async def analyze_audio(
    username: str = Form(...),
    user_id: Optional[str] = Form(None),
    reference_text: str = Form(...),
    duration: Optional[float] = Form(None),
    grade: Optional[int] = Form(None),
    level: Optional[int] = Form(None),
    sentence_index: Optional[int] = Form(None),
    eye_metrics: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    tmp_path = None
    try:
        audio_bytes = await file.read()

        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )

        transcript_text = transcription.text.strip()
        metrics = compute_metrics(reference_text, transcript_text, duration)

        return {"ok": True, "metrics": metrics, "sentence_index": sentence_index}

    except Exception as e:
        return {"ok": False, "error": f"Analyze failed: {e}"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------- 2) OLD ENDPOINT (MOVED) ----------
@router.post("/submit-audio")
async def submit_audio(
    username: str = Form(...),
    user_id: Optional[str] = Form(None),
    reference_text: str = Form(...),
    duration: Optional[float] = Form(None),
    grade: Optional[int] = Form(None),
    level: Optional[int] = Form(None),
    eye_metrics: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    tmp_path = None
    try:
        audio_bytes = await file.read()

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

        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )

        transcript_text = transcription.text.strip()
        metrics = compute_metrics(reference_text, transcript_text, duration)

        eye_data = {}
        if eye_metrics:
            try:
                eye_data = json.loads(eye_metrics)
            except Exception:
                eye_data = {}

        try:
            dyslexia_risk = predict_dyslexia_risk_ml(
                audio_metrics=metrics,
                eye_metrics=eye_data,
                duration=duration
            )
        except Exception:
            dyslexia_risk = compute_dyslexia_risk(metrics, eye_data)

        reading_doc = {
            "username": username,
            "user_id": user_id,
            "audio_file_id": audio_id,
            "grade": grade,
            "level": level,
            "duration": duration,

            "audio_id": str(audio_id),
            "audio_url": f"http://localhost:8000/audio/{audio_id}",
            "audio_metrics": metrics,

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
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------- 3) FINAL SESSION SUBMIT ----------
class SessionPayload(BaseModel):
    username: str
    user_id: Optional[str] = None
    grade: int
    level: int

    total_words: int
    total_correct: int
    total_time_seconds: int
    overall_accuracy: float

    mean_sentence_accuracy: float
    sentence_accuracy_std_dev: float

    avg_WER: float
    avg_CER: float
    avg_words_per_second: float

    incorrect_words_all: List[str]

    avg_fixation_time: float
    avg_regression_count: float
    avg_saccade_count: float
    avg_blink_rate_per_min: float

    sentences: List[Dict[str, Any]]


@router.post("/generate-tts")
async def generate_tts(text: str = Form(...)):
    """
    Generate Sinhala speech audio using OpenAI TTS
    Returns mp3 file
    """
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
        )

        # ✅ Correct way to get audio bytes
        audio_bytes = response.content

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_file.write(audio_bytes)
        tmp_file.close()

        return FileResponse(
            tmp_file.name,
            media_type="audio/mpeg",
            filename="tts.mp3"
        )

    except Exception as e:
        print("TTS ERROR:", e)
        return {"ok": False, "error": str(e)}


@router.post("/submit-session")
async def submit_session(payload: SessionPayload):
    try:
        # Step 1: Calculate the 0-1 Risk Scores required by the model
        ml_input = calculate_session_metrics(payload)

        # Step 2: Get the prediction from the Trained PKL Model
        prediction_result = predict_dyslexia_risk_ml(ml_input)
        
        # Step 3: Prepare the final document for MongoDB
        session_doc = payload.model_dump()
        session_doc["dyslexia_assessment"] = prediction_result
        session_doc["created_at"] = datetime.utcnow()

        # Step 4: Save to Database
        result = db["reading_sessions"].insert_one(session_doc)
        
        # Save summary stats
        db["reading_session_stats"].insert_one({
            "username": payload.username,
            "user_id": payload.user_id,
            "risk_level": prediction_result["risk_level"],
            "created_at": session_doc["created_at"]
        })

        return {
            "ok": True,
            "session_id": str(result.inserted_id),
            "risk_level": prediction_result["risk_level"],
            "confidence": prediction_result["confidence"]
        }

    except Exception as e:
        return {"ok": False, "error": f"Session processing failed: {str(e)}"}

@router.get("/check-task-lock")
async def check_task_lock(user_id: str, grade: int, level: int):
    # 1. Check if they have ever done a task for this level
    last_stat = db["reading_session_stats"].find_one(
        {"user_id": user_id, "grade": grade, "level": level},
        sort=[("created_at", -1)]
    )

    if not last_stat:
        # No previous task = NOT LOCKED (They need to do their first assessment)
        return {"is_locked": False}

    # 2. If they have a task, check if the Learning Path for it is finished
    # We look for a 'module_completed' flag in your progress collection
    progress = db["learning_progress"].find_one({
        "user_id": user_id,
        "grade": grade,
        "level": level,
        "module_number": 1
    })

    if progress and progress.get("is_completed", False):
        # Path is finished = UNLOCKED (They can re-assess to move to next level)
        return {"is_locked": False}
    else:
        # Path exists but not finished = LOCKED
        return {"is_locked": True}

@router.post("/learning/complete-module")
async def complete_module(data: dict):
    user_id = data.get("user_id")
    grade = data.get("grade")
    level = data.get("level")
    module_number = data.get("module_number")

    # Update or insert the completion status
    db["learning_progress"].update_one(
        {
            "user_id": user_id,
            "grade": grade,
            "level": level,
            "module_number": module_number
        },
        {
            "$set": {
                "is_completed": True,
                "completed_at": datetime.utcnow()
            }
        },
        upsert=True
    )
    
    return {"ok": True, "message": "Module marked as completed"}


@router.get("/dyslexia/check-task-lock")
async def check_task_lock(user_id: str, grade: int, level: int):
    # Check if there is a session record for this level
    last_session = db["reading_session_stats"].find_one(
        {"user_id": user_id, "grade": grade, "level": level},
        sort=[("created_at", -1)]
    )

    # If no session exists, the user is new to this level: NOT LOCKED
    if not last_session:
        return {"is_locked": False}

    # If a session exists, check if the corresponding learning module is finished
    progress = db["learning_progress"].find_one({
        "user_id": user_id,
        "grade": grade,
        "level": level,
        "module_number": 1 # Assuming Module 1 corresponds to the assessment
    })

    # Lock the task if progress is missing or is_completed is False
    is_locked = True
    if progress and progress.get("is_completed") == True:
        is_locked = False

    return {"is_locked": is_locked}
