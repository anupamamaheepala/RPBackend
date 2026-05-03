#Backend
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
from models.dyslexia.predict import predict_dyslexia_risk_ml

#----------XAI PART
from models.dyslexia_request_model import ReadingRequest
from services.dyslexia.comparator import compare_text
from services.dyslexia.explainer import generate_explanations

from services.dyslexia.skill_analyzer import analyze_skill_weakness
from models.dyslexia.model_loader import thresholds

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

        # transcript_text = transcription.text.strip()
        # metrics = compute_metrics(reference_text, transcript_text, duration)
        transcript_text = transcription.text.strip()

        metrics = compute_metrics(reference_text, transcript_text, duration)

        # 🔥 XAI PART
        # errors = compare_text(reference_text, transcript_text)
        # explanations = generate_explanations(errors)

        errors = compare_text(reference_text, transcript_text)
        explanations = generate_explanations(errors)

        eye_data = {}
        if eye_metrics:
            try:
              eye_data = json.loads(eye_metrics)
            except Exception:
              eye_data = {}
 
        skill_analysis = analyze_skill_weakness(
            reference_text=reference_text,
            transcript_text=transcript_text,
            metrics=metrics,
            eye_metrics=eye_data,
            xai_feedback=explanations,
   )

       # return {"ok": True, "metrics": metrics, "sentence_index": sentence_index}
        return {
            "ok": True,
            # "metrics": metrics,
             "metrics": {
                **metrics,
                "transcript": transcript_text,
                "xai_feedback": explanations,
                "skill_analysis": skill_analysis,
            },
            "transcript": transcript_text,
            "xai_feedback": explanations,   # 🔥 NEW
            "skill_analysis": skill_analysis,
            "sentence_index": sentence_index
    }

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

        # 🔥 XAI PART
        errors = compare_text(reference_text, transcript_text)
        explanations = generate_explanations(errors)

        skill_analysis = analyze_skill_weakness(
                reference_text=reference_text,
                transcript_text=transcript_text,
                metrics=metrics,
                eye_metrics=eye_data,
                xai_feedback=explanations,
        )

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
            "xai_feedback": explanations,
            "skill_analysis": skill_analysis,
            "transcript" : transcript_text,
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
            "xai_feedback": explanations,
            "skill_analysis": skill_analysis,
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
    session_type: str = "detection"

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
def submit_session(payload: SessionPayload):

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except:
            return default

    # -------------------------------
    # Validate required fields
    # -------------------------------
    required_fields = [
        "grade", "level", "total_words",
        "overall_accuracy", "avg_WER", "avg_CER",
        "total_time_seconds", "avg_words_per_second",
        "avg_regression_count"
    ]

    missing = [f for f in required_fields if getattr(payload, f) is None]

    if missing:
        return {
            "ok": False,
            "error": f"Missing required fields: {missing}"
        }

    # -------------------------------
    # Prepare ML input
    # -------------------------------
    ml_input = {
        "grade": int(payload.grade),
        "level": int(payload.level),
        "total_words": safe_float(payload.total_words),

        "overall_accuracy": safe_float(payload.overall_accuracy),
        "avg_WER": safe_float(payload.avg_WER),
        "avg_CER": safe_float(payload.avg_CER),
        "total_time_seconds": safe_float(payload.total_time_seconds),

        "avg_words_per_second": safe_float(payload.avg_words_per_second),
        "avg_regression_count": safe_float(payload.avg_regression_count)
    }

    print("ML INPUT:", ml_input)

    # -------------------------------
    # ML Prediction
    # -------------------------------
    try:
        dyslexia_risk = predict_dyslexia_risk_ml(ml_input)
        print("PREDICTION:", dyslexia_risk)
    except Exception as e:
        print("CRITICAL ML ERROR:", e)
        dyslexia_risk = {
            "risk_level": "UNKNOWN",
            "confidence": 0,
            "method": "ML Failed"
        }

    print("ML OUTPUT:", dyslexia_risk)

    # -------------------------------
    # Save to DB
    # -------------------------------
    created_at = datetime.utcnow()
    session_doc = payload.model_dump()
    session_doc["dyslexia_assessment"] = dyslexia_risk
    session_doc["created_at"] = created_at

    result = db["reading_sessions"].insert_one(session_doc)

    learning_progress_doc = {
        "user_id": payload.user_id,
        "username": payload.username,
        "grade": payload.grade,
        "level": payload.level,
        "risk_level": dyslexia_risk.get("risk_level", "UNKNOWN"),
        "created_at": created_at,
        "is_complete": False
    }

    db["learning_progress"].insert_one(learning_progress_doc)

    stats_doc = {
        "username": payload.username,
        "user_id": payload.user_id,
        "grade": payload.grade,
        "level": payload.level,
        "session_type": payload.session_type,
        "overall_accuracy": payload.overall_accuracy,
        "full_session_id": str(result.inserted_id),
        "created_at": created_at,
    }

    stats_result = db["reading_session_stats"].insert_one(stats_doc)

    # -------------------------------
    # Response
    # -------------------------------
    return {
        "ok": True,
        "session_id": str(result.inserted_id),
        "stats_id": str(stats_result.inserted_id),
        "dyslexia_assessment": dyslexia_risk
    }

@router.get("/check-task-lock")
async def check_task_lock(user_id: str, grade: int, level: int):

    # 1️⃣ Check if the user has ever done detection for this grade/level
    last_session = db["reading_session_stats"].find_one(
        {"user_id": user_id, "grade": grade, "level": level},
        sort=[("created_at", -1)]
    )

    # If no session exists → first attempt → allow detection
    if not last_session:
        return {"is_locked": False}

    # 2️⃣ Fetch learning modules assigned for this level
    progress_records = list(db["learning_progress"].find({
        "user_id": user_id,
        "grade": grade,
        "level": level
    }))

    # If no modules assigned yet → allow
    if not progress_records:
        return {"is_locked": False}

    # 3️⃣ Check if all modules completed
    all_completed = all(p.get("is_complete", False) for p in progress_records)

    if all_completed:
        return {"is_locked": False}

    # Otherwise lock detection
    return {"is_locked": True}

# @router.get("/check-task-lock")
# async def check_task_lock(user_id: str, grade: int, level: int):
#     # 1. Check if they have ever done a task for this level
#     last_stat = db["reading_session_stats"].find_one(
#         {"user_id": user_id, "grade": grade, "level": level},
#         sort=[("created_at", -1)]
#     )

#     if not last_stat:
#         # No previous task = NOT LOCKED (They need to do their first assessment)
#         return {"is_locked": False}

#     # 2. If they have a task, check if the Learning Path for it is finished
#     # We look for a 'module_completed' flag in your progress collection
#     progress = db["learning_progress"].find_one({
#         "user_id": user_id,
#         "grade": grade,
#         "level": level,
#         "module_number": 1
#     })

#     if progress and progress.get("is_completed", False):
#         # Path is finished = UNLOCKED (They can re-assess to move to next level)
#         return {"is_locked": False}
#     else:
#         # Path exists but not finished = LOCKED
#         return {"is_locked": True}

@router.post("/learning/complete-module")
async def complete_module(data: dict):
    user_id = data.get("user_id")
    grade = data.get("grade")
    level = data.get("level")
    risklevel = data.get("risk_level")
       # module_number = data.get("module_number")

    # Update or insert the completion status
    learning_progress = db["learning_progress"].find_one(
        {
            "user_id": user_id,
            "grade": grade,
            "level": level,
            #"module_number": module_number
            "risk_level": risklevel
        })
    if learning_progress:
        db["learning_progress"].update_one(
           {"_id": learning_progress["_id"]},
           {"$set": {
                "is_complete": True,
                "completed_at": datetime.utcnow()
            }}
    )
        return {"ok": True, "message": "Module marked as complete"}

    return {"ok": False, "error": "Module progress not found"}


@router.get("/has-attempt")
async def has_attempt(user_id: str, grade: int, level: int):

    session = db["reading_session_stats"].find_one({
        "user_id": user_id,
        "grade": grade,
        "level": level
    })

    return {
        "has_attempt": True if session else False
    }

# ---------- 7) GET USER DYSLEXIA HISTORY ----------
@router.get("/history")
async def get_user_history(user_id: str, session_type: Optional[str] = None):
    query = {"user_id": user_id}

    if session_type:
        query["session_type"] = session_type

    sessions = list(
        db["reading_sessions"]
        .find(query)
        .sort("created_at", -1)
    )

    results = []

    for s in sessions:
        assessment = s.get("dyslexia_assessment", {})

        sentence_results = []

        for sentence in s.get("sentences", []):
            metrics = sentence.get("metrics", {})

            total_words = metrics.get("total_words", 1)
            correct_words = metrics.get("correct_words", 0)

            accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0

            sentence_results.append({
                "sentence_index": sentence.get("sentence_index"),
                "accuracy": round(accuracy, 2),
                "correct_words": correct_words,
                "total_words": total_words
            })

        results.append({
            "grade": s.get("grade"),
            "level": s.get("level"),
            "session_type": s.get("session_type", "detection"),
            "overall_accuracy": s.get("overall_accuracy", 0),
            "avg_WER": s.get("avg_WER", 0),
            "avg_CER": s.get("avg_CER", 0),
            "avg_words_per_second": s.get("avg_words_per_second", 0),
            "total_time_seconds": s.get("total_time_seconds", 0),
            "risk_level": assessment.get("risk_level", "UNKNOWN"),
            "confidence": assessment.get("confidence", 0),
            "created_at": s.get("created_at").strftime("%Y-%m-%d"),
            "sentence_results": sentence_results
        })

    return {
        "ok": True,
        "sessions": results
    }

@router.post("/learning/complete-activity")
async def complete_activity(data: dict):
    user_id = data.get("user_id")
    grade = data.get("grade")
    level = data.get("level")
    risk_level = data.get("risk_level")

    # Find the learning progress entry for the user
    learning_progress = db["learning_progress"].find_one({
        "user_id": user_id,
        "grade": grade,
        "level": level,
        "risk_level": risk_level
    })

    if learning_progress:
        # Update the current_activity index to unlock the next activity
        current_activity = learning_progress.get("current_activity", 0)
        db["learning_progress"].update_one(
            {"_id": learning_progress["_id"]},
            {"$set": {"current_activity": current_activity + 1}}
        )
        return {"ok": True, "message": "Module marked as complete"}

    return {"ok": False, "error": "Progress not found"}

@router.get("/learning/progress")
async def get_learning_progress(user_id: str, grade: int, level: int, risk_level: str):
    # Find the learning progress entry for the user, grade, level, and risk level
    learning_progress = db["learning_progress"].find_one({
        "user_id": user_id,
        "grade": grade,
        "level": level,
        "risk_level": risk_level
    })

    if learning_progress:
        return {
            "ok": True,
            "progress": {
                "current_activity": learning_progress.get("current_activity", 0), 
                "is_complete": learning_progress.get("is_complete", False),  
                "completed_at": learning_progress.get("completed_at", None),  
            }
        }

    return {"ok": False, "error": "Progress not found"}

# FOR Dyslexia XAI PART
@router.post("/analyze-reading")
def analyze_reading(req: ReadingRequest):

    errors = compare_text(req.reference, req.student)
    explanations = generate_explanations(errors)

    return {
        "reference": req.reference,
        "student": req.student,
        "errors": explanations
    }