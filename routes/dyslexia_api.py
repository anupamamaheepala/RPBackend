# # routes/dyslexia_api.py
# import os
# import json
# import tempfile
# from datetime import datetime
# from typing import Optional, Any, Dict, List

# from fastapi import APIRouter, UploadFile, File, Form
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from bson import Binary

# from services.db_service import get_db
# from config.settings import settings
# from openai import OpenAI

# from services.dyslexia_metrics import compute_metrics, compute_dyslexia_risk
# from models.dyslexia.predict import predict_dyslexia_risk_ml

# router = APIRouter(prefix="/dyslexia", tags=["Dyslexia"])

# db = get_db()
# client = OpenAI(api_key=settings.OPENAI_API_KEY)

# db["reading_session_stats"].create_index([("user_id", 1), ("created_at", -1)])

# # ---------- 1) PER SENTENCE ANALYZE ----------
# @router.post("/analyze-audio")
# async def analyze_audio(
#     username: str = Form(...),
#     user_id: Optional[str] = Form(None),
#     reference_text: str = Form(...),
#     duration: Optional[float] = Form(None),
#     grade: Optional[int] = Form(None),
#     level: Optional[int] = Form(None),
#     sentence_index: Optional[int] = Form(None),
#     eye_metrics: Optional[str] = Form(None),
#     file: UploadFile = File(...),
# ):
#     tmp_path = None
#     try:
#         audio_bytes = await file.read()

#         suffix = os.path.splitext(file.filename)[1] or ".wav"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
#             tmp_file.write(audio_bytes)
#             tmp_path = tmp_file.name

#         with open(tmp_path, "rb") as audio_file:
#             transcription = client.audio.transcriptions.create(
#                 model="gpt-4o-transcribe",
#                 file=audio_file
#             )

#         transcript_text = transcription.text.strip()
#         metrics = compute_metrics(reference_text, transcript_text, duration)

#         return {"ok": True, "metrics": metrics, "sentence_index": sentence_index}

#     except Exception as e:
#         return {"ok": False, "error": f"Analyze failed: {e}"}

#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)


# # ---------- 2) OLD ENDPOINT (MOVED) ----------
# @router.post("/submit-audio")
# async def submit_audio(
#     username: str = Form(...),
#     user_id: Optional[str] = Form(None),
#     reference_text: str = Form(...),
#     duration: Optional[float] = Form(None),
#     grade: Optional[int] = Form(None),
#     level: Optional[int] = Form(None),
#     eye_metrics: Optional[str] = Form(None),
#     file: UploadFile = File(...),
# ):
#     tmp_path = None
#     try:
#         audio_bytes = await file.read()

#         audio_doc = {
#             "filename": file.filename,
#             "content_type": file.content_type,
#             "data": Binary(audio_bytes),
#             "grade": grade,
#             "level": level,
#             "duration": duration,
#             "created_at": datetime.utcnow(),
#         }
#         audio_result = db["audio_files"].insert_one(audio_doc)
#         audio_id = audio_result.inserted_id

#         suffix = os.path.splitext(file.filename)[1] or ".wav"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
#             tmp_file.write(audio_bytes)
#             tmp_path = tmp_file.name

#         with open(tmp_path, "rb") as audio_file:
#             transcription = client.audio.transcriptions.create(
#                 model="gpt-4o-transcribe",
#                 file=audio_file
#             )

#         transcript_text = transcription.text.strip()
#         metrics = compute_metrics(reference_text, transcript_text, duration)

#         eye_data = {}
#         if eye_metrics:
#             try:
#                 eye_data = json.loads(eye_metrics)
#             except Exception:
#                 eye_data = {}

#         try:
#             dyslexia_risk = predict_dyslexia_risk_ml(
#                 audio_metrics=metrics,
#                 eye_metrics=eye_data,
#                 duration=duration
#             )
#         except Exception:
#             dyslexia_risk = compute_dyslexia_risk(metrics, eye_data)

#         reading_doc = {
#             "username": username,
#             "user_id": user_id,
#             "audio_file_id": audio_id,
#             "grade": grade,
#             "level": level,
#             "duration": duration,

#             "audio_id": str(audio_id),
#             "audio_url": f"http://localhost:8000/audio/{audio_id}",
#             "audio_metrics": metrics,

#             "eye_tracking": {
#                 "fixation_count": eye_data.get("fixation_count", 0),
#                 "avg_fixation_ms": eye_data.get("avg_fixation_ms", 0),
#                 "regression_count": eye_data.get("regression_count", 0),
#                 "saccade_count": eye_data.get("saccade_count", 0),
#                 "blink_rate_per_min": eye_data.get("blink_rate_per_min", 0),
#             },
#             "dyslexia_assessment": dyslexia_risk,
#             "created_at": datetime.utcnow(),
#         }

#         reading_result = db["readings"].insert_one(reading_doc)

#         return {
#             "ok": True,
#             "reading_id": str(reading_result.inserted_id),
#             "metrics": metrics,
#             "eye_tracking": eye_data,
#             "dyslexia_assessment": dyslexia_risk,
#         }

#     except Exception as e:
#         return {"ok": False, "error": f"Transcription failed: {e}"}

#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)


# # ---------- 3) FINAL SESSION SUBMIT ----------
# class SessionPayload(BaseModel):
#     username: str
#     user_id: Optional[str] = None
#     grade: int
#     level: int

#     total_words: int
#     total_correct: int
#     total_time_seconds: int
#     overall_accuracy: float

#     mean_sentence_accuracy: float
#     sentence_accuracy_std_dev: float

#     avg_WER: float
#     avg_CER: float
#     avg_words_per_second: float

#     incorrect_words_all: List[str]

#     avg_fixation_time: float
#     avg_regression_count: float
#     avg_saccade_count: float
#     avg_blink_rate_per_min: float

#     sentences: List[Dict[str, Any]]


# @router.post("/generate-tts")
# async def generate_tts(text: str = Form(...)):
#     """
#     Generate Sinhala speech audio using OpenAI TTS
#     Returns mp3 file
#     """
#     try:
#         response = client.audio.speech.create(
#             model="gpt-4o-mini-tts",
#             voice="alloy",
#             input=text,
#         )

#         # ✅ Correct way to get audio bytes
#         audio_bytes = response.content

#         tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#         tmp_file.write(audio_bytes)
#         tmp_file.close()

#         return FileResponse(
#             tmp_file.name,
#             media_type="audio/mpeg",
#             filename="tts.mp3"
#         )

#     except Exception as e:
#         print("TTS ERROR:", e)
#         return {"ok": False, "error": str(e)}

# @router.post("/submit-session")
# def submit_session(payload: SessionPayload):
#     audio_metrics = {
#         "accuracy_percent": payload.overall_accuracy,
#         "wer": payload.avg_WER,
#         "cer": payload.avg_CER,
#         "words_per_second": payload.avg_words_per_second,
#         "correct_words": payload.total_correct,
#         "total_words": payload.total_words,
#     }

#     eye_metrics = {
#         "avg_fixation_ms": payload.avg_fixation_time,
#         "regression_count": payload.avg_regression_count,
#         "saccade_count": payload.avg_saccade_count,
#         "blink_rate_per_min": payload.avg_blink_rate_per_min,
#     }

#     try:
#         dyslexia_risk = predict_dyslexia_risk_ml(
#             audio_metrics=audio_metrics,
#             eye_metrics=eye_metrics,
#             duration=None
#         )
#     except Exception:
#         dyslexia_risk = compute_dyslexia_risk(audio_metrics, eye_metrics)

#     created_at = datetime.utcnow()

# # (A) Save full session (existing behavior)
#     session_doc = payload.model_dump()
#     session_doc["dyslexia_assessment"] = dyslexia_risk
#     session_doc["created_at"] = created_at

#     result = db["reading_sessions"].insert_one(session_doc)

# # (B) Save summary stats separately (NEW)
#     stats_doc = {
#         "username": payload.username,
#         "user_id": payload.user_id,
#         "grade": payload.grade,
#         "level": payload.level,

#     "total_words": payload.total_words,
#     "total_correct": payload.total_correct,
#     "overall_accuracy": payload.overall_accuracy,
#     "total_time_seconds": payload.total_time_seconds,

#     "mean_sentence_accuracy": payload.mean_sentence_accuracy,
#     "sentence_accuracy_std_dev": payload.sentence_accuracy_std_dev,

#     "avg_WER": payload.avg_WER,
#     "avg_CER": payload.avg_CER,
#     "avg_words_per_second": payload.avg_words_per_second,

#     # link to full session (nice to have)
#     "full_session_id": str(result.inserted_id),

#     "created_at": created_at,
# }

#     stats_result = db["reading_session_stats"].insert_one(stats_doc)

#     return {
#         "ok": True,
#         "session_id": str(result.inserted_id),
#         "stats_id": str(stats_result.inserted_id),
#         "dyslexia_assessment": dyslexia_risk
#     }

import os
import json
import tempfile
from datetime import datetime
from typing import Optional, Any, Dict, List

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from bson import Binary, ObjectId

from services.db_service import get_db
from config.settings import settings
from openai import OpenAI

from services.dyslexia_metrics import compute_metrics, compute_dyslexia_risk
from models.dyslexia.predict import predict_dyslexia_risk_ml

router = APIRouter(prefix="/dyslexia", tags=["Dyslexia"])

db = get_db()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Ensure index for faster lookups
db["reading_session_stats"].create_index([("user_id", 1), ("created_at", -1)])

class SessionPayload(BaseModel):
    username: str
    user_id: Optional[str] = None
    grade: int
    level: int
    total_words: int
    total_correct: int
    total_time_seconds: float
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
    eye_metrics: Optional[str] = Form(None), # Add this back
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
            # Note: Ensure model name is "whisper-1" or "gpt-4o-audio-preview" depending on your OpenAI tier
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )

        transcript_text = transcription.text.strip()
        
        # Calculate metrics for THIS sentence
        metrics = compute_metrics(reference_text, transcript_text, duration)

        # Return metrics so the frontend can store them for the final session
        return {
            "ok": True, 
            "metrics": metrics, 
            "transcript": transcript_text,
            "sentence_index": sentence_index
        }

    except Exception as e:
        print(f"Analyze Error: {e}")
        return {"ok": False, "error": f"Analyze failed: {str(e)}"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

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


# ---------- 2) SESSION SUBMISSION (HYBRID ML) ----------
@router.post("/submit-session")
def submit_session(payload: SessionPayload):
    # Step 1: Prepare raw data for pillar calculation
    audio_metrics = {
        "accuracy_percent": payload.overall_accuracy,
        "wer": payload.avg_WER,
        "words_per_second": payload.avg_words_per_second,
        "total_words": payload.total_words
    }
    eye_metrics = {
        "avg_fixation_ms": payload.avg_fixation_time,
        "regression_count": payload.avg_regression_count
    }

    # Step 2: Use your existing logic to get the 'Pillar Risks' (the inputs for the ML)
    base_risks = compute_dyslexia_risk(audio_metrics, eye_metrics)

    # Step 3: Map everything to the 10 features the .pkl model expects
    ml_input = {
        'grade': payload.grade,
        'level': payload.level,
        'total_words': payload.total_words,
        'overall_accuracy': payload.overall_accuracy,
        'avg_WER': payload.avg_WER,
        'avg_CER': payload.avg_CER,
        'total_time_seconds': payload.total_time_seconds,
        'dyslexia_assessment.phonological_risk': base_risks["phonological_risk"],
        'dyslexia_assessment.fluency_risk': base_risks["fluency_risk"],
        'dyslexia_assessment.eye_risk': base_risks["eye_risk"]
    }

    # Step 4: RUN THE TRAINED MODEL
    try:
        # This uses your dyslexia_model.pkl
        ml_result = predict_dyslexia_risk_ml(ml_input)
        
        # If the ML model returns a confidence, we use that
        final_assessment = ml_result
    except Exception as e:
        print(f"ML Error, falling back to basic logic: {e}")
        # Fallback if .pkl files are missing or version mismatch
        final_assessment = {
            "risk_level": base_risks["risk_level"],
            "confidence": 0.50, # Low confidence for fallback
            "method": "Formula Fallback"
        }

    # Step 5: Save and Return
    session_doc = payload.model_dump()
    session_doc["dyslexia_assessment"] = final_assessment
    session_doc["created_at"] = datetime.utcnow()
    
    db_result = db["reading_sessions"].insert_one(session_doc)

    return {
        "ok": True,
        "session_id": str(db_result.inserted_id),
        "dyslexia_assessment": final_assessment
    }

# @router.post("/submit-session")
# def submit_session(payload: SessionPayload):
#     # Step 1: Pre-calculate the 'Risk Pillars' required by the ML Model
#     audio_metrics_for_formula = {
#         "accuracy_percent": payload.overall_accuracy,
#         "wer": payload.avg_WER,
#         "words_per_second": payload.avg_words_per_second,
#         "correct_words": payload.total_correct,
#         "total_words": payload.total_words,
#     }

#     eye_metrics_for_formula = {
#         "avg_fixation_ms": payload.avg_fixation_time,
#         "regression_count": payload.avg_regression_count,
#     }

#     # Use existing formulas to get the input features for the ML model
#     base_calculation = compute_dyslexia_risk(audio_metrics_for_formula, eye_metrics_for_formula)

#     # Step 2: Prepare exact feature set for ML prediction
#     ml_features = {
#         'grade': payload.grade,
#         'level': payload.level,
#         'total_words': payload.total_words,
#         'overall_accuracy': payload.overall_accuracy,
#         'avg_WER': payload.avg_WER,
#         'avg_CER': payload.avg_CER,
#         'total_time_seconds': payload.total_time_seconds,
#         'dyslexia_assessment.phonological_risk': base_calculation["phonological_risk"],
#         'dyslexia_assessment.fluency_risk': base_calculation["fluency_risk"],
#         'dyslexia_assessment.eye_risk': base_calculation["eye_risk"]
#     }

#     # Step 3: Execute ML Prediction using .pkl files
#     try:
#         dyslexia_assessment = predict_dyslexia_risk_ml(ml_features)
#         # Ensure we didn't get an error dict back
#         if "error" in dyslexia_assessment:
#             raise Exception(dyslexia_assessment["error"])
#     except Exception as e:
#         print(f"ML Prediction failed, falling back to formula: {e}")
#         dyslexia_assessment = base_calculation
#         dyslexia_assessment["method"] = "Formula Fallback"

#     # Step 4: Save to MongoDB
#     created_at = datetime.utcnow()
    
#     session_doc = payload.model_dump()
#     session_doc["dyslexia_assessment"] = dyslexia_assessment
#     session_doc["created_at"] = created_at
#     result = db["reading_sessions"].insert_one(session_doc)

#     stats_doc = {
#         "username": payload.username,
#         "user_id": payload.user_id,
#         "grade": payload.grade,
#         "level": payload.level,
#         "total_words": payload.total_words,
#         "overall_accuracy": payload.overall_accuracy,
#         "avg_WER": payload.avg_WER,
#         "risk_level": dyslexia_assessment.get("risk_level"),
#         "full_session_id": str(result.inserted_id),
#         "created_at": created_at,
#     }
#     db["reading_session_stats"].insert_one(stats_doc)

#     return {
#         "ok": True, 
#         "session_id": str(result.inserted_id), 
#         "dyslexia_assessment": dyslexia_assessment
#     }

# ---------- 3) UTILITIES ----------
@router.post("/generate-tts")
async def generate_tts(text: str = Form(...)):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )
        audio_bytes = response.content
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_file.write(audio_bytes)
        tmp_file.close()

        return FileResponse(tmp_file.name, media_type="audio/mpeg", filename="tts.mp3")
    except Exception as e:
        return {"ok": False, "error": str(e)}