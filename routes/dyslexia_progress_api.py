import json
from datetime import datetime
from typing import Optional, Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel
from services.db_service import get_db

router = APIRouter(prefix="/learning", tags=["Learning Progress"])

db = get_db()

# Create index for faster querying
db["learning_progress_results"].create_index([("user_id", 1), ("created_at", -1)])


# ===============================
# PAYLOAD MODEL
# ===============================

class LearningProgressPayload(BaseModel):
    username: str
    user_id: Optional[str] = None

    grade: int
    level: int
    module_number: int
    activity: int

    total_words: int
    total_correct: int
    overall_accuracy: float
    avg_words_per_second: float

    sentences: List[Dict[str, Any]]


# ===============================
# SUBMIT LEARNING PROGRESS
# ===============================

@router.post("/submit-progress")
def submit_learning_progress(payload: LearningProgressPayload):
    try:
        created_at = datetime.utcnow()

        progress_doc = payload.model_dump()
        progress_doc["created_at"] = created_at

        result = db["learning_progress_results"].insert_one(progress_doc)

        return {
            "ok": True,
            "progress_id": str(result.inserted_id),
            "message": "Learning progress saved successfully"
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


# ===============================
# GET PROGRESS HISTORY
# ===============================

@router.get("/get-progress-history")
def get_progress_history(user_id: str):
    try:
        records = list(
            db["learning_progress_results"]
            .find({"user_id": user_id})
            .sort("created_at", -1)
        )

        # Convert ObjectId to string
        for r in records:
            r["_id"] = str(r["_id"])

        return {
            "ok": True,
            "records": records
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }