import json
from datetime import datetime
from typing import Optional, Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel
from services.db_service import get_db

router = APIRouter(prefix="/learning", tags=["Learning Progress"])

db = get_db()

# Create index for faster querying
db["dyslexia_learning_progress_results"].create_index([("user_id", 1), ("created_at", -1)])


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

        # ---------- (A) Save detailed progress result ----------
        progress_doc = payload.model_dump()
        progress_doc["created_at"] = created_at

        result = db["dyslexia_learning_progress_results"].insert_one(progress_doc)

        # ---------- (B) Update module progress tracking ----------
        db["module_progress"].update_one(
            {
                "user_id": payload.user_id,
                "grade": payload.grade,
                "level": payload.level,
                "module_number": payload.module_number
            },
            {
                "$set": {
                    f"activities.Activity{payload.activity}": True,
                    "updated_at": created_at
                }
            },
            upsert=True
        )

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
            db["dyslexia_learning_progress_results"]
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
    

@router.get("/get-module-progress")
def get_module_progress(
    user_id: str,
    grade: int,
    level: int,
    module_number: int
):
    doc = db["dyslexia_module_progress"].find_one({
        "user_id": user_id,
        "grade": grade,
        "level": level,
        "module_number": module_number
    })

    if not doc:
        return {
            "ok": True,
            "progress": {}
        }

    return {
        "ok": True,
        "progress": doc.get("activities", {})
    }

#get the grade , level, risk level to assign learning paths

@router.get("/get-assigned-learning-path")
def get_assigned_learning_path(
    user_id: str,
    grade: int,
    level: int
):
    session = db["reading_sessions"].find_one(
        {
            "user_id": user_id,
            "grade": grade,
            "level": level
        },
        sort=[("created_at", -1)]
    )

    if not session:
        return {
            "ok": True,
            "eligible": False
        }

    risk_level = session.get("dyslexia_assessment", {}).get("risk_level")

    if not risk_level:
        return {
            "ok": True,
            "eligible": False
        }

    return {
        "ok": True,
        "eligible": True,
        "risk_level": risk_level
    }


# Check if the user is allowed to test or must do learning first
@router.get("/learning/check-completion")
async def check_completion(user_id: str, grade: int, level: int):
    # Look for an assignment for this specific grade and level
    assignment = db["learning_assignments"].find_one({
        "user_id": user_id, 
        "grade": grade, 
        "level": level
    })
    
    # If no assignment exists, or if it is marked as COMPLETED, they can re-test
    if not assignment or assignment.get("status") == "COMPLETED":
        return {"can_retest": True}
        
    return {"can_retest": False}

# Update the status to COMPLETED when they finish activities
@router.post("/learning/update-status")
async def update_status(data: dict):
    db["learning_assignments"].update_one(
        {
            "user_id": data.get("user_id"), 
            "grade": data.get("grade"), 
            "level": data.get("level")
        },
        {"$set": {"status": data.get("status"), "completed_at": datetime.utcnow()}},
        upsert=True
    )
    return {"ok": True}