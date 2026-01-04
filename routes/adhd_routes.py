# routes/adhd_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from services.db_service import get_db

router = APIRouter(prefix="/adhd", tags=["ADHD Assessment"])

class ADHDSubmission(BaseModel):
    grade: int = 3
    total_correct: int
    total_premature: int
    total_wrong: int
    overall_accuracy: Optional[float] = None
    timestamp: Optional[str] = None
    child_id: Optional[str] = None

@router.post("/submit-results")
async def submit_adhd_results(submission: ADHDSubmission):
    print(">>> ADHD submission received:", submission.dict())  # ← Debug line

    db = get_db()

    doc = {
        "grade": submission.grade,
        "total_correct": submission.total_correct,
        "total_premature": submission.total_premature,
        "total_wrong": submission.total_wrong,
        "overall_accuracy": submission.overall_accuracy,
        "timestamp": submission.timestamp or datetime.utcnow().isoformat(),
        "child_id": submission.child_id,
        "created_at": datetime.utcnow(),
    }

    try:
        result = db["adhd_submissions"].insert_one(doc)
        print(">>> Saved to MongoDB with ID:", result.inserted_id)
        return {
            "ok": True,
            "message": "ADHD assessment saved successfully",
            "assessment_id": str(result.inserted_id)
        }
    except Exception as e:
        print(">>> MongoDB insert error:", str(e))
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")