# routes/dysgraphia_routes.py

from fastapi import APIRouter, HTTPException
from models.dysgraphia import DysgraphiaSubmission
from services.dysgraphia_service import save_dysgraphia_submission, get_dysgraphia_stats
from typing import Dict, Any

router = APIRouter(prefix="/dysgraphia", tags=["dysgraphia"])

@router.post("/submit-writing")
async def submit_writing(submission: DysgraphiaSubmission) -> Dict[str, Any]:
    """
    Submit dysgraphia writing data (strokes, times, etc.).
    
    Expects JSON payload validated by DysgraphiaSubmission model.
    Stores in MongoDB 'dysgraphia_submissions' collection.
    """
    result = save_dysgraphia_submission(submission)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@router.get("/stats")
async def dysgraphia_stats() -> Dict[str, Any]:
    """
    Get aggregated stats (e.g., submissions by grade/activity_type).
    """
    return get_dysgraphia_stats()

@router.get("/analysis/{submission_id}")
async def get_submission_analysis(submission_id: str) -> Dict[str, Any]:
    """
    Get analysis for a specific submission.
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    doc = collection.find_one({"_id": submission_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Submission not found")
    return {"ok": True, "analysis": doc.get("analysis", {})}