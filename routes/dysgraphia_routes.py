# routes/dysgraphia_routes.py

from fastapi import APIRouter, HTTPException
from models.dysgraphia import DysgraphiaSubmission
from services.dysgraphia_service import (
    save_dysgraphia_submission, 
    get_dysgraphia_stats,
    recalculate_all_risks
)
from typing import Dict, Any

router = APIRouter(prefix="/dysgraphia", tags=["dysgraphia"])

@router.post("/submit-writing")
async def submit_writing(submission: DysgraphiaSubmission) -> Dict[str, Any]:
    """
    Submit dysgraphia writing data (strokes, times, etc.).
    
    Expects JSON payload validated by DysgraphiaSubmission model.
    Automatically calculates and stores risk level (none/low/medium/high).
    Stores in MongoDB 'dysgraphia_submissions' collection.
    
    Returns:
        - ok: bool
        - submission_id: str
        - risk_level: str (none/low/medium/high)
        - risk_score: float (0-100)
        - message: str
    """
    result = save_dysgraphia_submission(submission)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@router.get("/stats")
async def dysgraphia_stats() -> Dict[str, Any]:
    """
    Get aggregated stats including:
    - Submissions by grade/activity_type
    - Risk level distribution (none/low/medium/high counts)
    - Average risk scores by grade
    """
    return get_dysgraphia_stats()

@router.post("/recalculate-risks")
async def recalculate_risks() -> Dict[str, Any]:
    """
    Recalculate risk levels for all existing submissions.
    Useful when updating the risk calculation algorithm.
    
    WARNING: This updates all records in the database.
    """
    result = recalculate_all_risks()
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result