# routes/dysgraphia_routes.py

from fastapi import APIRouter, HTTPException
from models.dysgraphia import DysgraphiaSubmission
from services.dysgraphia_service import (
    save_dysgraphia_submission, 
    get_dysgraphia_stats,
    recalculate_all_risks,
    get_db  # ADD THIS
)
from typing import Dict, Any, List  # ADD List HERE

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

@router.get("/export-data")
async def export_data() -> List[Dict[str, Any]]:
    """
    Export all dysgraphia data for ML training.
    Returns cleaned data ready for ML.
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    
    submissions = list(collection.find({}))
    
    ml_data = []
    for sub in submissions:
        details = sub.get("risk_details", {})
        ml_data.append({
            # Features
            "avg_time": details.get("avg_time_per_prompt", 0),
            "avg_strokes": details.get("avg_strokes", 0),
            "avg_clears": details.get("avg_clears_per_prompt", 0),
            "time_inconsistency": details.get("time_inconsistency", 0),
            "excessive_strokes_count": details.get("excessive_strokes_count", 0),
            "excessive_clears_count": details.get("excessive_clears_count", 0),
            "time_deviation_percent": details.get("time_deviation_percent", 0),
            
            # Labels & Metadata
            "risk_level": sub.get("risk_level", "none"),
            "risk_score": sub.get("risk_score", 0),
            "grade": sub.get("grade", 3),
            "activity_type": sub.get("activity_type", "letters"),
            
            # Timestamps
            "created_at": sub.get("created_at").isoformat() if sub.get("created_at") else None
        })
    
    return ml_data