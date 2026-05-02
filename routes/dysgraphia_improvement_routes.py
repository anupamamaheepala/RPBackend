from fastapi import APIRouter, HTTPException
from models.dysgraphia_improvement_models import DysgraphiaImprovementSubmission
from services.dysgraphia_improvement_service import (
    save_improvement_session,
    get_user_improvement_results,
)
from typing import Dict, Any

router = APIRouter(prefix="/dysgraphia-improvement", tags=["dysgraphia-improvement"])

@router.post("/submit-session")
async def submit_improvement_session(submission: DysgraphiaImprovementSubmission):
    """Submits results and updates the child's progress state."""
    result = save_improvement_session(submission)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@router.get("/user-results/{user_id}")
async def get_improvement_results(user_id: str):
    """
    Returns the current tier, mastery progress, and whether the 
    Detection Gate is unlocked.
    """
    result = get_user_improvement_results(user_id)
    return result