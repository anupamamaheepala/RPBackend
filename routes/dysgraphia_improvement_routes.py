# routes/dysgraphia_improvement_routes.py

from fastapi import APIRouter, HTTPException
from models.dysgraphia_improvement_models import DysgraphiaImprovementSubmission
from services.dysgraphia_improvement_service import (
    save_improvement_session,
    get_user_improvement_results,
)
from typing import Dict, Any

router = APIRouter(prefix="/dysgraphia-improvement", tags=["dysgraphia-improvement"])


@router.post("/submit-session")
async def submit_improvement_session(
    submission: DysgraphiaImprovementSubmission,
) -> Dict[str, Any]:
    """
    Submit a completed improvement activity session.

    Records: user_id, grade, risk_level, activity_name, activity_label,
             total_items, correct_count, duration_seconds.
    Calculates and stores score_percent automatically.

    Returns:
        - ok: bool
        - session_id: str
        - score_percent: float (0-100)
        - message: str
    """
    result = save_improvement_session(submission)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/user-results/{user_id}")
async def get_improvement_results(user_id: str) -> Dict[str, Any]:
    """
    Returns all improvement activity sessions for a specific user,
    newest first. Includes a summary with average score, best score,
    activity bests, and risk level counts.

    Returns:
        - ok: bool
        - user_id: str
        - total_sessions: int
        - summary: { average_score, best_score, latest_activity, latest_score,
                     risk_counts, activity_bests }
        - sessions: list of session objects
    """
    return get_user_improvement_results(user_id)