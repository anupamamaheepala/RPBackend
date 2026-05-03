# routes/dysgraphia_improvement_routes.py
# FastAPI router for dysgraphia improvement sessions.
# Endpoint paths are used as-is by the Flutter dashboard.

from fastapi import APIRouter, HTTPException
from models.dysgraphia_improvement_models import DysgraphiaImprovementSubmission
from services.dysgraphia_improvement_service import (
    save_improvement_session,
    get_user_improvement_results,
)
from typing import Dict, Any

router = APIRouter(
    prefix="/dysgraphia-improvement",
    tags=["dysgraphia-improvement"],
)


@router.post("/submit-session")
async def submit_improvement_session(
    submission: DysgraphiaImprovementSubmission,
) -> Dict[str, Any]:
    """
    Submit a completed improvement activity session from Flutter.

    Body fields (DysgraphiaImprovementSubmission):
        user_id, grade, risk_level, activity_name, activity_label,
        total_items, correct_count, duration_seconds

    Returns:
        ok            : bool
        session_id    : str
        score_percent : float  (0–100, computed server-side)
        message       : str
    """
    result = save_improvement_session(submission)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    return result


@router.get("/user-results/{user_id}")
async def get_improvement_results(user_id: str) -> Dict[str, Any]:
    """
    Returns all improvement sessions for a user plus a rich summary block.
    Called by the Flutter DysgraphiaDashboardMain on init and pull-to-refresh.

    Response shape (consumed directly by Flutter dashboard):
    {
        ok             : bool,
        user_id        : str,
        total_sessions : int,
        sessions       : [ { user_id, grade, risk_level, activity_name,
                              activity_label, total_items, correct_count,
                              score_percent, duration_seconds, created_at } ],
        summary: {
            latest_risk_level     : str,
            avg_accuracy          : float,
            this_month_accuracy   : float,
            last_month_accuracy   : float,
            latest_accuracy       : float,
            latest_activity_label : str,
            latest_duration       : float | null,
            total_duration_seconds: float,
            current_streak        : int,
            week_practiced        : [bool x7],   # Mon–Sun
            activity_bests        : {
                "<activity_name>": {
                    activity_name, activity_label,
                    best_accuracy, avg_accuracy,
                    session_count, last_played_at
                }
            },
            risk_counts: { low, medium, high }
        }
    }
    """
    result = get_user_improvement_results(user_id)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail="Failed to fetch user results")
    return result