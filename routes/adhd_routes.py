from fastapi import APIRouter, HTTPException
from models.adhd_model import ADHDSubmissionRequest, ADHDSubmissionResponse
from services.adhd_service import compute_metrics, save_assessment

router = APIRouter(prefix="/adhd", tags=["ADHD Assessment"])


@router.post("/submit-results", response_model=ADHDSubmissionResponse)
async def submit_adhd_results(submission: ADHDSubmissionRequest):
    try:
        metrics       = compute_metrics(submission)
        assessment_id = save_assessment(submission, metrics)

        return ADHDSubmissionResponse(
            ok=True,
            message="Assessment saved successfully",
            assessment_id=assessment_id,
            computed_metrics=metrics,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))