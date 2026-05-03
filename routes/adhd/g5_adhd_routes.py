from fastapi import APIRouter, HTTPException
from models.adhd.g5_adhd_model import G5ADHDSubmissionRequest, G5ADHDSubmissionResponse, G5DiagnosticHistoryResponse
from services.adhd.g5_adhd_service import g5_compute_metrics, g5_save_assessment, g5_get_diagnostic_history

router = APIRouter(prefix="/adhd/g5", tags=["ADHD Grade 5"])

@router.post("/submit-results", response_model=G5ADHDSubmissionResponse)
async def g5_submit(submission: G5ADHDSubmissionRequest):
    try:
        metrics = g5_compute_metrics(submission)
        aid     = g5_save_assessment(submission, metrics)
        return G5ADHDSubmissionResponse(ok=True, message="Grade 5 assessment saved",
                                        assessment_id=aid, computed_metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{child_id}", response_model=G5DiagnosticHistoryResponse)
async def g5_history(child_id: str):
    try:
        return g5_get_diagnostic_history(child_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
