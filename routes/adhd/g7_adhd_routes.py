from fastapi import APIRouter, HTTPException
from models.adhd.g7_adhd_model import G7ADHDSubmissionRequest, G7ADHDSubmissionResponse, G7DiagnosticHistoryResponse
from services.adhd.g7_adhd_service import g7_compute_metrics, g7_save_assessment, g7_get_diagnostic_history

router = APIRouter(prefix="/adhd/g7", tags=["ADHD Grade 7"])

@router.post("/submit-results", response_model=G7ADHDSubmissionResponse)
async def g7_submit(submission: G7ADHDSubmissionRequest):
    try:
        metrics = g7_compute_metrics(submission)
        aid     = g7_save_assessment(submission, metrics)
        return G7ADHDSubmissionResponse(ok=True, message="Grade 7 assessment saved",
                                        assessment_id=aid, computed_metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{child_id}", response_model=G7DiagnosticHistoryResponse)
async def g7_history(child_id: str):
    try:    return g7_get_diagnostic_history(child_id)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
