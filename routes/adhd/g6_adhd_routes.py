from fastapi import APIRouter, HTTPException
from models.adhd.g6_adhd_model import G6ADHDSubmissionRequest, G6ADHDSubmissionResponse, G6DiagnosticHistoryResponse
from services.adhd.g6_adhd_service import g6_compute_metrics, g6_save_assessment, g6_get_history

router = APIRouter(prefix="/adhd/g6", tags=["ADHD Grade 6"])

@router.post("/submit-results", response_model=G6ADHDSubmissionResponse)
async def g6_submit(req: G6ADHDSubmissionRequest):
    try:
        metrics = g6_compute_metrics(req)
        aid     = g6_save_assessment(req, metrics)
        return G6ADHDSubmissionResponse(ok=True, message="Grade 6 assessment saved",
                                        assessment_id=aid, computed_metrics=metrics)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{child_id}")
async def g6_history(child_id: str):
    try:    return g6_get_history(child_id)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
