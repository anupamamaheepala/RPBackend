from fastapi import APIRouter, HTTPException
from models.adhd.g4_adhd_model import (
    G4ADHDSubmissionRequest, G4ADHDSubmissionResponse,
    G4DiagnosticHistoryResponse,
)
from services.adhd.g4_adhd_service import (
    g4_compute_metrics, g4_save_assessment, g4_get_diagnostic_history,
)

router = APIRouter(prefix="/adhd/g4", tags=["ADHD Grade 4"])


@router.post("/submit-results", response_model=G4ADHDSubmissionResponse)
async def g4_submit(submission: G4ADHDSubmissionRequest):
    try:
        metrics       = g4_compute_metrics(submission)
        assessment_id = g4_save_assessment(submission, metrics)
        return G4ADHDSubmissionResponse(
            ok=True,
            message="Grade 4 assessment saved",
            assessment_id=assessment_id,
            computed_metrics=metrics,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{child_id}", response_model=G4DiagnosticHistoryResponse)
async def g4_history(child_id: str):
    try:
        return g4_get_diagnostic_history(child_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
