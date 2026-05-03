from fastapi import APIRouter, HTTPException
from models.adhd.g4_learning_task_model import (
    G4LearningTaskAssignRequest, G4LearningTaskResult,
)
from services.adhd.g4_learning_task_service import (
    g4_assign_tasks, g4_save_task_result, g4_get_progress,
)

router = APIRouter(prefix="/learning-tasks/g4", tags=["Learning Tasks Grade 4"])


@router.post("/assign")
async def g4_assign(req: G4LearningTaskAssignRequest):
    try:
        return g4_assign_tasks(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit-result")
async def g4_submit_result(result: G4LearningTaskResult):
    try:
        return g4_save_task_result(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{child_id}")
async def g4_progress(child_id: str):
    try:
        return g4_get_progress(child_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
