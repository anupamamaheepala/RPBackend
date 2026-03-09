from fastapi import APIRouter, HTTPException
from models.learning_task_model import (
    LearningTaskAssignRequest, LearningTaskResult
)
from services.learning_task_service import (
    assign_tasks, save_task_result, get_progress
)

router = APIRouter(prefix="/learning-tasks", tags=["Learning Tasks"])


@router.post("/assign")
async def assign(req: LearningTaskAssignRequest):
    try:
        return assign_tasks(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit-result")
async def submit_result(result: LearningTaskResult):
    try:
        return save_task_result(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{child_id}")
async def progress(child_id: str):
    try:
        return get_progress(child_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))