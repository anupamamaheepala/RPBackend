from fastapi import APIRouter, HTTPException
from models.adhd.g5_learning_task_model import G5LearningTaskAssignRequest, G5LearningTaskResult
from services.adhd.g5_learning_task_service import g5_assign_tasks, g5_save_task_result, g5_get_progress

router = APIRouter(prefix="/learning-tasks/g5", tags=["Learning Tasks Grade 5"])

@router.post("/assign")
async def g5_assign(req: G5LearningTaskAssignRequest):
    try:    return g5_assign_tasks(req)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit-result")
async def g5_submit_result(result: G5LearningTaskResult):
    try:    return g5_save_task_result(result)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{child_id}")
async def g5_progress(child_id: str):
    try:    return g5_get_progress(child_id)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
