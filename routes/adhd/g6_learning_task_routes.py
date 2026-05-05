from fastapi import APIRouter, HTTPException
from models.adhd.g6_learning_task_model import G6LearningTaskAssignRequest, G6LearningTaskResult
from services.adhd.g6_learning_task_service import g6_assign_tasks, g6_save_task_result, g6_get_progress

router = APIRouter(prefix="/learning-tasks/g6", tags=["Learning Tasks Grade 6"])

@router.post("/assign")
async def g6_assign(req: G6LearningTaskAssignRequest):
    try:    return g6_assign_tasks(req)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit-result")
async def g6_result(result: G6LearningTaskResult):
    try:    return g6_save_task_result(result)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{child_id}")
async def g6_progress(child_id: str):
    try:    return g6_get_progress(child_id)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
