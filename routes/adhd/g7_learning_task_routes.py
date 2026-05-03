from fastapi import APIRouter, HTTPException
from models.adhd.g7_learning_task_model import G7LearningTaskAssignRequest, G7LearningTaskResult
from services.adhd.g7_learning_task_service import g7_assign_tasks, g7_save_task_result, g7_get_progress

router = APIRouter(prefix="/learning-tasks/g7", tags=["Learning Tasks Grade 7"])

@router.post("/assign")
async def g7_assign(req: G7LearningTaskAssignRequest):
    try:    return g7_assign_tasks(req)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit-result")
async def g7_submit_result(result: G7LearningTaskResult):
    try:    return g7_save_task_result(result)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{child_id}")
async def g7_progress(child_id: str):
    try:    return g7_get_progress(child_id)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
