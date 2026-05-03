from fastapi import APIRouter, HTTPException
from models.adhd.g5_learning_plan_model import G5LearningPlanRequest, G5LearningPlanResponse
from services.adhd.g5_learning_plan_service import g5_generate_learning_plan, g5_get_latest_plan

router = APIRouter(prefix="/learning-plan/g5", tags=["Learning Plan Grade 5"])

@router.post("/generate", response_model=G5LearningPlanResponse)
async def g5_generate(req: G5LearningPlanRequest):
    try:
        return g5_generate_learning_plan(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest/{child_id}")
async def g5_get_plan(child_id: str):
    plan = g5_get_latest_plan(child_id)
    if not plan:
        raise HTTPException(status_code=404, detail="No Grade 5 plan found")
    return plan
