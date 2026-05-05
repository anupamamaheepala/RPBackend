from fastapi import APIRouter, HTTPException
from models.adhd.g7_learning_plan_model import G7LearningPlanRequest, G7LearningPlanResponse
from services.adhd.g7_learning_plan_service import g7_generate_learning_plan, g7_get_latest_plan

router = APIRouter(prefix="/learning-plan/g7", tags=["Learning Plan Grade 7"])

@router.post("/generate", response_model=G7LearningPlanResponse)
async def g7_generate(req: G7LearningPlanRequest):
    try:    return g7_generate_learning_plan(req)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest/{child_id}")
async def g7_get_plan(child_id: str):
    plan = g7_get_latest_plan(child_id)
    if not plan: raise HTTPException(status_code=404, detail="No Grade 7 plan found")
    return plan
