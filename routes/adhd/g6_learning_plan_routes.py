from fastapi import APIRouter, HTTPException
from models.adhd.g6_learning_plan_model import G6LearningPlanRequest, G6LearningPlanResponse
from services.adhd.g6_learning_plan_service import g6_generate_learning_plan, g6_get_latest_plan

router = APIRouter(prefix="/learning-plan/g6", tags=["Learning Plan Grade 6"])

@router.post("/generate", response_model=G6LearningPlanResponse)
async def g6_generate(req: G6LearningPlanRequest):
    try:    return g6_generate_learning_plan(req)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest/{child_id}")
async def g6_latest(child_id: str):
    plan = g6_get_latest_plan(child_id)
    if not plan: raise HTTPException(status_code=404, detail="No Grade 6 plan found")
    return plan
