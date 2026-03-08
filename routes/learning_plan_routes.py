from fastapi import APIRouter, HTTPException
from models.learning_plan_model import LearningPlanRequest, LearningPlanResponse
from services.learning_plan_service import generate_learning_plan, get_latest_plan

router = APIRouter(prefix="/learning-plan", tags=["Learning Plan"])


@router.post("/generate", response_model=LearningPlanResponse)
async def generate_plan(req: LearningPlanRequest):
    try:
        return generate_learning_plan(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/{child_id}")
async def get_plan(child_id: str):
    plan = get_latest_plan(child_id)
    if not plan:
        raise HTTPException(status_code=404, detail="No plan found")
    return plan