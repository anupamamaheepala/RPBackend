from fastapi import APIRouter, HTTPException
from models.adhd.g4_learning_plan_model import G4LearningPlanRequest, G4LearningPlanResponse
from services.adhd.g4_learning_plan_service import g4_generate_learning_plan, g4_get_latest_plan

router = APIRouter(prefix="/learning-plan/g4", tags=["Learning Plan Grade 4"])


@router.post("/generate", response_model=G4LearningPlanResponse)
async def g4_generate(req: G4LearningPlanRequest):
    try:
        return g4_generate_learning_plan(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/{child_id}")
async def g4_get_plan(child_id: str):
    plan = g4_get_latest_plan(child_id)
    if not plan:
        raise HTTPException(status_code=404, detail="No Grade 4 plan found")
    return plan
