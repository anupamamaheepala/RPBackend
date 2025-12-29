from fastapi import APIRouter, HTTPException, Body
from services.db_service import get_db
from models.dyscalculia_models import DyscalculiaResult
from datetime import datetime

router = APIRouter()
db = get_db()

@router.post("/dyscalculia/submit-result")
async def submit_dyscalculia_result(result: DyscalculiaResult):
    try:
        result_dict = result.dict()
        result_dict["created_at"] = datetime.utcnow()
        
        # Insert into 'dyscalculia_results' collection
        insert_result = db["dyscalculia_results"].insert_one(result_dict)
        
        return {
            "ok": True, 
            "id": str(insert_result.inserted_id),
            "message": "Results saved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))