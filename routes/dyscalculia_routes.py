import os
import numpy as np
from fastapi import APIRouter, HTTPException
from services.db_service import get_db
from models.dyscalculia_models import DyscalculiaResult
from datetime import datetime

# ML MODEL TEMPORARILY COMMENTED OUT FOR TESTING
# import joblib
# current_dir = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(current_dir, "dyscalculia_rf_model.pkl")
# rf_model = None
# try:
#     rf_model = joblib.load(MODEL_PATH)
#     print(f"Dyscalculia RF Model loaded successfully")
# except Exception as e:
#     print(f"Warning: Could not load ML model at {MODEL_PATH}. Error: {e}")

router = APIRouter()
db = get_db()

@router.post("/dyscalculia/submit-result")
async def submit_dyscalculia_result(result: DyscalculiaResult):
    try:
        # Default string since we are skipping the ML step for now
        risk_level_str = "Pending AI Setup"
        
        # --- ML PREDICTION COMMENTED OUT ---
        # if rf_model is not None:
        #     features = np.array([[
        #         result.grade, result.task_number, result.accuracy,
        #         result.response_time_avg, result.hesitation_time_avg,
        #         result.retries, result.backtracks, result.skipped_items,
        #         result.completion_time # NOTE: We will need to decide if wrong_count gets added to features later!
        #     ]])
        #     prediction = rf_model.predict(features)[0]
        #     ... (risk mapping logic) ...

        # 2. Prepare data for MongoDB
        result_dict = result.dict()
        result_dict["risk_level"] = risk_level_str
        result_dict["created_at"] = datetime.utcnow()
        
        # 3. Insert into database
        insert_result = db["dyscalculia_results"].insert_one(result_dict)
        
        # 4. Return the response
        return {
            "ok": True, 
            "id": str(insert_result.inserted_id),
            "risk_level": risk_level_str,
            "message": "Results saved successfully. Data collection active."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))