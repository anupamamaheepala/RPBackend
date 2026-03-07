import os
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from services.db_service import get_db
from models.dyscalculia_models import DyscalculiaResult
from datetime import datetime

router = APIRouter()
db = get_db()

# --- LOAD THE ML MODEL ---
# This safely finds the .pkl file as long as it is in the same folder as this script
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "dyscalculia_rf_model.pkl")

rf_model = None

try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"Dyscalculia RF Model loaded successfully with joblib from: {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load ML model at {MODEL_PATH}. Error: {e}")

@router.post("/dyscalculia/submit-result")
async def submit_dyscalculia_result(result: DyscalculiaResult):
    try:
        risk_level_str = "Pending/Error"
        
        # 1. Predict Risk Level using the ML Model
        if rf_model is not None:
            # Prepare features array (MUST match the exact order of your CSV dataset)
            features = np.array([[
                result.grade,
                result.task_number,
                result.accuracy,
                result.response_time_avg,
                result.hesitation_time_avg,
                result.retries,
                result.backtracks,
                result.skipped_items,
                result.completion_time
            ]])
            
            # Predict
            prediction = rf_model.predict(features)[0]
            
            # Handle the output depending on how your dataset was labeled
            if str(prediction) == "0":
                risk_level_str = "No Dyscalculia"
            elif str(prediction) == "1":
                risk_level_str = "Mild Dyscalculia"
            elif str(prediction) == "2":
                risk_level_str = "Severe Dyscalculia"
            else:
                # If your dataset labels were already strings like 'Mild Dyscalculia'
                risk_level_str = str(prediction) 
        else:
            print("Model not loaded. Skipping ML prediction.")

        # 2. Prepare data for MongoDB
        result_dict = result.dict()
        result_dict["risk_level"] = risk_level_str
        result_dict["created_at"] = datetime.utcnow()
        
        # 3. Insert into database
        insert_result = db["dyscalculia_results"].insert_one(result_dict)
        
        # 4. Return the response including the ML prediction
        return {
            "ok": True, 
            "id": str(insert_result.inserted_id),
            "risk_level": risk_level_str,
            "message": "Results and prediction saved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))