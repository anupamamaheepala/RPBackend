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
# This looks for the dyscalculia_rf_model.pkl file in the same directory as this script
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "dyscalculia_rf_model.pkl")

rf_model = None
try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"Dyscalculia RF Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load ML model at {MODEL_PATH}. Error: {e}")

@router.post("/dyscalculia/submit-result")
async def submit_dyscalculia_result(result: DyscalculiaResult):
    try:
        risk_level_str = "Pending/Error"
        
        # 1. Predict Risk Level using the ML Model
        if rf_model is not None:
            # Prepare features array (MUST match the exact order of your V3 CSV dataset)
            features = np.array([[
                result.grade,
                result.task_number,
                result.accuracy,
                result.response_time_avg,
                result.hesitation_time_avg,
                result.retries,
                result.backtracks,
                result.skipped_items,
                result.wrong_count,       # <-- New Feature!
                result.completion_time
            ]])
            
            # Predict
            prediction = rf_model.predict(features)[0]
            
            # Since your dataset labels were trained as strings ("No Dyscalculia", etc.)
            # The model will output exactly that string.
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

@router.get("/dyscalculia/results/{user_id}")
async def get_user_dyscalculia_results(user_id: str):
    try:
        # 1. Fetch all results for this user, sorted by newest first (-1)
        # This ensures the first time we see a (grade, task) it is the latest one.
        cursor = db["dyscalculia_results"].find({"user_id": user_id}).sort("created_at", -1)
        all_results = list(cursor)

        # 2. Filter to keep only the latest result per Grade & Task Number
        latest_results_map = {}
        for res in all_results:
            key = f"grade_{res['grade']}_task_{res['task_number']}"
            
            if key not in latest_results_map:
                # Convert ObjectId and datetime for JSON serialization
                res["_id"] = str(res["_id"])
                if "created_at" in res and res["created_at"]:
                    res["created_at"] = res["created_at"].isoformat()
                
                latest_results_map[key] = res

        # 3. Convert dictionary back to a list
        final_results = list(latest_results_map.values())
        
        # 4. Optional: Sort the final list sequentially by Grade, then Task Number
        final_results.sort(key=lambda x: (x["grade"], x["task_number"]))

        return {"ok": True, "results": final_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))