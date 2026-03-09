import os
import joblib
import numpy as np
import random
import __main__  # Required to map the class for joblib
from fastapi import APIRouter, HTTPException
from services.db_service import get_db
from models.dyscalculia_models import DyscalculiaResult, LearningMetrics
from datetime import datetime

router = APIRouter()
db = get_db()

# ==========================================
# 1. DEFINE RULE ENGINE CLASS FOR JOBLIB
# ==========================================
# We must define the class here so the .pkl file knows how to reconstruct itself 
# when the FastAPI server starts.
class AdaptiveLearningPathEngine:
    def __init__(self, math_questions_data):
        self.question_bank = math_questions_data
        self.levels = ["easy", "medium", "hard"]
        
    def evaluate_performance(self, current_level, metrics):
        try:
            current_idx = self.levels.index(current_level.lower())
        except ValueError:
            current_idx = 0 
            
        is_mastery = (
            metrics['accuracy'] >= 4 and
            metrics['retries'] <= 2 and
            metrics['backtracks'] <= 1 and
            metrics['hesitation_time_avg'] < 8.0 and
            metrics['skipped_items'] == 0
        )
        
        is_struggling = (
            metrics['accuracy'] <= 2 or
            metrics['skipped_items'] >= 2 or
            metrics['retries'] >= 6 or
            metrics['hesitation_time_avg'] > 18.0 or
            metrics['wrong_count'] >= 5
        )
        
        if is_mastery:
            new_idx = min(len(self.levels) - 1, current_idx + 1)
            action = "Promote"
            message = "නියමයි! ඔයා ගොඩක් දක්ෂයි! (Excellent! You are doing great!)"
        elif is_struggling:
            new_idx = max(0, current_idx - 1)
            action = "Regress"
            message = "අපි තවත් ටිකක් පුහුණු වෙමු! (Let's practice a bit more!)"
        else:
            new_idx = current_idx
            action = "Stay"
            message = "හොඳ උත්සාහයක්! මේ වගේ තවත් ගැටලු විසඳමු. (Good try! Let's solve more like this.)"
            
        next_level = self.levels[new_idx]
        
        return {
            "action": action,
            "next_level": next_level,
            "message": message
        }
        
    def get_questions_for_level(self, level, count=5):
        level_key = level.lower()
        all_grade_3 = self.question_bank.get("math_tasks_grade_03", {})
        questions_pool = all_grade_3.get(level_key, [])
        if len(questions_pool) < count:
            return questions_pool
        return random.sample(questions_pool, count)

# Map the class to __main__ so joblib finds it exactly where it expects to
__main__.AdaptiveLearningPathEngine = AdaptiveLearningPathEngine

# ==========================================
# 2. LOAD AI & RULE MODELS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- LOAD THE ML MODEL (DETECTION) ---
MODEL_PATH = os.path.join(current_dir, "dyscalculia_rf_model.pkl")
rf_model = None
try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"Dyscalculia RF Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load ML model at {MODEL_PATH}. Error: {e}")

# --- LOAD THE RULE ENGINE MODEL (LEARNING PATH) ---
RULE_ENGINE_PATH = os.path.join(current_dir, "learning_path_rule_engine.pkl")
rule_engine = None
try:
    rule_engine = joblib.load(RULE_ENGINE_PATH)
    print(f"Rule Engine loaded successfully from: {RULE_ENGINE_PATH}")
except Exception as e:
    print(f"Warning: Could not load Rule Engine at {RULE_ENGINE_PATH}. Error: {e}")


# ==========================================
# 3. DETECTION ROUTES
# ==========================================
@router.post("/dyscalculia/submit-result")
async def submit_dyscalculia_result(result: DyscalculiaResult):
    try:
        risk_level_str = "Pending/Error"
        
        # Predict Risk Level using the ML Model
        if rf_model is not None:
            features = np.array([[
                result.grade,
                result.task_number,
                result.accuracy,
                result.response_time_avg,
                result.hesitation_time_avg,
                result.retries,
                result.backtracks,
                result.skipped_items,
                result.wrong_count,
                result.completion_time
            ]])
            
            prediction = rf_model.predict(features)[0]
            risk_level_str = str(prediction)
        else:
            print("Model not loaded. Skipping ML prediction.")

        # Prepare and insert data
        result_dict = result.dict()
        result_dict["risk_level"] = risk_level_str
        result_dict["created_at"] = datetime.utcnow()
        
        insert_result = db["dyscalculia_results"].insert_one(result_dict)
        
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
        cursor = db["dyscalculia_results"].find({"user_id": user_id}).sort("created_at", -1)
        all_results = list(cursor)

        latest_results_map = {}
        for res in all_results:
            key = f"grade_{res['grade']}_task_{res['task_number']}"
            
            if key not in latest_results_map:
                res["_id"] = str(res["_id"])
                if "created_at" in res and res["created_at"]:
                    res["created_at"] = res["created_at"].isoformat()
                
                latest_results_map[key] = res

        final_results = list(latest_results_map.values())
        final_results.sort(key=lambda x: (x["grade"], x["task_number"]))

        return {"ok": True, "results": final_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 4. LEARNING PATH ROUTES
# ==========================================
@router.get("/dyscalculia/learning-state/{user_id}")
async def get_learning_state(user_id: str):
    """Fetches the student's current difficulty level and task count."""
    try:
        state = db["dyscalculia_learning_state"].find_one({"user_id": user_id})
        
        if not state:
            # First time playing: Check ML detection history to set start level
            detection = db["dyscalculia_results"].find_one({"user_id": user_id}, sort=[("created_at", -1)])
            start_level = "easy"
            
            if detection:
                if detection["risk_level"] == "No Dyscalculia":
                    start_level = "hard"
                elif detection["risk_level"] == "Mild Dyscalculia":
                    start_level = "medium"
                    
            state = {
                "user_id": user_id, 
                "current_level": start_level, 
                "tasks_completed": 0
            }
            db["dyscalculia_learning_state"].insert_one(state)
            
        return {"level": state["current_level"], "tasks_completed": state.get("tasks_completed", 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dyscalculia/learning-questions/{grade}/{level}")
async def get_learning_questions(grade: int, level: str):
    """Fetches exactly 5 random questions for the specified grade and level."""
    try:
        # Ensure collection name dynamically matches the grade (e.g., grade_03_maths)
        collection_name = f"grade_{grade:02d}_maths"
        
        # MongoDB $sample aggregation for fast random fetching
        pipeline = [
            {"$match": {"difficulty": level.lower()}},
            {"$sample": {"size": 5}}
        ]
        questions = list(db[collection_name].aggregate(pipeline))
        
        for q in questions:
            q["_id"] = str(q["_id"])
            
        return {"ok": True, "questions": questions}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))


@router.post("/dyscalculia/submit-learning-task")
async def submit_learning_task(metrics: LearningMetrics):
    """Evaluates a 5-question micro-mission and updates the student's learning path."""
    try:
        if rule_engine is None:
            raise HTTPException(status_code=500, detail="Rule Engine Model is not loaded.")
            
        state = db["dyscalculia_learning_state"].find_one({"user_id": metrics.user_id})
        current_level = state["current_level"] if state else "easy"
        
        # 1. Evaluate using Rule Engine
        metrics_dict = metrics.dict()
        evaluation = rule_engine.evaluate_performance(current_level, metrics_dict)
        
        action = evaluation["action"]       
        next_level = evaluation["next_level"] 
        message = evaluation["message"]
        
        # 2. Update the student's overall state
        new_tasks_completed = state.get("tasks_completed", 0) + 1
        
        db["dyscalculia_learning_state"].update_one(
            {"user_id": metrics.user_id},
            {"$set": {
                "current_level": next_level, 
                "tasks_completed": new_tasks_completed
            }}
        )
        
        # 3. Save to learning history
        history_record = metrics_dict.copy()
        history_record["evaluated_action"] = action
        history_record["level_played"] = current_level
        history_record["next_level"] = next_level
        history_record["created_at"] = datetime.utcnow()
        
        db["dyscalculia_learning_history"].insert_one(history_record)
        
        return {
            "ok": True,
            "action": action,
            "next_level": next_level,
            "message": message,
            "tasks_completed": new_tasks_completed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))