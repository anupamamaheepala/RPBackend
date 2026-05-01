import os
import joblib
import numpy as np
import random
import __main__ 
from fastapi import APIRouter, HTTPException
from services.db_service import get_db
from models.dyscalculia_models import DyscalculiaResult, LearningMetrics
from datetime import datetime

router = APIRouter()
db = get_db()

# ==========================================
# 1. DEFINE RULE ENGINE CLASS FOR JOBLIB
# ==========================================
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

__main__.AdaptiveLearningPathEngine = AdaptiveLearningPathEngine

# ==========================================
# 2. LOAD AI & RULE MODELS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(current_dir, "dyscalculia_rf_model.pkl")
rf_model = None
try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"Dyscalculia RF Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load ML model at {MODEL_PATH}. Error: {e}")

RULE_ENGINE_G03_PATH = os.path.join(current_dir, "learning_path_rule_engine.pkl")
rule_engine_g03 = None
try:
    rule_engine_g03 = joblib.load(RULE_ENGINE_G03_PATH)
    print(f"Grade 3 Rule Engine loaded successfully")
except Exception as e:
    print(f"Warning: Could not load G03 Rule Engine. Error: {e}")

RULE_ENGINE_G04_PATH = os.path.join(current_dir, "learning_path_rule_engine_g04.pkl")
rule_engine_g04 = None
try:
    rule_engine_g04 = joblib.load(RULE_ENGINE_G04_PATH)
    print(f"Grade 4 Rule Engine loaded successfully")
except Exception as e:
    print(f"Warning: Could not load G04 Rule Engine. Error: {e}")


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def clean_risk_level(raw_prediction) -> str:
    """
    Clean the RF model prediction output to a standardized risk level string.
    Handles numpy arrays, lists, and various string formats.
    """
    if raw_prediction is None:
        return "Unknown"
    
    # Convert to string and clean
    risk_str = str(raw_prediction)
    
    # Remove common wrappers
    risk_str = risk_str.strip()
    risk_str = risk_str.replace("[", "").replace("]", "")
    risk_str = risk_str.replace("'", "").replace('"', "")
    risk_str = risk_str.strip()
    
    # Also handle numpy array specific formatting
    risk_str = risk_str.replace("\n", "").replace("\r", "")
    
    # Standardize the risk level
    risk_str_lower = risk_str.lower()
    if "no dyscalculia" in risk_str_lower or "no" == risk_str_lower.strip():
        return "No Dyscalculia"
    elif "severe" in risk_str_lower:
        return "Severe Dyscalculia"
    elif "mild" in risk_str_lower:
        return "Mild Dyscalculia"
    else:
        # If we can't determine, return the cleaned string
        print(f"WARNING: Unknown risk level raw output: {raw_prediction} -> cleaned: {risk_str}")
        return risk_str if risk_str else "Unknown"


def determine_start_level(risk_level_str: str) -> str:
    """
    Determine the starting level for learning path based on detection result.
    LOGIC:
    - Severe Dyscalculia → easy
    - Mild Dyscalculia → medium
    - No Dyscalculia → hard (still need to learn/practice)
    - Unknown → easy (safest default)
    """
    risk_lower = risk_level_str.lower()
    
    if "severe" in risk_lower:
        return "easy"
    elif "mild" in risk_lower:
        return "medium"
    elif "no dyscalculia" in risk_lower or "no" == risk_lower.strip():
        return "hard"
    else:
        # Default to easy for safety
        print(f"WARNING: Unknown risk level '{risk_level_str}', defaulting to 'easy'")
        return "easy"


# ==========================================
# 4. DETECTION ROUTES
# ==========================================
@router.post("/dyscalculia/submit-result")
async def submit_dyscalculia_result(result: DyscalculiaResult):
    try:
        risk_level_str = "Unknown"
        
        if rf_model is not None:
            features = np.array([[
                result.grade, result.task_number, result.accuracy,
                result.response_time_avg, result.hesitation_time_avg,
                result.retries, result.backtracks, result.skipped_items,
                result.wrong_count, result.completion_time
            ]])
            raw_prediction = rf_model.predict(features)[0]
            risk_level_str = clean_risk_level(raw_prediction)
            print(f"Detection submission - Grade {result.grade} Task {result.task_number}: {risk_level_str}")
        else:
            print("WARNING: RF Model not loaded, using default risk level")

        result_dict = result.dict()
        result_dict["risk_level"] = risk_level_str
        result_dict["created_at"] = datetime.utcnow()
        
        insert_result = db["dyscalculia_results"].insert_one(result_dict)
        
        # IMPORTANT: After new detection, DELETE old learning state
        # This forces re-initialization based on the new detection result
        delete_result = db["dyscalculia_learning_state"].delete_one(
            {"user_id": result.user_id, "grade": result.grade}
        )
        print(f"Deleted learning state for user {result.user_id} grade {result.grade}: deleted={delete_result.deleted_count}")
        
        return {"ok": True, "id": str(insert_result.inserted_id), "risk_level": risk_level_str}
    except Exception as e:
        print(f"ERROR in submit-result: {e}")
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
# 5. LEARNING PATH ROUTES
# ==========================================
@router.get("/dyscalculia/learning-state/{user_id}/{grade}")
async def get_learning_state(user_id: str, grade: int):
    """
    Get or create the learning state for a user+grade.
    
    IMPORTANT LOGIC:
    1. Student MUST complete detection for this grade first (gatekeeper)
    2. Detection result determines starting level:
       - Severe Dyscalculia → easy
       - Mild Dyscalculia → medium
       - No Dyscalculia → hard (student should still practice)
    3. If learning state already exists, return it (preserves ongoing progress)
    """
    try:
        # Gatekeeper: student must have at least one detection record for this grade
        detection = db["dyscalculia_results"].find_one(
            {"user_id": user_id, "grade": grade}, 
            sort=[("created_at", -1)]
        )
        
        if not detection:
            print(f"Access denied: User {user_id} has no detection result for grade {grade}")
            return {"ok": False, "message": f"Must complete Grade {grade} detection first."}

        # Get the risk level from the latest detection
        risk_level_str = detection.get("risk_level", "Unknown")
        print(f"Learning state request - User: {user_id}, Grade: {grade}, Detection Risk Level: '{risk_level_str}'")
        
        # Determine the correct starting level based on detection
        correct_start_level = determine_start_level(risk_level_str)
        print(f"Determined start level: {correct_start_level}")

        # Check if learning state already exists
        state = db["dyscalculia_learning_state"].find_one({"user_id": user_id, "grade": grade})
        
        if not state:
            # No state exists — create a fresh one based on detection result
            state = {
                "user_id": user_id,
                "grade": grade,
                "current_level": correct_start_level,
                "tasks_completed": 0
            }
            db["dyscalculia_learning_state"].insert_one(state)
            print(f"Created new learning state: level={correct_start_level}, tasks_completed=0")
        else:
            print(f"Found existing learning state: level={state['current_level']}, tasks_completed={state.get('tasks_completed', 0)}")
            
        return {
            "ok": True,
            "level": state["current_level"],
            "tasks_completed": state.get("tasks_completed", 0)
        }
    except Exception as e:
        print(f"ERROR in learning-state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dyscalculia/learning-questions/{grade}/{level}")
async def get_learning_questions(grade: int, level: str):
    try:
        collection_name = f"math_questions_g{grade:02d}"
        grade_key = f"math_tasks_grade_{grade:02d}"
        level_key = level.lower()
        
        doc = db[collection_name].find_one({})
        if not doc or grade_key not in doc or level_key not in doc[grade_key]:
            return {"ok": False, "questions": []}
            
        questions_pool = doc[grade_key][level_key]
        if len(questions_pool) >= 5:
            selected_questions = random.sample(questions_pool, 5)
        else:
            selected_questions = questions_pool
            
        return {"ok": True, "questions": selected_questions}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))


@router.post("/dyscalculia/submit-learning-task")
async def submit_learning_task(metrics: LearningMetrics):
    try:
        if metrics.grade == 3:
            active_rule_engine = rule_engine_g03
        elif metrics.grade == 4:
            active_rule_engine = rule_engine_g04
        else:
            raise HTTPException(status_code=400, detail=f"No learning path rule engine available for grade {metrics.grade}")

        if active_rule_engine is None: 
            raise HTTPException(status_code=500, detail=f"Rule Engine Model for grade {metrics.grade} not loaded.")
            
        state = db["dyscalculia_learning_state"].find_one({"user_id": metrics.user_id, "grade": metrics.grade})
        current_level = state["current_level"] if state else "easy"
        
        print(f"Learning task submission - User: {metrics.user_id}, Grade: {metrics.grade}, Current Level: {current_level}")
        print(f"Metrics: accuracy={metrics.accuracy}, retries={metrics.retries}, wrongs={metrics.wrong_count}, hesitation={metrics.hesitation_time_avg:.2f}s")
        
        metrics_dict = metrics.dict()
        
        evaluation = active_rule_engine.evaluate_performance(current_level, metrics_dict)
        
        action = evaluation["action"]       
        next_level = evaluation["next_level"] 
        message = evaluation["message"]
        new_tasks_completed = state.get("tasks_completed", 0) + 1
        
        print(f"Evaluation: action={action}, next_level={next_level}, tasks_completed={new_tasks_completed}")
        
        db["dyscalculia_learning_state"].update_one(
            {"user_id": metrics.user_id, "grade": metrics.grade},
            {"$set": {"current_level": next_level, "tasks_completed": new_tasks_completed}},
            upsert=True
        )
        
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
        print(f"ERROR in submit-learning-task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 6. SPECIAL TASK & RESULTS ROUTES
# ==========================================
@router.post("/dyscalculia/submit-special-task")
async def submit_special_task(result: DyscalculiaResult):
    try:
        risk_level_str = "Unknown"
        
        if rf_model is not None:
            features = np.array([[
                result.grade, result.task_number, result.accuracy,
                result.response_time_avg, result.hesitation_time_avg,
                result.retries, result.backtracks, result.skipped_items,
                result.wrong_count, result.completion_time
            ]])
            raw_prediction = rf_model.predict(features)[0]
            risk_level_str = clean_risk_level(raw_prediction)
            print(f"Special task submission - Grade {result.grade}: {risk_level_str}")
        else:
            print("WARNING: RF Model not loaded for special task")

        # Save Special Result
        result_dict = result.dict()
        result_dict["risk_level"] = risk_level_str
        result_dict["created_at"] = datetime.utcnow()
        db["dyscalculia_special_results"].insert_one(result_dict)
        
        # Determine next starting level based on special task result
        start_level = determine_start_level(risk_level_str)
        
        # Reset tasks_completed to 0 so the special task dialog won't re-trigger
        # immediately on the next _initLearningPath() call
        db["dyscalculia_learning_state"].update_one(
            {"user_id": result.user_id, "grade": result.grade},
            {"$set": {
                "current_level": start_level, 
                "tasks_completed": 0
            }},
            upsert=True
        )
        
        print(f"Special task complete - New start level: {start_level}, tasks reset to 0")
        
        return {"ok": True, "risk_level": risk_level_str, "next_level": start_level}
    except Exception as e:
        print(f"ERROR in submit-special-task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dyscalculia/learning-history/{user_id}")
async def get_learning_history(user_id: str):
    try:
        special_result = db["dyscalculia_special_results"].find_one(
            {"user_id": user_id}, sort=[("created_at", -1)]
        )
        if special_result:
            special_result["_id"] = str(special_result["_id"])
            if "created_at" in special_result and special_result["created_at"]:
                special_result["created_at"] = special_result["created_at"].isoformat()
                
        cursor = db["dyscalculia_learning_history"].find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(5)
        
        history_list = list(cursor)
        for h in history_list:
            h["_id"] = str(h["_id"])
            if "created_at" in h and h["created_at"]:
                h["created_at"] = h["created_at"].isoformat()
                
        return {
            "ok": True,
            "special_result": special_result,
            "history": history_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))