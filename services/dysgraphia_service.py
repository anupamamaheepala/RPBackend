# services/dysgraphia_service.py
from services.db_service import get_db
from bson.binary import Binary
from datetime import datetime
from typing import Dict, Any, List
import math

import pickle
import numpy as np

# Load model once at startup (module level)
_ml_model = None

def get_ml_model():
    global _ml_model
    if _ml_model is None:
        with open("models/dysgraphia_model.pkl", "rb") as f:
            _ml_model = pickle.load(f)
    return _ml_model

def predict_risk_ml(details: dict) -> dict:
    """
    Use XGBoost model to predict dysgraphia risk.
    Input: the 'details' dict from calculate_risk_score()
    Output: { risk_level, risk_score, confidence }
    """
    model = get_ml_model()

    # Build feature vector — ORDER MUST MATCH TRAINING
    features = np.array([[
        details["avg_time_per_prompt"],
        details["avg_strokes"],
        details["avg_clears_per_prompt"],
        details["time_inconsistency"],
        details["excessive_strokes_count"],
        details["excessive_clears_count"],
        details["time_deviation_percent"],
    ]])

    # Get prediction + probability
    predicted_class = model.predict(features)[0]          # e.g. "high"
    probabilities = model.predict_proba(features)[0]      # e.g. [0.1, 0.2, 0.6, 0.1]
    confidence = round(float(max(probabilities)) * 100, 1)

    # Map numeric class to label if model outputs integers
    class_map = {0: "none", 1: "low", 2: "medium", 3: "high"}
    if isinstance(predicted_class, (int, np.integer)):
        risk_level = class_map[predicted_class]
    else:
        risk_level = str(predicted_class)

    # Convert risk_level to a 0-100 score for consistency
    score_map = {"none": 10, "low": 30, "medium": 57, "high": 85}
    risk_score = score_map.get(risk_level, 0)

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence": confidence
    }



def save_dysgraphia_submission(submission_data) -> dict:
    db = get_db()
    collection = db["dysgraphia_submissions"]

    try:
        doc = submission_data.dict()

        # Step 1: Always calculate rule-based details (needed as ML features)
        rule_assessment = calculate_risk_score(doc)
        details = rule_assessment["details"]

        # Step 2: Try ML prediction, fall back to rule-based
        try:
            ml_assessment = predict_risk_ml(details)
            final_risk_level = ml_assessment["risk_level"]
            final_risk_score = ml_assessment["risk_score"]
            prediction_source = "ml"
            confidence = ml_assessment["confidence"]
        except Exception as ml_error:
            # Fallback: use rule-based if model fails
            final_risk_level = rule_assessment["risk_level"]
            final_risk_score = rule_assessment["risk_score"]
            prediction_source = "rules"
            confidence = None

        doc["risk_level"] = final_risk_level
        doc["risk_score"] = final_risk_score
        doc["risk_details"] = details
        doc["prediction_source"] = prediction_source  # Track which system decided
        doc["confidence"] = confidence
        doc["rule_based_score"] = rule_assessment["risk_score"]  # Keep for comparison
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()

        result = collection.insert_one(doc)

        return {
            "ok": True,
            "submission_id": str(result.inserted_id),
            "risk_level": final_risk_level,
            "risk_score": final_risk_score,
            "confidence": confidence,
            "prediction_source": prediction_source,
            "message": "Submission saved successfully"
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}



def calculate_risk_score(submission_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate dysgraphia risk score based on multiple factors including clears.
    Returns risk_level (none/low/medium/high) and detailed metrics.
    
    Scoring breakdown (0-100 scale):
    - Time Factor: 0-35 points (slower than grade benchmark)
    - Stroke Efficiency: 0-20 points (too many strokes = poor motor planning)
    - Time Inconsistency: 0-20 points (variance indicates attention/control issues)
    - Clears Factor: 0-25 points (CRITICAL - high clears = strong dysgraphia indicator)
    """
    grade = submission_data.get("grade", 3)
    activity_type = submission_data.get("activity_type", "letters")
    prompts_data = submission_data.get("prompts_data", [])
    
    if not prompts_data:
        return {"risk_level": "none", "risk_score": 0, "details": {}}
    
    # Grade-specific time benchmarks (in seconds)
    # Based on typical development milestones for Sinhala handwriting
    time_benchmarks = {
        'letters': {3: 3.0, 4: 2.5, 5: 2.0, 6: 1.8, 7: 1.5},
        'words': {3: 8.0, 4: 6.5, 5: 5.5, 6: 4.5, 7: 4.0},
        'sentences': {3: 20.0, 4: 19.0, 5: 18.0, 6: 15.0, 7: 12.0}  # UPDATED
    }
    
    # Get benchmark for this grade and activity
    benchmark_time = time_benchmarks.get(activity_type, {}).get(grade, 5.0)
    
    # Initialize metrics
    total_time = 0
    total_strokes = 0
    total_clears = 0
    time_deviations = []
    excessive_strokes_count = 0
    excessive_clears_count = 0
    
    # Expected stroke ranges by activity type
    expected_strokes = {
        'letters': (1, 5),    # Simple letters: 1-5 strokes
        'words': (3, 20),     # Words: 3-20 strokes
        'sentences': (10, 60) # Sentences: 10-60 strokes
    }
    min_strokes, max_strokes = expected_strokes.get(activity_type, (1, 10))
    
    # Analyze each prompt
    for prompt_data in prompts_data:
        time_taken = prompt_data.get("time_taken", 0)
        strokes = prompt_data.get("strokes", [])
        clears = prompt_data.get("clears", 0)  # Get clears for this prompt
        stroke_count = len(strokes)
        
        total_time += time_taken
        total_strokes += stroke_count
        total_clears += clears
        
        # Calculate time deviation from benchmark
        if time_taken > 0:
            deviation = (time_taken - benchmark_time) / benchmark_time
            time_deviations.append(deviation)
        
        # Check for excessive strokes (indicates poor motor planning)
        if stroke_count > max_strokes * 1.5:
            excessive_strokes_count += 1
        
        # Check for excessive clears (indicates difficulty/lack of confidence)
        # 3+ clears on a single prompt is considered excessive
        if clears >= 3:
            excessive_clears_count += 1
    
    num_prompts = len(prompts_data)
    avg_time = total_time / num_prompts if num_prompts > 0 else 0
    avg_strokes = total_strokes / num_prompts if num_prompts > 0 else 0
    avg_clears = total_clears / num_prompts if num_prompts > 0 else 0
    avg_time_deviation = sum(time_deviations) / len(time_deviations) if time_deviations else 0
    
    # Calculate time variance (inconsistency indicator)
    if len(time_deviations) > 1:
        variance = sum((d - avg_time_deviation) ** 2 for d in time_deviations) / len(time_deviations)
        time_inconsistency = math.sqrt(variance)
    else:
        time_inconsistency = 0
    
    # ========================================================================
    # SCORING SYSTEM (0-100 scale)
    # ========================================================================
    risk_score = 0
    
    # 1. TIME FACTOR (0-35 points)
    # Significantly slower than benchmark indicates processing/motor difficulty
    if avg_time_deviation > 1.5:  # 150% slower than benchmark
        risk_score += 35
    elif avg_time_deviation > 1.0:  # 100% slower (2x time)
        risk_score += 27
    elif avg_time_deviation > 0.5:  # 50% slower
        risk_score += 18
    elif avg_time_deviation > 0.2:  # 20% slower
        risk_score += 9
    
    # 2. STROKE EFFICIENCY (0-20 points)
    # Too many strokes suggests poor motor planning and spatial awareness
    stroke_ratio = avg_strokes / max_strokes if max_strokes > 0 else 0
    if stroke_ratio > 1.5:  # 50% more strokes than expected
        risk_score += 20
    elif stroke_ratio > 1.2:  # 20% more strokes
        risk_score += 15
    elif stroke_ratio > 1.0:  # Slightly more than expected
        risk_score += 10
    elif stroke_ratio > 0.8:  # Near expected range
        risk_score += 5
    
    # 3. TIME INCONSISTENCY (0-20 points)
    # High variance indicates attention issues or inconsistent motor control
    if time_inconsistency > 1.0:
        risk_score += 20
    elif time_inconsistency > 0.7:
        risk_score += 15
    elif time_inconsistency > 0.4:
        risk_score += 10
    elif time_inconsistency > 0.2:
        risk_score += 5
    
    # 4. CLEARS/ERASES FACTOR (0-25 points) - CRITICAL INDICATOR
    # High number of clears is one of the STRONGEST indicators of dysgraphia
    # Shows: difficulty with motor planning, lack of confidence, poor spatial awareness,
    # difficulty forming letters correctly, perfectionism due to motor difficulties
    if avg_clears >= 4:
        risk_score += 25  # Very high concern - persistent difficulty
    elif avg_clears >= 3:
        risk_score += 20  # High concern - frequent restarts
    elif avg_clears >= 2:
        risk_score += 15  # Moderate concern - regular difficulty
    elif avg_clears >= 1:
        risk_score += 10  # Some concern - occasional restarts
    elif avg_clears >= 0.5:
        risk_score += 5   # Slight concern - infrequent issues
    
    # Additional penalty for consistency in clearing behavior
    # If many prompts had excessive clears (3+), it shows persistent pattern
    clear_consistency_ratio = excessive_clears_count / num_prompts if num_prompts > 0 else 0
    if clear_consistency_ratio > 0.5:  # More than half of prompts had 3+ clears
        risk_score += 10  # Pattern of persistent difficulty
    elif clear_consistency_ratio > 0.3:  # 30-50% had excessive clears
        risk_score += 5   # Frequent difficulty pattern
    
    # ========================================================================
    # RISK LEVEL CLASSIFICATION
    # ========================================================================
    # Based on total score (0-100)
    if risk_score >= 70:
        risk_level = "high"      # Immediate attention needed
    elif risk_score >= 45:
        risk_level = "medium"    # Intervention recommended
    elif risk_score >= 20:
        risk_level = "low"       # Minor concerns, monitor progress
    else:
        risk_level = "none"      # Normal development
    
    # ========================================================================
    # DETAILED BREAKDOWN FOR ANALYSIS
    # ========================================================================
    details = {
        # Time metrics
        "avg_time_per_prompt": round(avg_time, 2),
        "benchmark_time": benchmark_time,
        "time_deviation_percent": round(avg_time_deviation * 100, 1),
        "time_inconsistency": round(time_inconsistency, 2),
        
        # Stroke metrics
        "avg_strokes": round(avg_strokes, 1),
        "expected_max_strokes": max_strokes,
        "excessive_strokes_count": excessive_strokes_count,
        
        # Clears metrics (NEW - CRITICAL DATA)
        "total_clears": total_clears,
        "avg_clears_per_prompt": round(avg_clears, 2),
        "excessive_clears_count": excessive_clears_count,
        "clear_consistency_ratio": round(clear_consistency_ratio, 2),
        
        # General metrics
        "total_prompts": num_prompts
    }
    
    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 1),
        "details": details
    }


def save_dysgraphia_submission(submission_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save dysgraphia submission to MongoDB with automatic risk assessment.
    
    Args:
        submission_data: Dict from Pydantic model (validated).
    
    Returns:
        Dict with 'ok' status, 'submission_id', 'risk_level', and 'risk_score'.
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    
    try:
        # Convert to dict (Pydantic model to dict)
        doc = submission_data.dict()
        
        # Calculate risk assessment using our algorithm
        risk_assessment = calculate_risk_score(doc)
        
        # Add risk data to document
        doc["risk_level"] = risk_assessment["risk_level"]
        doc["risk_score"] = risk_assessment["risk_score"]
        doc["risk_details"] = risk_assessment["details"]
        
        # Add metadata timestamps
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        
        # Insert into MongoDB and get ID
        result = collection.insert_one(doc)
        submission_id = str(result.inserted_id)
        
        return {
            "ok": True,
            "submission_id": submission_id,
            "risk_level": risk_assessment["risk_level"],
            "risk_score": risk_assessment["risk_score"],
            "message": "Submission saved successfully with risk assessment"
        }
    
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to save submission: {str(e)}"
        }


def get_dysgraphia_stats() -> Dict[str, Any]:
    """
    Get comprehensive statistics including risk level distributions.
    
    Returns:
        - Submissions count by grade and activity type
        - Risk level distribution (none/low/medium/high counts)
        - Average risk scores by grade
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    
    # Stats by grade and activity
    pipeline = [
        {"$group": {
            "_id": {"grade": "$grade", "activity_type": "$activity_type"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id.grade": 1}}
    ]
    
    # Risk level distribution
    risk_pipeline = [
        {"$group": {
            "_id": "$risk_level",
            "count": {"$sum": 1}
        }}
    ]
    
    # Average risk scores by grade
    grade_risk_pipeline = [
        {"$group": {
            "_id": "$grade",
            "avg_risk_score": {"$avg": "$risk_score"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    
    stats = list(collection.aggregate(pipeline))
    risk_distribution = list(collection.aggregate(risk_pipeline))
    grade_risk_stats = list(collection.aggregate(grade_risk_pipeline))
    
    return {
        "ok": True,
        "stats": stats,
        "risk_distribution": risk_distribution,
        "grade_risk_stats": grade_risk_stats
    }


def recalculate_all_risks() -> Dict[str, Any]:
    """
    Utility function to recalculate risk levels for all existing submissions.
    Useful when you update the risk calculation algorithm.
    
    WARNING: This updates all records in the database.
    
    Returns:
        Dict with 'ok' status and 'updated_count'.
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    
    try:
        submissions = collection.find({})
        updated_count = 0
        
        for submission in submissions:
            # Calculate new risk assessment with updated algorithm
            risk_assessment = calculate_risk_score(submission)
            
            # Update document in database
            collection.update_one(
                {"_id": submission["_id"]},
                {
                    "$set": {
                        "risk_level": risk_assessment["risk_level"],
                        "risk_score": risk_assessment["risk_score"],
                        "risk_details": risk_assessment["details"],
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            updated_count += 1
        
        return {
            "ok": True,
            "updated_count": updated_count,
            "message": f"Recalculated risk levels for {updated_count} submissions"
        }
    
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to recalculate risks: {str(e)}"
        }