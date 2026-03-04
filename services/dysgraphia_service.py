# services/dysgraphia_service.py

from services.db_service import get_db
from datetime import datetime
from typing import Dict, Any
import math

def calculate_risk_score(submission_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate dysgraphia risk score based on:
    - Time Factor:          0-35 pts
    - Stroke Efficiency:    0-20 pts
    - Time Inconsistency:   0-20 pts
    - Clears Factor:        0-25 pts (+10 consistency bonus)
    - Formation Accuracy:   0-20 pts  ← NEW from ML Kit
    Total possible: 130 pts, capped at 100
    """
    grade = submission_data.get("grade", 3)
    activity_type = submission_data.get("activity_type", "letters")
    prompts_data = submission_data.get("prompts_data", [])
    formation_accuracy = submission_data.get("formation_accuracy", None)  # NEW

    if not prompts_data:
        return {"risk_level": "none", "risk_score": 0, "details": {}}

    time_benchmarks = {
        'letters':   {3: 3.0, 4: 2.5, 5: 2.0, 6: 1.8, 7: 1.5},
        'words':     {3: 8.0, 4: 6.5, 5: 5.5, 6: 4.5, 7: 4.0},
        'sentences': {3: 20.0, 4: 19.0, 5: 18.0, 6: 15.0, 7: 12.0},
    }
    benchmark_time = time_benchmarks.get(activity_type, {}).get(grade, 5.0)

    total_time = 0
    total_strokes = 0
    total_clears = 0
    time_deviations = []
    excessive_strokes_count = 0
    excessive_clears_count = 0

    expected_strokes = {
        'letters':   (1, 5),
        'words':     (3, 20),
        'sentences': (10, 60),
    }
    min_strokes, max_strokes = expected_strokes.get(activity_type, (1, 10))

    for prompt_data in prompts_data:
        time_taken = prompt_data.get("time_taken", 0)
        strokes = prompt_data.get("strokes", [])
        clears = prompt_data.get("clears", 0)
        stroke_count = len(strokes)

        total_time += time_taken
        total_strokes += stroke_count
        total_clears += clears

        if time_taken > 0:
            deviation = (time_taken - benchmark_time) / benchmark_time
            time_deviations.append(deviation)

        if stroke_count > max_strokes * 1.5:
            excessive_strokes_count += 1

        if clears >= 3:
            excessive_clears_count += 1

    num_prompts = len(prompts_data)
    avg_time = total_time / num_prompts if num_prompts > 0 else 0
    avg_strokes = total_strokes / num_prompts if num_prompts > 0 else 0
    avg_clears = total_clears / num_prompts if num_prompts > 0 else 0
    avg_time_deviation = sum(time_deviations) / len(time_deviations) if time_deviations else 0

    if len(time_deviations) > 1:
        variance = sum((d - avg_time_deviation) ** 2 for d in time_deviations) / len(time_deviations)
        time_inconsistency = math.sqrt(variance)
    else:
        time_inconsistency = 0

    risk_score = 0

    # 1. TIME FACTOR (0-35 pts)
    if avg_time_deviation > 1.5:   risk_score += 35
    elif avg_time_deviation > 1.0: risk_score += 27
    elif avg_time_deviation > 0.5: risk_score += 18
    elif avg_time_deviation > 0.2: risk_score += 9

    # 2. STROKE EFFICIENCY (0-20 pts)
    stroke_ratio = avg_strokes / max_strokes if max_strokes > 0 else 0
    if stroke_ratio > 1.5:   risk_score += 20
    elif stroke_ratio > 1.2: risk_score += 15
    elif stroke_ratio > 1.0: risk_score += 10
    elif stroke_ratio > 0.8: risk_score += 5

    # 3. TIME INCONSISTENCY (0-20 pts)
    if time_inconsistency > 1.0:   risk_score += 20
    elif time_inconsistency > 0.7: risk_score += 15
    elif time_inconsistency > 0.4: risk_score += 10
    elif time_inconsistency > 0.2: risk_score += 5

    # 4. CLEARS FACTOR (0-25 pts + 10 consistency bonus)
    if avg_clears >= 4:     risk_score += 25
    elif avg_clears >= 3:   risk_score += 20
    elif avg_clears >= 2:   risk_score += 15
    elif avg_clears >= 1:   risk_score += 10
    elif avg_clears >= 0.5: risk_score += 5

    clear_consistency_ratio = excessive_clears_count / num_prompts if num_prompts > 0 else 0
    if clear_consistency_ratio > 0.5:   risk_score += 10
    elif clear_consistency_ratio > 0.3: risk_score += 5

    # 5. FORMATION ACCURACY (0-20 pts) ← NEW
    # Only applied when ML Kit data is available (letters & words only)
    formation_penalty = 0
    if formation_accuracy is not None:
        # Low formation accuracy = strong dysgraphia indicator
        if formation_accuracy < 0.2:    formation_penalty = 20  # <20% correct = very high concern
        elif formation_accuracy < 0.4:  formation_penalty = 15  # <40% correct = high concern
        elif formation_accuracy < 0.6:  formation_penalty = 10  # <60% correct = moderate concern
        elif formation_accuracy < 0.8:  formation_penalty = 5   # <80% correct = slight concern
        risk_score += formation_penalty

    # Cap at 100
    risk_score = min(risk_score, 100)

    # Risk level classification
    if risk_score >= 70:   risk_level = "high"
    elif risk_score >= 45: risk_level = "medium"
    elif risk_score >= 20: risk_level = "low"
    else:                  risk_level = "none"

    details = {
        "avg_time_per_prompt":     round(avg_time, 2),
        "benchmark_time":          benchmark_time,
        "time_deviation_percent":  round(avg_time_deviation * 100, 1),
        "time_inconsistency":      round(time_inconsistency, 2),
        "avg_strokes":             round(avg_strokes, 1),
        "expected_max_strokes":    max_strokes,
        "excessive_strokes_count": excessive_strokes_count,
        "total_clears":            total_clears,
        "avg_clears_per_prompt":   round(avg_clears, 2),
        "excessive_clears_count":  excessive_clears_count,
        "clear_consistency_ratio": round(clear_consistency_ratio, 2),
        "total_prompts":           num_prompts,
        # NEW formation fields
        "formation_accuracy":      round(formation_accuracy, 2) if formation_accuracy is not None else None,
        "formation_penalty":       formation_penalty,
        "formation_source":        "ml_kit" if formation_accuracy is not None else "not_available",
    }

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 1),
        "details": details,
    }


def save_dysgraphia_submission(submission_data) -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_submissions"]

    try:
        doc = submission_data.dict()
        risk_assessment = calculate_risk_score(doc)

        doc["risk_level"] = risk_assessment["risk_level"]
        doc["risk_score"] = risk_assessment["risk_score"]
        doc["risk_details"] = risk_assessment["details"]
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()

        result = collection.insert_one(doc)

        return {
            "ok": True,
            "submission_id": str(result.inserted_id),
            "risk_level": risk_assessment["risk_level"],
            "risk_score": risk_assessment["risk_score"],
            "formation_accuracy": doc.get("formation_accuracy"),  # echo back to Flutter
            "message": "Submission saved successfully",
        }

    except Exception as e:
        return {"ok": False, "error": f"Failed to save submission: {str(e)}"}


def get_dysgraphia_stats() -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_submissions"]

    pipeline = [
        {"$group": {"_id": {"grade": "$grade", "activity_type": "$activity_type"}, "count": {"$sum": 1}}},
        {"$sort": {"_id.grade": 1}},
    ]
    risk_pipeline = [
        {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
    ]
    grade_risk_pipeline = [
        {"$group": {"_id": "$grade", "avg_risk_score": {"$avg": "$risk_score"}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]

    return {
        "ok": True,
        "stats": list(collection.aggregate(pipeline)),
        "risk_distribution": list(collection.aggregate(risk_pipeline)),
        "grade_risk_stats": list(collection.aggregate(grade_risk_pipeline)),
    }


def recalculate_all_risks() -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_submissions"]

    try:
        updated_count = 0
        for submission in collection.find({}):
            risk_assessment = calculate_risk_score(submission)
            collection.update_one(
                {"_id": submission["_id"]},
                {"$set": {
                    "risk_level": risk_assessment["risk_level"],
                    "risk_score": risk_assessment["risk_score"],
                    "risk_details": risk_assessment["details"],
                    "updated_at": datetime.utcnow(),
                }},
            )
            updated_count += 1

        return {"ok": True, "updated_count": updated_count,
                "message": f"Recalculated risk levels for {updated_count} submissions"}

    except Exception as e:
        return {"ok": False, "error": f"Failed to recalculate risks: {str(e)}"}