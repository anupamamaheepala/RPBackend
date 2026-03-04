# services/dysgraphia_service.py
from services.db_service import get_db
from datetime import datetime
from typing import Dict, Any
import math

def calculate_risk_score(submission_data: Dict[str, Any]) -> Dict[str, Any]:
    grade              = submission_data.get("grade", 3)
    activity_type      = submission_data.get("activity_type", "letters")
    prompts_data       = submission_data.get("prompts_data", [])
    formation_accuracy = submission_data.get("formation_accuracy", None)

    if not prompts_data:
        return {"risk_level": "none", "risk_score": 0, "details": {}}

    # Grade-specific time benchmarks (seconds)
    time_benchmarks = {
        'letters':   {3: 3.0, 4: 2.5, 5: 2.0, 6: 1.8, 7: 1.5},
        'words':     {3: 8.0, 4: 6.5, 5: 5.5, 6: 4.5, 7: 4.0},
        'sentences': {3: 20.0, 4: 19.0, 5: 18.0, 6: 15.0, 7: 12.0},
    }
    benchmark_time = time_benchmarks.get(activity_type, {}).get(grade, 5.0)

    expected_strokes = {
        'letters':   (1, 5),
        'words':     (3, 20),
        'sentences': (10, 60),
    }
    _, max_strokes = expected_strokes.get(activity_type, (1, 10))

    # Collect metrics from each prompt
    total_time              = 0
    total_strokes           = 0
    total_clears            = 0
    time_deviations         = []
    excessive_strokes_count = 0
    excessive_clears_count  = 0

    for prompt_data in prompts_data:
        time_taken   = prompt_data.get("time_taken", 0)
        strokes      = prompt_data.get("strokes", [])
        clears       = prompt_data.get("clears", 0)
        stroke_count = len(strokes)

        total_time    += time_taken
        total_strokes += stroke_count
        total_clears  += clears

        if time_taken > 0:
            deviation = (time_taken - benchmark_time) / benchmark_time
            time_deviations.append(deviation)

        if stroke_count > max_strokes * 1.5:
            excessive_strokes_count += 1
        if clears >= 3:
            excessive_clears_count += 1

    num_prompts        = len(prompts_data)
    avg_time           = total_time    / num_prompts if num_prompts > 0 else 0
    avg_strokes        = total_strokes / num_prompts if num_prompts > 0 else 0
    avg_clears         = total_clears  / num_prompts if num_prompts > 0 else 0
    avg_time_dev       = sum(time_deviations) / len(time_deviations) if time_deviations else 0

    if len(time_deviations) > 1:
        variance           = sum((d - avg_time_dev) ** 2 for d in time_deviations) / len(time_deviations)
        time_inconsistency = math.sqrt(variance)
    else:
        time_inconsistency = 0

    clear_consistency_ratio = excessive_clears_count / num_prompts if num_prompts > 0 else 0

    risk_score = 0

    # 1. FORMATION ACCURACY (0-45 pts) — PRIMARY SIGNAL
    # This is now the most important factor. A child writing unrecognisable
    # letters is the clearest indicator of dysgraphia regardless of speed.
    formation_pts = 0
    if formation_accuracy is not None:
        if formation_accuracy < 0.20:   formation_pts = 45  # <20% correct — very severe
        elif formation_accuracy < 0.40: formation_pts = 35  # <40% correct — severe
        elif formation_accuracy < 0.60: formation_pts = 25  # <60% correct — moderate-high
        elif formation_accuracy < 0.80: formation_pts = 12  # <80% correct — mild
        elif formation_accuracy < 0.90: formation_pts = 5   # <90% correct — slight
    risk_score += formation_pts

    # 2. CLEARS / ERASES (0-25 pts + 10 consistency bonus)
    clears_pts = 0
    if avg_clears >= 4:     clears_pts = 25
    elif avg_clears >= 3:   clears_pts = 20
    elif avg_clears >= 2:   clears_pts = 15
    elif avg_clears >= 1:   clears_pts = 10
    elif avg_clears >= 0.5: clears_pts = 5
    risk_score += clears_pts

    consistency_pts = 0
    if clear_consistency_ratio > 0.5:   consistency_pts = 10
    elif clear_consistency_ratio > 0.3: consistency_pts = 5
    risk_score += consistency_pts

    # 3. TIME FACTOR (0-20 pts) — reduced from 35, supporting signal only
    time_pts = 0
    if avg_time_dev > 1.5:   time_pts = 20
    elif avg_time_dev > 1.0: time_pts = 15
    elif avg_time_dev > 0.5: time_pts = 10
    elif avg_time_dev > 0.2: time_pts = 5
    risk_score += time_pts

    # 4. STROKE EFFICIENCY (0-15 pts) — reduced from 20
    stroke_ratio = avg_strokes / max_strokes if max_strokes > 0 else 0
    stroke_pts = 0
    if stroke_ratio > 1.5:   stroke_pts = 15
    elif stroke_ratio > 1.2: stroke_pts = 10
    elif stroke_ratio > 1.0: stroke_pts = 7
    elif stroke_ratio > 0.8: stroke_pts = 3
    risk_score += stroke_pts

    # 5. TIME INCONSISTENCY (0-10 pts) — reduced from 20
    inconsistency_pts = 0
    if time_inconsistency > 1.0:   inconsistency_pts = 10
    elif time_inconsistency > 0.7: inconsistency_pts = 7
    elif time_inconsistency > 0.4: inconsistency_pts = 4
    elif time_inconsistency > 0.2: inconsistency_pts = 2
    risk_score += inconsistency_pts

    # Cap at 100 before applying override
    risk_score = min(risk_score, 100)

    # FORMATION OVERRIDE
    # When letter shapes are clearly wrong, force a minimum score.
    # This is the key fix: a fast confident child writing wrong letters
    # must still score high — speed and low erases don't cancel bad formation.
    formation_override_applied = False
    if formation_accuracy is not None:
        if formation_accuracy < 0.20 and risk_score < 70:
            risk_score = 70   # Force HIGH — letters totally unrecognisable
            formation_override_applied = True
        elif formation_accuracy < 0.40 and risk_score < 50:
            risk_score = 50   # Force MEDIUM — majority of letters wrong
            formation_override_applied = True
        elif formation_accuracy < 0.60 and risk_score < 35:
            risk_score = 35   # Lift to LOW-MEDIUM boundary
            formation_override_applied = True

    # Risk level classification
    if risk_score >= 70:   risk_level = "high"
    elif risk_score >= 45: risk_level = "medium"
    elif risk_score >= 20: risk_level = "low"
    else:                  risk_level = "none"

    details = {
        "formation_accuracy":          round(formation_accuracy, 2) if formation_accuracy is not None else None,
        "formation_pts":               formation_pts,
        "formation_override_applied":  formation_override_applied,
        "formation_source":            "ml_kit" if formation_accuracy is not None else "not_available",
        "avg_time_per_prompt":         round(avg_time, 2),
        "benchmark_time":              benchmark_time,
        "time_deviation_percent":      round(avg_time_dev * 100, 1),
        "time_inconsistency":          round(time_inconsistency, 2),
        "time_pts":                    time_pts,
        "avg_strokes":                 round(avg_strokes, 1),
        "expected_max_strokes":        max_strokes,
        "excessive_strokes_count":     excessive_strokes_count,
        "stroke_pts":                  stroke_pts,
        "total_clears":                total_clears,
        "avg_clears_per_prompt":       round(avg_clears, 2),
        "excessive_clears_count":      excessive_clears_count,
        "clear_consistency_ratio":     round(clear_consistency_ratio, 2),
        "clears_pts":                  clears_pts,
        "total_prompts":               num_prompts,
        "inconsistency_pts":           inconsistency_pts,
    }

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 1),
        "details":    details,
    }


def save_dysgraphia_submission(submission_data) -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_submissions"]

    try:
        doc = submission_data.dict()
        risk_assessment = calculate_risk_score(doc)

        doc["risk_level"]   = risk_assessment["risk_level"]
        doc["risk_score"]   = risk_assessment["risk_score"]
        doc["risk_details"] = risk_assessment["details"]
        doc["created_at"]   = datetime.utcnow()
        doc["updated_at"]   = datetime.utcnow()

        result = collection.insert_one(doc)

        return {
            "ok":                 True,
            "submission_id":      str(result.inserted_id),
            "risk_level":         risk_assessment["risk_level"],
            "risk_score":         risk_assessment["risk_score"],
            "formation_accuracy": doc.get("formation_accuracy"),
            "message":            "Submission saved successfully",
        }

    except Exception as e:
        return {"ok": False, "error": f"Failed to save submission: {str(e)}"}


def get_dysgraphia_stats() -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_submissions"]

    return {
        "ok":                True,
        "stats":             list(collection.aggregate([
            {"$group": {"_id": {"grade": "$grade", "activity_type": "$activity_type"}, "count": {"$sum": 1}}},
            {"$sort": {"_id.grade": 1}},
        ])),
        "risk_distribution": list(collection.aggregate([
            {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
        ])),
        "grade_risk_stats":  list(collection.aggregate([
            {"$group": {"_id": "$grade", "avg_risk_score": {"$avg": "$risk_score"}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ])),
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
                    "risk_level":   risk_assessment["risk_level"],
                    "risk_score":   risk_assessment["risk_score"],
                    "risk_details": risk_assessment["details"],
                    "updated_at":   datetime.utcnow(),
                }},
            )
            updated_count += 1

        return {
            "ok":            True,
            "updated_count": updated_count,
            "message":       f"Recalculated risk levels for {updated_count} submissions",
        }

    except Exception as e:
        return {"ok": False, "error": f"Failed to recalculate risks: {str(e)}"}