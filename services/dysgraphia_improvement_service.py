# services/dysgraphia_improvement_service.py

from services.db_service import get_db
from datetime import datetime
from typing import Dict, Any


def save_improvement_session(submission_data) -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_improvement_sessions"]

    try:
        doc = submission_data.dict()
        total   = doc.get("total_items", 0)
        correct = doc.get("correct_count", 0)
        doc["score_percent"] = round((correct / total * 100), 1) if total > 0 else 0.0
        doc["created_at"]    = datetime.utcnow()

        result = collection.insert_one(doc)
        return {
            "ok":            True,
            "session_id":    str(result.inserted_id),
            "score_percent": doc["score_percent"],
            "message":       "Improvement session saved successfully",
        }

    except Exception as e:
        return {"ok": False, "error": f"Failed to save improvement session: {str(e)}"}


def get_user_improvement_results(user_id: str) -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_improvement_sessions"]

    sessions_raw = list(
        collection.find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("created_at", -1)
    )

    if not sessions_raw:
        return {
            "ok":             True,
            "user_id":        user_id,
            "total_sessions": 0,
            "summary":        None,
            "sessions":       [],
        }

    sessions = []
    for s in sessions_raw:
        sessions.append({
            "grade":            s.get("grade"),
            "risk_level":       s.get("risk_level"),
            "activity_name":    s.get("activity_name"),
            "activity_label":   s.get("activity_label"),
            "total_items":      s.get("total_items"),
            "correct_count":    s.get("correct_count"),
            "score_percent":    s.get("score_percent"),
            "duration_seconds": s.get("duration_seconds"),
            "created_at":       s["created_at"].isoformat() if s.get("created_at") else None,
        })

    # Summary stats
    avg_score = round(sum(s["score_percent"] for s in sessions) / len(sessions), 1)
    best_score = max(s["score_percent"] for s in sessions)

    # Activity breakdown — best score per activity
    activity_bests: Dict[str, float] = {}
    for s in sessions:
        name = s["activity_name"]
        if name not in activity_bests or s["score_percent"] > activity_bests[name]:
            activity_bests[name] = s["score_percent"]

    # Risk level counts
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    for s in sessions:
        lvl = s.get("risk_level", "")
        if lvl in risk_counts:
            risk_counts[lvl] += 1

    return {
        "ok":             True,
        "user_id":        user_id,
        "total_sessions": len(sessions),
        "summary": {
            "average_score":   avg_score,
            "best_score":      best_score,
            "latest_activity": sessions[0]["activity_label"],
            "latest_score":    sessions[0]["score_percent"],
            "risk_counts":     risk_counts,
            "activity_bests":  activity_bests,
        },
        "sessions": sessions,
    }