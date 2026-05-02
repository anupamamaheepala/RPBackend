from services.db_service import get_db
from datetime import datetime
from typing import Dict, Any

def save_improvement_session(submission_data) -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_improvement_sessions"]

    try:
        doc = submission_data.dict()
        total = doc.get("total_items", 0)
        correct = doc.get("correct_count", 0)
        doc["score_percent"] = round((correct / total * 100), 1) if total > 0 else 0.0
        doc["created_at"] = datetime.utcnow()

        result = collection.insert_one(doc)
        return {
            "ok": True,
            "session_id": str(result.inserted_id),
            "score_percent": doc["score_percent"],
            "message": "Session recorded successfully",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def get_user_improvement_results(user_id: str) -> Dict[str, Any]:
    db = get_db()
    collection = db["dysgraphia_improvement_sessions"]

    sessions = list(collection.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1))

    if not sessions:
        return {"ok": True, "user_id": user_id, "current_tier": "high", "can_detect": False, "sessions": []}

    # Latest risk level determines the current Tier
    latest_session = sessions[0]
    current_risk = latest_session.get("risk_level", "high")
    
    # Calculate Mastery for the current tier
    # We check if the last 3 activities in this tier have scores > 80%
    current_tier_key = f"{current_risk}_support"
    tier_sessions = [s for s in sessions if s.get("activity_category") == current_tier_key]
    
    mastery_count = sum(1 for s in tier_sessions if s.get("score_percent", 0) >= 80)
    # Threshold: Need at least 3 high-score sessions to unlock Detection
    can_detect = mastery_count >= 3 

    return {
        "ok": True,
        "user_id": user_id,
        "summary": {
            "current_risk": current_risk,
            "total_sessions": len(sessions),
            "can_detect_now": can_detect,
            "mastery_progress": f"{mastery_count}/3",
            "latest_score": latest_session.get("score_percent")
        },
        "sessions": sessions
    }