# services/dysgraphia_service.py
from services.db_service import get_db
from bson.binary import Binary
from datetime import datetime
from typing import Dict, Any

def save_dysgraphia_submission(submission_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save dysgraphia submission to MongoDB.
    
    Args:
        submission_data: Dict from Pydantic model (validated).
    
    Returns:
        Dict with 'ok' status and 'submission_id' on success, or 'error' on failure.
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    
    try:
        # Add metadata
        doc = submission_data.dict()
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        
        # Insert and get ID
        result = collection.insert_one(doc)
        submission_id = str(result.inserted_id)
        
        return {
            "ok": True,
            "submission_id": submission_id,
            "message": "Submission saved successfully"
        }
    
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to save submission: {str(e)}"
        }

def get_dysgraphia_stats() -> Dict[str, Any]:
    """
    Optional: Get basic stats (e.g., total submissions by grade/activity).
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    
    pipeline = [
        {"$group": {
            "_id": {"grade": "$grade", "activity_type": "$activity_type"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id.grade": 1}}
    ]
    
    stats = list(collection.aggregate(pipeline))
    return {"ok": True, "stats": stats}