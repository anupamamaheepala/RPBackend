from pydantic import BaseModel
from typing import Optional

class DysgraphiaImprovementSubmission(BaseModel):
    user_id: Optional[str] = None          
    grade: int                             
    risk_level: str                        # "low" | "medium" | "high"
    activity_category: str                 # NEW: "high_support" | "medium_support" | "low_support" | "detection"
    activity_name: str                     
    activity_label: str                    
    total_items: int                       
    correct_count: int                     
    duration_seconds: Optional[float] = None  

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "abc123",
                "grade": 3,
                "risk_level": "high",
                "activity_category": "high_support",
                "activity_name": "basic_strokes",
                "activity_label": "මූලික රේඛා",
                "total_items": 10,
                "correct_count": 8,
                "duration_seconds": 150.0
            }
        }

    }