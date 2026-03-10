# models/dysgraphia_improvement_models.py

from pydantic import BaseModel
from typing import Optional

class DysgraphiaImprovementSubmission(BaseModel):
    user_id: Optional[str] = None          # logged-in user ID from Flutter session
    grade: int                             # e.g. 3
    risk_level: str                        # "low" | "medium" | "high"
    activity_name: str                     # e.g. "confusable_pairs"
    activity_label: str                    # human-readable e.g. "සමාන අකුරු"
    total_items: int                       # total prompts/items in the activity
    correct_count: int                     # how many the child got right
    duration_seconds: Optional[float] = None  # total time spent (optional)

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "abc123",
                "grade": 3,
                "risk_level": "low",
                "activity_name": "confusable_pairs",
                "activity_label": "සමාන අකුරු",
                "total_items": 5,
                "correct_count": 4,
                "duration_seconds": 120.5
            }
        }
    }