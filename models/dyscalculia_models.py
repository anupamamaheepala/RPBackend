from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DyscalculiaResult(BaseModel):
    user_id: str             # <-- Added to track the specific student
    grade: int
    task_number: int
    accuracy: int            # Out of 5
    response_time_avg: float # Average seconds per problem
    hesitation_time_avg: float # Average hesitation (seconds)
    retries: int             # Total retries
    backtracks: int          # Total backtracks
    skipped_items: int       # Total skipped
    completion_time: float   # Total task time in seconds
    risk_level: Optional[str] = None  # <-- Added to store the ML prediction
    created_at: Optional[datetime] = None