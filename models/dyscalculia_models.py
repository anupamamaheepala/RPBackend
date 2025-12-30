from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DyscalculiaResult(BaseModel):
    grade: int
    task_number: int
    accuracy: int            # Out of 5
    response_time_avg: float # Average seconds per problem
    hesitation_time_avg: float # Average hesitation (seconds)
    retries: int             # Total retries
    backtracks: int          # Total backtracks
    skipped_items: int       # Total skipped
    completion_time: float   # Total task time in seconds
    created_at: Optional[datetime] = None