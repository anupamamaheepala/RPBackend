from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DyscalculiaResult(BaseModel):
    user_id: str             
    grade: int
    task_number: int
    accuracy: int            
    response_time_avg: float 
    hesitation_time_avg: float 
    retries: int             
    backtracks: int          
    skipped_items: int       
    wrong_count: int         # <-- ADDED: Tracks total incorrect check attempts
    completion_time: float   
    risk_level: Optional[str] = None  
    created_at: Optional[datetime] = None