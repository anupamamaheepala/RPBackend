from pydantic import BaseModel
from typing import List


class G7LearningTaskAssignRequest(BaseModel):
    child_id: str
    grade:    int = 7


class G7AssignedTask(BaseModel):
    task_id:        str
    task_name:      str
    difficulty:     int
    target_deficit: str
    instructions:   str


class G7LearningTaskAssignResponse(BaseModel):
    child_id:         str
    grade:            int
    session_number:   int
    tasks:            List[G7AssignedTask]
    dominant_deficit: str
    severity_scores:  dict


class G7LearningTaskResult(BaseModel):
    child_id:          str
    grade:             int
    task_id:           str
    difficulty:        int
    correct:           int
    wrong:             int
    premature:         int
    total_trials:      int
    response_times_ms: List[int]
    session_number:    int


class G7LearningTaskResultResponse(BaseModel):
    ok:              bool
    message:         str
    score_percent:   float
    next_difficulty: int
    encouragement:   str
