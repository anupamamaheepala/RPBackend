from pydantic import BaseModel
from typing import List, Optional


class LearningTaskAssignRequest(BaseModel):
    child_id:  str
    grade:     int


class AssignedTask(BaseModel):
    task_id:        str   # "gonogo" / "wait_match" / "audio_sequence" / "spot_change" / "attention_grid"
    task_name:      str
    difficulty:     int   # 1 / 2 / 3
    target_deficit: str   # "impulsivity" / "inattention" / "accuracy" / "maintenance"
    instructions:   str


class LearningTaskAssignResponse(BaseModel):
    child_id:       str
    grade:          int
    session_number: int
    tasks:          List[AssignedTask]
    dominant_deficit: str
    severity_scores: dict


class LearningTaskResult(BaseModel):
    child_id:       str
    grade:          int
    task_id:        str
    difficulty:     int
    correct:        int
    wrong:          int
    premature:      int
    total_trials:   int
    response_times_ms: List[int]
    session_number: int


class LearningTaskResultResponse(BaseModel):
    ok:             bool
    message:        str
    score_percent:  float
    next_difficulty: int
    encouragement:  str