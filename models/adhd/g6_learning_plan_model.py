from pydantic import BaseModel
from typing import List, Optional


class G6LearningPlanRequest(BaseModel):
    child_id:          str
    grade:             int = 6
    attention_profile: str


class G6AdaptationParams(BaseModel):
    chunk_size:          int
    session_minutes:     int
    break_frequency:     int
    modality:            str
    pacing:              str
    visual_complexity:   str
    feedback_style:      str
    encouragement_level: str


class G6LearningActivity(BaseModel):
    title:        str
    description:  str
    type:         str
    duration_min: int
    delivery:     str
    instructions: str


class G6LearningPlanResponse(BaseModel):
    child_id:          str
    grade:             int
    profile:           str
    profile_label:     str
    adaptation_params: G6AdaptationParams
    activities:        List[G6LearningActivity]
    teacher_note:      str
    parent_note:       str
    generated_at:      str
