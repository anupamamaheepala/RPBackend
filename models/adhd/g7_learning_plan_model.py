from pydantic import BaseModel
from typing import List, Optional


class G7LearningPlanRequest(BaseModel):
    child_id:          str
    grade:             int = 7
    attention_profile: str


class G7AdaptationParams(BaseModel):
    chunk_size:          int
    session_minutes:     int
    break_frequency:     int
    modality:            str
    pacing:              str
    visual_complexity:   str
    feedback_style:      str
    encouragement_level: str


class G7LearningActivity(BaseModel):
    title:        str
    description:  str
    type:         str
    duration_min: int
    delivery:     str
    instructions: str


class G7LearningPlanResponse(BaseModel):
    child_id:          str
    grade:             int
    profile:           str
    profile_label:     str
    adaptation_params: G7AdaptationParams
    activities:        List[G7LearningActivity]
    teacher_note:      str
    parent_note:       str
    generated_at:      str
