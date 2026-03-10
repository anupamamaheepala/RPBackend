from pydantic import BaseModel
from typing import List, Optional


class LearningPlanRequest(BaseModel):
    child_id:          str
    grade:             int
    attention_profile: str


class AdaptationParams(BaseModel):
    chunk_size:           int
    session_minutes:      int
    break_frequency:      int
    modality:             str
    pacing:               str
    visual_complexity:    str
    feedback_style:       str
    encouragement_level:  str   # "standard" / "high" / "very_high"


class LearningActivity(BaseModel):
    title:            str
    description:      str
    type:             str        # "focus_builder" / "memory" / "inhibition" / "comprehension"
    duration_min:     int
    delivery:         str        # "in_app" / "teacher_led" / "independent"
    instructions:     str        # step-by-step what child/teacher does


class LearningPlanResponse(BaseModel):
    child_id:          str
    grade:             int
    profile:           str
    profile_label:     str
    adaptation_params: AdaptationParams
    activities:        List[LearningActivity]
    teacher_note:      str
    parent_note:       str
    generated_at:      str