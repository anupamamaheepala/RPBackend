# models/dysgraphia.py

from pydantic import BaseModel
from typing import List, Dict, Optional

class Point(BaseModel):
    x: float
    y: float

class Stroke(BaseModel):
    points: List[Point]

class PromptData(BaseModel):
    prompt: str
    strokes: List[Stroke]
    time_taken: Optional[float] = None
    clears: Optional[int] = 0
    formation_correct: Optional[bool] = None  # NEW: from ML Kit (True/False/None)

class DysgraphiaSubmission(BaseModel):
    grade: int
    activity_type: str
    prompts_data: List[PromptData]
    formation_accuracy: Optional[float] = None  # NEW: 0.0-1.0 overall score from ML Kit

    model_config = {
        "json_schema_extra": {
            "example": {
                "grade": 3,
                "activity_type": "letters",
                "formation_accuracy": 0.75,
                "prompts_data": [
                    {
                        "prompt": "අ",
                        "strokes": [{"points": [{"x": 10.0, "y": 20.0}]}],
                        "time_taken": 2.5,
                        "clears": 1,
                        "formation_correct": True
                    }
                ]
            }
        }
    }