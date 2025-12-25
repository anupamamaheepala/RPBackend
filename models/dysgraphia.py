# models/dysgraphia.py (updated)

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

class DysgraphiaSubmission(BaseModel):
    grade: int
    activity_type: str  # e.g., 'letters', 'words', 'sentences'
    prompts_data: List[PromptData]  # One per prompt: prompt text, strokes, time

    model_config = {  # Updated: V2 syntax (replaces 'class Config: schema_extra')
        "json_schema_extra": {
            "example": {
                "grade": 3,
                "activity_type": "letters",
                "prompts_data": [
                    {
                        "prompt": "අ",
                        "strokes": [
                            {
                                "points": [
                                    {"x": 10.0, "y": 20.0},
                                    {"x": 15.0, "y": 25.0}
                                ]
                            }
                        ],
                        "time_taken": 2.5
                    }
                ]
            }
        }
    }