from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class TaskResponseTimes(BaseModel):
    task1: List[int] = Field(default_factory=list)
    task2: List[int] = Field(default_factory=list)
    task3: List[int] = Field(default_factory=list)


class ADHDSubmissionRequest(BaseModel):
    grade:               int                         = 3
    total_correct:       int
    total_premature:     int
    total_wrong:         int
    overall_accuracy:    Optional[float]             = None
    timestamp:           Optional[str]               = None
    child_id:            Optional[str]               = None
    task_response_times: Optional[TaskResponseTimes] = None  # ── NEW


class ComputedMetrics(BaseModel):
    total_attempts:      int
    overall_accuracy:    float   # recomputed server-side
    impulsivity_ratio:   float   # premature / attempts
    inattention_score:   float   # wrong     / attempts
    attention_label:     str     # "high" | "medium" | "low"
    rt_mean_ms:          Optional[float] = None   # ── NEW
    rt_cv:               Optional[float] = None   # ── NEW: key ML feature
    rt_cv_per_task:      Optional[Dict[str, Optional[float]]] = None  # ── NEW


class ADHDSubmissionResponse(BaseModel):
    ok:               bool
    message:          str
    assessment_id:    str
    computed_metrics: ComputedMetrics