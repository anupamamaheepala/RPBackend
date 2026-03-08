from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class TaskResponseTimes(BaseModel):
    task1: List[int] = Field(default_factory=list)
    task2: List[int] = Field(default_factory=list)
    task3: List[int] = Field(default_factory=list)


class ADHDSubmissionRequest(BaseModel):
    grade:               int                          = 3
    total_correct:       int
    total_premature:     int
    total_wrong:         int
    overall_accuracy:    Optional[float]              = None
    timestamp:           Optional[str]                = None
    child_id:            Optional[str]                = None
    task_response_times: Optional[TaskResponseTimes]  = None


class ComputedMetrics(BaseModel):
    total_attempts:    int
    overall_accuracy:  float
    impulsivity_ratio: float
    inattention_score: float
    attention_label:   str
    rt_mean_ms:        Optional[float]                     = None
    rt_cv:             Optional[float]                     = None
    rt_cv_per_task:    Optional[Dict[str, Optional[float]]] = None


class ADHDSubmissionResponse(BaseModel):
    ok:               bool
    message:          str
    assessment_id:    str
    computed_metrics: ComputedMetrics