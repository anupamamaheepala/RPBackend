from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class G5TaskResponseTimes(BaseModel):
    task1: List[int] = Field(default_factory=list)  # Ladder
    task2: List[int] = Field(default_factory=list)  # Filter
    task3: List[int] = Field(default_factory=list)  # Stillness
    task4: List[int] = Field(default_factory=list)  # Switch Go


class G5ADHDSubmissionRequest(BaseModel):
    grade:               int                          = 5
    child_id:            Optional[str]                = None
    timestamp:           Optional[str]                = None
    total_correct:       int
    total_premature:     int
    total_wrong:         int
    overall_accuracy:    Optional[float]              = None
    task_response_times: Optional[G5TaskResponseTimes] = None
    # Grade 5 specific
    total_steps_completed: int = 0   # Ladder — sequential_completion_rate
    total_steps_skipped:   int = 0   # Ladder
    total_breaks_count:    int = 0   # Stillness — motor_inhibition_score
    total_hold_duration_ms: int = 0  # Stillness
    total_switch_errors:   int = 0   # Switch Go — switch_error_rate
    total_switch_trials:   int = 0   # Switch Go


class G5ComputedMetrics(BaseModel):
    total_attempts:               int
    overall_accuracy:             float
    impulsivity_ratio:            float
    inattention_score:            float
    attention_label:              str
    rt_mean_ms:                   Optional[float]                      = None
    rt_cv:                        Optional[float]                      = None
    rt_cv_per_task:               Optional[Dict[str, Optional[float]]] = None
    switch_error_rate:            Optional[float]                      = None
    motor_inhibition_score:       Optional[float]                      = None
    sequential_completion_rate:   Optional[float]                      = None


class G5ADHDSubmissionResponse(BaseModel):
    ok:               bool
    message:          str
    assessment_id:    str
    computed_metrics: G5ComputedMetrics


class G5DiagnosticSession(BaseModel):
    child_id:         str
    grade:            int
    timestamp:        str
    total_correct:    int
    total_premature:  int
    total_wrong:      int
    computed_metrics: Optional[Dict[str, Any]] = None


class G5DiagnosticHistoryResponse(BaseModel):
    child_id:       str
    total_sessions: int
    history:        List[G5DiagnosticSession]
