from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class G6TaskResponseTimes(BaseModel):
    task1: List[int] = Field(default_factory=list)  # Stroop
    task2: List[int] = Field(default_factory=list)  # N-Back
    task3: List[int] = Field(default_factory=list)  # Visual Search
    task4: List[int] = Field(default_factory=list)  # Dual Go/No-Go


class G6ADHDSubmissionRequest(BaseModel):
    grade:               int                            = 6
    child_id:            Optional[str]                  = None
    timestamp:           Optional[str]                  = None
    total_correct:       int
    total_premature:     int
    total_wrong:         int
    task_response_times: Optional[G6TaskResponseTimes] = None

    # Task 1 — Stroop Interference
    stroop_congruent_trials:   int = 8
    stroop_incongruent_trials: int = 12
    stroop_incongruent_errors: int = 0
    stroop_total_errors:       int = 0

    # Task 2 — N-Back Lite (1-back)
    nback_correct:      int = 0
    nback_false_alarms: int = 0
    nback_misses:       int = 0
    nback_total_trials: int = 20

    # Task 3 — Rapid Visual Search
    search_correct:  int       = 0
    search_wrong:    int       = 0
    search_times_ms: List[int] = Field(default_factory=list)

    # Task 4 — Dual Condition Go/No-Go
    dual_correct:      int = 0
    dual_false_alarms: int = 0
    dual_missed:       int = 0
    dual_total_trials: int = 30

    # Task 5 — Sustained Counting
    counting_actual: int = 0
    counting_user:   int = 0


class G6ComputedMetrics(BaseModel):
    total_attempts:          int
    overall_accuracy:        float
    impulsivity_ratio:       float
    inattention_score:       float
    attention_label:         str
    rt_mean_ms:              Optional[float]                      = None
    rt_cv:                   Optional[float]                      = None
    rt_cv_per_task:          Optional[Dict[str, Optional[float]]] = None
    interference_error_rate: Optional[float]                      = None
    nback_accuracy:          Optional[float]                      = None
    visual_search_speed:     Optional[float]                      = None


class G6ADHDSubmissionResponse(BaseModel):
    ok:               bool
    message:          str
    assessment_id:    str
    computed_metrics: G6ComputedMetrics


class G6DiagnosticHistoryResponse(BaseModel):
    child_id:       str
    total_sessions: int
    history:        List[Dict[str, Any]]
