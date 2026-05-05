from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class G4TaskResponseTimes(BaseModel):
    task1: List[int] = Field(default_factory=list)  # Listen & Extract
    task2: List[int] = Field(default_factory=list)  # Stop/Go Signals
    task3: List[int] = Field(default_factory=list)  # Follow Card
    task4: List[int] = Field(default_factory=list)  # Stay & Complete


class G4ADHDSubmissionRequest(BaseModel):
    grade:               int                           = 4
    child_id:            Optional[str]                 = None
    timestamp:           Optional[str]                 = None

    # Raw counts aggregated across all 4 tasks
    total_correct:       int
    total_premature:     int
    total_wrong:         int
    overall_accuracy:    Optional[float]               = None

    # Per-task response times for CV features
    task_response_times: Optional[G4TaskResponseTimes] = None

    # Grade 4 specific raw inputs
    total_rule_views:         int = 0   # Follow Card help-button presses
    total_items_completed:    int = 0   # Stay & Complete items finished
    total_replay_count:       int = 0   # Listen & Extract replay presses


class G4ComputedMetrics(BaseModel):
    total_attempts:              int
    overall_accuracy:            float
    impulsivity_ratio:           float
    inattention_score:           float
    attention_label:             str
    rt_mean_ms:                  Optional[float]                      = None
    rt_cv:                       Optional[float]                      = None
    rt_cv_per_task:              Optional[Dict[str, Optional[float]]] = None
    # Grade 4 extra features
    rule_follow_score:           Optional[float]                      = None
    sustained_completion_rate:   Optional[float]                      = None
    replay_dependency:           Optional[float]                      = None


class G4ADHDSubmissionResponse(BaseModel):
    ok:               bool
    message:          str
    assessment_id:    str
    computed_metrics: G4ComputedMetrics


class G4DiagnosticSession(BaseModel):
    child_id:         str
    grade:            int
    timestamp:        str
    total_correct:    int
    total_premature:  int
    total_wrong:      int
    computed_metrics: Optional[Dict[str, Any]] = None


class G4DiagnosticHistoryResponse(BaseModel):
    child_id:       str
    total_sessions: int
    history:        List[G4DiagnosticSession]
