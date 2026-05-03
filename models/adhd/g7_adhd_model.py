from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class G7TaskResponseTimes(BaseModel):
    task1: List[int] = Field(default_factory=list)  # Vigilance
    task3: List[int] = Field(default_factory=list)  # Rule Switching
    task5: List[int] = Field(default_factory=list)  # Go/No-Go Inhibition


class G7ADHDSubmissionRequest(BaseModel):
    grade:               int                          = 7
    child_id:            Optional[str]                = None
    timestamp:           Optional[str]                = None
    total_correct:       int
    total_premature:     int
    total_wrong:         int
    overall_accuracy:    Optional[float]              = None
    task_response_times: Optional[G7TaskResponseTimes] = None

    # Task 1 — Vigilance
    vigilance_correct_hits:  int = 0
    vigilance_misses:        int = 0
    vigilance_false_alarms:  int = 0
    vigilance_targets_shown: int = 0
    vigilance_total_trials:  int = 10

    # Task 3 — Rule Switching
    switching_correct:       int = 0
    switching_errors:        int = 0   # includes switch errors
    switching_switch_errors: int = 0   # errors specifically after rule change
    switching_total_trials:  int = 60

    # Task 4 — Divided Attention
    divided_visual_hits:          int = 0
    divided_visual_false_alarms:  int = 0
    divided_visual_targets_shown: int = 0
    divided_beep_count:           int = 0   # beeps presented

    # Task 5 — Go/No-Go Inhibition
    inhibition_correct_go:     int = 0
    inhibition_false_alarms:   int = 0
    inhibition_total_trials:   int = 150

    # Task 6 — Distraction Resistance
    distraction_correct_taps:    int = 0
    distraction_taps:            int = 0   # wrong taps on distractor
    distraction_misses:          int = 0


class G7ComputedMetrics(BaseModel):
    total_attempts:          int
    overall_accuracy:        float
    impulsivity_ratio:       float
    inattention_score:       float
    attention_label:         str
    rt_mean_ms:              Optional[float]                      = None
    rt_cv:                   Optional[float]                      = None
    rt_cv_per_task:          Optional[Dict[str, Optional[float]]] = None
    switch_error_rate:       Optional[float]                      = None
    divided_attention_score: Optional[float]                      = None
    distraction_resistance:  Optional[float]                      = None
    vigilance_hit_rate:      Optional[float]                      = None
    false_alarm_rate:        Optional[float]                      = None


class G7ADHDSubmissionResponse(BaseModel):
    ok:               bool
    message:          str
    assessment_id:    str
    computed_metrics: G7ComputedMetrics


class G7DiagnosticSession(BaseModel):
    child_id:         str
    grade:            int
    timestamp:        str
    total_correct:    int
    total_premature:  int
    total_wrong:      int
    computed_metrics: Optional[Dict[str, Any]] = None


class G7DiagnosticHistoryResponse(BaseModel):
    child_id:       str
    total_sessions: int
    history:        List[G7DiagnosticSession]
