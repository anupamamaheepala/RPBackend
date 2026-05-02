# models/dysgraphia_improvement_models.py
# Pydantic models for submission + all response shapes the dashboard expects.

from pydantic import BaseModel, Field, computed_field
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Submission (Flutter → API)
# ─────────────────────────────────────────────────────────────────────────────

class DysgraphiaImprovementSubmission(BaseModel):
    """Single activity session submitted from the Flutter app."""

    user_id:           Optional[str]   = None        # logged-in user ID
    grade:             int                           # e.g. 3
    risk_level:        str                           # "low" | "medium" | "high"
    activity_name:     str                           # e.g. "confusable_pairs"
    activity_label:    str                           # Sinhala label, e.g. "සමාන අකුරු"
    total_items:       int                           # total prompts in the activity
    correct_count:     int                           # correct answers
    duration_seconds:  Optional[float] = None        # total time spent (seconds)
    created_at:        Optional[datetime] = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def accuracy(self) -> float:
        """Accuracy as a percentage 0–100."""
        if self.total_items == 0:
            return 0.0
        return round((self.correct_count / self.total_items) * 100, 2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "abc123",
                "grade": 3,
                "risk_level": "low",
                "activity_name": "confusable_pairs",
                "activity_label": "සමාන අකුරු",
                "total_items": 5,
                "correct_count": 4,
                "duration_seconds": 120.5,
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-activity best stats (used in summary)
# ─────────────────────────────────────────────────────────────────────────────

class ActivityBest(BaseModel):
    """Best-ever stats for a single activity type."""
    activity_name:    str
    activity_label:   str
    best_accuracy:    float   # 0–100
    avg_accuracy:     float   # 0–100
    session_count:    int
    last_played_at:   Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard summary (computed server-side, returned to Flutter)
# ─────────────────────────────────────────────────────────────────────────────

class DysgraphiaDashboardSummary(BaseModel):
    """
    Aggregated summary block returned as `data['summary']`.
    All fields are consumed by the Flutter dashboard components.
    """

    # ── Risk ──────────────────────────────────────────────────────────────────
    latest_risk_level:     str            # "low" | "medium" | "high"

    # ── Accuracy ─────────────────────────────────────────────────────────────
    avg_accuracy:          float          # overall average across all sessions
    this_month_accuracy:   float          # average accuracy this calendar month
    last_month_accuracy:   float          # average accuracy last calendar month
    latest_accuracy:       float          # accuracy of the most recent session
    latest_activity_label: str            # Sinhala label of the most recent activity
    latest_duration:       Optional[float] = None  # duration of latest session (s)

    # ── Streak ────────────────────────────────────────────────────────────────
    current_streak:        int            # consecutive days with at least one session
    week_practiced:        List[bool]     # Mon–Sun flags, True = practiced that day

    # ── Activity bests ───────────────────────────────────────────────────────
    activity_bests:        dict[str, ActivityBest]
    # key = activity_name, value = ActivityBest


# ─────────────────────────────────────────────────────────────────────────────
# Full API response shape  (data['sessions'] + data['summary'] + total)
# ─────────────────────────────────────────────────────────────────────────────

class DysgraphiaUserResultsResponse(BaseModel):
    """
    Top-level response from  GET /dysgraphia-improvement/user-results/{user_id}
    This is what the Flutter dashboard decodes with jsonDecode(response.body).
    """

    total_sessions: int
    summary:        DysgraphiaDashboardSummary
    sessions:       List[DysgraphiaImprovementSubmission]
    # Most-recent first. Flutter Journey & Activities tabs consume this list
    # directly for chart data (accuracy over time, sessions per week, etc.).