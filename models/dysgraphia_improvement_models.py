# models/dysgraphia_improvement_models.py
# Pydantic models for submission + all response shapes the dashboard expects.

from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Submission (Flutter → API)
# ─────────────────────────────────────────────────────────────────────────────

class DysgraphiaImprovementSubmission(BaseModel):
    """Single activity session submitted from the Flutter app."""

    user_id:           Optional[str]   = None
    grade:             int
    risk_level:        str                        # "low" | "medium" | "high"
    activity_name:     str
    activity_label:    str                        # Sinhala label, e.g. "සමාන අකුරු"
    total_items:       int
    correct_count:     int
    duration_seconds:  Optional[float] = None

    # NOTE: created_at is intentionally NOT included here.
    # It is always set server-side in the service to avoid client clock issues.
    # The @computed_field accuracy was also removed — it caused Pydantic to
    # inject an "accuracy" key into .dict() which corrupted the MongoDB document
    # and broke round-trip deserialization of stored sessions.

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
    best_accuracy:    float   # 0-100
    avg_accuracy:     float   # 0-100
    session_count:    int
    last_played_at:   Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard summary (computed server-side, returned to Flutter)
# ─────────────────────────────────────────────────────────────────────────────

class DysgraphiaDashboardSummary(BaseModel):
    """
    Aggregated summary block returned as data['summary'].
    All fields are consumed by the Flutter dashboard components.
    """

    # Risk
    latest_risk_level:     str

    # Accuracy
    avg_accuracy:          float
    this_month_accuracy:   float
    last_month_accuracy:   float
    latest_accuracy:       float
    latest_activity_label: str
    latest_duration:       Optional[float] = None

    # Streak
    current_streak:        int
    week_practiced:        List[bool]     # Mon-Sun flags, True = practiced that day

    # Activity bests
    # Dict[str, ActivityBest] instead of dict[str, ActivityBest]
    # for Python 3.8/3.9 compatibility (lowercase dict[] requires 3.10+)
    activity_bests:        Dict[str, ActivityBest]


# ─────────────────────────────────────────────────────────────────────────────
# Session row returned in the sessions list
# ─────────────────────────────────────────────────────────────────────────────

class DysgraphiaSessionRow(BaseModel):
    """
    A single stored session as returned from the DB.
    Separate from DysgraphiaImprovementSubmission so that server-set fields
    (score_percent, created_at) are included without polluting the inbound model.
    """
    user_id:           Optional[str]   = None
    grade:             Optional[int]   = None
    risk_level:        Optional[str]   = None
    activity_name:     Optional[str]   = None
    activity_label:    Optional[str]   = None
    total_items:       Optional[int]   = None
    correct_count:     Optional[int]   = None
    score_percent:     Optional[float] = None
    duration_seconds:  Optional[float] = None
    created_at:        Optional[str]   = None   # ISO string after serialisation


# ─────────────────────────────────────────────────────────────────────────────
# Full API response shape  (data['sessions'] + data['summary'] + total)
# ─────────────────────────────────────────────────────────────────────────────

class DysgraphiaUserResultsResponse(BaseModel):
    """
    Top-level response from GET /dysgraphia-improvement/user-results/{user_id}
    This is what the Flutter dashboard decodes with jsonDecode(response.body).
    """
    total_sessions: int
    summary:        DysgraphiaDashboardSummary
    # Uses DysgraphiaSessionRow (not DysgraphiaImprovementSubmission) so that
    # score_percent and created_at are included and no computed_field mismatch occurs.
    sessions:       List[DysgraphiaSessionRow]