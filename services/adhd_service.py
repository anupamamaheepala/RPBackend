import statistics
from datetime import datetime
from typing import List, Optional, Dict
from models.adhd_model import ADHDSubmissionRequest, ComputedMetrics
from services.db_service import get_db


def _compute_attention_label(premature: int, wrong: int) -> str:
    if premature >= 8 or wrong >= 10:
        return "low"
    elif premature >= 5 or wrong >= 6:
        return "medium"
    else:
        return "high"


def _compute_cv(times: List[int]) -> Optional[float]:
    """
    Coefficient of Variation = stdev / mean.
    Returns None if not enough data points.
    High CV means inconsistent response times → strong inattention marker.
    """
    if len(times) < 2:
        return None
    mean = statistics.mean(times)
    if mean == 0:
        return None
    return round(statistics.stdev(times) / mean, 4)


def _compute_mean(times: List[int]) -> Optional[float]:
    if not times:
        return None
    return round(statistics.mean(times), 2)


def compute_metrics(req: ADHDSubmissionRequest) -> ComputedMetrics:
    total_attempts    = req.total_correct + req.total_wrong + req.total_premature
    accuracy          = req.total_correct   / total_attempts if total_attempts else 0.0
    impulsivity_ratio = req.total_premature / total_attempts if total_attempts else 0.0
    inattention_score = req.total_wrong     / total_attempts if total_attempts else 0.0

    # ── Response time computations ────────────────────────────────────────────
    rt_mean_ms     = None
    rt_cv          = None
    rt_cv_per_task = None

    if req.task_response_times:
        all_times = (
            req.task_response_times.task1 +
            req.task_response_times.task2 +
            req.task_response_times.task3
        )
        rt_mean_ms = _compute_mean(all_times)
        rt_cv      = _compute_cv(all_times)

        # Per-task CV — useful later for identifying which task type
        # causes the most inconsistency
        rt_cv_per_task = {
            "task1": _compute_cv(req.task_response_times.task1),
            "task2": _compute_cv(req.task_response_times.task2),
            "task3": _compute_cv(req.task_response_times.task3),
        }

    return ComputedMetrics(
        total_attempts=total_attempts,
        overall_accuracy=round(accuracy, 4),
        impulsivity_ratio=round(impulsivity_ratio, 4),
        inattention_score=round(inattention_score, 4),
        attention_label=_compute_attention_label(
            req.total_premature, req.total_wrong
        ),
        rt_mean_ms=rt_mean_ms,
        rt_cv=rt_cv,
        rt_cv_per_task=rt_cv_per_task,
    )


def save_assessment(req: ADHDSubmissionRequest,
                    metrics: ComputedMetrics) -> str:
    db = get_db()

    doc = {
        # ── Raw data from Flutter ─────────────────────────────────────────
        "child_id":          req.child_id,
        "grade":             req.grade,
        "total_correct":     req.total_correct,
        "total_premature":   req.total_premature,
        "total_wrong":       req.total_wrong,
        # ── Raw response times stored for future reprocessing ─────────────
        "task_response_times": req.task_response_times.dict()
            if req.task_response_times else None,
        # ── Server-computed metrics (these feed the ML classifier later) ──
        "metrics":           metrics.dict(),
        # ── Timestamps ───────────────────────────────────────────────────
        "client_timestamp":  req.timestamp,
        "created_at":        datetime.utcnow(),
    }

    result = db["adhd_submissions"].insert_one(doc)
    return str(result.inserted_id)