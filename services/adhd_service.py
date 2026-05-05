import pickle
import statistics
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from models.adhd_model import ADHDSubmissionRequest, ComputedMetrics
from services.db_service import get_db


# ── Load ML model once at startup ─────────────────────────────────────────────
_PKL_PATH = Path(__file__).parent.parent / "ml_models" / "attention_classifier.pkl"

try:
    with open(_PKL_PATH, "rb") as f:
        _bundle = pickle.load(f)
    _model    = _bundle["model"]
    _le       = _bundle["label_encoder"]
    _features = _bundle["feature_cols"]
    print("✅ Attention classifier loaded successfully")
except FileNotFoundError:
    _model = None
    _le    = None
    print("⚠️  attention_classifier.pkl not found — falling back to rule-based labels")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _compute_cv(times: List[int]) -> Optional[float]:
    """Coefficient of Variation = stdev / mean. Key inattention marker."""
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


def _rule_based_label(
    premature:         int,
    wrong:             int,
    impulsivity_ratio: float,
    inattention_score: float,
) -> str:
    """Fallback label if pkl is not loaded."""
    high_impulsive   = premature >= 5 or impulsivity_ratio >= 0.25
    high_inattentive = wrong     >= 6 or inattention_score >= 0.30

    if not high_impulsive and not high_inattentive:
        return "profile_a"
    elif high_inattentive and not high_impulsive:
        return "profile_b"
    elif high_impulsive and not high_inattentive:
        return "profile_c"
    else:
        return "profile_d"


def _ml_predict_profile(
    overall_accuracy:  float,
    impulsivity_ratio: float,
    inattention_score: float,
    rt_mean_ms:        float,
    rt_cv:             float,
    rt_cv_task1:       float,
    rt_cv_task2:       float,
    rt_cv_task3:       float,
) -> str:
    """ML-based profile prediction using the trained Random Forest."""
    row = pd.DataFrame([{
        "overall_accuracy":   overall_accuracy,
        "impulsivity_ratio":  impulsivity_ratio,
        "inattention_score":  inattention_score,
        "rt_mean_ms":         rt_mean_ms,
        "rt_cv":              rt_cv,
        "rt_cv_task1":        rt_cv_task1,
        "rt_cv_task2":        rt_cv_task2,
        "rt_cv_task3":        rt_cv_task3,
    }])
    pred_enc = _model.predict(row)[0]
    return _le.inverse_transform([pred_enc])[0]


# ── Main service functions ────────────────────────────────────────────────────
def compute_metrics(req: ADHDSubmissionRequest) -> ComputedMetrics:
    total_attempts    = req.total_correct + req.total_wrong + req.total_premature
    accuracy          = req.total_correct   / total_attempts if total_attempts else 0.0
    impulsivity_ratio = req.total_premature / total_attempts if total_attempts else 0.0
    inattention_score = req.total_wrong     / total_attempts if total_attempts else 0.0

    # ── Response time metrics ─────────────────────────────────────────────────
    rt_mean_ms     = None
    rt_cv          = None
    rt_cv_per_task = None
    rt_cv_t1       = 0.0
    rt_cv_t2       = 0.0
    rt_cv_t3       = 0.0

    if req.task_response_times:
        all_times = (
            req.task_response_times.task1 +
            req.task_response_times.task2 +
            req.task_response_times.task3
        )
        rt_mean_ms = _compute_mean(all_times)
        rt_cv      = _compute_cv(all_times)

        rt_cv_t1 = _compute_cv(req.task_response_times.task1) or 0.0
        rt_cv_t2 = _compute_cv(req.task_response_times.task2) or 0.0
        rt_cv_t3 = _compute_cv(req.task_response_times.task3) or 0.0

        rt_cv_per_task = {
            "task1": rt_cv_t1,
            "task2": rt_cv_t2,
            "task3": rt_cv_t3,
        }

    # ── Profile classification — ML if available, else rule-based ────────────
    if _model is not None:
        attention_label = _ml_predict_profile(
            overall_accuracy  = round(accuracy, 4),
            impulsivity_ratio = round(impulsivity_ratio, 4),
            inattention_score = round(inattention_score, 4),
            rt_mean_ms        = rt_mean_ms or 0.0,
            rt_cv             = rt_cv      or 0.0,
            rt_cv_task1       = rt_cv_t1,
            rt_cv_task2       = rt_cv_t2,
            rt_cv_task3       = rt_cv_t3,
        )
    else:
        attention_label = _rule_based_label(
            req.total_premature,
            req.total_wrong,
            round(impulsivity_ratio, 4),
            round(inattention_score, 4),
        )

    return ComputedMetrics(
        total_attempts    = total_attempts,
        overall_accuracy  = round(accuracy, 4),
        impulsivity_ratio = round(impulsivity_ratio, 4),
        inattention_score = round(inattention_score, 4),
        attention_label   = attention_label,
        rt_mean_ms        = rt_mean_ms,
        rt_cv             = rt_cv,
        rt_cv_per_task    = rt_cv_per_task,
    )


def save_assessment(req: ADHDSubmissionRequest,
                    metrics: ComputedMetrics) -> str:
    db = get_db()

    doc = {
        "child_id":          req.child_id,
        "grade":             req.grade,
        "total_correct":     req.total_correct,
        "total_premature":   req.total_premature,
        "total_wrong":       req.total_wrong,
        "task_response_times": req.task_response_times.dict()
            if req.task_response_times else None,
        "metrics":           metrics.dict(),
        "client_timestamp":  req.timestamp,
        "created_at":        datetime.utcnow(),
    }

    result = db["adhd_submissions"].insert_one(doc)
    return str(result.inserted_id)


def get_diagnostic_history(child_id: str) -> dict:
    """
    Returns last 10 diagnostic submissions for a child, newest first.

    MongoDB stores:
      "metrics"           → remapped to "computed_metrics" for Flutter
      "client_timestamp"  → preferred timestamp string (sent by Flutter)
      "created_at"        → datetime fallback
    """
    db = get_db()

    records = list(
        db["adhd_submissions"]
        .find({"child_id": child_id})
        .sort("created_at", -1)
        .limit(10)
    )

    clean = []
    for r in records:
        # Remap "metrics" → "computed_metrics"
        raw_metrics = r.get("metrics", {})

        # Prefer client_timestamp (Flutter ISO string), fall back to created_at
        ts = r.get("client_timestamp")
        if not ts:
            created = r.get("created_at")
            ts = created.isoformat() if hasattr(created, "isoformat") else str(created)

        clean.append({
            "child_id":        r.get("child_id", child_id),
            "grade":           r.get("grade", 3),
            "timestamp":       ts,
            "total_correct":   r.get("total_correct",   0),
            "total_premature": r.get("total_premature", 0),
            "total_wrong":     r.get("total_wrong",     0),
            "computed_metrics": {
                "attention_label":   raw_metrics.get("attention_label",   ""),
                "overall_accuracy":  raw_metrics.get("overall_accuracy",  0.0),
                "impulsivity_ratio": raw_metrics.get("impulsivity_ratio", 0.0),
                "inattention_score": raw_metrics.get("inattention_score", 0.0),
                "rt_mean_ms":        raw_metrics.get("rt_mean_ms"),
                "rt_cv":             raw_metrics.get("rt_cv"),
            },
        })

    return {
        "child_id":       child_id,
        "total_sessions": len(clean),
        "history":        clean,
    }