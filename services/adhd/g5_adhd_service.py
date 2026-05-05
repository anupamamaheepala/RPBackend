"""
Grade 5 ADHD Service
Features: overall_accuracy, impulsivity_ratio, inattention_score,
          rt_mean_ms, rt_cv, switch_error_rate,
          motor_inhibition_score, sequential_completion_rate
"""
import pickle, statistics, pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from models.adhd.g5_adhd_model import G5ADHDSubmissionRequest, G5ComputedMetrics
from services.db_service import get_db

_PKL_PATH = Path(__file__).parent.parent.parent / "ml_models" / "adhd" / "g5_attention_classifier.pkl"
try:
    with open(_PKL_PATH, "rb") as f:
        _bundle = pickle.load(f)
    _model    = _bundle["model"]
    _le       = _bundle["label_encoder"]
    _features = _bundle["feature_cols"]
    print("✅ G5 attention classifier loaded")
except FileNotFoundError:
    _model = None
    _le    = None
    print("⚠️  g5_attention_classifier.pkl not found — rule-based fallback")


def _cv(times: List[int]) -> Optional[float]:
    if len(times) < 2: return None
    mean = statistics.mean(times)
    return round(statistics.stdev(times) / mean, 4) if mean else None

def _mean(times: List[int]) -> Optional[float]:
    return round(statistics.mean(times), 2) if times else None

def _rule_based(imp: float, inat: float) -> str:
    hi = imp  >= 0.25
    ia = inat >= 0.25
    if not hi and not ia: return "profile_a"
    elif ia and not hi:   return "profile_b"
    elif hi and not ia:   return "profile_c"
    return "profile_d"


def g5_compute_metrics(req: G5ADHDSubmissionRequest) -> G5ComputedMetrics:
    total = req.total_correct + req.total_wrong + req.total_premature
    acc   = req.total_correct   / total if total else 0.0
    imp   = req.total_premature / total if total else 0.0
    inat  = req.total_wrong     / total if total else 0.0

    # Grade 5 specific features
    # switch_error_rate: switch errors / total switch trials
    sw_err = (req.total_switch_errors / req.total_switch_trials
              if req.total_switch_trials > 0 else 0.0)

    # motor_inhibition_score: inverse of break ratio (30s task, breaks penalise)
    motor = max(0.0, 1.0 - (req.total_breaks_count / 10.0))

    # sequential_completion_rate: steps completed / expected 5 steps
    expected_steps = 5
    seq = min(1.0, req.total_steps_completed / expected_steps)

    # RT features
    rt_mean = rt_cv_all = None
    rt_cv_t1 = rt_cv_t2 = rt_cv_t3 = rt_cv_t4 = 0.0
    rt_cv_per_task = None

    if req.task_response_times:
        all_times = (req.task_response_times.task1 + req.task_response_times.task2 +
                     req.task_response_times.task3 + req.task_response_times.task4)
        rt_mean    = _mean(all_times)
        rt_cv_all  = _cv(all_times)
        rt_cv_t1   = _cv(req.task_response_times.task1) or 0.0
        rt_cv_t2   = _cv(req.task_response_times.task2) or 0.0
        rt_cv_t3   = _cv(req.task_response_times.task3) or 0.0
        rt_cv_t4   = _cv(req.task_response_times.task4) or 0.0
        rt_cv_per_task = {"task1":rt_cv_t1,"task2":rt_cv_t2,"task3":rt_cv_t3,"task4":rt_cv_t4}

    if _model is not None:
        row = pd.DataFrame([{
            "overall_accuracy":           round(acc,   4),
            "impulsivity_ratio":          round(imp,   4),
            "inattention_score":          round(inat,  4),
            "rt_mean_ms":                 rt_mean or 0.0,
            "rt_cv":                      rt_cv_all or 0.0,
            "switch_error_rate":          round(sw_err, 4),
            "motor_inhibition_score":     round(motor,  4),
            "sequential_completion_rate": round(seq,    4),
        }])
        label = _le.inverse_transform(_model.predict(row))[0]
    else:
        label = _rule_based(imp, inat)

    return G5ComputedMetrics(
        total_attempts              = total,
        overall_accuracy            = round(acc,    4),
        impulsivity_ratio           = round(imp,    4),
        inattention_score           = round(inat,   4),
        attention_label             = label,
        rt_mean_ms                  = rt_mean,
        rt_cv                       = rt_cv_all,
        rt_cv_per_task              = rt_cv_per_task,
        switch_error_rate           = round(sw_err, 4),
        motor_inhibition_score      = round(motor,  4),
        sequential_completion_rate  = round(seq,    4),
    )


def g5_save_assessment(req: G5ADHDSubmissionRequest, metrics: G5ComputedMetrics) -> str:
    db  = get_db()
    doc = {
        "child_id":              req.child_id,
        "grade":                 5,
        "total_correct":         req.total_correct,
        "total_premature":       req.total_premature,
        "total_wrong":           req.total_wrong,
        "total_steps_completed": req.total_steps_completed,
        "total_steps_skipped":   req.total_steps_skipped,
        "total_breaks_count":    req.total_breaks_count,
        "total_hold_duration_ms": req.total_hold_duration_ms,
        "total_switch_errors":   req.total_switch_errors,
        "total_switch_trials":   req.total_switch_trials,
        "task_response_times":   req.task_response_times.dict() if req.task_response_times else None,
        "metrics":               metrics.dict(),
        "client_timestamp":      req.timestamp,
        "created_at":            datetime.utcnow(),
    }
    return str(db["adhd_submissions"].insert_one(doc).inserted_id)


def g5_get_diagnostic_history(child_id: str) -> dict:
    db      = get_db()
    records = list(db["adhd_submissions"].find({"child_id": child_id, "grade": 5})
                   .sort("created_at", -1).limit(10))
    clean   = []
    for r in records:
        raw = r.get("metrics", {})
        ts  = r.get("client_timestamp")
        if not ts:
            created = r.get("created_at")
            ts = created.isoformat() if hasattr(created, "isoformat") else str(created)
        clean.append({
            "child_id":  r.get("child_id", child_id),
            "grade":     5,
            "timestamp": ts,
            "total_correct":   r.get("total_correct",   0),
            "total_premature": r.get("total_premature", 0),
            "total_wrong":     r.get("total_wrong",     0),
            "total_hold_duration_ms": r.get("total_hold_duration_ms", 0),
            "computed_metrics": {
                "attention_label":            raw.get("attention_label",           ""),
                "overall_accuracy":           raw.get("overall_accuracy",          0.0),
                "impulsivity_ratio":          raw.get("impulsivity_ratio",         0.0),
                "inattention_score":          raw.get("inattention_score",         0.0),
                "rt_mean_ms":                 raw.get("rt_mean_ms"),
                "rt_cv":                      raw.get("rt_cv"),
                "switch_error_rate":          raw.get("switch_error_rate"),
                "motor_inhibition_score":     raw.get("motor_inhibition_score"),
                "sequential_completion_rate": raw.get("sequential_completion_rate"),
            },
        })
    return {"child_id": child_id, "total_sessions": len(clean), "history": clean}