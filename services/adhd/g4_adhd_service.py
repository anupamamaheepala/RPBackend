"""
Grade 4 ADHD Service
Features: overall_accuracy, impulsivity_ratio, inattention_score,
          rt_mean_ms, rt_cv, rule_follow_score,
          sustained_completion_rate, replay_dependency
"""

import pickle
import statistics
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from models.adhd.g4_adhd_model import (
    G4ADHDSubmissionRequest, G4ComputedMetrics,
)
from services.db_service import get_db

# ── Load G4 model ─────────────────────────────────────────────────────────────
_PKL_PATH = Path(__file__).parent.parent.parent / "ml_models" / "adhd" / "g4_attention_classifier.pkl"

try:
    with open(_PKL_PATH, "rb") as f:
        _bundle = pickle.load(f)
    _model    = _bundle["model"]
    _le       = _bundle["label_encoder"]
    _features = _bundle["feature_cols"]
    print("✅ G4 attention classifier loaded")
except FileNotFoundError:
    _model = None
    _le    = None
    print("⚠️  g4_attention_classifier.pkl not found — using rule-based fallback")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cv(times: List[int]) -> Optional[float]:
    if len(times) < 2:
        return None
    mean = statistics.mean(times)
    return round(statistics.stdev(times) / mean, 4) if mean else None


def _mean(times: List[int]) -> Optional[float]:
    return round(statistics.mean(times), 2) if times else None


def _rule_based(impulsivity_ratio: float, inattention_score: float) -> str:
    hi = impulsivity_ratio >= 0.25
    ia = inattention_score >= 0.25
    if not hi and not ia:
        return "profile_a"
    elif ia and not hi:
        return "profile_b"
    elif hi and not ia:
        return "profile_c"
    return "profile_d"


# ── Main functions ────────────────────────────────────────────────────────────
def g4_compute_metrics(req: G4ADHDSubmissionRequest) -> G4ComputedMetrics:
    total   = req.total_correct + req.total_wrong + req.total_premature
    acc     = req.total_correct   / total if total else 0.0
    imp     = req.total_premature / total if total else 0.0
    inat    = req.total_wrong     / total if total else 0.0

    # Grade 4 specific features
    # rule_follow_score: inverse of help presses ratio (max ~10 presses assumed)
    rule_follow  = max(0.0, 1.0 - (req.total_rule_views / 10.0))

    # sustained_completion_rate: items completed out of expected 5 (Stay & Complete)
    sustained    = min(1.0, req.total_items_completed / 5.0)

    # replay_dependency: ratio of replays to total listen tasks (max ~5 assumed)
    replay_dep   = min(1.0, req.total_replay_count / 5.0)

    # RT features
    rt_mean, rt_cv_all = None, None
    rt_cv_t1 = rt_cv_t2 = rt_cv_t3 = rt_cv_t4 = 0.0
    rt_cv_per_task = None

    if req.task_response_times:
        all_times = (
            req.task_response_times.task1 +
            req.task_response_times.task2 +
            req.task_response_times.task3 +
            req.task_response_times.task4
        )
        rt_mean    = _mean(all_times)
        rt_cv_all  = _cv(all_times)
        rt_cv_t1   = _cv(req.task_response_times.task1) or 0.0
        rt_cv_t2   = _cv(req.task_response_times.task2) or 0.0
        rt_cv_t3   = _cv(req.task_response_times.task3) or 0.0
        rt_cv_t4   = _cv(req.task_response_times.task4) or 0.0
        rt_cv_per_task = {
            "task1": rt_cv_t1, "task2": rt_cv_t2,
            "task3": rt_cv_t3, "task4": rt_cv_t4,
        }

    # Profile classification
    if _model is not None:
        row = pd.DataFrame([{
            "overall_accuracy":          round(acc,         4),
            "impulsivity_ratio":         round(imp,         4),
            "inattention_score":         round(inat,        4),
            "rt_mean_ms":                rt_mean or 0.0,
            "rt_cv":                     rt_cv_all or 0.0,
            "rule_follow_score":         round(rule_follow, 4),
            "sustained_completion_rate": round(sustained,   4),
            "replay_dependency":         round(replay_dep,  4),
        }])
        label = _le.inverse_transform(_model.predict(row))[0]
    else:
        label = _rule_based(imp, inat)

    return G4ComputedMetrics(
        total_attempts              = total,
        overall_accuracy            = round(acc,         4),
        impulsivity_ratio           = round(imp,         4),
        inattention_score           = round(inat,        4),
        attention_label             = label,
        rt_mean_ms                  = rt_mean,
        rt_cv                       = rt_cv_all,
        rt_cv_per_task              = rt_cv_per_task,
        rule_follow_score           = round(rule_follow, 4),
        sustained_completion_rate   = round(sustained,   4),
        replay_dependency           = round(replay_dep,  4),
    )


def g4_save_assessment(req: G4ADHDSubmissionRequest,
                       metrics: G4ComputedMetrics) -> str:
    db  = get_db()
    doc = {
        "child_id":            req.child_id,
        "grade":               4,
        "total_correct":       req.total_correct,
        "total_premature":     req.total_premature,
        "total_wrong":         req.total_wrong,
        "total_rule_views":    req.total_rule_views,
        "total_items_completed": req.total_items_completed,
        "total_replay_count":  req.total_replay_count,
        "task_response_times": req.task_response_times.dict()
            if req.task_response_times else None,
        "metrics":             metrics.dict(),
        "client_timestamp":    req.timestamp,
        "created_at":          datetime.utcnow(),
    }
    result = db["adhd_submissions"].insert_one(doc)
    return str(result.inserted_id)


def g4_get_diagnostic_history(child_id: str) -> dict:
    db      = get_db()
    records = list(
        db["adhd_submissions"]
        .find({"child_id": child_id, "grade": 4})
        .sort("created_at", -1)
        .limit(10)
    )
    clean = []
    for r in records:
        raw = r.get("metrics", {})
        ts  = r.get("client_timestamp")
        if not ts:
            created = r.get("created_at")
            ts = created.isoformat() if hasattr(created, "isoformat") else str(created)
        clean.append({
            "child_id":        r.get("child_id", child_id),
            "grade":           4,
            "timestamp":       ts,
            "total_correct":   r.get("total_correct",   0),
            "total_premature": r.get("total_premature", 0),
            "total_wrong":     r.get("total_wrong",     0),
            "total_rule_views":      r.get("total_rule_views",      0),
            "total_items_completed": r.get("total_items_completed", 0),
            "total_replay_count":    r.get("total_replay_count",    0),
            "computed_metrics": {
                "attention_label":             raw.get("attention_label",           ""),
                "overall_accuracy":            raw.get("overall_accuracy",          0.0),
                "impulsivity_ratio":           raw.get("impulsivity_ratio",         0.0),
                "inattention_score":           raw.get("inattention_score",         0.0),
                "rt_mean_ms":                  raw.get("rt_mean_ms"),
                "rt_cv":                       raw.get("rt_cv"),
                "rule_follow_score":           raw.get("rule_follow_score"),
                "sustained_completion_rate":   raw.get("sustained_completion_rate"),
                "replay_dependency":           raw.get("replay_dependency"),
            },
        })
    return {"child_id": child_id, "total_sessions": len(clean), "history": clean}