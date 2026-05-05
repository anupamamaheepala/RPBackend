"""
Grade 6 ADHD Service — 8 ML features
Features: overall_accuracy, impulsivity_ratio, inattention_score,
          rt_mean_ms, rt_cv, interference_error_rate, nback_accuracy,
          visual_search_speed
"""
import pickle, statistics, pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from models.adhd.g6_adhd_model import G6ADHDSubmissionRequest, G6ComputedMetrics
from services.db_service import get_db

_PKL = Path(__file__).parent.parent.parent / "ml_models" / "adhd" / "g6_attention_classifier.pkl"
try:
    with open(_PKL, "rb") as f:
        _b = pickle.load(f)
    _model, _le, _features = _b["model"], _b["label_encoder"], _b["feature_cols"]
    print("✅ G6 classifier loaded")
except FileNotFoundError:
    _model = None; print("⚠️  g6_attention_classifier.pkl not found")


def _cv(times: List[int]) -> Optional[float]:
    if len(times) < 2: return None
    m = statistics.mean(times)
    return round(statistics.stdev(times) / m, 4) if m else None

def _mean(times: List[int]) -> Optional[float]:
    return round(statistics.mean(times), 2) if times else None

def _rule_based(imp: float, inat: float) -> str:
    hi = imp >= 0.25; ia = inat >= 0.25
    if not hi and not ia: return "profile_a"
    elif ia and not hi:   return "profile_b"
    elif hi and not ia:   return "profile_c"
    return "profile_d"


def g6_compute_metrics(req: G6ADHDSubmissionRequest) -> G6ComputedMetrics:
    total = req.total_correct + req.total_wrong + req.total_premature
    acc   = req.total_correct   / total if total else 0.0
    imp   = req.total_premature / total if total else 0.0
    inat  = req.total_wrong     / total if total else 0.0

    # ── Grade 6 specific features ─────────────────────────────────────────────
    # interference_error_rate: incongruent errors / incongruent trials (Stroop)
    ife = (req.stroop_incongruent_errors / req.stroop_incongruent_trials
           if req.stroop_incongruent_trials > 0 else 0.0)

    # nback_accuracy: correct / total N-Back trials
    nba = (req.nback_correct / req.nback_total_trials
           if req.nback_total_trials > 0 else 0.0)

    # visual_search_speed: 1 - normalized mean search time (faster = higher)
    avg_search = (_mean(req.search_times_ms) or 5000.0)
    vss = round(max(0.0, 1.0 - min(1.0, avg_search / 5000.0)), 4)

    # ── RT features ───────────────────────────────────────────────────────────
    rt_mean = rt_cv_all = None; rt_cv_per = None
    if req.task_response_times:
        all_rts = (req.task_response_times.task1 + req.task_response_times.task2 +
                   req.task_response_times.task3 + req.task_response_times.task4)
        rt_mean  = _mean(all_rts)
        rt_cv_all = _cv(all_rts)
        rt_cv_per = {
            "task1": _cv(req.task_response_times.task1),
            "task2": _cv(req.task_response_times.task2),
            "task3": _cv(req.task_response_times.task3),
            "task4": _cv(req.task_response_times.task4),
        }

    # ── Profile classification ─────────────────────────────────────────────────
    if _model is not None:
        row = pd.DataFrame([{
            "overall_accuracy":       round(acc,   4),
            "impulsivity_ratio":      round(imp,   4),
            "inattention_score":      round(inat,  4),
            "rt_mean_ms":             rt_mean or 0.0,
            "rt_cv":                  rt_cv_all or 0.0,
            "interference_error_rate":round(ife,   4),
            "nback_accuracy":         round(nba,   4),
            "visual_search_speed":    vss,
        }])
        label = _le.inverse_transform(_model.predict(row))[0]
    else:
        label = _rule_based(imp, inat)

    return G6ComputedMetrics(
        total_attempts          = total,
        overall_accuracy        = round(acc,  4),
        impulsivity_ratio       = round(imp,  4),
        inattention_score       = round(inat, 4),
        attention_label         = label,
        rt_mean_ms              = rt_mean,
        rt_cv                   = rt_cv_all,
        rt_cv_per_task          = rt_cv_per,
        interference_error_rate = round(ife, 4),
        nback_accuracy          = round(nba, 4),
        visual_search_speed     = vss,
    )


def g6_save_assessment(req: G6ADHDSubmissionRequest,
                        metrics: G6ComputedMetrics) -> str:
    db  = get_db()
    doc = {
        "child_id":          req.child_id,
        "grade":             6,
        "total_correct":     req.total_correct,
        "total_premature":   req.total_premature,
        "total_wrong":       req.total_wrong,
        # Task 1 — Stroop
        "stroop_congruent_trials":   req.stroop_congruent_trials,
        "stroop_incongruent_trials": req.stroop_incongruent_trials,
        "stroop_incongruent_errors": req.stroop_incongruent_errors,
        "stroop_total_errors":       req.stroop_total_errors,
        # Task 2 — N-Back
        "nback_correct":      req.nback_correct,
        "nback_false_alarms": req.nback_false_alarms,
        "nback_misses":       req.nback_misses,
        "nback_total_trials": req.nback_total_trials,
        # Task 3 — Visual Search
        "search_correct":  req.search_correct,
        "search_wrong":    req.search_wrong,
        "search_times_ms": req.search_times_ms,
        # Task 4 — Dual Go/No-Go
        "dual_correct":      req.dual_correct,
        "dual_false_alarms": req.dual_false_alarms,
        "dual_missed":       req.dual_missed,
        "dual_total_trials": req.dual_total_trials,
        # Task 5 — Sustained Counting
        "counting_actual": req.counting_actual,
        "counting_user":   req.counting_user,
        "task_response_times": req.task_response_times.dict()
            if req.task_response_times else None,
        "metrics":          metrics.dict(),
        "client_timestamp": req.timestamp,
        "created_at":       datetime.utcnow(),
    }
    return str(db["adhd_submissions"].insert_one(doc).inserted_id)


def g6_get_history(child_id: str) -> dict:
    db  = get_db()
    recs = list(db["adhd_submissions"]
                .find({"child_id": child_id, "grade": 6})
                .sort("created_at", -1).limit(10))
    clean = []
    for r in recs:
        raw = r.get("metrics", {})
        ts  = r.get("client_timestamp")
        if not ts:
            c = r.get("created_at")
            ts = c.isoformat() if hasattr(c, "isoformat") else str(c)
        clean.append({
            "child_id":  r.get("child_id", child_id),
            "grade":     6,
            "timestamp": ts,
            "total_correct":   r.get("total_correct",   0),
            "total_premature": r.get("total_premature", 0),
            "total_wrong":     r.get("total_wrong",     0),
            # raw G6 counts for progress page
            "stroop_incongruent_errors": r.get("stroop_incongruent_errors", 0),
            "nback_correct":             r.get("nback_correct", 0),
            "search_correct":            r.get("search_correct", 0),
            "dual_false_alarms":         r.get("dual_false_alarms", 0),
            "counting_actual":           r.get("counting_actual", 0),
            "counting_user":             r.get("counting_user", 0),
            "computed_metrics": {
                "attention_label":         raw.get("attention_label", ""),
                "overall_accuracy":        raw.get("overall_accuracy", 0.0),
                "impulsivity_ratio":       raw.get("impulsivity_ratio", 0.0),
                "inattention_score":       raw.get("inattention_score", 0.0),
                "rt_mean_ms":              raw.get("rt_mean_ms"),
                "rt_cv":                   raw.get("rt_cv"),
                "interference_error_rate": raw.get("interference_error_rate"),
                "nback_accuracy":          raw.get("nback_accuracy"),
                "visual_search_speed":     raw.get("visual_search_speed"),
            },
        })
    return {"child_id": child_id, "total_sessions": len(clean), "history": clean}
