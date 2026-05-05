"""
Grade 7 ADHD Service — 10 ML features
Features: overall_accuracy, impulsivity_ratio, inattention_score,
          rt_mean_ms, rt_cv, switch_error_rate, divided_attention_score,
          distraction_resistance, vigilance_hit_rate, false_alarm_rate
"""
import pickle, statistics, pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from models.adhd.g7_adhd_model import G7ADHDSubmissionRequest, G7ComputedMetrics
from services.db_service import get_db

_PKL_PATH = Path(__file__).parent.parent.parent / "ml_models" / "adhd" / "g7_attention_classifier.pkl"
try:
    with open(_PKL_PATH, "rb") as f:
        _bundle = pickle.load(f)
    _model    = _bundle["model"]
    _le       = _bundle["label_encoder"]
    _features = _bundle["feature_cols"]
    print("✅ G7 attention classifier loaded")
except FileNotFoundError:
    _model = None; _le = None
    print("⚠️  g7_attention_classifier.pkl not found — rule-based fallback")


def _cv(times: List[int]) -> Optional[float]:
    if len(times) < 2: return None
    mean = statistics.mean(times)
    return round(statistics.stdev(times) / mean, 4) if mean else None

def _mean(times: List[int]) -> Optional[float]:
    return round(statistics.mean(times), 2) if times else None

def _rule_based(imp: float, inat: float) -> str:
    hi = imp  >= 0.25; ia = inat >= 0.25
    if not hi and not ia: return "profile_a"
    elif ia and not hi:   return "profile_b"
    elif hi and not ia:   return "profile_c"
    return "profile_d"


def g7_compute_metrics(req: G7ADHDSubmissionRequest) -> G7ComputedMetrics:
    total = req.total_correct + req.total_wrong + req.total_premature
    acc   = req.total_correct   / total if total else 0.0
    imp   = req.total_premature / total if total else 0.0
    inat  = req.total_wrong     / total if total else 0.0

    # ── Grade 7 specific features ─────────────────────────────────────────
    # vigilance_hit_rate: correct hits / targets shown (Task 1)
    vig_hit = (req.vigilance_correct_hits / req.vigilance_targets_shown
               if req.vigilance_targets_shown > 0 else 0.0)

    # switch_error_rate: switch-specific errors / switching trials (Task 3)
    sw_err = (req.switching_switch_errors / req.switching_total_trials
              if req.switching_total_trials > 0 else 0.0)

    # divided_attention_score: visual hits / visual targets shown (Task 4)
    div_att = (req.divided_visual_hits / req.divided_visual_targets_shown
               if req.divided_visual_targets_shown > 0 else 0.0)

    # distraction_resistance: correct / (correct + distraction_taps) (Task 6)
    dist_total = req.distraction_correct_taps + req.distraction_taps
    dist_res = (req.distraction_correct_taps / dist_total
                if dist_total > 0 else 1.0)

    # false_alarm_rate: all false alarms across tasks / total trials
    all_false = (req.vigilance_false_alarms +
                 req.divided_visual_false_alarms +
                 req.inhibition_false_alarms)
    all_trials = (req.vigilance_total_trials +
                  req.switching_total_trials +
                  req.inhibition_total_trials)
    fa_rate = (all_false / all_trials if all_trials > 0 else 0.0)

    # ── RT features ───────────────────────────────────────────────────────
    rt_mean = rt_cv_all = None
    rt_cv_t1 = rt_cv_t3 = rt_cv_t5 = 0.0
    rt_cv_per_task = None
    if req.task_response_times:
        all_times = (req.task_response_times.task1 +
                     req.task_response_times.task3 +
                     req.task_response_times.task5)
        rt_mean    = _mean(all_times)
        rt_cv_all  = _cv(all_times)
        rt_cv_t1   = _cv(req.task_response_times.task1) or 0.0
        rt_cv_t3   = _cv(req.task_response_times.task3) or 0.0
        rt_cv_t5   = _cv(req.task_response_times.task5) or 0.0
        rt_cv_per_task = {"task1": rt_cv_t1, "task3": rt_cv_t3, "task5": rt_cv_t5}

    # ── Profile classification ────────────────────────────────────────────
    if _model is not None:
        row = pd.DataFrame([{
            "overall_accuracy":        round(acc,      4),
            "impulsivity_ratio":       round(imp,      4),
            "inattention_score":       round(inat,     4),
            "rt_mean_ms":              rt_mean or 0.0,
            "rt_cv":                   rt_cv_all or 0.0,
            "switch_error_rate":       round(sw_err,   4),
            "divided_attention_score": round(div_att,  4),
            "distraction_resistance":  round(dist_res, 4),
            "vigilance_hit_rate":      round(vig_hit,  4),
            "false_alarm_rate":        round(fa_rate,  4),
        }])
        label = _le.inverse_transform(_model.predict(row))[0]
    else:
        label = _rule_based(imp, inat)

    return G7ComputedMetrics(
        total_attempts          = total,
        overall_accuracy        = round(acc,      4),
        impulsivity_ratio       = round(imp,      4),
        inattention_score       = round(inat,     4),
        attention_label         = label,
        rt_mean_ms              = rt_mean,
        rt_cv                   = rt_cv_all,
        rt_cv_per_task          = rt_cv_per_task,
        switch_error_rate       = round(sw_err,   4),
        divided_attention_score = round(div_att,  4),
        distraction_resistance  = round(dist_res, 4),
        vigilance_hit_rate      = round(vig_hit,  4),
        false_alarm_rate        = round(fa_rate,  4),
    )


def g7_save_assessment(req: G7ADHDSubmissionRequest,
                       metrics: G7ComputedMetrics) -> str:
    db  = get_db()
    doc = {
        "child_id":          req.child_id,
        "grade":             7,
        "total_correct":     req.total_correct,
        "total_premature":   req.total_premature,
        "total_wrong":       req.total_wrong,
        # Task 1
        "vigilance_correct_hits":  req.vigilance_correct_hits,
        "vigilance_misses":        req.vigilance_misses,
        "vigilance_false_alarms":  req.vigilance_false_alarms,
        "vigilance_targets_shown": req.vigilance_targets_shown,
        # Task 3
        "switching_correct":       req.switching_correct,
        "switching_errors":        req.switching_errors,
        "switching_switch_errors": req.switching_switch_errors,
        # Task 4
        "divided_visual_hits":          req.divided_visual_hits,
        "divided_visual_false_alarms":  req.divided_visual_false_alarms,
        "divided_visual_targets_shown": req.divided_visual_targets_shown,
        "divided_beep_count":           req.divided_beep_count,
        # Task 5
        "inhibition_correct_go":   req.inhibition_correct_go,
        "inhibition_false_alarms": req.inhibition_false_alarms,
        # Task 6
        "distraction_correct_taps": req.distraction_correct_taps,
        "distraction_taps":         req.distraction_taps,
        "distraction_misses":       req.distraction_misses,
        "task_response_times":      req.task_response_times.dict()
            if req.task_response_times else None,
        "metrics":             metrics.dict(),
        "client_timestamp":    req.timestamp,
        "created_at":          datetime.utcnow(),
    }
    return str(db["adhd_submissions"].insert_one(doc).inserted_id)


def g7_get_diagnostic_history(child_id: str) -> dict:
    db      = get_db()
    records = list(db["adhd_submissions"]
                   .find({"child_id": child_id, "grade": 7})
                   .sort("created_at", -1).limit(10))
    clean = []
    for r in records:
        raw = r.get("metrics", {})
        ts  = r.get("client_timestamp")
        if not ts:
            created = r.get("created_at")
            ts = created.isoformat() if hasattr(created, "isoformat") else str(created)
        clean.append({
            "child_id":  r.get("child_id", child_id),
            "grade":     7,
            "timestamp": ts,
            "total_correct":   r.get("total_correct",   0),
            "total_premature": r.get("total_premature", 0),
            "total_wrong":     r.get("total_wrong",     0),
            "vigilance_correct_hits":  r.get("vigilance_correct_hits", 0),
            "vigilance_false_alarms":  r.get("vigilance_false_alarms", 0),
            "switching_switch_errors": r.get("switching_switch_errors", 0),
            "divided_visual_hits":     r.get("divided_visual_hits", 0),
            "distraction_correct_taps": r.get("distraction_correct_taps", 0),
            "distraction_taps":         r.get("distraction_taps", 0),
            "inhibition_false_alarms":  r.get("inhibition_false_alarms", 0),
            "computed_metrics": {
                "attention_label":         raw.get("attention_label",         ""),
                "overall_accuracy":        raw.get("overall_accuracy",        0.0),
                "impulsivity_ratio":       raw.get("impulsivity_ratio",       0.0),
                "inattention_score":       raw.get("inattention_score",       0.0),
                "rt_mean_ms":              raw.get("rt_mean_ms"),
                "rt_cv":                   raw.get("rt_cv"),
                "switch_error_rate":       raw.get("switch_error_rate"),
                "divided_attention_score": raw.get("divided_attention_score"),
                "distraction_resistance":  raw.get("distraction_resistance"),
                "vigilance_hit_rate":      raw.get("vigilance_hit_rate"),
                "false_alarm_rate":        raw.get("false_alarm_rate"),
            },
        })
    return {"child_id": child_id, "total_sessions": len(clean), "history": clean}
