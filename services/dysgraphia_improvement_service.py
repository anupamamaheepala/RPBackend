# services/dysgraphia_improvement_service.py
# Handles DB reads/writes for dysgraphia improvement sessions.
# Returns the exact JSON shape the Flutter dashboard expects.

from services.db_service import get_db
from datetime import datetime, timezone, timedelta
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────────────────────────────────────

def save_improvement_session(submission_data) -> Dict[str, Any]:
    """
    Insert a single activity session into MongoDB.
    Calculates score_percent before saving.
    """
    db         = get_db()
    collection = db["dysgraphia_improvement_sessions"]

    try:
        # Exclude any computed fields (e.g. accuracy) and let the server
        # always set created_at — never trust the client clock.
        doc     = submission_data.dict(exclude={"accuracy", "created_at"})
        total   = doc.get("total_items",   0)
        correct = doc.get("correct_count", 0)

        doc["score_percent"] = round((correct / total * 100), 1) if total > 0 else 0.0
        doc["created_at"]    = datetime.now(timezone.utc)

        result = collection.insert_one(doc)
        return {
            "ok":            True,
            "session_id":    str(result.inserted_id),
            "score_percent": doc["score_percent"],
            "message":       "Improvement session saved successfully",
        }

    except Exception as e:
        return {"ok": False, "error": f"Failed to save improvement session: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# READ
# ─────────────────────────────────────────────────────────────────────────────

def get_user_improvement_results(user_id: str) -> Dict[str, Any]:
    """
    Returns all sessions for a user (newest first) plus a rich summary block
    that matches the Flutter DysgraphiaDashboardSummary model exactly.

    Summary fields returned:
      latest_risk_level, avg_accuracy, this_month_accuracy, last_month_accuracy,
      latest_accuracy, latest_activity_label, latest_duration,
      current_streak, week_practiced, total_duration_seconds,
      activity_bests  (dict[name → ActivityBest object])

    Sessions list fields returned (per session):
      user_id, grade, risk_level, activity_name, activity_label,
      total_items, correct_count, score_percent, duration_seconds, created_at
    """
    db         = get_db()
    collection = db["dysgraphia_improvement_sessions"]

    sessions_raw = list(
        collection.find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("created_at", -1)   # newest first
    )

    # ── Empty state ────────────────────────────────────────────────────────────
    if not sessions_raw:
        return {
            "ok":             True,
            "user_id":        user_id,
            "total_sessions": 0,
            "summary":        None,
            "sessions":       [],
        }

    # ── Normalise sessions ─────────────────────────────────────────────────────
    sessions = []
    for s in sessions_raw:
        created_at = s.get("created_at")
        # Make timezone-aware if naive
        if isinstance(created_at, datetime) and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        sessions.append({
            "user_id":          s.get("user_id"),
            "grade":            s.get("grade"),
            "risk_level":       s.get("risk_level", "medium"),
            "activity_name":    s.get("activity_name", ""),
            "activity_label":   s.get("activity_label", ""),
            "total_items":      s.get("total_items", 0),
            "correct_count":    s.get("correct_count", 0),
            "score_percent":    s.get("score_percent", 0.0),
            "duration_seconds": s.get("duration_seconds"),
            "created_at":       created_at.isoformat() if created_at else None,
            "_created_at_dt":   created_at,          # kept for internal calcs only
        })

    # ── Accuracy helpers ───────────────────────────────────────────────────────
    scores = [s["score_percent"] for s in sessions]
    avg_accuracy = round(sum(scores) / len(scores), 1)

    # Month-over-month
    now        = datetime.now(timezone.utc)
    this_month = now.month
    this_year  = now.year
    last_month = (now.replace(day=1) - timedelta(days=1))  # last day of prev month

    this_month_scores = [
        s["score_percent"] for s in sessions
        if s["_created_at_dt"]
        and s["_created_at_dt"].month == this_month
        and s["_created_at_dt"].year  == this_year
    ]
    last_month_scores = [
        s["score_percent"] for s in sessions
        if s["_created_at_dt"]
        and s["_created_at_dt"].month == last_month.month
        and s["_created_at_dt"].year  == last_month.year
    ]

    this_month_accuracy = round(sum(this_month_scores) / len(this_month_scores), 1) if this_month_scores else 0.0
    last_month_accuracy = round(sum(last_month_scores) / len(last_month_scores), 1) if last_month_scores else 0.0

    # ── Total practice time ────────────────────────────────────────────────────
    total_duration = sum(
        s["duration_seconds"] for s in sessions if s["duration_seconds"] is not None
    )

    # ── Latest session fields ──────────────────────────────────────────────────
    latest          = sessions[0]     # newest first
    latest_accuracy = latest["score_percent"]
    latest_activity = latest["activity_label"]
    latest_duration = latest["duration_seconds"]
    latest_risk     = latest["risk_level"]

    # ── Streak calculation ─────────────────────────────────────────────────────
    # Count consecutive calendar days (backwards from today) with ≥1 session
    practiced_dates = set()
    for s in sessions:
        dt = s["_created_at_dt"]
        if dt:
            practiced_dates.add(dt.date())

    streak = 0
    check_date = now.date()
    while check_date in practiced_dates:
        streak    += 1
        check_date -= timedelta(days=1)

    # ── Current week (Mon–Sun) practice flags ─────────────────────────────────
    # weekday(): Mon=0 … Sun=6
    week_start = now.date() - timedelta(days=now.weekday())  # Monday of current week
    week_practiced = [
        (week_start + timedelta(days=i)) in practiced_dates
        for i in range(7)
    ]

    # ── Per-activity breakdown ─────────────────────────────────────────────────
    activity_map: Dict[str, dict] = {}

    for s in sessions:
        name  = s["activity_name"]
        label = s["activity_label"]
        score = s["score_percent"]
        dt    = s["_created_at_dt"]

        if name not in activity_map:
            activity_map[name] = {
                "activity_name":  name,
                "activity_label": label,
                "scores":         [],
                "last_played_at": None,
            }

        activity_map[name]["scores"].append(score)

        # Track most recent session date per activity
        if dt and (
            activity_map[name]["last_played_at"] is None
            or dt > activity_map[name]["last_played_at"]
        ):
            activity_map[name]["last_played_at"] = dt

    # Build ActivityBest objects (as plain dicts — serialised to JSON by FastAPI)
    activity_bests = {}
    for name, data in activity_map.items():
        sc = data["scores"]
        last_played = data["last_played_at"]
        activity_bests[name] = {
            "activity_name":  name,
            "activity_label": data["activity_label"],
            "best_accuracy":  round(max(sc), 1),
            "avg_accuracy":   round(sum(sc) / len(sc), 1),
            "session_count":  len(sc),
            "last_played_at": last_played.isoformat() if last_played else None,
        }

    # ── Risk level counts (kept for backwards compat) ──────────────────────────
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    for s in sessions:
        lvl = s.get("risk_level", "")
        if lvl in risk_counts:
            risk_counts[lvl] += 1

    # ── Strip internal _created_at_dt before returning sessions ───────────────
    clean_sessions = [{k: v for k, v in s.items() if k != "_created_at_dt"} for s in sessions]

    # ── Final response ─────────────────────────────────────────────────────────
    return {
        "ok":             True,
        "user_id":        user_id,
        "total_sessions": len(sessions),
        "summary": {
            # Risk
            "latest_risk_level":     latest_risk,

            # Accuracy
            "avg_accuracy":          avg_accuracy,
            "this_month_accuracy":   this_month_accuracy,
            "last_month_accuracy":   last_month_accuracy,
            "latest_accuracy":       latest_accuracy,
            "latest_activity_label": latest_activity,
            "latest_duration":       latest_duration,

            # Time
            "total_duration_seconds": total_duration,

            # Streak
            "current_streak":  streak,
            "week_practiced":  week_practiced,

            # Per-activity
            "activity_bests": activity_bests,

            # Kept for backwards compat
            "risk_counts": risk_counts,
        },
        "sessions": clean_sessions,
    }