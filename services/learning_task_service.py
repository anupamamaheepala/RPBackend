from datetime import datetime
from models.learning_task_model import (
    LearningTaskAssignRequest, AssignedTask,
    LearningTaskAssignResponse, LearningTaskResult,
    LearningTaskResultResponse,
)
from services.db_service import get_db


# ── Task definitions ──────────────────────────────────────────────────────────

TASK_DEFINITIONS = {
    "gonogo": {
        "name": "යන්න / නොයන්න",
        "target": "impulsivity",
        "instructions": {
            1: "සතෙකු ශබ්දය ඇසෙන විට ස්පර්ශ කරන්න. බළලාගේ ශබ්දය ඇසෙන විට ස්පර්ශ නොකරන්න.",
            2: "සතෙකු ශබ්දය ඇසෙන විට ස්පර්ශ කරන්න. බළලා හෝ බල්ලාගේ ශබ්දය ඇසෙන විට ස්පර්ශ නොකරන්න.",
            3: "ශබ්දයට ඉක්මනින් ප්‍රතිචාර දක්වන්න. නමුත් 'නැවතුම' ශබ්දයට නොකරන්න.",
        },
    },
    "wait_match": {
        "name": "බලා ගැලපීම",
        "target": "impulsivity",
        "instructions": {
            1: "රූපය 3 තත්පර බලන්න. එය නැති වූ පසු ගැලපෙන රූපය ස්පර්ශ කරන්න.",
            2: "රූපය 2 තත්පර බලන්න. 'ස්පර්ශ කරන්න' සංඥාව දිස්වෙන තෙක් බලා සිටින්න.",
            3: "රූපය 1 තත්පරයක් පමණ. ඉක්මනින් ස්පර්ශ කළොත් ලකුණු අඩු වේ.",
        },
    },
    "audio_sequence": {
        "name": "කතාව අනුපිළිවෙල",
        "target": "inattention",
        "instructions": {
            1: "කෙටි කතාවට සවන් දෙන්න. රූප 3 නිවැරදි අනුපිළිවෙලට සකසන්න.",
            2: "කතාව 1 වරක් ශ්‍රවණය කළ හැක. රූප 3 නිවැරදිව සකසන්න.",
            3: "කතාවට 1 වරක් සවන් දෙන්න. සටහනක් නොගෙන රූප 4 සකසන්න.",
        },
    },
    "spot_change": {
        "name": "වෙනස සොයන්න",
        "target": "inattention",
        "instructions": {
            1: "රූප 2ක් සසඳා බලන්න. වෙනස් දෙය ස්පර්ශ කරන්න. ඕනෑ තරම් කාලය ගන්න.",
            2: "රූප 2ක් 5 තත්පර සසඳා වෙනස් දෙය ස්පර්ශ කරන්න.",
            3: "රූප 3ක සසඳා වෙනස් 2 සොයා ස්පර්ශ කරන්න.",
        },
    },
    "attention_grid": {
        "name": "අවධාන ජාලය",
        "target": "maintenance",
        "instructions": {
            1: "3x3 ජාලයෙන් ඉලක්ක සංකේතය සොයා ස්පර්ශ කරන්න. කාලය 30 තත්පර.",
            2: "4x4 ජාලයෙන් ඉලක්ක සංකේත සොයා ස්පර්ශ කරන්න. කාලය 25 තත්පර.",
            3: "5x5 ජාලයෙන් ඉලක්ක සංකේත ඉක්මනින් සොයන්න. කාලය 20 තත්පර.",
        },
    },
}


# ── Deficit-to-task mapping ───────────────────────────────────────────────────

def _select_tasks(impulsivity: float, inattention: float,
                  accuracy: float) -> tuple[list, str]:
    """
    Return (task_id_list, dominant_deficit).
    Always returns minimum 3 tasks so the child has enough work per session.
    Primary tasks are chosen by deficit profile; a third task is added if needed.
    """
    high_imp  = impulsivity > 0.25
    high_inat = inattention > 0.25
    low_acc   = accuracy    < 0.50

    if high_imp and high_inat:
        # Both deficits — cover impulsivity, inattention, plus accuracy check
        return ["gonogo", "audio_sequence", "wait_match"], "mixed"
    elif high_imp:
        # Impulsivity dominant — add spot_change as third (sustained attention)
        return ["gonogo", "wait_match", "spot_change"], "impulsivity"
    elif high_inat:
        # Inattention dominant — add attention_grid as third (maintenance)
        return ["audio_sequence", "spot_change", "attention_grid"], "inattention"
    elif low_acc:
        # Low accuracy — add gonogo to build response control
        return ["wait_match", "spot_change", "gonogo"], "accuracy"
    else:
        # Profile A / all normal — three maintenance tasks
        return ["attention_grid", "spot_change", "wait_match"], "maintenance"


# ── Difficulty logic ──────────────────────────────────────────────────────────

def _get_difficulty(child_id: str, task_id: str, session_number: int) -> int:
    """
    Level 1 by default.
    Level 2 if last 2 sessions scored >= 60%.
    Level 3 if last 2 sessions at level 2 scored >= 75%.
    """
    db = get_db()
    past = list(
        db["learning_task_results"]
        .find({"child_id": child_id, "task_id": task_id})
        .sort("session_number", -1)
        .limit(2)
    )

    if len(past) < 2:
        return 1

    scores       = [r["score_percent"] for r in past]
    current_diff = past[0].get("difficulty", 1)

    if current_diff == 1 and all(s >= 60 for s in scores):
        return 2
    elif current_diff == 2 and all(s >= 75 for s in scores):
        return 3
    return current_diff


# ── Session number ────────────────────────────────────────────────────────────

def _get_session_number(child_id: str) -> int:
    db    = get_db()
    count = db["learning_task_results"].count_documents({"child_id": child_id})
    return (count // 2) + 1


# ── Serializer ────────────────────────────────────────────────────────────────

def _serialize(doc: dict) -> dict:
    """Convert ObjectId and datetime fields so JSON serialization never crashes."""
    doc["_id"] = str(doc["_id"])
    if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
        # Keep as "timestamp" string — Flutter reads this key
        doc["timestamp"] = doc["created_at"].isoformat()
        del doc["created_at"]
    return doc


# ── Main service functions ────────────────────────────────────────────────────

def assign_tasks(req: LearningTaskAssignRequest) -> LearningTaskAssignResponse:
    db = get_db()

    # adhd_service.py stores under "metrics" not "computed_metrics"
    # server time is stored in "created_at" — use for correct sort
    latest = db["adhd_submissions"].find_one(
        {"child_id": req.child_id},
        sort=[("created_at", -1)],
    )

    if latest and "metrics" in latest:
        m           = latest["metrics"]
        impulsivity = m.get("impulsivity_ratio", 0.0)
        inattention = m.get("inattention_score", 0.0)
        accuracy    = m.get("overall_accuracy",  1.0)
    else:
        # No diagnostic yet — default to inattention tasks
        impulsivity, inattention, accuracy = 0.1, 0.3, 0.6

    task_ids, dominant = _select_tasks(impulsivity, inattention, accuracy)
    session_number     = _get_session_number(req.child_id)

    assigned = []
    for tid in task_ids:
        defn       = TASK_DEFINITIONS[tid]
        difficulty = _get_difficulty(req.child_id, tid, session_number)
        assigned.append(
            AssignedTask(
                task_id        = tid,
                task_name      = defn["name"],
                difficulty     = difficulty,
                target_deficit = defn["target"],
                instructions   = defn["instructions"][difficulty],
            )
        )

    return LearningTaskAssignResponse(
        child_id         = req.child_id,
        grade            = req.grade,
        session_number   = session_number,
        tasks            = assigned,
        dominant_deficit = dominant,
        severity_scores  = {
            "impulsivity": round(impulsivity, 3),
            "inattention": round(inattention, 3),
            "accuracy":    round(accuracy,    3),
        },
    )


def save_task_result(result: LearningTaskResult) -> LearningTaskResultResponse:
    db    = get_db()
    total = result.total_trials or 1
    score = round((result.correct / total) * 100, 1)

    # Determine next difficulty
    past = list(
        db["learning_task_results"]
        .find({"child_id": result.child_id, "task_id": result.task_id})
        .sort("session_number", -1)
        .limit(1)
    )
    current_diff = result.difficulty
    next_diff    = current_diff

    if past:
        prev_score = past[0].get("score_percent", 0)
        if score >= 75 and prev_score >= 75 and current_diff < 3:
            next_diff = current_diff + 1
    elif score >= 75 and current_diff < 3:
        next_diff = current_diff   # need 2 good sessions first

    # Encouragement message
    if score >= 80:
        msg = "ඉතා හොඳයි! ඔබ ඉතා හොඳින් කළා! 🌟"
    elif score >= 60:
        msg = "හොඳයි! ඔබ හොඳින් කළා! ⭐"
    else:
        msg = "ගොඩක් හොඳයි! නැවත උත්සාහ කරන්න! 💪"

    avg_rt = (
        round(sum(result.response_times_ms) / len(result.response_times_ms))
        if result.response_times_ms else 0
    )

    now = datetime.utcnow()

    db["learning_task_results"].insert_one({
        "child_id":          result.child_id,
        "grade":             result.grade,
        "task_id":           result.task_id,
        "difficulty":        result.difficulty,
        "correct":           result.correct,
        "wrong":             result.wrong,
        "premature":         result.premature,
        "total_trials":      result.total_trials,
        "score_percent":     score,
        "avg_rt_ms":         avg_rt,
        "response_times_ms": result.response_times_ms,
        "session_number":    result.session_number,
        "next_difficulty":   next_diff,
        "timestamp":         now.isoformat(),  # string — Flutter reads this
        "created_at":        now,              # datetime — for DB sorting
    })

    return LearningTaskResultResponse(
        ok              = True,
        message         = "ප්‍රතිඵල සුරකින ලදී",
        score_percent   = score,
        next_difficulty = next_diff,
        encouragement   = msg,
    )


def get_progress(child_id: str) -> dict:
    db      = get_db()
    results = list(
        db["learning_task_results"]
        .find({"child_id": child_id})
        # FIX 5: fallback sort on "timestamp" for old docs that pre-date "created_at" field
        .sort([("created_at", -1), ("timestamp", -1)])
        .limit(20)
    )
    # serialize all docs — converts _id + created_at safely
    clean = [_serialize(r) for r in results]
    return {"child_id": child_id, "sessions": clean}