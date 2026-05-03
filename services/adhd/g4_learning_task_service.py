"""
Grade 4 Learning Task Service
Same 5 task types as Grade 3 — difficulty parameters scaled for Grade 4 (9-10 years).
Task assignment logic identical to Grade 3 service.
"""

from datetime import datetime
from models.adhd.g4_learning_task_model import (
    G4LearningTaskAssignRequest, G4AssignedTask,
    G4LearningTaskAssignResponse, G4LearningTaskResult,
    G4LearningTaskResultResponse,
)
from services.db_service import get_db


# ── Grade 4 task definitions ──────────────────────────────────────────────────
# Instructions updated for Grade 4 age (9-10 years) with corrected Sinhala grammar.

TASK_DEFINITIONS = {
    "gonogo": {
        "name": "යන්න / නොයන්න",
        "target": "impulsivity",
        "instructions": {
            1: "ගමනාගමන ආලෝකය: කොළ පැහැය දුටු විට ස්පර්ශ කරන්න. රතු හෝ කහ පැහැයට ස්පර්ශ නොකරන්න. වාර 15ක්.",
            2: "ඉක්මනින් ප්‍රතිචාර දක්වන්න. රතු හෝ කහ පැහැයට ස්පර්ශ කිරීමෙන් වළකින්න. වාර 20ක්.",
            3: "වැඩි වේගයකින් ප්‍රතිචාර දක්වන්න. අවධානයෙන් සිට රතු සහ කහ ආලෝකයන් මඟ හරින්න. වාර 25ක්.",
        },
    },
    "wait_match": {
        "name": "බලා ගැලපීම",
        "target": "impulsivity",
        "instructions": {
            1: "රූපය දෙස තත්පර 2.5ක් බලා සිටින්න. ඉන්පසු දී ඇති රූප 3කින් නිවැරදි රූපය තෝරා ගලපන්න.",
            2: "රූපය දෙස තත්පර 2ක් බලා සිටින්න. ඉන්පසු නිවැරදි රූපය ඉක්මනින් තෝරා ගලපන්න.",
            3: "රූපය දෙස තත්පර 1ක් පමණක් බලා සිටින්න. ඉන්පසු ඉතා ඉක්මනින් නිවැරදි රූපය තෝරා ගලපන්න.",
        },
    },
    "audio_sequence": {
        "name": "කතාව අනුපිළිවෙල",
        "target": "inattention",
        "instructions": {
            1: "වාක්‍ය 3කින් යුත් කතාවට සවන් දෙන්න. රූප 3ක් නිවැරදි අනුපිළිවෙලට සකසන්න. වාර 2ක් නැවත සවන් දිය හැක.",
            2: "කතාවට සවන් දෙන්න. රූප 3ක් නිවැරදි අනුපිළිවෙලට සකසන්න. නැවත සවන් දිය හැක්කේ 1 වරක් පමණි.",
            3: "කතාවට අවධානයෙන් සවන් දෙන්න. රූප නිවැරදි අනුපිළිවෙලට සකසන්න. නැවත සවන් දීමට අවස්ථාවක් නොලැබේ.",
        },
    },
    "spot_change": {
        "name": "වෙනස සොයන්න",
        "target": "inattention",
        "instructions": {
            1: "රූප දෙක අතර ඇති වෙනස්කම් සොයා ස්පර්ශ කරන්න. තත්පර 8ක් ඇතුළත අවසන් කිරීමට උත්සාහ කරන්න.",
            2: "රූප අතර ඇති වෙනස්කම් ඉක්මනින් සොයන්න. තත්පර 6ක් ඇතුළත අවසන් කරන්න.",
            3: "ඉතා ඉක්මනින් වෙනස්කම් සොයන්න. තත්පර 5ක් ඇතුළත මෙය අවසන් කිරීමට උත්සාහ කරන්න.",
        },
    },
    "attention_grid": {
        "name": "අවධාන ජාලය",
        "target": "maintenance",
        "instructions": {
            1: "4x4 කොටු ජාලය තුළ ඇති ඉලක්කයන් සොයන්න. තත්පර 28ක් ඇතුළත අවසන් කරන්න.",
            2: "4x4 කොටු ජාලය තුළ ඇති ඉලක්කයන් ඉක්මනින් සොයන්න. තත්පර 23ක් ඇතුළත අවසන් කරන්න.",
            3: "5x5 කොටු ජාලය තුළ ඇති ඉලක්කයන් අවධානයෙන් සොයන්න. තත්පර 18ක් ඇතුළත අවසන් කරන්න.",
        },
    },
}


# ── Task selection (same logic as Grade 3) ───────────────────────────────────
def _select_tasks(imp: float, inat: float, acc: float) -> tuple:
    hi = imp  > 0.25
    ia = inat > 0.25
    la = acc  < 0.50
    if hi and ia:
        return ["gonogo", "audio_sequence", "wait_match"], "mixed"
    elif hi:
        return ["gonogo", "wait_match", "spot_change"], "impulsivity"
    elif ia:
        return ["audio_sequence", "spot_change", "attention_grid"], "inattention"
    elif la:
        return ["wait_match", "spot_change", "gonogo"], "accuracy"
    else:
        return ["attention_grid", "spot_change", "wait_match"], "maintenance"


# ── Difficulty logic ──────────────────────────────────────────────────────────
def _get_difficulty(child_id: str, task_id: str, grade: int) -> int:
    db   = get_db()
    past = list(
        db["learning_task_results"]
        .find({"child_id": child_id, "task_id": task_id, "grade": grade})
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


def _get_session_number(child_id: str, grade: int) -> int:
    db    = get_db()
    count = db["learning_task_results"].count_documents(
        {"child_id": child_id, "grade": grade}
    )
    return (count // 3) + 1


def _serialize(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])
    if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
        doc["timestamp"] = doc["created_at"].isoformat()
        del doc["created_at"]
    return doc


# ── Main service functions ────────────────────────────────────────────────────
def g4_assign_tasks(req: G4LearningTaskAssignRequest) -> G4LearningTaskAssignResponse:
    db     = get_db()
    latest = db["adhd_submissions"].find_one(
        {"child_id": req.child_id, "grade": 4},
        sort=[("created_at", -1)],
    )

    if latest and "metrics" in latest:
        m   = latest["metrics"]
        imp = m.get("impulsivity_ratio", 0.0)
        ia  = m.get("inattention_score", 0.0)
        acc = m.get("overall_accuracy",  1.0)
    else:
        imp, ia, acc = 0.1, 0.3, 0.6

    task_ids, dominant = _select_tasks(imp, ia, acc)
    session_num        = _get_session_number(req.child_id, 4)

    assigned = []
    for tid in task_ids:
        defn       = TASK_DEFINITIONS[tid]
        difficulty = _get_difficulty(req.child_id, tid, 4)
        assigned.append(G4AssignedTask(
            task_id        = tid,
            task_name      = defn["name"],
            difficulty     = difficulty,
            target_deficit = defn["target"],
            instructions   = defn["instructions"][difficulty],
        ))

    return G4LearningTaskAssignResponse(
        child_id         = req.child_id,
        grade            = 4,
        session_number   = session_num,
        tasks            = assigned,
        dominant_deficit = dominant,
        severity_scores  = {
            "impulsivity": round(imp, 3),
            "inattention": round(ia,  3),
            "accuracy":    round(acc, 3),
        },
    )


def g4_save_task_result(result: G4LearningTaskResult) -> G4LearningTaskResultResponse:
    db    = get_db()
    total = result.total_trials or 1
    score = round((result.correct / total) * 100, 1)

    past = list(
        db["learning_task_results"]
        .find({"child_id": result.child_id, "task_id": result.task_id, "grade": 4})
        .sort("session_number", -1)
        .limit(1)
    )
    next_diff = result.difficulty
    if past:
        if score >= 75 and past[0].get("score_percent", 0) >= 75 and result.difficulty < 3:
            next_diff = result.difficulty + 1
    
    if score >= 80:
        msg = "ඉතා හොඳයි! ඔබ ඉතා හොඳින් කළා! 🌟"
    elif score >= 60:
        msg = "හොඳයි! ඔබ හොඳින් කළා! ⭐"
    else:
        msg = "නැවත උත්සාහ කරමු! ඔබට මීට වඩා හොඳින් කළ හැකියි! 💪"

    avg_rt = (
        round(sum(result.response_times_ms) / len(result.response_times_ms))
        if result.response_times_ms else 0
    )
    now = datetime.utcnow()
    db["learning_task_results"].insert_one({
        "child_id":          result.child_id,
        "grade":             4,
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
        "timestamp":         now.isoformat(),
        "created_at":        now,
    })

    return G4LearningTaskResultResponse(
        ok=True, message="ප්‍රතිඵල සුරකින ලදී",
        score_percent=score, next_difficulty=next_diff, encouragement=msg,
    )


def g4_get_progress(child_id: str) -> dict:
    db      = get_db()
    results = list(
        db["learning_task_results"]
        .find({"child_id": child_id, "grade": 4})
        .sort([("created_at", -1), ("timestamp", -1)])
        .limit(20)
    )
    return {"child_id": child_id, "grade": 4, "sessions": [_serialize(r) for r in results]}