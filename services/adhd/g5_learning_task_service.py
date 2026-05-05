from datetime import datetime
from models.adhd.g5_learning_task_model import (
    G5LearningTaskAssignRequest, G5AssignedTask,
    G5LearningTaskAssignResponse, G5LearningTaskResult,
    G5LearningTaskResultResponse,
)
from services.db_service import get_db

TASK_DEFINITIONS = {
    "gonogo": {
        "name": "යන්න / නොයන්න",
        "target": "impulsivity",
        "instructions": {
            1: "සත්ව emoji = ස්පර්ශ කරන්න. නිශ්චිත සත්ව emoji = නොකරන්න. 20 ත්‍රිකෝණ.",
            2: "ඉක්මනින් ප්‍රතිචාර දක්වන්න. 25 ත්‍රිකෝණ. 1.5s window.",
            3: "ඉහළ වේගය. 30 ත්‍රිකෝණ. 1.5s window.",
        },
    },
    "wait_match": {
        "name": "බලා ගැලපීම",
        "target": "impulsivity",
        "instructions": {
            1: "රූපය 2s බලන්න. 3 options. 12 ත්‍රිකෝණ.",
            2: "රූපය 1.5s. 4 options. 14 ත්‍රිකෝණ.",
            3: "රූපය 1s. 4 options. 16 ත්‍රිකෝණ.",
        },
    },
    "audio_sequence": {
        "name": "කතාව අනුපිළිවෙල",
        "target": "inattention",
        "instructions": {
            1: "4 sentence කතාව. 2 plays. 4 items.",
            2: "4 sentence කතාව. 1 play. 4 items.",
            3: "4 sentence කතාව. 1 play. 4 items. Zero help.",
        },
    },
    "spot_change": {
        "name": "වෙනස සොයන්න",
        "target": "inattention",
        "instructions": {
            1: "2 changes. 7s limit.",
            2: "2 changes. 5s limit.",
            3: "2 changes. 4s limit.",
        },
    },
    "attention_grid": {
        "name": "අවධාන ජාලය",
        "target": "maintenance",
        "instructions": {
            1: "4x4 grid. 25s.",
            2: "5x5 grid. 20s.",
            3: "5x5 grid. 18s. ~25% targets.",
        },
    },
}


def _select_tasks(imp, inat, acc):
    hi = imp  > 0.25
    ia = inat > 0.25
    la = acc  < 0.50
    if hi and ia:  return ["gonogo","audio_sequence","wait_match"], "mixed"
    elif hi:       return ["gonogo","wait_match","spot_change"], "impulsivity"
    elif ia:       return ["audio_sequence","spot_change","attention_grid"], "inattention"
    elif la:       return ["wait_match","spot_change","gonogo"], "accuracy"
    else:          return ["attention_grid","spot_change","wait_match"], "maintenance"


def _get_difficulty(child_id, task_id, grade):
    db   = get_db()
    past = list(db["learning_task_results"]
                .find({"child_id":child_id,"task_id":task_id,"grade":grade})
                .sort("session_number",-1).limit(2))
    if len(past) < 2: return 1
    scores = [r["score_percent"] for r in past]
    diff   = past[0].get("difficulty", 1)
    if diff == 1 and all(s >= 60 for s in scores): return 2
    if diff == 2 and all(s >= 75 for s in scores): return 3
    return diff


def _get_session_number(child_id, grade):
    db = get_db()
    return (db["learning_task_results"].count_documents({"child_id":child_id,"grade":grade}) // 3) + 1


def _serialize(doc):
    doc["_id"] = str(doc["_id"])
    if "created_at" in doc and hasattr(doc["created_at"],"isoformat"):
        doc["timestamp"] = doc["created_at"].isoformat()
        del doc["created_at"]
    return doc


def g5_assign_tasks(req: G5LearningTaskAssignRequest) -> G5LearningTaskAssignResponse:
    db     = get_db()
    latest = db["adhd_submissions"].find_one({"child_id":req.child_id,"grade":5},
                                             sort=[("created_at",-1)])
    if latest and "metrics" in latest:
        m   = latest["metrics"]
        imp = m.get("impulsivity_ratio", 0.0)
        ia  = m.get("inattention_score", 0.0)
        acc = m.get("overall_accuracy",  1.0)
    else:
        imp, ia, acc = 0.1, 0.3, 0.6

    task_ids, dominant = _select_tasks(imp, ia, acc)
    session_num        = _get_session_number(req.child_id, 5)
    assigned = []
    for tid in task_ids:
        defn = TASK_DEFINITIONS[tid]
        diff = _get_difficulty(req.child_id, tid, 5)
        assigned.append(G5AssignedTask(
            task_id=tid, task_name=defn["name"], difficulty=diff,
            target_deficit=defn["target"], instructions=defn["instructions"][diff],
        ))
    return G5LearningTaskAssignResponse(
        child_id=req.child_id, grade=5, session_number=session_num,
        tasks=assigned, dominant_deficit=dominant,
        severity_scores={"impulsivity":round(imp,3),"inattention":round(ia,3),"accuracy":round(acc,3)},
    )


def g5_save_task_result(result: G5LearningTaskResult) -> G5LearningTaskResultResponse:
    db    = get_db()
    total = result.total_trials or 1
    score = round((result.correct / total) * 100, 1)
    past  = list(db["learning_task_results"]
                 .find({"child_id":result.child_id,"task_id":result.task_id,"grade":5})
                 .sort("session_number",-1).limit(1))
    next_diff = result.difficulty
    if past and score >= 75 and past[0].get("score_percent",0) >= 75 and result.difficulty < 3:
        next_diff = result.difficulty + 1
    msg = ("ඉතා හොඳයි! 🌟" if score >= 80 else "හොඳයි! ⭐" if score >= 60 else "නැවත උත්සාහ කරන්න! 💪")
    avg_rt = round(sum(result.response_times_ms)/len(result.response_times_ms)) if result.response_times_ms else 0
    now = datetime.utcnow()
    db["learning_task_results"].insert_one({
        "child_id":result.child_id,"grade":5,"task_id":result.task_id,
        "difficulty":result.difficulty,"correct":result.correct,"wrong":result.wrong,
        "premature":result.premature,"total_trials":result.total_trials,
        "score_percent":score,"avg_rt_ms":avg_rt,"response_times_ms":result.response_times_ms,
        "session_number":result.session_number,"next_difficulty":next_diff,
        "timestamp":now.isoformat(),"created_at":now,
    })
    return G5LearningTaskResultResponse(ok=True, message="ප්‍රතිඵල සුරකින ලදී",
                                        score_percent=score, next_difficulty=next_diff, encouragement=msg)


def g5_get_progress(child_id):
    db      = get_db()
    results = list(db["learning_task_results"].find({"child_id":child_id,"grade":5})
                   .sort([("created_at",-1),("timestamp",-1)]).limit(20))
    return {"child_id":child_id,"grade":5,"sessions":[_serialize(r) for r in results]}
