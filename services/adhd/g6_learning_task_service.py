from datetime import datetime
from models.adhd.g6_learning_task_model import (
    G6LearningTaskAssignRequest, G6AssignedTask,
    G6LearningTaskAssignResponse, G6LearningTaskResult,
    G6LearningTaskResultResponse,
)
from services.db_service import get_db

TASK_DEFINITIONS = {
    "gonogo":         {"name": "යන්න / නොයන්න",   "target": "impulsivity",
        "instructions": {1:"10 trials, 2s.", 2:"14 trials, 1.8s.", 3:"18 trials, 1.5s."}},
    "wait_match":     {"name": "බලා ගැලපීම",        "target": "impulsivity",
        "instructions": {1:"8 trials, 3s show.", 2:"10 trials, 2s.", 3:"12 trials, 1s."}},
    "audio_sequence": {"name": "කතාව අනුපිළිවෙල",  "target": "inattention",
        "instructions": {1:"2 sentences, 1 play.", 2:"3 sentences.", 3:"4 sentences."}},
    "spot_change":    {"name": "වෙනස සොයන්න",       "target": "inattention",
        "instructions": {1:"5 scenes, 8s.", 2:"8 scenes, 6s.", 3:"12 scenes, 5s."}},
    "attention_grid": {"name": "අවධාන ජාලය",       "target": "maintenance",
        "instructions": {1:"4x4, 28s.", 2:"5x5, 23s.", 3:"5x5, 18s."}},
}


def _select(imp, inat, acc):
    hi = imp > 0.25; ia = inat > 0.25; la = acc < 0.50
    if hi and ia:  return ["gonogo","audio_sequence","wait_match"], "mixed"
    elif hi:       return ["gonogo","wait_match","spot_change"], "impulsivity"
    elif ia:       return ["audio_sequence","spot_change","attention_grid"], "inattention"
    elif la:       return ["wait_match","spot_change","gonogo"], "accuracy"
    else:          return ["attention_grid","spot_change","wait_match"], "maintenance"


def _difficulty(child_id, task_id):
    db   = get_db()
    past = list(db["learning_task_results"]
                .find({"child_id":child_id,"task_id":task_id,"grade":6})
                .sort("session_number",-1).limit(2))
    if len(past) < 2: return 1
    d = past[0].get("difficulty", 1)
    s = [r["score_percent"] for r in past]
    if d == 1 and all(x >= 60 for x in s): return 2
    if d == 2 and all(x >= 75 for x in s): return 3
    return d


def _session_number(child_id):
    db = get_db()
    return (db["learning_task_results"].count_documents(
        {"child_id": child_id, "grade": 6}) // 3) + 1


def g6_assign_tasks(req: G6LearningTaskAssignRequest) -> G6LearningTaskAssignResponse:
    db  = get_db()
    lat = db["adhd_submissions"].find_one(
        {"child_id": req.child_id, "grade": 6}, sort=[("created_at", -1)])
    if lat and "metrics" in lat:
        m   = lat["metrics"]
        imp = m.get("impulsivity_ratio", 0.0)
        ia  = m.get("inattention_score", 0.0)
        acc = m.get("overall_accuracy", 1.0)
    else:
        imp, ia, acc = 0.1, 0.3, 0.6

    task_ids, dominant = _select(imp, ia, acc)
    assigned = [
        G6AssignedTask(
            task_id=tid, task_name=TASK_DEFINITIONS[tid]["name"],
            difficulty=_difficulty(req.child_id, tid),
            target_deficit=TASK_DEFINITIONS[tid]["target"],
            instructions=TASK_DEFINITIONS[tid]["instructions"][_difficulty(req.child_id, tid)],
        ) for tid in task_ids
    ]
    return G6LearningTaskAssignResponse(
        child_id=req.child_id, grade=6,
        session_number=_session_number(req.child_id),
        tasks=assigned, dominant_deficit=dominant,
        severity_scores={"impulsivity":round(imp,3),"inattention":round(ia,3),"accuracy":round(acc,3)},
    )


def g6_save_task_result(result: G6LearningTaskResult) -> G6LearningTaskResultResponse:
    db    = get_db()
    total = result.total_trials or 1
    score = round((result.correct / total) * 100, 1)
    past  = list(db["learning_task_results"]
                 .find({"child_id":result.child_id,"task_id":result.task_id,"grade":6})
                 .sort("session_number",-1).limit(1))
    nd = result.difficulty
    if past and score >= 75 and past[0].get("score_percent",0) >= 75 and result.difficulty < 3:
        nd = result.difficulty + 1
    msg = "ඉතා හොඳයි! 🌟" if score >= 80 else "හොඳයි! ⭐" if score >= 60 else "නැවත උත්සාහ කරන්න! 💪"
    avg = round(sum(result.response_times_ms)/len(result.response_times_ms)) if result.response_times_ms else 0
    now = datetime.utcnow()
    db["learning_task_results"].insert_one({
        "child_id":result.child_id, "grade":6, "task_id":result.task_id,
        "difficulty":result.difficulty, "correct":result.correct, "wrong":result.wrong,
        "premature":result.premature, "total_trials":result.total_trials,
        "score_percent":score, "avg_rt_ms":avg,
        "response_times_ms":result.response_times_ms,
        "session_number":result.session_number, "next_difficulty":nd,
        "timestamp":now.isoformat(), "created_at":now,
    })
    return G6LearningTaskResultResponse(ok=True, message="ප්‍රතිඵල සුරකින ලදී",
                                        score_percent=score, next_difficulty=nd, encouragement=msg)


def g6_get_progress(child_id):
    db = get_db()
    results = list(db["learning_task_results"].find({"child_id":child_id,"grade":6})
                   .sort("created_at",-1).limit(20))
    for r in results:
        r["_id"] = str(r["_id"])
        if "created_at" in r and hasattr(r["created_at"],"isoformat"):
            r["timestamp"] = r["created_at"].isoformat(); del r["created_at"]
    return {"child_id":child_id,"grade":6,"sessions":results}
