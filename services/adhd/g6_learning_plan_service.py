"""
Grade 6 Learning Plan Service — age 11-12 years
Activities reference Grade 6 diagnostic tasks:
  Stroop Interference, N-Back Lite, Rapid Visual Search,
  Dual Condition Go/No-Go, Sustained Counting.
"""
from datetime import datetime
from models.adhd.g6_learning_plan_model import (
    G6LearningPlanRequest, G6LearningPlanResponse,
    G6AdaptationParams, G6LearningActivity,
)
from services.db_service import get_db

PROFILE_PARAMS = {
    "profile_a": dict(chunk_size=5, session_minutes=20, break_frequency=20,
                      modality="visual_only",  pacing="timed",
                      visual_complexity="high", feedback_style="visual",
                      encouragement_level="standard"),
    "profile_b": dict(chunk_size=3, session_minutes=10, break_frequency=10,
                      modality="audio_visual", pacing="self_paced",
                      visual_complexity="low",  feedback_style="immediate_audio",
                      encouragement_level="high"),
    "profile_c": dict(chunk_size=4, session_minutes=12, break_frequency=12,
                      modality="visual_only",  pacing="timed_relaxed",
                      visual_complexity="medium", feedback_style="haptic_visual",
                      encouragement_level="high"),
    "profile_d": dict(chunk_size=2, session_minutes=8,  break_frequency=8,
                      modality="audio_visual", pacing="self_paced",
                      visual_complexity="low",  feedback_style="immediate_audio",
                      encouragement_level="very_high"),
}

PROFILE_LABELS = {
    "profile_a": "ඉහළ අවධානය ✨",
    "profile_b": "අවධානය වර්ධනය කරමු 📚",
    "profile_c": "ආවේගශීලීතාවය පාලනය කරමු 🧘",
    "profile_d": "සමබල ඉගෙනීම 🌱",
}

ACTIVITY_LIBRARY = {
    # Profile A — high attention: Stroop L2 + N-Back L2, timed reading, mind map
    "profile_a": [
        G6LearningActivity(
            title="Stroop L2 + N-Back L2",
            description="Advanced interference control and working memory — Grade 6 challenge",
            type="focus_builder", duration_min=12, delivery="in_app",
            instructions="Stroop Level 2: 20 trials, 60% incongruent, 2s per trial. "
                         "N-Back Level 2: 2-back, 20 items, 2s window. "
                         "interference_error_rate සහ nback_accuracy "
                         "session-by-session track කරන්න."),
        G6LearningActivity(
            title="කාල-නිශ්චිත ස්වාධීන කියවීම",
            description="Timed independent reading with comprehension — Grade 6 level",
            type="comprehension", duration_min=12, delivery="independent",
            instructions="ශ්‍රේෂ්ඨ ෙෙෙෙෙ 1 ෙෙෙෙෙෙ, ෙෙෙෙ 5ෙෙ ෙෙෙෙ ෙෙෙෙ. "
                         "ෙෙෙ ෙෙෙ, ෙෙෙෙෙ ෙෙෙෙ."),
        G6LearningActivity(
            title="Rapid Visual Search Advanced",
            description="Speed and accuracy under time pressure",
            type="focus_builder", duration_min=8, delivery="in_app",
            instructions="Visual Search Level 3: 4x4 grid, 3s limit, 10 trials. "
                         "visual_search_speed track කරන්න."),
        G6LearningActivity(
            title="දෛනික සැලසුම් ලිවීම",
            description="Structured daily planner — executive function",
            type="focus_builder", duration_min=8, delivery="independent",
            instructions="ෙෙෙෙ ෙෙෙෙෙ 4ෙෙ ෙෙෙ, ෙෙෙ-ෙෙෙෙෙ ෙෙෙ."),
    ],

    # Profile B — inattentive: N-Back L1 + Attention Grid, note-taking, paragraph writing
    "profile_b": [
        G6LearningActivity(
            title="N-Back Level 1 + Attention Grid",
            description="Working memory and sustained attention — Grade 6 inattention tasks",
            type="memory", duration_min=10, delivery="in_app",
            instructions="N-Back Level 1: 1-back, 20 items, 2s window. "
                         "Attention Grid 5x5, 20s. "
                         "nback_accuracy සහ grid score track කරන්න."),
        G6LearningActivity(
            title="ගුරු-නිර්දේශිත සටහන් ගැනීම",
            description="Guided note-taking — structured listening for inattention",
            type="comprehension", duration_min=10, delivery="teacher_led",
            instructions="ෙෙෙෙෙෙෙ: ෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙ. "
                         "ෙෙෙෙෙෙ ෙෙෙෙ ෙෙෙ 3 ෙෙෙෙ ෙෙෙ ෙෙෙෙ. "
                         "N-Back task ෙෙෙෙ ෙෙෙෙ ෙෙෙෙ."),
        G6LearningActivity(
            title="ෙෙෙෙෙ ෙෙෙෙ — ෙෙෙෙෙෙ",
            description="Paragraph writing from memory — Grade 6 level",
            type="comprehension", duration_min=8, delivery="independent",
            instructions="ෙෙෙ ෙෙෙෙෙ ෙෙෙ 1 ෙෙෙෙෙ ෙෙෙෙ. ෙෙෙ ෙෙෙ, ෙෙෙ ෙෙෙෙ."),
        G6LearningActivity(
            title="Sustained Counting Practice",
            description="Sustained attention training with counting",
            type="focus_builder", duration_min=8, delivery="in_app",
            instructions="Sustained Counting Level 1: 30s, ෙෙෙ ෙෙෙ ෙෙෙෙ. "
                         "counting_accuracy track කරන්න."),
    ],

    # Profile C — impulsive: Stroop L1 + Dual Go/No-Go, think-aloud, chess
    "profile_c": [
        G6LearningActivity(
            title="Stroop L1 + Dual Condition Go/No-Go",
            description="Interference control and dual-rule inhibition — Grade 6 impulsivity tasks",
            type="inhibition", duration_min=10, delivery="in_app",
            instructions="Stroop Level 1: 20 trials, 50% incongruent, 2.5s per trial. "
                         "Dual Go/No-Go: tap blue circles only, 30 trials, 1.5s. "
                         "interference_error_rate සහ dual_false_alarms track කරන්න."),
        G6LearningActivity(
            title="Think-Aloud ෙෙෙෙෙ ෙෙෙෙෙ",
            description="Slows impulsive responding — mandatory thinking before answering",
            type="inhibition", duration_min=10, delivery="teacher_led",
            instructions="ෙෙෙෙෙෙෙ: ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙ. "
                         "ෙෙෙෙෙෙ: 'ෙෙෙ ෙෙෙෙ...' ෙෙෙ ෙෙෙ ෙෙෙෙ. "
                         "Stroop incongruent error ෙෙෙෙ ෙෙෙෙ ෙෙෙෙ."),
        G6LearningActivity(
            title="Chess / Strategy Puzzle",
            description="Strategy game — plans before acting",
            type="inhibition", duration_min=12, delivery="independent",
            instructions="Chess ෙෙෙෙ strategy puzzle. "
                         "ෙෙෙෙෙෙෙෙ ෙෙෙ ෙෙෙෙ 3ෙෙ ෙෙෙෙෙ."),
        G6LearningActivity(
            title="Mindfulness Journal",
            description="Daily reflection on impulse control",
            type="inhibition", duration_min=8, delivery="independent",
            instructions="ෙෙෙෙ ෙෙෙෙෙ: ෙෙෙ ෙෙෙෙෙ ෙෙෙෙ ෙෙෙ ෙෙෙ. "
                         "Stroop score trend ෙෙෙ ෙෙෙෙ."),
    ],

    # Profile D — mixed: Visual Search + Sustained Counting L1, 1-on-1, structured homework
    "profile_d": [
        G6LearningActivity(
            title="Visual Search L1 + Sustained Counting L1",
            description="Basic attention speed and sustained focus — Grade 6 mixed profile",
            type="focus_builder", duration_min=8, delivery="in_app",
            instructions="Visual Search Level 1: 4x4 grid, 5s limit, 10 trials. "
                         "Sustained Counting: 30s, ෙෙෙ ෙෙෙ ෙෙෙෙ. "
                         "visual_search_speed සහ counting_accuracy track කරන්න."),
        G6LearningActivity(
            title="1-on-1 ගුරු ෙෙෙෙ",
            description="One-to-one teacher check-in — structured support",
            type="focus_builder", duration_min=8, delivery="teacher_led",
            instructions="ෙෙෙෙෙෙෙ: ෙෙෙෙෙෙ 1-on-1. "
                         "'ෙෙෙෙ ෙෙෙෙෙ ෙෙෙ' ෙෙෙ ෙෙෙෙ. "
                         "Stroop error rate trend ෙෙෙ ෙෙෙෙ."),
        G6LearningActivity(
            title="ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙෙ",
            description="Structured homework schedule — accountability system",
            type="focus_builder", duration_min=8, delivery="independent",
            instructions="ෙෙෙෙෙ 8 + ෙෙෙෙෙෙ 3 + ෙෙෙෙෙ 8 + ෙෙෙෙෙෙ 3. "
                         "Sticker chart ෙෙෙ ෙෙෙෙ."),
        G6LearningActivity(
            title="Physical Movement + Short Task",
            description="Movement break then short analytic task",
            type="focus_builder", duration_min=10, delivery="independent",
            instructions="ෙෙෙෙෙ 5ෙෙ ෙෙෙෙෙෙ + ෙෙෙෙෙ 5ෙෙ ෙෙෙෙෙ 3. "
                         "N-Back ෙෙෙෙ ෙෙෙෙ ෙෙෙ ෙෙෙෙ."),
    ],
}

TEACHER_NOTES = {
    "profile_a": (
        "Grade 6 — ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ. "
        "interference_error_rate සහ nback_accuracy session-by-session track කරන්න. "
        "visual_search_speed trend ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ."
    ),
    "profile_b": (
        "Grade 6 — ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ. "
        "nback_accuracy track කරන්න — ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ. "
        "ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ."
    ),
    "profile_c": (
        "Grade 6 — ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ. "
        "interference_error_rate track කරන්න. "
        "dual_false_alarms ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ."
    ),
    "profile_d": (
        "Grade 6 — ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ. "
        "visual_search_speed සහ counting_accuracy track කරන්න. "
        "1-on-1 check-in ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ ශ්‍රේෂ්ඨ."
    ),
}

PARENT_NOTES = {
    "profile_a":  "Chess, N-Back ෙෙෙෙ ෙෙෙෙ ෙෙෙෙ. ෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙ.",
    "profile_b":  "ෙෙෙ 10 ෙෙෙ + ෙෙෙෙෙෙ. Sticker ෙෙෙෙ. nback score ෙෙෙෙ.",
    "profile_c":  "Chess, Sudoku ෙෙෙෙ. Stroop ෙෙෙ ෙෙෙෙ ෙෙෙෙ ෙෙෙෙ.",
    "profile_d":  "Movement + ෙෙෙෙ ෙෙෙෙ. 1 ෙෙෙෙ ෙෙෙෙ. score ෙෙෙ ෙෙෙෙ.",
}


def g6_generate_learning_plan(req: G6LearningPlanRequest) -> G6LearningPlanResponse:
    profile = req.attention_profile.lower()
    if profile not in PROFILE_PARAMS:
        profile = "profile_b"
    plan = G6LearningPlanResponse(
        child_id          = req.child_id,
        grade             = 6,
        profile           = profile,
        profile_label     = PROFILE_LABELS.get(profile, profile),
        adaptation_params = G6AdaptationParams(**PROFILE_PARAMS[profile]),
        activities        = ACTIVITY_LIBRARY.get(profile, []),
        teacher_note      = TEACHER_NOTES.get(profile, ""),
        parent_note       = PARENT_NOTES.get(profile, ""),
        generated_at      = datetime.utcnow().isoformat(),
    )
    db = get_db()
    db["learning_plans"].insert_one({**plan.dict(), "created_at": datetime.utcnow()})
    return plan


def g6_get_latest_plan(child_id: str):
    db   = get_db()
    plan = db["learning_plans"].find_one(
        {"child_id": child_id, "grade": 6}, sort=[("created_at", -1)])
    if not plan: return None
    plan.pop("_id", None); plan.pop("created_at", None)
    return plan
