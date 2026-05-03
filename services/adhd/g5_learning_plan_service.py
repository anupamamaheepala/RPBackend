"""
Grade 5 Learning Plan Service — age 10-11 years
Profile A: chunk=5, session=20min
Profile B: chunk=3, session=10min
Profile C: chunk=4, session=12min
Profile D: chunk=2, session=7min
"""
from datetime import datetime
from models.adhd.g5_learning_plan_model import (
    G5LearningPlanRequest, G5LearningPlanResponse,
    G5AdaptationParams, G5LearningActivity,
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
    "profile_d": dict(chunk_size=2, session_minutes=7,  break_frequency=7,
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
    "profile_a": [
        G5LearningActivity(title="Spot the Change + Filter Level 2",
            description="Advanced visual attention and filtering — Grade 5 challenge",
            type="focus_builder", duration_min=12, delivery="in_app",
            instructions="Spot the Change: 2 changes in 7s. Filter: tap green, ignore blue in 4x4 grid. 20 trials."),
        G5LearningActivity(title="ගුරු-නිර්දේශිත සටහන් ගැනීම",
            description="Guided note-taking — structured listening and writing",
            type="comprehension", duration_min=10, delivery="teacher_led",
            instructions="ගුරුවරයා කෙටි කොටසක් කියවයි. ශිෂ්‍යයා ප්‍රධාන කරුණු 3ක් ලියයි. කිසිදු යෙදුම් සහය නොමැතිව."),
        G5LearningActivity(title="මතකයෙන් මනස-සිතියම",
            description="Mind map from memory — Grade 5 comprehension extension",
            type="memory", duration_min=8, delivery="independent",
            instructions="කෙටි ඡේදයක් 1 වතාවක් කියවා, පොත වසා, අදහස් සිතියමකින් ලියන්න."),
        G5LearningActivity(title="Switch Go Level 2 Strategy",
            description="Cognitive flexibility training with self-monitoring",
            type="inhibition", duration_min=10, delivery="independent",
            instructions="Switch Go task: නීතිය සෑම trial 10කට වෙනස් වේ. ස්වයං-ලකුණු දිනපොතේ සටහන් කරන්න."),
    ],
    "profile_b": [
        G5LearningActivity(title="Audio Sequence Level 3 + Filter",
            description="Inattention — auditory sequencing and selective attention",
            type="memory", duration_min=10, delivery="in_app",
            instructions="Audio Sequence: 4 sentences, 1 play only, 4 items to order. Filter: tap green only in 4x4 grid."),
        G5LearningActivity(title="ගුරු-නිර්දේශිත ස්මරණ",
            description="Teacher-directed recall — structured comprehension",
            type="comprehension", duration_min=8, delivery="teacher_led",
            instructions="ගුරුවරයා ප්‍රශ්නයක් ඇසීමෙන් ශිෂ්‍යයාගේ අවධානය නැවත ගෙන යයි. සෑම විනාඩි 10කට."),
        G5LearningActivity(title="ඡේදය ලිවීම — මතකයෙන්",
            description="Paragraph writing from memory — 5th grade level",
            type="comprehension", duration_min=8, delivery="independent",
            instructions="කෙටි ඡේදයක් කියවා, වසා, 3 වාක්‍යයෙන් ලියන්න. ශබ්ද කෝශය භාවිතා නොකරන්න."),
        G5LearningActivity(title="Sticker ත්‍යාග ක්‍රමය — Grade 5",
            description="Daily focus reward chart — age-appropriate",
            type="focus_builder", duration_min=5, delivery="independent",
            instructions="දිනකට ඉලක්ක 3 ලිවා ඒවා සම්පූර්ණ කළ විට sticker දමන්න. සතියේ අවසානයේ ත්‍යාගයක්."),
    ],
    "profile_c": [
        G5LearningActivity(title="Switch Go + Stillness",
            description="Cognitive flexibility and motor inhibition — targets impulsivity",
            type="inhibition", duration_min=10, delivery="in_app",
            instructions="Switch Go Level 1: animal/vehicle rules. Stillness: hold 30s without lifting finger."),
        G5LearningActivity(title="පාලිත විවාදය — පිළිතුරු රැඳීම",
            description="Controlled debate — must wait turn before responding",
            type="inhibition", duration_min=10, delivery="teacher_led",
            instructions="ශිෂ්‍යයා අදහස් ප්‍රකාශ කිරීමට පෙර සෙසු අය නිමා කරන තෙක් බලා සිටිය යුතුය."),
        G5LearningActivity(title="Mindfulness දිනපොත",
            description="Daily mindfulness journal — impulse control practice",
            type="inhibition", duration_min=8, delivery="independent",
            instructions="ඉක්මන් ප්‍රතිචාරයක් දෙන්නට පෙර: 3 ගැඹුරු හුස්ම. දිනපොතේ 'මා ඉවසූ දේ' ලියන්න."),
        G5LearningActivity(title="Strategy Board Game",
            description="Chess or strategy puzzle — plans before acting",
            type="inhibition", duration_min=10, delivery="independent",
            instructions="Chess, Sudoku, හෝ strategy puzzle. ගෙමෙවීමකට පෙර තත්පර 5ක් සිතන්න."),
    ],
    "profile_d": [
        G5LearningActivity(title="Filter Level 1 + Ladder",
            description="Basic filtering and sequential following — both deficits",
            type="focus_builder", duration_min=8, delivery="in_app",
            instructions="Filter Level 1: 4x4 grid, 20 trials. Ladder: step-by-step instructions, 1 step at a time."),
        G5LearningActivity(title="අත්-ව්‍යවහාරික වර්ගීකරණ කාර්ය",
            description="Hands-on sorting — physical engagement for mixed profile",
            type="focus_builder", duration_min=7, delivery="teacher_led",
            instructions="ගුරුවරයා: කාඩ් හෝ වස්තු 2 කාණ්ඩ 2කට වර්ගකිරීමට ශිෂ්‍යයාට ලබාදෙයි. 1 කාර්ය, 1 නිහඬ නිර්දේශයක් පමණි."),
        G5LearningActivity(title="Drawing + 1 Sentence",
            description="Drawing and one sentence description — creative low-load task",
            type="focus_builder", duration_min=7, delivery="independent",
            instructions="මෑතකදී ඉගෙනූ දෙයක් රූපයකින් ඇඳ, 1 වාක්‍යයකින් විස්තර කරන්න."),
        G5LearningActivity(title="Physical Movement Break + Short Task",
            description="Movement then focus — resets attention for mixed profile",
            type="focus_builder", duration_min=10, delivery="independent",
            instructions="විනාඩි 5ක් ශාරීරික ව්‍යායාම. ඉන්පසු විනාඩි 5ක් 3 ගැටළු විසඳන්න. ඒ ආකාරයෙන් ප්‍රත්‍යාවර්ත කරන්න."),
    ],
}

TEACHER_NOTES = {
    "profile_a": "Grade 5 ශිෂ්‍යයා ඉහළ අවධානයක් පෙන්වයි. දිගු project කාර්ය, critical thinking ගැටළු ලබා දෙන්න. Switch Go Level 3 සහ Spot the Change Level 3 ද ලබා දෙන්න.",
    "profile_b": "Grade 5 — ශිෂ්‍යයාට දිගු අවධානය ගැටළුවකි. පාඩම් කොටස් 3-4 කට බෙදන්න. සෑම විනාඩි 10කට ප්‍රශ්නයක් ඇසීමෙන් අවධානය නැවත ගෙනෙන්න. Audio replay ගණන නිරීක්ෂිත කරන්න.",
    "profile_c": "Grade 5 — ශිෂ්‍යයාගේ ආවේගශීලීතාවය ගැටළුවකි. Think-Aloud ක්‍රමය ගුරු-නිර්දේශිත කාර්ය සෑම කටයුත්තකදීම ොයිතා ගන්න. Switch Go error rate නිරීක්ෂිත කරන්න.",
    "profile_d": "Grade 5 — ශිෂ්‍යයා ආවේගශීලී + අවධානය දෙකෙහිම ගැටළු ඇත. 1 පියවරක් පමණක් ලබා දෙන්න. Motor inhibition score නිරීක්ෂිත කරන්න. Stillness task breaks count ගැන දෙමාපියන්ට දන්වන්න.",
}

PARENT_NOTES = {
    "profile_a": "ඔබේ දරුවා Grade 5 මට්ටමේ හොඳ අවධානයක් පෙන්වයි. Switch Go, Filter tasks ගෙදර ෙකෙෙෙ ක්‍රීඩා ලෙස ෙෙෙෙෙ.",
    "profile_b": "ගෙදර ඉගෙනීමේදී: Audio tasks replay කරන ගණන ලිහිල් ලෙස නිරීක්ෂිත කරන්න. කෙටි, පැහැදිලි ඉලක්ක ලබා දෙන්න. Sticker reward chart ෙොෙෙ ොෙෙ.",
    "profile_c": "ගෙදර ඉගෙනීමේදී: Mindfulness breathing, Chess, Sudoku ෙෙෙ strategy ෙෙෙෙ ෙෙෙෙෙෙ. ෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙෙ.",
    "profile_d": "ගෙදර ඉගෙනීමේදී: ෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙ. ෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙ + ෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙෙ.",
}


def g5_generate_learning_plan(req: G5LearningPlanRequest) -> G5LearningPlanResponse:
    profile = req.attention_profile.lower()
    if profile not in PROFILE_PARAMS:
        profile = "profile_b"
    plan = G5LearningPlanResponse(
        child_id          = req.child_id,
        grade             = 5,
        profile           = profile,
        profile_label     = PROFILE_LABELS.get(profile, profile),
        adaptation_params = G5AdaptationParams(**PROFILE_PARAMS[profile]),
        activities        = ACTIVITY_LIBRARY.get(profile, []),
        teacher_note      = TEACHER_NOTES.get(profile, ""),
        parent_note       = PARENT_NOTES.get(profile, ""),
        generated_at      = datetime.utcnow().isoformat(),
    )
    db = get_db()
    db["learning_plans"].insert_one({**plan.dict(), "created_at": datetime.utcnow()})
    return plan


def g5_get_latest_plan(child_id: str):
    db   = get_db()
    plan = db["learning_plans"].find_one({"child_id": child_id, "grade": 5},
                                         sort=[("created_at", -1)])
    if not plan:
        return None
    plan.pop("_id", None)
    plan.pop("created_at", None)
    return plan
