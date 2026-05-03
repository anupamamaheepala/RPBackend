"""
Grade 4 Learning Plan Service
Same structure as Grade 3 learning_plan_service.py.
Grade 4 specific: chunk sizes, session durations, and age-appropriate activities.
"""

from datetime import datetime
from models.adhd.g4_learning_plan_model import (
    G4LearningPlanRequest, G4LearningPlanResponse,
    G4AdaptationParams, G4LearningActivity,
)
from services.db_service import get_db

# ── Profile parameters (Grade 4 scaled) ──────────────────────────────────────
PROFILE_PARAMS = {
    "profile_a": {
        "chunk_size": 5, "session_minutes": 18, "break_frequency": 20,
        "modality": "visual_only",    "pacing": "timed",
        "visual_complexity": "high",  "feedback_style": "visual",
        "encouragement_level": "standard",
    },
    "profile_b": {
        "chunk_size": 2, "session_minutes": 8,  "break_frequency": 8,
        "modality": "audio_visual",   "pacing": "self_paced",
        "visual_complexity": "low",   "feedback_style": "immediate_audio",
        "encouragement_level": "high",
    },
    "profile_c": {
        "chunk_size": 3, "session_minutes": 12, "break_frequency": 12,
        "modality": "visual_only",    "pacing": "timed_relaxed",
        "visual_complexity": "medium","feedback_style": "haptic_visual",
        "encouragement_level": "high",
    },
    "profile_d": {
        "chunk_size": 2, "session_minutes": 6,  "break_frequency": 6,
        "modality": "audio_visual",   "pacing": "self_paced",
        "visual_complexity": "low",   "feedback_style": "immediate_audio",
        "encouragement_level": "very_high",
    },
}

# ── Profile labels ────────────────────────────────────────────────────────────
PROFILE_LABELS = {
    "profile_a": "ඉහළ අවධානය ✨",
    "profile_b": "අවධානය වර්ධනය කරමු 📚",
    "profile_c": "ආවේගශීලීතාවය පාලනය කරමු 🧘",
    "profile_d": "සමබර ඉගෙනීම 🌱",
}

# ── Activity library (Grade 4, age-appropriate) ───────────────────────────────
ACTIVITY_LIBRARY = {
    "profile_a": [
        G4LearningActivity(
            title="Stop/Go Level 2 ක්‍රීඩාව",
            description="Grade 4 traffic light task — fast reaction with multiple no-go signals",
            type="inhibition",
            duration_min=10,
            delivery="in_app",
            instructions="ගමනාගමන ආලෝකය දිස් වන විට: කොළ = ස්පර්ශ කරන්න, රතු/කහ = නොකරන්න. ඉක්මනින් ප්‍රතිචාර දක්වන්න.",
        ),
        G4LearningActivity(
            title="ජෝඩු කරමු — Think-Aloud",
            description="Pair reading with comprehension questions — teacher guided",
            type="comprehension",
            duration_min=10,
            delivery="teacher_led",
            instructions="ගුරුවරයා සමඟ ජෝඩු වී කියවන්න. සෑම ඡේදයකට පසු ප්‍රශ්නයක් විසඳන්න.",
        ),
        G4LearningActivity(
            title="කෙටි සාරාංශ ලිවීම",
            description="Independent — short paragraph summarisation",
            type="comprehension",
            duration_min=8,
            delivery="independent",
            instructions="කියෙව්ව දේ 3 වාක්‍යයෙන් ලියන්න. ශබ්ද කෝශය නොබලා.",
        ),
        G4LearningActivity(
            title="Chess / ගැටළු ෙකළිය",
            description="Strategy game — builds planning and inhibitory control",
            type="inhibition",
            duration_min=10,
            delivery="independent",
            instructions="Chess හෝ Sudoku ක්‍රීඩාව දිනකට විනාඩි 10ක් ක්‍රීඩා කරන්න.",
        ),
    ],
    "profile_b": [
        G4LearningActivity(
            title="Audio Sequence Level 2 + Listen & Extract",
            description="Auditory sequencing and detail extraction — targets inattention",
            type="memory",
            duration_min=8,
            delivery="in_app",
            instructions="කතාව 2 වාක්‍ය දිගයෙන් සවන් දෙන්න. 3 රූප නිවැරදිව සකසන්න. නැවත 1 වරක් ශ්‍රවණය කළ හැක.",
        ),
        G4LearningActivity(
            title="ජෝඩු කියවීම — ප්‍රශ්න සමඟ",
            description="Pair reading with comprehension questions — teacher guided",
            type="comprehension",
            duration_min=8,
            delivery="teacher_led",
            instructions="ගුරුවරයා සමඟ ජෝඩු කියවීමෙදී සෑම ඡේදයකට ප්‍රශ්නයකට පිළිතුරු දෙන්න.",
        ),
        G4LearningActivity(
            title="කෙටි සාරාංශ ලිවීම",
            description="Write a short summary independently after reading",
            type="comprehension",
            duration_min=7,
            delivery="independent",
            instructions="කියෙව්ව දේ ගැන 3 වාක්‍යයකින් ලියන්න.",
        ),
        G4LearningActivity(
            title="Sticker ත්‍යාග ක්‍රමය",
            description="Reward chart to reinforce sustained attention daily",
            type="focus_builder",
            duration_min=5,
            delivery="independent",
            instructions="දිනකට ඉලක්ක 3 ලිවා ඒවා සම්පූර්ණ කළ විට sticker එකක් දමන්න.",
        ),
    ],
    "profile_c": [
        G4LearningActivity(
            title="Stop/Go Training + Follow Card",
            description="Traffic light inhibition + rule memory — targets impulsivity",
            type="inhibition",
            duration_min=10,
            delivery="in_app",
            instructions="Stop/Go task Level 1 සම්පූර්ණ කරන්න. Follow Card task: නීති 3 කටපාඩම් කර ක්‍රියාත්මක කරන්න.",
        ),
        G4LearningActivity(
            title="Think-Aloud ගැටළු විසඳීම",
            description="Teacher-guided think-aloud problem solving — slows impulsive responding",
            type="inhibition",
            duration_min=8,
            delivery="teacher_led",
            instructions="ගුරුවරයා ගැටළුවක් දෙයි. ශිෂ්‍යයා: 'මම හිතන්නේ...' කියා 声に出して (声出して) කියමින් විසඳීම කරයි.",
        ),
        G4LearningActivity(
            title="Chess / Strategy ෙකළිය",
            description="Chess or puzzle — builds planning before acting",
            type="inhibition",
            duration_min=10,
            delivery="independent",
            instructions="Chess, Sudoku, හෝ strategy puzzle ක්‍රීඩා කරන්න. ඉක්මනින් ගෙමෙවීම ගාමු නොකරන්න.",
        ),
        G4LearningActivity(
            title="One-step-at-a-time කාර්ය",
            description="Break assignments into single steps — reduces impulsive rushing",
            type="focus_builder",
            duration_min=5,
            delivery="independent",
            instructions="ගෙදර කාර්යය කොටස් 3කට කඩා, එකිනෙකක් ඉවර කළ මතු ඊළඟ කොටස ආරම්භ කරන්න.",
        ),
    ],
    "profile_d": [
        G4LearningActivity(
            title="Simple Go/No-Go + Stay Complete",
            description="Basic inhibition + sustained attention tasks — targets both deficits",
            type="inhibition",
            duration_min=6,
            delivery="in_app",
            instructions="Go/No-Go Level 1 කරන්න. Stay & Complete: ගණිත ගැටළු 5 ශ්‍රේණිගත ලෙස ඉවර කරන්න.",
        ),
        G4LearningActivity(
            title="One-step-at-a-time ගුරු සහාය",
            description="Teacher breaks task into one step at a time",
            type="focus_builder",
            duration_min=6,
            delivery="teacher_led",
            instructions="ගුරුවරයා: පළමු පියවර පමණක් කියන්න. ශිෂ්‍යයා ඉවර කළ පසු ඊළඟ පියවර දෙන්න.",
        ),
        G4LearningActivity(
            title="Sticker ත්‍යාග ක්‍රමය",
            description="Daily reward chart to motivate task completion",
            type="focus_builder",
            duration_min=5,
            delivery="independent",
            instructions="සෑම කාර්යයක් ඉවර කළ විට sticker දමන්න. දිනකට 3 sticker = විශේෂ ත්‍යාගයක්.",
        ),
        G4LearningActivity(
            title="Physical Movement + කෙටි කාර්ය",
            description="Short physical break then resume short task — resets attention",
            type="focus_builder",
            duration_min=10,
            delivery="independent",
            instructions="විනාඩි 5ක් ශාරීරික ව්‍යායාම + විනාඩි 5ක් ගණිත ගැළළු 3. ඒ ආකාරයෙන් වෙනස් කරන්න.",
        ),
    ],
}

# ── Teacher & parent notes ────────────────────────────────────────────────────
TEACHER_NOTES = {
    "profile_a": (
        "මෙෙ ශිෂ්‍යයා Grade 4 මට්ටමේ හොඳ අවධානයක් පෙන්වයි. "
        "දිගු කාලීන project කාර්යයන් ලබා දෙන්න. "
        "ඉහළ cognitive load ඇති ගැටළු ලබා දී critical thinking දිරිගන්වන්න."
    ),
    "profile_b": (
        "මෙෙ ශිෂ්‍යයාට දිගු අවධානය ගැටළුවකි. "
        "පාඩම් කොටස් 2-3 කට බෙදන්න. සෑෙ විනාඩි 8 කට ප්‍රශ්නයක් ඇසීෙෙන් "
        "ශිෂ්‍යයාෙේ අවධානය නැවත ෙෙොඩ නං ෙෙො. "
        "ශිෂ්‍යයාට ඉදිරිපස ආසනයක් ලො ෙෙෙදෙෙෙෙෙ. "
        "Listen & Extract task නැවත ශ්‍රවණය කිරීමෙ ගණන නිරීක්ෂිතකරන්න."
    ),
    "profile_c": (
        "මෙෙ ශිෂ්‍යයාෙේ ආෙෙගශීලීතාෙය ෙෙෙෙ. "
        "ෙෙ Think-Aloud ක්‍රමය ොෙිතා ෙෙරෙෙෙෙෙ. "
        "ෙෙ Follow Card ෙෙෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙෙෙ. "
        "ෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙෙ — ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙ ෙෙ ෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙ."
    ),
    "profile_d": (
        "මෙෙ ශිෂ්‍යයාෙේ ෙෙෙෙෙෙෙෙ ෙෙ ෙෙෙෙෙෙ ෙෙ ෙෙෙෙෙ. "
        "ෙෙ ෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙෙ. "
        "ෙෙෙෙ ෙෙෙෙෙ 1ෙෙ ෙෙෙෙෙ ෙෙෙෙෙ. "
        "ෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙ."
    ),
}

PARENT_NOTES = {
    "profile_a": (
        "ඔෙෙ දරෙෙ Grade 4 ෙෙ ෙෙෙෙෙ ෙෙෙෙෙෙෙ ෙෙෙෙෙෙ. "
        "ෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ Chess, ෙෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ."
    ),
    "profile_b": (
        "ෙෙෙෙ ෙෙෙෙෙෙෙෙ ෙෙෙෙෙෙ: ෙෙෙෙෙ ෙෙෙෙෙෙ 20 ෙෙෙෙෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙෙ. "
        "ෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ."
    ),
    "profile_c": (
        "ෙෙෙෙ ෙෙෙෙෙෙෙෙ ෙෙෙෙෙෙ: ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ. "
        "Chess, Sudoku, ෙෙෙෙ strategy ෙෙෙෙෙ ෙෙෙෙෙෙෙ ෙෙෙෙෙෙෙ."
    ),
    "profile_d": (
        "ෙෙෙෙ ෙෙෙෙෙෙෙෙ ෙෙෙෙෙෙ: ෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙ. "
        "ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙෙ ෙෙෙෙෙ ෙෙෙෙ ෙෙෙෙෙ."
    ),
}


# ── Service functions ─────────────────────────────────────────────────────────
def g4_generate_learning_plan(req: G4LearningPlanRequest) -> G4LearningPlanResponse:
    profile = req.attention_profile.lower()
    if profile not in PROFILE_PARAMS:
        profile = "profile_b"

    params     = PROFILE_PARAMS[profile]
    activities = ACTIVITY_LIBRARY.get(profile, [])

    plan = G4LearningPlanResponse(
        child_id          = req.child_id,
        grade             = 4,
        profile           = profile,
        profile_label     = PROFILE_LABELS.get(profile, profile),
        adaptation_params = G4AdaptationParams(**params),
        activities        = activities,
        teacher_note      = TEACHER_NOTES.get(profile, ""),
        parent_note       = PARENT_NOTES.get(profile, ""),
        generated_at      = datetime.utcnow().isoformat(),
    )

    db = get_db()
    db["learning_plans"].insert_one({
        **plan.dict(),
        "created_at": datetime.utcnow(),
    })
    return plan


def g4_get_latest_plan(child_id: str):
    db   = get_db()
    plan = db["learning_plans"].find_one(
        {"child_id": child_id, "grade": 4},
        sort=[("created_at", -1)],
    )
    if not plan:
        return None
    plan.pop("_id", None)
    plan.pop("created_at", None)
    return plan
