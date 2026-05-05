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
    "profile_d": "සමබල ඉගෙනීම 🌱",
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
            instructions="ගමනාගමන ආලෝකය දිස් වන විට: කොළ = ස්පර්ශ කරන්න, රතු/කහ = නතර වන්න. ඉක්මනින් ප්‍රතිචාර දක්වන්න.",
        ),
        G4LearningActivity(
            title="ජෝඩු කරමු — Think-Aloud",
            description="Pair reading with comprehension questions — teacher guided",
            type="comprehension",
            duration_min=10,
            delivery="teacher_led",
            instructions="ගුරුවරයා සමඟ එක්ව කියවන්න. සෑම ඡේදයකටම පසු ප්‍රශ්නයකට පිළිතුරු දෙන්න.",
        ),
        G4LearningActivity(
            title="කෙටි සාරාංශ ලිවීම",
            description="Independent — short paragraph summarisation",
            type="comprehension",
            duration_min=8,
            delivery="independent",
            instructions="කියවූ දේ වාක්‍ය 3කින් ලියන්න. ශබ්දකෝෂය භාවිතා නොකරන්න.",
        ),
        G4LearningActivity(
            title="චෙස් හෝ ප්‍රහේලිකා ක්‍රීඩා",
            description="Strategy game — builds planning and inhibitory control",
            type="inhibition",
            duration_min=10,
            delivery="independent",
            instructions="චෙස් හෝ සුඩෝකු (Sudoku) වැනි ක්‍රීඩාවක දිනකට විනාඩි 10ක් නිරත වන්න.",
        ),
    ],
    "profile_b": [
        G4LearningActivity(
            title="Audio Sequence Level 2 + Listen & Extract",
            description="Auditory sequencing and detail extraction — targets inattention",
            type="memory",
            duration_min=8,
            delivery="in_app",
            instructions="වාක්‍ය 2ක දිගින් යුත් කතාවට සවන් දෙන්න. රූප 3ක් නිවැරදි පිළිවෙලට සකසන්න. එක් වරක් නැවත සවන් දිය හැක.",
        ),
        G4LearningActivity(
            title="ජෝඩු කියවීම — ප්‍රශ්න සමඟ",
            description="Pair reading with comprehension questions — teacher guided",
            type="comprehension",
            duration_min=8,
            delivery="teacher_led",
            instructions="ගුරුවරයා සමඟ එක්ව කියවන අතරතුර සෑම ඡේදයකටම අදාළ ප්‍රශ්න වලට පිළිතුරු දෙන්න.",
        ),
        G4LearningActivity(
            title="කෙටි සාරාංශ ලිවීම",
            description="Write a short summary independently after reading",
            type="comprehension",
            duration_min=7,
            delivery="independent",
            instructions="කියවූ දේ ගැන වාක්‍ය 3ක් ස්වාධීනව ලියන්න.",
        ),
        G4LearningActivity(
            title="ස්ටිකර් ත්‍යාග ක්‍රමය",
            description="Reward chart to reinforce sustained attention daily",
            type="focus_builder",
            duration_min=5,
            delivery="independent",
            instructions="දිනකට ඉලක්ක 3ක් ලියා ඒවා සම්පූර්ණ කළ විට ස්ටිකර් එකක් අලවන්න.",
        ),
    ],
    "profile_c": [
        G4LearningActivity(
            title="Stop/Go Training + Follow Card",
            description="Traffic light inhibition + rule memory — targets impulsivity",
            type="inhibition",
            duration_min=10,
            delivery="in_app",
            instructions="Stop/Go task Level 1 සම්පූර්ණ කරන්න. පසුව නීති 3ක් මතක තබාගෙන ක්‍රියාත්මක කරන්න.",
        ),
        G4LearningActivity(
            title="Think-Aloud ගැටළු විසඳීම",
            description="Teacher-guided think-aloud problem solving — slows impulsive responding",
            type="inhibition",
            duration_min=8,
            delivery="teacher_led",
            instructions="ගුරුවරයා ගැටළුවක් ලබා දෙයි. ශිෂ්‍යයා එය විසඳන ආකාරය ශබ්ද නගා පවසමින් සිදු කළ යුතුය.",
        ),
        G4LearningActivity(
            title="උපායමාර්ගික ක්‍රීඩා",
            description="Chess or puzzle — builds planning before acting",
            type="inhibition",
            duration_min=10,
            delivery="independent",
            instructions="චෙස් හෝ සුඩෝකු ක්‍රීඩා කරන්න. පියවරක් ගැනීමට පෙර හොඳින් සිතීමට පුහුණු වන්න.",
        ),
        G4LearningActivity(
            title="පියවරෙන් පියවර කාර්යයන්",
            description="Break assignments into single steps — reduces impulsive rushing",
            type="focus_builder",
            duration_min=5,
            delivery="independent",
            instructions="ගෙදර වැඩ කොටස් කිහිපයකට කඩා, එකක් අවසන් වූ පසු පමණක් අනෙක ආරම්භ කරන්න.",
        ),
    ],
    "profile_d": [
        G4LearningActivity(
            title="Simple Go/No-Go + Stay Complete",
            description="Basic inhibition + sustained attention tasks — targets both deficits",
            type="inhibition",
            duration_min=6,
            delivery="in_app",
            instructions="Go/No-Go Level 1 කරන්න. පසුව ලබා දෙන කෙටි ගණිත ගැටළු 5 අවසන් කරන්න.",
        ),
        G4LearningActivity(
            title="පියවරෙන් පියවර ගුරු සහාය",
            description="Teacher breaks task into one step at a time",
            type="focus_builder",
            duration_min=6,
            delivery="teacher_led",
            instructions="ගුරුවරයා එක් වරකට එක් පියවරක් පමණක් පවසයි. එය අවසන් වූ පසු ඊළඟ පියවර ලබා දෙයි.",
        ),
        G4LearningActivity(
            title="ස්ටිකර් ත්‍යාග ක්‍රමය",
            description="Daily reward chart to motivate task completion",
            type="focus_builder",
            duration_min=5,
            delivery="independent",
            instructions="සෑම කුඩා කාර්යයක් අවසානයේම ස්ටිකරයක් ලබා දෙන්න. දිනකට ස්ටිකර් 3ක් ලැබූ විට විශේෂ ත්‍යාගයක් ලබා දෙන්න.",
        ),
        G4LearningActivity(
            title="ශාරීරික විවේකය + කෙටි කාර්යයන්",
            description="Short physical break then resume short task — resets attention",
            type="focus_builder",
            duration_min=10,
            delivery="independent",
            instructions="විනාඩි 5ක ශාරීරික ව්‍යායාමයකින් පසු විනාඩි 5ක් ඉගෙනීමේ කටයුතු වල නිරත වන්න.",
        ),
    ],
}

# ── Teacher & parent notes ────────────────────────────────────────────────────
TEACHER_NOTES = {
    "profile_a": (
        "මෙම ශිෂ්‍යයා 4 වන ශ්‍රේණියේ මට්ටමට සාපේක්ෂව ඉහළ අවධානයක් පෙන්වයි. "
        "ඔහුට දිගුකාලීන ව්‍යාපෘති සහ විවේචනාත්මක චින්තනය (critical thinking) අවශ්‍ය වන ගැටළු ලබා දෙන්න."
    ),
    "profile_b": (
        "මෙම ශිෂ්‍යයාට දිගු වේලාවක් අවධානය පවත්වා ගැනීම අපහසුය. "
        "එබැවින් පාඩම් කොටස් 2-3 කට බෙදා උගන්වන්න. සෑම විනාඩි 8කටම වරක් ප්‍රශ්නයක් අසා අවධානය පරීක්ෂා කරන්න. "
        "පන්තියේ ඉදිරිපස අසුනක් ලබා දීම වඩාත් සුදුසුයි."
    ),
    "profile_c": (
        "මෙම ශිෂ්‍යයාගේ ආවේගශීලීතාවය පාලනය කිරීම සඳහා 'Think-Aloud' ක්‍රමය භාවිතා කරන්න. "
        "වැඩක් ආරම්භ කිරීමට පෙර සැලසුම් කිරීමට සහ උපදෙස් හොඳින් කියවීමට හුරු කරන්න."
    ),
    "profile_d": (
        "මෙම ශිෂ්‍යයාට අවධානය සහ ආවේගශීලීතාවය යන දෙකෙහිම ගැටළු පවතී. "
        "එබැවින් ඉතා සරල, එක් පියවරක උපදෙස් පමණක් ලබා දෙන්න. ස්ටිකර් වැනි දිරිගැන්වීමේ ක්‍රම නිතර භාවිතා කරන්න."
    ),
}

PARENT_NOTES = {
    "profile_a": (
        "ඔබේ දරුවා 4 වන ශ්‍රේණියේ මට්ටමින් හොඳ අවධානයක් පෙන්වයි. "
        "නිවසේදී චෙස් වැනි උපායමාර්ගික ක්‍රීඩා සඳහා දරුවා යොමු කරන්න."
    ),
    "profile_b": (
        "නිවසේ ඉගෙනීමේ කටයුතු වලදී විනාඩි 20කට වඩා එක දිගට වැඩ කිරීමට බල නොකරන්න. "
        "පැහැදිලි කෙටි ඉලක්ක ලබා දී ඒවා අවසන් කිරීමට සහාය වන්න."
    ),
    "profile_c": (
        "නිවසේදී ඕනෑම ක්‍රියාවක් කිරීමට පෙර තත්පර කිහිපයක් සිතීමට දරුවාට උගන්වන්න. "
        "හුස්ම ගැනීමේ ව්‍යායාම සහ සුඩෝකු වැනි ක්‍රීඩා මෙයට උදව් වේ."
    ),
    "profile_d": (
        "දරුවාට වැඩ කිරීමේදී නිතර විවේක ලබා දෙන්න. "
        "කුඩා ජයග්‍රහණ පවා අගය කරන්න. පියවරෙන් පියවර වැඩ කිරීමට හුරු කරවන්න."
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