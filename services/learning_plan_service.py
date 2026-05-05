from datetime import datetime
from models.learning_plan_model import (
    LearningPlanRequest, AdaptationParams,
    LearningActivity, LearningPlanResponse
)
from services.db_service import get_db


# ── Adaptation parameters per profile ────────────────────────────────────────
PROFILE_PARAMS = {

    # Profile A — High Attention
    "profile_a": AdaptationParams(
        chunk_size=4,
        session_minutes=15,
        break_frequency=4,
        modality="visual_only",
        pacing="timed",
        visual_complexity="rich",
        feedback_style="visual_only",
        encouragement_level="standard",
    ),

    # Profile B — Inattention Dominant
    "profile_b": AdaptationParams(
        chunk_size=2,
        session_minutes=7,
        break_frequency=2,
        modality="audio_visual",
        pacing="self_paced",
        visual_complexity="minimal",
        feedback_style="immediate_audio",
        encouragement_level="high",
    ),

    # Profile C — Impulsivity Dominant
    "profile_c": AdaptationParams(
        chunk_size=3,
        session_minutes=10,
        break_frequency=3,
        modality="visual_only",
        pacing="timed_relaxed",
        visual_complexity="standard",
        feedback_style="haptic_visual",
        encouragement_level="high",
    ),

    # Profile D — Mixed/Low
    "profile_d": AdaptationParams(
        chunk_size=2,
        session_minutes=5,
        break_frequency=2,
        modality="audio_visual",
        pacing="self_paced",
        visual_complexity="minimal",
        feedback_style="immediate_audio",
        encouragement_level="very_high",
    ),
}


# ── Grade 3 Activity Library ──────────────────────────────────────────────────

ACTIVITY_LIBRARY = {

    "profile_a": [
        LearningActivity(
            title="කතාව කියවා ප්‍රශ්නවලට පිළිතුරු දෙන්න",
            description="කෙටි සිංහල ඡේදයක් කියවා ප්‍රශ්න 3කට පිළිතුරු දෙන්න",
            type="comprehension",
            duration_min=10,
            delivery="in_app",
            instructions=(
                "1. ඡේදය සම්පූර්ණයෙන් කියවන්න\n"
                "2. ප්‍රශ්නය කියවන්න\n"
                "3. ඡේදයෙන් පිළිතුර සොයා ගෙන තෝරන්න\n"
                "4. සෑම ප්‍රශ්නයකටම ඉහළ ලකුණු ලබා ගැනීමට උත්සාහ කරන්න"
            ),
        ),
        LearningActivity(
            title="රටාව සම්පූර්ණ කරන්න",
            description="දී ඇති රටාවේ හිස් ස්ථානය නිවැරදිව පුරවන්න",
            type="focus_builder",
            duration_min=8,
            delivery="in_app",
            instructions=(
                "1. රටාව හොඳින් බලන්න\n"
                "2. ඊළඟ රූපය හෝ අංකය කුමක්දැයි සිතන්න\n"
                "3. නිවැරදි පිළිතුර තෝරන්න\n"
                "4. රටා 5ක් නිවැරදිව සම්පූර්ණ කරන්න"
            ),
        ),
        LearningActivity(
            title="වචන අගය සෙල්ලම",
            description="වචනය කියවා නිවැරදි රූපය ඉක්මනින් ස්පර්ශ කරන්න",
            type="memory",
            duration_min=7,
            delivery="in_app",
            instructions=(
                "1. තිරයේ වචනය කියවන්න\n"
                "2. රූප 4 අතරින් ගැලපෙන රූපය සොයන්න\n"
                "3. හැකිතාක් ඉක්මනින් නිවැරදි රූපය ස්පර්ශ කරන්න\n"
                "4. 10 වතාවක් නිවැරදිව කරන්න"
            ),
        ),
        LearningActivity(
            title="ගුරුවරයා සමඟ: මනස් සිතියම",
            description="ගුරුවරයා සමඟ පාඩමක් ගැන මනස් සිතියමක් ඇඳීම",
            type="comprehension",
            duration_min=15,
            delivery="teacher_led",
            instructions=(
                "ගුරුවරයා සඳහා:\n"
                "1. පාඩමේ ප්‍රධාන වචනය මැද ලියන්න\n"
                "2. දරුවාගෙන් සම්බන්ධ වචන ඇසීම\n"
                "3. දරුවාට රේඛා ඇඳ සම්බන්ධ කිරීමට ඉඩ දෙන්න\n"
                "4. සම්පූර්ණ සිතියම ගැන කතා කරන්න"
            ),
        ),
    ],

    "profile_b": [
        LearningActivity(
            title="ශබ්දයට සවන් දී රූපය ස්පර්ශ කරන්න",
            description=(
                "ශබ්දයට සවන් දෙන්න — ශබ්දය නිම වූ පසු "
                "ගැලපෙන රූපය ස්පර්ශ කරන්න"
            ),
            type="focus_builder",
            duration_min=5,
            delivery="in_app",
            instructions=(
                "1. හෙඩ්ෆෝන් හෝ ශබ්දය සාදාගන්න\n"
                "2. ශබ්දය සම්පූර්ණයෙන් සවන් දෙන්න\n"
                "3. ශබ්දය නිම වූ පසු පමණක් ස්පර්ශ කරන්න\n"
                "4. ශබ්දය නැවත ඇසීමට ශබ්ද බොත්තම ස්පර්ශ කළ හැක"
            ),
        ),
        LearningActivity(
            title="කෙටි කතාව: මතක ක්‍රීඩාව",
            description="වාක්‍ය 2ක් කියවා ප්‍රශ්නයකට පිළිතුරු දෙන්න",
            type="memory",
            duration_min=5,
            delivery="in_app",
            instructions=(
                "1. කෙටි වාක්‍ය 2ක් ශ්‍රව්‍ය සමඟ කියවන්න\n"
                "2. කියවීම නිම වූ පසු ප්‍රශ්නය දිස්වේ\n"
                "3. ඉහළට scroll කිරීමකින් තොරව ප්‍රශ්නයට පිළිතුරු දෙන්න\n"
                "4. 5 ප්‍රශ්න නිවැරදිව කරන්න try කරන්න"
            ),
        ),
        LearningActivity(
            title="ගුරුවරයා සමඟ: හඬ නගා කියවීම",
            description="ගුරුවරයා සමඟ ඡේදයක් හඬ නගා කියවීම",
            type="comprehension",
            duration_min=7,
            delivery="teacher_led",
            instructions=(
                "ගුරුවරයා සඳහා:\n"
                "1. ගුරුවරයා පළමු වාක්‍යය කියවන්න\n"
                "2. දරුවා ඊළඟ වාක්‍යය කියවන්න (ආදේශ කිරීම)\n"
                "3. වචන 5ට වරක් නවතා තේරුම ඇසීම\n"
                "4. කියවීම අවසානයේ ප්‍රශ්න 2ක් ඇසීම\n"
                "නිර්දේශ: ඡේදය 4-5 වාක්‍ය ප්‍රමාණයට සීමා කරන්න"
            ),
        ),
        LearningActivity(
            title="දෛනික ක්‍රියාකාරකම: පින්තූර දිනපොත",
            description="දවස ගැන රූප 2ක් ඇඳ කෙටි ලෙස ලිවීම",
            type="focus_builder",
            duration_min=10,
            delivery="independent",
            instructions=(
                "දරුවා සඳහා:\n"
                "1. අද දවසේ සිදු වූ දේ 2ක් ගැන සිතන්න\n"
                "2. එක් එක් දේ ගැන රූපයක් ඇඳීම\n"
                "3. රූපය යටින් වාක්‍යයක් ලිවීම\n"
                "4. ගෙදර කෙනෙකුට පෙන්වා කතා කිරීම\n"
                "නිර්දේශ: දිනපතා 10 මිනිත්තු"
            ),
        ),
    ],

    "profile_c": [
        LearningActivity(
            title="නවතා — සිතා — ස්පර්ශ කරන්න",
            description=(
                "රූපය දිස්වූ වහාම ස්පර්ශ නොකරන්න — "
                "3 ගණන් කර ස්පර්ශ කරන්න"
            ),
            type="inhibition",
            duration_min=6,
            delivery="in_app",
            instructions=(
                "1. රූපය දිස්වූ විට 1, 2, 3 ගණන් කරන්න\n"
                "2. 3 ගණන් කිරීමෙන් පසු පමණක් ස්පර්ශ කරන්න\n"
                "3. ඉක්මනට ස්පර්ශ කළොත් ලකුණු අඩු වේ\n"
                "4. ඉවසීමෙන් ස්පර්ශ කළොත් ලකුණු දෙගුණ වේ"
            ),
        ),
        LearningActivity(
            title="සියල්ල බලා තෝරන්න",
            description=(
                "රූප 4ම හොඳින් බලා — "
                "ශ්‍රේෂ්ඨ ගැලපීම තෝරන්න"
            ),
            type="inhibition",
            duration_min=7,
            delivery="in_app",
            instructions=(
                "1. පළමු රූපය පමණක් නොබලන්න\n"
                "2. රූප 4ම ඉහළ සිට පහළට බලන්න\n"
                "3. හොඳම ගැලපීම ලකුනු කරගන්න\n"
                "4. ඉන්පසු ස්පර්ශ කරන්න\n"
                "ඉවසිලිවන්ත ලකුණු: සෑම නිවැරදි පිළිතුරකට +2"
            ),
        ),
        LearningActivity(
            title="ගුරුවරයා සමඟ: රතු එළිය — කොළ එළිය",
            description="ගුරුවරයා සමඟ ශාරීරික ඉවසීම් ක්‍රීඩාව",
            type="inhibition",
            duration_min=8,
            delivery="teacher_led",
            instructions=(
                "ගුරුවරයා සඳහා:\n"
                "1. 'කොළ' කිවහොත් — දරුවා ගෙ/ඇඟිල්ල ඔසවයි\n"
                "2. 'රතු' කිවහොත් — දරුවා නිහඩව සිටියි\n"
                "3. ගුරුවරයා ඉක්මනින් වෙනස් කිරීම\n"
                "4. රතු කිවහොත් ගෙ/ඇඟිල්ල ඔසවූවොත් restart\n"
                "5. 10 ක් නිවැරදිව කළොත් ජය\n"
                "නිර්දේශ: දිනකට 5 මිනිත්තු"
            ),
        ),
        LearningActivity(
            title="හුස්ම ගැනීමේ ව්‍යායාම",
            description="කාර්යය ආරම්භ කිරීමට පෙර හුස්ම ව්‍යායාම 3ක්",
            type="inhibition",
            duration_min=3,
            delivery="independent",
            instructions=(
                "දරුවා සඳහා:\n"
                "1. ඇස් වසාගන්න\n"
                "2. 4 ගණන් ලෙස හුස්ම ගන්න\n"
                "3. 4 ගණන් ලෙස හිඳගන්න\n"
                "4. 4 ගණන් ලෙස හුස්ම මුදා හරින්න\n"
                "5. 3 වතාවක් නැවත කරන්න\n"
                "ඕනෑම කාර්යයකට පෙර කළ හැක"
            ),
        ),
    ],

    "profile_d": [
        LearningActivity(
            title="එක් රූපයක් — ස්පර්ශ කරන්න",
            description=(
                "ශබ්දයට සවන් දී — "
                "ගැලපෙන රූපය ස්පර්ශ කරන්න (රූප 2ක් පමණ)"
            ),
            type="focus_builder",
            duration_min=4,
            delivery="in_app",
            instructions=(
                "1. ශබ්දයට සවන් දෙන්න\n"
                "2. රූප 2 ඇත — ගැලපෙන රූපය 1 ස්පර්ශ කරන්න\n"
                "3. නිවැරදි නම් — ⭐ ලකුණ ලැබේ\n"
                "4. 5 ⭐ ලැබෙනතෙක් කරන්න\n"
                "සටහන: ශබ්දය නැවත ඇසීමට බොත්තම ස්පර්ශ කළ හැක"
            ),
        ),
        LearningActivity(
            title="වර්ණ ගැලපීම",
            description="ගැලපෙන වර්ණ ගුලි ඇදලා ගැලපෙන ස්ථානයට දමන්න",
            type="categorization",
            duration_min=4,
            delivery="in_app",
            instructions=(
                "1. වම් පසේ ගුලිය ඇදගන්න\n"
                "2. දකුණු පසේ ගැලපෙන ස්ථානයට දමන්න\n"
                "3. ගුලි 4ම ගැලපෙන ස්ථානවලට දමන්න\n"
                "4. නිවැරදිව ගැලපුනොත් හොඳ ශබ්දයක් ඇසේ"
            ),
        ),
        LearningActivity(
            title="ගුරුවරයා සමඟ: ඇඟිලි ගණිතය",
            description="ගුරුවරයා සමඟ ඇඟිලි භාවිතා කර ගණිත ක්‍රීඩාව",
            type="memory",
            duration_min=5,
            delivery="teacher_led",
            instructions=(
                "ගුරුවරයා සඳහා:\n"
                "1. ඇඟිලි 1-5 ගිනිය\n"
                "2. ගුරුවරයා ඇඟිලි ගණනක් ඔසවයි\n"
                "3. දරුවා ගණන කියයි\n"
                "4. 3 ක් නිවැරදිව කිවහොත් ප්‍රශංසා කිරීම\n"
                "5. ඉන්පසු දරුවා ඇඟිලි ඔසවයි\n"
                "නිර්දේශ: ශ්‍රේෂ්ඨ ලකුණු සඳහා ස්ටිකර් ලබා දෙන්න"
            ),
        ),
        LearningActivity(
            title="දෛනික ක්‍රියාකාරකම: ශාරීරික ව්‍යායාම",
            description="10 මිනිත්තු ශාරීරික ක්‍රීඩාව",
            type="focus_builder",
            duration_min=10,
            delivery="independent",
            instructions=(
                "දරුවා සඳහා:\n"
                "1. 10 වාරය ඉහළ පිම්ම (Jumping jacks)\n"
                "2. 10 වාරය squats\n"
                "3. 10 වාරය ශ්‍රේෂ්ඨ ශ්වාස ව්‍යායාම\n"
                "4. ඉන්පසු ඉගෙනුම් කාර්යයට\n"
                "පර්යේෂණ: ශාරීරික ක්‍රීඩාවෙන් අවධානය 20% ඉහළ යයි"
            ),
        ),
    ],
}

PROFILE_LABELS = {
    "profile_a": "ඉහළ අවධානය 🌟",
    "profile_b": "අවධානය ගොඩනගමු 🎧",
    "profile_c": "ඉවසීම ගොඩනගමු ⏸️",
    "profile_d": "අවධානය වර්ධනය කරමු 💪",
}

TEACHER_NOTES = {
    "profile_a": (
        "මෙම දරුවාට ඉහළ අවධානය ඇත. සාමාන්‍ය "
        "පන්ති ක්‍රමය ප්‍රමාණවත් වේ. වඩාත් අභියෝගාත්මක "
        "කාර්යයන් ලබා දීමෙන් දරුවාගේ හැකියාව වර්ධනය කළ හැක."
    ),
    "profile_b": (
        "දරුවාට දිගු කාලය අවධානය පවත්වා ගැනීම අපහසු වේ. "
        "කාර්යයන් කෙටි කොටස්වලට බෙදා ලබා දෙන්න. "
        "ශ්‍රව්‍ය + දෘශ්‍ය ක්‍රම (audio + visual) භාවිතා කරන්න. "
        "සෑම විනාඩි 7කට වරක් කෙටි විවේකයක් ලබා දෙන්න. "
        "හැකිනම් ඉදිරිපස ආසනයක් ලබා දෙන්න."
    ),
    "profile_c": (
        "දරුවා ඉක්මනින් ප්‍රතිචාර දක්වයි — නිවැරදිව සිතීමට "
        "කාලය නොගනී. ප්‍රශ්නයට පිළිතුරු දීමට පෙර 3 ගණන් "
        "කිරීමට ඉල්ලා සිටීම. ශ්‍රේෂ්ඨ ඉවසීම ප්‍රශංසා කිරීම "
        "ඉතා වැදගත්. ඉක්මන් ප්‍රතිචාර දැක්වූ විට 'හොඳයි, "
        "නමුත් ඊළඟ වතාවේ ටිකක් සිතා පිළිතුරු දෙන්න' ලෙස "
        "කරුණාකාරව මඟ පෙන්වන්න."
    ),
    "profile_d": (
        "දරුවාට විශේෂ අවධානයක් අවශ්‍ය වේ. "
        "සියලු කාර්යයන් ඉතා කෙටිව (විනාඩි 5ට අඩු) "
        "ලබා දෙන්න. ශ්‍රේෂ්ඨ ලකුණු සඳහා ස්ටිකර් / "
        "ත්‍යාග ක්‍රමයක් භාවිතා කරන්න. "
        "දෙමාපියන් සමඟ සම්බන්ධ වී ගෙදර පුහුණුව "
        "ද සිදු කරන්න. සෞඛ්‍ය වෘත්තිකයෙකුගේ "
        "ඇගයීමක් සලකා බලන්න."
    ),
}

PARENT_NOTES = {
    "profile_a": (
        "ඔබේ දරුවා අවධානය ඉතා හොඳින් පවත්වා ගනී. "
        "කියවීම, ගණිතය, හෝ නිර්මාණශීලී ක්‍රියාකාරකම් "
        "සඳහා දිනකට 20-30 මිනිත්තු ලබා දෙන්න."
    ),
    "profile_b": (
        "ඔබේ දරුවාට ගෙදර ඉගෙනීමේදී:\n"
        "• කාර්යයන් විනාඩි 10 කොටස්වලට බෙදන්න\n"
        "• TV / ජංගම දුරකථන ඉගෙනීමේදී අවම කරන්න\n"
        "• ශාන්ත ස්ථානයක ඉගෙනීමට ඉඩ දෙන්න\n"
        "• රාත්‍රී 9ට පෙර නිදාගන්නා ලෙස සලස්වන්න"
    ),
    "profile_c": (
        "ඔබේ දරුවාට ගෙදර:\n"
        "• ස්පර්ශ කිරීමට / කිරීමට පෙර 'සිතීම' ගැන "
        "ත්‍යාග ලබා දෙන්න\n"
        "• ශාරීරික ක්‍රීඩාවට (sport) නිතිපතා ඉඩ දෙන්න\n"
        "• ඉවසීම ප්‍රශංසා කිරීම\n"
        "• Chess, puzzles ක්‍රීඩා දිරිමත් කරන්න"
    ),
    "profile_d": (
        "ඔබේ දරුවාට ගෙදර:\n"
        "• දිනකට 10 මිනිත්තු ශාරීරික ව්‍යායාම\n"
        "• ඉගෙනීම කෙටි (5 min) කොටස්වලට\n"
        "• ශ්‍රේෂ්ඨ ලකුණු සඳහා ස්ටිකර් ලබා දෙන්න\n"
        "• ගුරුවරයා සමඟ නිතිපතා සම්බන්ධ වෙන්න\n"
        "• ළමා රෝගී විශේෂඥ ඇගයීමක් ගැන සලකා බලන්න"
    ),
}


def generate_learning_plan(req: LearningPlanRequest) -> LearningPlanResponse:
    profile    = req.attention_profile
    params     = PROFILE_PARAMS.get(profile,     PROFILE_PARAMS["profile_d"])
    activities = ACTIVITY_LIBRARY.get(profile, ACTIVITY_LIBRARY["profile_d"])
    note       = TEACHER_NOTES.get(profile,     TEACHER_NOTES["profile_d"])
    p_note     = PARENT_NOTES.get(profile,      PARENT_NOTES["profile_d"])
    label      = PROFILE_LABELS.get(profile,    "ඔබේ ඉගෙනුම් සැලැස්ම")

    plan = LearningPlanResponse(
        child_id          = req.child_id,
        grade             = req.grade,
        profile           = profile,
        profile_label     = label,
        adaptation_params = params,
        activities        = activities,
        teacher_note      = note,
        parent_note       = p_note,
        generated_at      = datetime.utcnow().isoformat(),
    )

    db = get_db()
    db["learning_plans"].insert_one({
        **plan.model_dump(),            # ✅ Fixed: was plan.dict() — Pydantic v2
        "created_at": datetime.utcnow(),
    })

    return plan


def get_latest_plan(child_id: str):
    db   = get_db()
    plan = db["learning_plans"].find_one(
        {"child_id": child_id},
        sort=[("created_at", -1)],
    )
    if plan:
        plan["_id"] = str(plan["_id"])
        # ✅ Fixed: convert raw datetime to string before returning
        if "created_at" in plan and hasattr(plan["created_at"], "isoformat"):
            plan["created_at"] = plan["created_at"].isoformat()
    return plan