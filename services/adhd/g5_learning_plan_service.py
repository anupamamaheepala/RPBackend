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
            instructions="Spot the Change: Identify 2 changes within 7s. Filter: Tap green, ignore blue in a 4x4 grid. 20 trials."),
        G5LearningActivity(title="ගුරු-නිර්දේශිත සටහන් ගැනීම",
            description="Guided note-taking — structured listening and writing",
            type="comprehension", duration_min=10, delivery="teacher_led",
            instructions="ගුරුවරයා කෙටි කොටසක් කියවයි. ශිෂ්‍යයා ප්‍රධාන කරුණු 3ක් ලියයි. කිසිදු බාහිර සහයකින් තොරව මෙය සිදු කරන්න."),
        G5LearningActivity(title="මතකයෙන් මනස-සිතියම",
            description="Mind map from memory — Grade 5 comprehension extension",
            type="memory", duration_min=8, delivery="independent",
            instructions="කෙටි ඡේදයක් එක් වරක් කියවා, පොත වසා, එහි අදහස් මනස-සිතියමකින් (Mind map) දක්වන්න."),
        G5LearningActivity(title="Switch Go Level 2 Strategy",
            description="Cognitive flexibility training with self-monitoring",
            type="inhibition", duration_min=10, delivery="independent",
            instructions="Switch Go task: නීති සෑම වාර 10කටම වරක් වෙනස් වේ. ඔබේ ප්‍රගතිය ස්වයං-ලකුණු දිනපොතේ සටහන් කරන්න."),
    ],
    "profile_b": [
        G5LearningActivity(title="Audio Sequence Level 3 + Filter",
            description="Inattention — auditory sequencing and selective attention",
            type="memory", duration_min=10, delivery="in_app",
            instructions="Audio Sequence: වාක්‍ය 4කට සවන් දී (එක් වරක් පමණි) ඒවා නිවැරදි පිළිවෙලට සකසන්න. Filter: 4x4 grid එකක කොළ පැහැය පමණක් තෝරන්න."),
        G5LearningActivity(title="ගුරු-නිර්දේශිත ස්මරණ",
            description="Teacher-directed recall — structured comprehension",
            type="comprehension", duration_min=8, delivery="teacher_led",
            instructions="ගුරුවරයා ප්‍රශ්න ඇසීමෙන් ශිෂ්‍යයාගේ අවධානය යොමු කරවයි. මෙය සෑම විනාඩි 10කටම වරක් සිදු කරන්න."),
        G5LearningActivity(title="ඡේදය ලිවීම — මතකයෙන්",
            description="Paragraph writing from memory — 5th grade level",
            type="comprehension", duration_min=8, delivery="independent",
            instructions="කෙටි ඡේදයක් කියවා, පොත වසා, එය වාක්‍ය 3කින් ලියන්න. වෙනත් සහායක පොත් භාවිතා නොකරන්න."),
        G5LearningActivity(title="ස්ටිකර් ත්‍යාග ක්‍රමය — Grade 5",
            description="Daily focus reward chart — age-appropriate",
            type="focus_builder", duration_min=5, delivery="independent",
            instructions="දිනකට ඉලක්ක 3ක් ලියා ඒවා සම්පූර්ණ කළ විට ස්ටිකරයක් අලවන්න. සතිය අවසානයේ විශේෂ ත්‍යාගයක් ලබා දෙන්න."),
    ],
    "profile_c": [
        G5LearningActivity(title="Switch Go + Stillness",
            description="Cognitive flexibility and motor inhibition — targets impulsivity",
            type="inhibition", duration_min=10, delivery="in_app",
            instructions="Switch Go Level 1: සතුන්/වාහන නීති අනුගමනය කරන්න. Stillness: තත්පර 30ක් ඇඟිල්ල එසවීමෙන් තොරව නිශ්චලව සිටින්න."),
        G5LearningActivity(title="පාලිත විවාදය — පිළිතුරු රැඳීම",
            description="Controlled debate — must wait turn before responding",
            type="inhibition", duration_min=10, delivery="teacher_led",
            instructions="ශිෂ්‍යයා අදහස් ප්‍රකාශ කිරීමට පෙර අනෙක් අය තම අදහස් දක්වා අවසන් වන තෙක් ඉවසිල්ලෙන් බලා සිටිය යුතුය."),
        G5LearningActivity(title="Mindfulness දිනපොත",
            description="Daily mindfulness journal — impulse control practice",
            type="inhibition", duration_min=8, delivery="independent",
            instructions="ප්‍රතිචාර දැක්වීමට පෙර ගැඹුරු හුස්ම 3ක් ගන්න. දිනපොතේ 'මා ඉවසීම පුරුදු කළ අවස්ථා' ලියන්න."),
        G5LearningActivity(title="Strategy Board Game",
            description="Chess or strategy puzzle — plans before acting",
            type="inhibition", duration_min=10, delivery="independent",
            instructions="චෙස්, සුඩෝකු හෝ වෙනත් උපායමාර්ගික ප්‍රහේලිකා විසඳන්න. පියවරක් ගැනීමට පෙර තත්පර 5ක් සිතන්න."),
    ],
    "profile_d": [
        G5LearningActivity(title="Filter Level 1 + Ladder",
            description="Basic filtering and sequential following — both deficits",
            type="focus_builder", duration_min=8, delivery="in_app",
            instructions="Filter Level 1: 4x4 grid එකක වාර 20ක් සිදු කරන්න. Ladder: පියවරෙන් පියවර උපදෙස් අනුගමනය කරන්න."),
        G5LearningActivity(title="අත්-ව්‍යවහාරික වර්ගීකරණ කාර්ය",
            description="Hands-on sorting — physical engagement for mixed profile",
            type="focus_builder", duration_min=7, delivery="teacher_led",
            instructions="ගුරුවරයා විසින් ලබාදෙන කාඩ් හෝ වස්තු කාණ්ඩ 2කට වර්ග කරන්න. එක් වරකට එක් සරල උපදෙසක් පමණක් අනුගමනය කරන්න."),
        G5LearningActivity(title="Drawing + 1 Sentence",
            description="Drawing and one sentence description — creative low-load task",
            type="focus_builder", duration_min=7, delivery="independent",
            instructions="මෑතකදී ඉගෙන ගත් දෙයක් රූපයකින් ඇඳ, එය එක් වාක්‍යයකින් විස්තර කරන්න."),
        G5LearningActivity(title="ශාරීරික විවේකය + කෙටි කාර්යයන්",
            description="Movement then focus — resets attention for mixed profile",
            type="focus_builder", duration_min=10, delivery="independent",
            instructions="විනාඩි 5ක් ශාරීරික ව්‍යායාම කරන්න. ඉන්පසු විනාඩි 5ක් ගැටළු 3ක් විසඳන්න. මෙය මාරුවෙන් මාරුවට සිදු කරන්න."),
    ],
}

TEACHER_NOTES = {
    "profile_a": "5 වන ශ්‍රේණියේ මෙම ශිෂ්‍යයා ඉහළ අවධානයක් පෙන්වයි. ඔහුට සංකීර්ණ ව්‍යාපෘති සහ විවේචනාත්මක චින්තනය (critical thinking) අවශ්‍ය ගැටළු ලබා දෙන්න. Switch Go සහ Spot the Change Level 3 නිර්දේශ කෙරේ.",
    "profile_b": "5 වන ශ්‍රේණිය — ශිෂ්‍යයාට දිගු වේලාවක් අවධානය තබා ගැනීම අපහසුය. පාඩම් කොටස් 3-4 කට බෙදන්න. සෑම විනාඩි 10කටම වරක් ප්‍රශ්නයක් අසා අවධානය පරීක්ෂා කරන්න.",
    "profile_c": "5 වන ශ්‍රේණිය — ශිෂ්‍යයාගේ ආවේගශීලීතාවය පාලනය කළ යුතුය. 'Think-Aloud' ක්‍රමය ගුරු සහාය ඇතිව සිදු කරන සෑම කාර්යයකදීම භාවිතා කරන්න. Switch Go වැරදි අනුපාතය (error rate) නිරීක්ෂණය කරන්න.",
    "profile_d": "5 වන ශ්‍රේණිය — ශිෂ්‍යයාට ආවේගශීලීතාවය සහ අවධානය යන දෙකෙහිම ගැටළු ඇත. එක් වරකදී එක් පියවරක් පමණක් ලබා දෙන්න. Stillness task සාර්ථකත්වය පිළිබඳව දෙමාපියන් දැනුවත් කරන්න.",
}

PARENT_NOTES = {
    "profile_a": "ඔබේ දරුවා 5 වන ශ්‍රේණියේ මට්ටමට අනුව ඉතා හොඳ අවධානයක් පෙන්වයි. Switch Go සහ Filter වැනි ක්‍රියාකාරකම් නිවසේදී ක්‍රීඩා ලෙස පුහුණු කරවන්න.",
    "profile_b": "නිවසේ ඉගෙනීමේදී: ශ්‍රව්‍ය කාර්යයන් (Audio tasks) නැවත ශ්‍රවණය කරන වාර ගණන නිරීක්ෂණය කරන්න. කෙටි සහ පැහැදිලි ඉලක්ක ලබා දෙන්න. ස්ටිකර් ත්‍යාග ක්‍රමය භාවිතා කරන්න.",
    "profile_c": "නිවසේ ඉගෙනීමේදී: හුස්ම ගැනීමේ ව්‍යායාම, චෙස් සහ සුඩෝකු වැනි උපායමාර්ගික ක්‍රීඩා සඳහා දරුවා යොමු කරන්න. ඉක්මන් ප්‍රතිචාර දැක්වීමට පෙර සිතීමට හුරු කරන්න.",
    "profile_d": "නිවසේ ඉගෙනීමේදී: වැඩ සහ විවේකය මාරුවෙන් මාරුවට ලබා දෙන්න. ඉතා කෙටි පියවර වශයෙන් උපදෙස් ලබා දෙන්න. දරුවාගේ කුඩා ප්‍රගතිය පවා අගය කරන්න.",
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