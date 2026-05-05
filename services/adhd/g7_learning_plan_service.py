"""
Grade 7 Learning Plan Service — age 12-13 years
Profile A: chunk=6, session=25min
Profile B: chunk=3, session=12min
Profile C: chunk=5, session=15min
Profile D: chunk=3, session=10min
"""
from datetime import datetime
from models.adhd.g7_learning_plan_model import (
    G7LearningPlanRequest, G7LearningPlanResponse,
    G7AdaptationParams, G7LearningActivity,
)
from services.db_service import get_db

PROFILE_PARAMS = {
    "profile_a": dict(chunk_size=6, session_minutes=25, break_frequency=25,
                      modality="visual_only",  pacing="timed",
                      visual_complexity="high", feedback_style="visual",
                      encouragement_level="standard"),
    "profile_b": dict(chunk_size=3, session_minutes=12, break_frequency=12,
                      modality="audio_visual", pacing="self_paced",
                      visual_complexity="low",  feedback_style="immediate_audio",
                      encouragement_level="high"),
    "profile_c": dict(chunk_size=5, session_minutes=15, break_frequency=15,
                      modality="visual_only",  pacing="timed_relaxed",
                      visual_complexity="medium", feedback_style="haptic_visual",
                      encouragement_level="high"),
    "profile_d": dict(chunk_size=2, session_minutes=10,  break_frequency=10,
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
        G7LearningActivity(
            title="Vigilance Level 3 + Distraction Resistance",
            description="Advanced sustained attention under high distraction — Grade 7 challenge",
            type="focus_builder", duration_min=15, delivery="in_app",
            instructions="Vigilance: 20% target rate, 150 trials. Distraction Resistance: 3 minutes, ignore yellow popups. Track hit rate and errors across sessions."),
        G7LearningActivity(
            title="ස්වාධීන කාල-නිශ්චිත කියවීම",
            description="Independent timed reading with comprehension questions",
            type="comprehension", duration_min=12, delivery="independent",
            instructions="සංකීර්ණ ඡේදයක් එක් වරක් කියවා, නියමිත කාලය තුළ ප්‍රශ්න 5කට පිළිතුරු දෙන්න. ශබ්දකෝෂ සහාය ලබා නොගන්න."),
        G7LearningActivity(
            title="ශිෂ්‍ය-මෙහෙයවන ඉගැන්වීම් කාර්යය",
            description="Peer teaching with defined roles — builds sustained attention",
            type="comprehension", duration_min=12, delivery="teacher_led",
            instructions="ශිෂ්‍යයාට 'විග්‍රහ කරන්නා' භූමිකාව ලබා දෙන්න. ඔහු විසින් සෙසු සිසුන්ට අදාළ විෂය කොටස් 3ක් පැහැදිලි කළ යුතුය."),
        G7LearningActivity(
            title="දෛනික සැලසුම්කරණය",
            description="Daily structured planner — executive function builder",
            type="focus_builder", duration_min=10, delivery="independent",
            instructions="දිනකට ඉලක්ක 5ක් ලියා ප්‍රමුඛතාවය අනුව පෙළගස්වන්න. සෑම ඉලක්කයකටම නිශ්චිත කාලසීමාවක් වෙන් කරන්න."),
    ],
    "profile_b": [
        G7LearningActivity(
            title="Divided Attention Training + Attention Grid",
            description="Dual-task and sustained attention — targets inattention",
            type="memory", duration_min=12, delivery="in_app",
            instructions="Divided Attention Level 2: Visual and audio dual-task. Attention Grid 5x5: 20s duration. Record scores in the app diary."),
        G7LearningActivity(
            title="ගුරු-නිර්දේශිත ස්මරණ ක්‍රියාකාරකම",
            description="Teacher-directed recall game — structured attention rebuilding",
            type="comprehension", duration_min=12, delivery="teacher_led",
            instructions="ගුරුවරයා සෑම විනාඩි 12කට වරක් ශිෂ්‍යයාගෙන් ඉගෙන ගත් කරුණු පිළිබඳ ප්‍රශ්න අසා අවධානය පරීක්ෂා කළ යුතුය."),
        G7LearningActivity(
            title="ස්වාධීන කාලානුක්‍රමික ප්‍රශ්නාවලිය",
            description="Independent timed quiz — builds sustained reading focus",
            type="comprehension", duration_min=12, delivery="independent",
            instructions="ප්‍රශ්න 10කින් යුත් ප්‍රශ්නාවලියකට නියමිත කාලය තුළ පිළිතුරු දෙන්න. සටහන් බැලීමෙන් තොරව මෙය සිදු කරන්න."),
        G7LearningActivity(
            title="ව්‍යුහගත ගෙදර වැඩ සැලසුම",
            description="Structured daily homework plan — accountability and focus",
            type="focus_builder", duration_min=10, delivery="independent",
            instructions="නිවසේදී: විනාඩි 12ක වැඩ + විනාඩි 3ක විවේකය යන චක්‍රය අනුගමනය කරන්න. ප්‍රගතිය ස්ටිකර් සටහනක සලකුණු කරන්න."),
    ],
    "profile_c": [
        G7LearningActivity(
            title="Complex Inhibition + Wait & Match Level 3",
            description="Advanced inhibition training — targets impulsivity",
            type="inhibition", duration_min=12, delivery="in_app",
            instructions="Go/No-Go Level 3: Dual-rule stimuli, 40 trials. Wait & Match Level 3: 5 options. Monitor the false alarm rate closely."),
        G7LearningActivity(
            title="මන්දගාමී විග්‍රහ කිරීමේ පුහුණුව",
            description="Peer teaching with mandatory slow explanation — impulse control",
            type="inhibition", duration_min=15, delivery="teacher_led",
            instructions="ශිෂ්‍යයා විසින් සෙසු සිසුවෙකුට ගණිත ක්‍රමයක් ඉගැන්විය යුතුය. මෙහිදී ඉතා සෙමින් සහ පැහැදිලිව විග්‍රහ කිරීමට උපදෙස් දෙන්න."),
        G7LearningActivity(
            title="උපායමාර්ගික බෝඩ් ක්‍රීඩා",
            description="Chess or advanced puzzle — plans before acting",
            type="inhibition", duration_min=15, delivery="independent",
            instructions="චෙස් හෝ සංකීර්ණ ප්‍රහේලිකා විසඳන්න. සෑම පියවරකටම පෙර විය හැකි ප්‍රතිඵල 3ක් ගැන සිතීමේ රීතිය අනුගමනය කරන්න."),
        G7LearningActivity(
            title="දෛනික ආවර්ජනය සහ සැලසුම්කරණය",
            description="Structured daily planner with impulse reflection",
            type="inhibition", duration_min=10, delivery="independent",
            instructions="දිනකට ඉලක්ක 5ක් ලියන්න. පසුව ඊයේ දිනයේදී ආවේගශීලීව ගත් එක් තීරණයක් සහ එය පාලනය කළ හැකිව තිබූ ආකාරය සටහන් කරන්න."),
    ],
    "profile_d": [
        G7LearningActivity(
            title="Vigilance Level 1 + Wait & Match Level 1",
            description="Basic sustained attention and impulse control — mixed deficits",
            type="focus_builder", duration_min=10, delivery="in_app",
            instructions="Vigilance: 10 trials, 20% target. Wait & Match: 15 trials. Record scores and show the trend to the teacher."),
        G7LearningActivity(
            title="ගුරු-ශිෂ්‍ය පෞද්ගලික සාකච්ඡාව",
            description="One-to-one teacher check-in — structured support",
            type="focus_builder", duration_min=10, delivery="teacher_led",
            instructions="ගුරුවරයා ශිෂ්‍යයා සමඟ පෞද්ගලිකව සාකච්ඡා කර පසුගිය සතියේ අභියෝග සහ ඉදිරි සතියේ ඉලක්ක 2ක් ලේඛනගත කළ යුතුය."),
        G7LearningActivity(
            title="ව්‍යුහගත ගෙදර ඉගෙනුම් කාලසටහන",
            description="Structured homework schedule — accountability system",
            type="focus_builder", duration_min=10, delivery="independent",
            instructions="විනාඩි 10ක වැඩ + විනාඩි 3ක විවේකය. නිහඬ ස්ථානයක පමණක් වැඩ කරන්න. ස්ටිකර් පුවරුවක් මගින් දිරිගැන්වීම් ලබා දෙන්න."),
        G7LearningActivity(
            title="ශාරීරික ක්‍රියාකාරකම් සහ විශ්ලේෂණය",
            description="Movement break then short analytical task — resets dual deficit",
            type="focus_builder", duration_min=10, delivery="independent",
            instructions="විනාඩි 5ක ව්‍යායාමයකින් පසු වහාම ගණිත ගැටළු 3ක් විසඳන්න. ව්‍යායාමයෙන් පසු ප්‍රමාදයකින් තොරව වැඩ ආරම්භ කිරීම වැදගත් වේ."),
    ],
}

TEACHER_NOTES = {
    "profile_a": (
        "7 වන ශ්‍රේණියේ මෙම ශිෂ්‍යයා ඉතා හොඳ අවධානයක් පෙන්වයි. ඔහුට සංකීර්ණ ව්‍යාපෘති සහ විවේචනාත්මක චින්තනය අවශ්‍ය ගැටළු ලබා දෙන්න. "
        "Vigilance hit rate සහ distraction resistance ලකුණු මට්ටම් පිළිබඳව අවධානයෙන් සිටින්න. "
        "ඔහුව 'Peer Teacher' ලෙස යොදා ගැනීමෙන් ඔහුගේ නායකත්ව ගුණාංග ද වර්ධනය වේ."
    ),
    "profile_b": (
        "මෙම ශිෂ්‍යයාට දිගු වේලාවක් අවධානය පවත්වා ගැනීම අභියෝගාත්මක විය හැක. "
        "එබැවින් පාඩම් කොටස් කිහිපයකට බෙදා උගන්වන්න. සෑම විනාඩි 12කටම වරක් ප්‍රශ්නයක් අසා අවධානය පරීක්ෂා කරන්න. "
        "පන්තියේ ඉදිරිපස අසුනක් ලබා දීම සහ විභාග වලදී අමතර කාලය ලබා දීම සලකා බලන්න."
    ),
    "profile_c": (
        "ශිෂ්‍යයාගේ ආවේගශීලීතාවය පාලනය කිරීමට 'Think-Aloud' ක්‍රමය ඉතා වැදගත් වේ. "
        "වැඩක් ආරම්භ කිරීමට පෙර සැලසුම් කිරීමට ඔහුට උපකාර කරන්න. "
        "Peer teaching ක්‍රියාකාරකම් මගින් ඔහුගේ ඉක්මන් ප්‍රතිචාර දැක්වීමේ ස්වභාවය අඩු කළ හැක."
    ),
    "profile_d": (
        "මෙම ශිෂ්‍යයාට අවධානය සහ ආවේගශීලීතාවය යන දෙකෙහිම ගැටළු පවතී. "
        "එක් වරකදී එක් සරල පියවරක් පමණක් අඩංගු කාර්යයන් ලබා දෙන්න. "
        "සෑම සතියකම පෞද්ගලිකව සාකච්ඡා කර (1-on-1 check-in) ඔහුගේ ප්‍රගතිය ඇගයීමට ලක් කරන්න."
    ),
}

PARENT_NOTES = {
    "profile_a": (
        "ඔබේ දරුවා 7 වන ශ්‍රේණියේ මට්ටමින් ඉතා හොඳ අවධානයක් පෙන්වයි. "
        "චෙස්, උපායමාර්ගික ක්‍රීඩා සහ දෛනික සැලසුම්කරණය සඳහා ඔහු දිරිගන්වන්න. "
        "Vigilance app එකෙහි ඇති ඉහළ මට්ටමේ කාර්යයන් ඔහුට ලබා දෙන්න."
    ),
    "profile_b": (
        "නිවසේ ඉගෙනීමේදී: නියමිත කාල සීමාවන් (Time blocks) සහිතව වැඩ කිරීමට හුරු කරන්න. "
        "ව්‍යුහගත ගෙදර වැඩ කාලසටහනක් (Structured homework schedule) සකස් කර ස්ටිකර් මගින් ප්‍රගතිය ඇගයීමට ලක් කරන්න."
    ),
    "profile_c": (
        "නිවසේ ඉගෙනීමේදී: චෙස්, සුඩෝකු සහ උපායමාර්ගික ප්‍රහේලිකා විසඳීමට අවස්ථාව ලබා දෙන්න. "
        "තීරණයක් ගැනීමට පෙර අවම වශයෙන් තත්පර කිහිපයක් හෝ සිතීමට ඔහුව පුහුණු කරවන්න."
    ),
    "profile_d": (
        "නිවසේ ඉගෙනීමේදී: ඉතා කෙටි කාල සීමාවන් තුළ වැඩ කිරීමට ඉඩ දෙන්න. "
        "වැඩ මාරු කිරීමේදී ශාරීරික ක්‍රියාකාරකමකට (Movement break) අවස්ථාව ලබා දෙන්න. "
        "Vigilance score ලකුණු මට්ටම් පිළිබඳව නිරන්තර අවධානයෙන් සිටින්න."
    ),
}

def g7_generate_learning_plan(req: G7LearningPlanRequest) -> G7LearningPlanResponse:
    profile = req.attention_profile.lower()
    if profile not in PROFILE_PARAMS:
        profile = "profile_b"
    plan = G7LearningPlanResponse(
        child_id          = req.child_id,
        grade             = 7,
        profile           = profile,
        profile_label     = PROFILE_LABELS.get(profile, profile),
        adaptation_params = G7AdaptationParams(**PROFILE_PARAMS[profile]),
        activities        = ACTIVITY_LIBRARY.get(profile, []),
        teacher_note      = TEACHER_NOTES.get(profile, ""),
        parent_note       = PARENT_NOTES.get(profile, ""),
        generated_at      = datetime.utcnow().isoformat(),
    )
    db = get_db()
    db["learning_plans"].insert_one({**plan.dict(), "created_at": datetime.utcnow()})
    return plan

def g7_get_latest_plan(child_id: str):
    db   = get_db()
    plan = db["learning_plans"].find_one(
        {"child_id": child_id, "grade": 7}, sort=[("created_at", -1)])
    if not plan: return None
    plan.pop("_id", None); plan.pop("created_at", None)
    return plan