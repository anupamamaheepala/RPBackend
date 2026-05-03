"""
Grade 5 Learning Plan Service — Ages 10-11 years
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
            instructions="Spot the Change: Identify 2 changes in 7s. Filter: Tap green, ignore blue in 4x4 grid. 20 trials."),
        G5LearningActivity(title="ගුරු-නිර්දේශිත සටහන් ගැනීම",
            description="Guided note-taking — structured listening and writing",
            type="comprehension", duration_min=10, delivery="teacher_led",
            instructions="ගුරුවරයා කෙටි කොටසක් කියවයි. ශිෂ්‍යයා ප්‍රධාන කරුණු 3ක් ලියයි. කිසිදු යෙදුම් සහය නොමැතිව."),
        G5LearningActivity(title="මතකයෙන් මනස-සිතියම",
            description="Mind map from memory — Grade 5 comprehension extension",
            type="memory", duration_min=8, delivery="independent",
            instructions="කෙටි ඡේදයක් 1 වතාවක් කියවා, පොත වසා, අදහස් සිතියමක් අඳින්න."),
        G5LearningActivity(title="Switch Go Level 2 Strategy",
            description="Cognitive flexibility training with self-monitoring",
            type="inhibition", duration_min=10, delivery="independent",
            instructions="Switch Go task: Rules change every 10 trials. Record results in the self-monitoring diary."),
    ],
    "profile_b": [
        G5LearningActivity(title="Audio Sequence Level 3 + Filter",
            description="Inattention — auditory sequencing and selective attention",
            type="memory", duration_min=10, delivery="in_app",
            instructions="Audio Sequence: 4 sentences, 1 play only, order 4 items. Filter: Tap green only in 4x4 grid."),
        G5LearningActivity(title="ගුරු-නිර්දේශිත ස්මරණ",
            description="Teacher-directed recall — structured comprehension",
            type="comprehension", duration_min=8, delivery="teacher_led",
            instructions="ගුරුවරයා ප්‍රශ්න ඇසීමෙන් ශිෂ්‍යයාගේ අවධානය යොමු කරවයි. මෙය සෑම විනාඩි 10කටම වරක් සිදු කරන්න."),
        G5LearningActivity(title="ඡේදය ලිවීම — මතකයෙන්",
            description="Paragraph writing from memory — 5th grade level",
            type="comprehension", duration_min=8, delivery="independent",
            instructions="කෙටි ඡේදයක් කියවා, පොත වසා, වාක්‍ය 3කින් ලියන්න. ශබ්දකෝෂ භාවිතා නොකරන්න."),
        G5LearningActivity(title="Sticker ත්‍යාග ක්‍රමය — Grade 5",
            description="Daily focus reward chart — age-appropriate",
            type="focus_builder", duration_min=5, delivery="independent",
            instructions="දිනකට ඉලක්ක 3ක් ලියා ඒවා සම්පූර්ණ කළ විට ස්ටිකර් අලවන්න. සතිය අවසානයේ ත්‍යාගයක් ලබා දෙන්න."),
    ],
    "profile_c": [
        G5LearningActivity(title="Switch Go + Stillness",
            description="Cognitive flexibility and motor inhibition — targets impulsivity",
            type="inhibition", duration_min=10, delivery="in_app",
            instructions="Switch Go Level 1: Animal/vehicle rules. Stillness: Hold for 30s without lifting a finger."),
        G5LearningActivity(title="පාලිත විවාදය — පිළිතුරු රැඳීම",
            description="Controlled debate — waiting turn before responding",
            type="inhibition", duration_min=10, delivery="teacher_led",
            instructions="ශිෂ්‍යයා අදහස් ප්‍රකාශ කිරීමට පෙර සෙසු අය අවසන් කරන තෙක් බලා සිටිය යුතුය."),
        G5LearningActivity(title="Mindfulness දිනපොත",
            description="Daily mindfulness journal — impulse control practice",
            type="inhibition", duration_min=8, delivery="independent",
            instructions="ප්‍රතිචාර දැක්වීමට පෙර ගැඹුරු හුස්ම 3ක් ගන්න. 'මා ඉවසූ අවස්ථා' දිනපොතේ ලියන්න."),
        G5LearningActivity(title="Strategy Board Game",
            description="Chess or strategy puzzles — planning before acting",
            type="inhibition", duration_min=10, delivery="independent",
            instructions="චෙස්, සුඩෝකු හෝ උපායමාර්ගික ප්‍රහේලිකා. පියවරක් ගැනීමට පෙර තත්පර 5ක් සිතන්න."),
    ],
    "profile_d": [
        G5LearningActivity(title="Filter Level 1 + Ladder",
            description="Basic filtering and sequential following — mixed deficits",
            type="focus_builder", duration_min=8, delivery="in_app",
            instructions="Filter Level 1: 4x4 grid, 20 trials. Ladder: Step-by-step instructions, one step at a time."),
        G5LearningActivity(title="අත්-ව්‍යවහාරික වර්ගීකරණ කාර්ය",
            description="Hands-on sorting — physical engagement for mixed profile",
            type="focus_builder", duration_min=7, delivery="teacher_led",
            instructions="කාඩ් හෝ වස්තු කාණ්ඩ 2කට වර්ග කිරීමට ශිෂ්‍යයාට ලබා දෙන්න. එක් වරකට එක් සරල උපදෙසක් පමණක් ලබා දෙන්න."),
        G5LearningActivity(title="Drawing + 1 Sentence",
            description="Drawing and one-sentence description — creative low-load task",
            type="focus_builder", duration_min=7, delivery="independent",
            instructions="මෑතකදී ඉගෙන ගත් දෙයක් රූපයකින් ඇඳ, එක් වාක්‍යයකින් විස්තර කරන්න."),
        G5LearningActivity(title="Physical Movement Break + Short Task",
            description="Movement then focus — resets attention for mixed profile",
            type="focus_builder", duration_min=10, delivery="independent",
            instructions="විනාඩි 5ක් ශාරීරික ව්‍යායාම කරන්න. ඉන්පසු විනාඩි 5ක් ගැටළු 3ක් විසඳන්න. මෙය මාරුවෙන් මාරුවට සිදු කරන්න."),
    ],
}

TEACHER_NOTES = {
    "profile_a": "Grade 5 ශිෂ්‍යයා ඉහළ අවධානයක් පෙන්වයි. සංකීර්ණ ව්‍යාපෘති සහ විවේචනාත්මක චින්තනය අවශ්‍ය ගැටළු ලබා දෙන්න. Switch Go Level 3 සහ Spot the Change Level 3 නිර්දේශ කෙරේ.",
    "profile_b": "Grade 5 — ශිෂ්‍යයාට දිගු වේලාවක් අවධානය තබා ගැනීම අපහසුය. පාඩම් කොටස් 3-4 කට බෙදන්න. සෑම විනාඩි 10කටම වරක් ප්‍රශ්නයක් අසා අවධානය පරීක්ෂා කරන්න.",
    "profile_c": "Grade 5 — ආවේගශීලීතාවය පාලනය කිරීම අවශ්‍ය වේ. 'Think-Aloud' ක්‍රමය භාවිතා කරන්න. Switch Go වැරදි අනුපාතය (error rate) කෙරෙහි අවධානය යොමු කරන්න.",
    "profile_d": "Grade 5 — ආවේගශීලීතාවය සහ අවධානය යන දෙකෙහිම ගැටළු ඇත. එක් වරකදී එක් පියවරක් පමණක් ලබා දෙන්න. Stillness task සාර්ථකත්වය පිළිබඳව දෙමාපියන් දැනුවත් කරන්න.",
}

PARENT_NOTES = {
    "profile_a": "ඔබේ දරුවා Grade 5 මට්ටමේ හොඳ අවධානයක් පෙන්වයි. Switch Go සහ Filter tasks නිවසේදී ක්‍රීඩා ලෙස පුහුණු කරවන්න.",
    "profile_b": "නිවසේ ඉගෙනීමේදී: කෙටි සහ පැහැදිලි ඉලක්ක ලබා දෙන්න. ස්ටිකර් ත්‍යාග ක්‍රමය (Sticker reward chart) භාවිතා කිරීම දිරිමත් කරන්න.",
    "profile_c": "නිවසේ ඉගෙනීමේදී: හුස්ම ගැනීමේ ව්‍යායාම (Mindfulness), චෙස් සහ සුඩෝකු වැනි උපායමාර්ගික ක්‍රීඩා සඳහා දරුවා යොමු කරන්න.",
    "profile_d": "නිවසේ ඉගෙනීමේදී: වැඩ සහ විවේකය මාරුවෙන් මාරුවට ලබා දෙන්න. පියවරෙන් පියවර උපදෙස් ලබා දීම වඩාත් සාර්ථක වේ.",
}