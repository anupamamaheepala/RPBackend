# services/dyslexia/skill_analyzer.py

from typing import Dict, List, Any


def analyze_skill_weakness(
    reference_text: str,
    transcript_text: str,
    metrics: Dict[str, Any],
    eye_metrics: Dict[str, Any] | None = None,
    xai_feedback: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Advanced XAI layer:
    Converts reading mistakes into understandable improvement areas.

    This does not replace the ML dyslexia risk model.
    It explains what reading skill the student should improve.
    """

    eye_metrics = eye_metrics or {}
    xai_feedback = xai_feedback or []

    ref_words = reference_text.strip().split()
    stu_words = transcript_text.strip().split()

    wer = float(metrics.get("wer", metrics.get("WER", 0)) or 0)
    cer = float(metrics.get("cer", metrics.get("CER", 0)) or 0)
    wps = float(metrics.get("words_per_second", metrics.get("wps", 0)) or 0)

    regression_count = float(
        eye_metrics.get("regression_count", eye_metrics.get("avg_regression_count", 0)) or 0
    )

    avg_fixation_ms = float(
        eye_metrics.get("avg_fixation_ms", eye_metrics.get("avg_fixation_time", 0)) or 0
    )

    scores = {
        "phonological_awareness": 0,
        "word_recognition": 0,
        "reading_fluency": 0,
        "sentence_tracking": 0,
        "visual_tracking": 0,
        "character_discrimination": 0,
    }

    reasons = {
        "phonological_awareness": [],
        "word_recognition": [],
        "reading_fluency": [],
        "sentence_tracking": [],
        "visual_tracking": [],
        "character_discrimination": [],
    }

    # -------------------------------
    # 1. Word-level mistake checking
    # -------------------------------
    min_len = min(len(ref_words), len(stu_words))

    substitution_count = 0

    for i in range(min_len):
        if ref_words[i] != stu_words[i]:
            substitution_count += 1
            scores["word_recognition"] += 2
            reasons["word_recognition"].append(
                "Some words were read differently from the given sentence."
            )

    # Missing words
    if len(stu_words) < len(ref_words):
        missing_count = len(ref_words) - len(stu_words)
        scores["sentence_tracking"] += missing_count * 2
        reasons["sentence_tracking"].append(
            "Some words may have been skipped while reading."
        )

    # Extra words
    if len(stu_words) > len(ref_words):
        extra_count = len(stu_words) - len(ref_words)
        scores["sentence_tracking"] += extra_count * 2
        reasons["sentence_tracking"].append(
            "Extra words may have been added while reading."
        )

    # -------------------------------
    # 2. Character-level mistake checking
    # -------------------------------
    if cer > 0.15:
        scores["phonological_awareness"] += 3
        scores["character_discrimination"] += 2

        reasons["phonological_awareness"].append(
            "Several character-level reading mistakes were detected."
        )
        reasons["character_discrimination"].append(
            "The student may have confused Sinhala letters or letter sounds."
        )

    if cer > 0.30:
        scores["phonological_awareness"] += 2
        reasons["phonological_awareness"].append(
            "The character error rate is high, so letter-sound practice is needed."
        )

    # Use XAI feedback count also
    if len(xai_feedback) >= 2:
        scores["phonological_awareness"] += 2
        scores["character_discrimination"] += 1
        reasons["phonological_awareness"].append(
            "Multiple detailed character mistakes were found in the explanation feedback."
        )

    # -------------------------------
    # 3. WER-based word recognition
    # -------------------------------
    if wer > 0.20:
        scores["word_recognition"] += 3
        reasons["word_recognition"].append(
            "The word error rate is high, so word recognition practice is needed."
        )

    if wer > 0.40:
        scores["word_recognition"] += 2
        reasons["word_recognition"].append(
            "Many words were different from the expected sentence."
        )

    # -------------------------------
    # 4. Fluency checking
    # -------------------------------
    if wps > 0 and wps < 0.8:
        scores["reading_fluency"] += 3
        reasons["reading_fluency"].append(
            "The reading speed is lower than expected."
        )

    if wps > 0 and wps < 0.5:
        scores["reading_fluency"] += 2
        reasons["reading_fluency"].append(
            "The student may need more practice to read smoothly."
        )

    # -------------------------------
    # 5. Eye-tracking based analysis
    # -------------------------------
    if regression_count > 10:
        scores["visual_tracking"] += 3
        scores["sentence_tracking"] += 2
        reasons["visual_tracking"].append(
            "The eye movement pattern shows repeated backward movements."
        )
        reasons["sentence_tracking"].append(
            "The student may be losing the reading position in the sentence."
        )

    if avg_fixation_ms > 800:
        scores["visual_tracking"] += 2
        reasons["visual_tracking"].append(
            "Long fixation time shows that the student may be spending more time on words."
        )

    # -------------------------------
    # 6. Select main area
    # -------------------------------
    main_area = max(scores, key=scores.get)
    main_score = scores[main_area]

    if main_score == 0:
        return {
            "main_improvement_area": "Good Reading Performance",
            "main_improvement_area_si": "හොඳ කියවීමේ කාර්යසාධනය",
            "priority": "Low",
            "confidence": 0.95,
            "simple_explanation": "No major reading difficulty was detected in this sentence.",
            "simple_explanation_si": "මෙම වාක්‍යයේ විශාල කියවීමේ දුෂ්කරතාවයක් හඳුනාගෙන නැත.",
            "advice": "Continue regular reading practice.",
            "advice_si": "නිතිපතා කියවීමේ පුහුණුව දිගටම කරගෙන යන්න.",
            "scores": scores,
        }

    priority = "Low"
    if main_score >= 6:
        priority = "High"
    elif main_score >= 3:
        priority = "Medium"

    confidence = min(0.95, 0.50 + (main_score * 0.07))

    area_details = _get_area_details(main_area)

    return {
        "main_improvement_area": area_details["name"],
        "main_improvement_area_si": area_details["name_si"],
        "priority": priority,
        "confidence": round(confidence, 2),
        "simple_explanation": _build_simple_explanation(main_area),
        "simple_explanation_si": area_details["explanation_si"],
        "advice": area_details["advice"],
        "advice_si": area_details["advice_si"],
        "recommended_activity_type": area_details["activity_type"],
        "reasons": list(set(reasons[main_area])),
        "scores": scores,
    }


def _get_area_details(area: str) -> Dict[str, str]:
    details = {
        "phonological_awareness": {
            "name": "Phonological Awareness",
            "name_si": "අකුරු-ශබ්ද හඳුනාගැනීම",
            "explanation_si": "ශිෂ්‍යයා කියවීමේදී සමාන ශබ්ද ඇති අකුරු හෝ වචන වරදවා කියවා ඇත.",
            "advice": "Practise Sinhala letter sounds, especially similar-sounding letters.",
            "advice_si": "සමාන ශබ්ද ඇති සිංහල අකුරු නැවත නැවත කියවීමට පුහුණු වන්න.",
            "activity_type": "letter_sound_practice",
        },
        "word_recognition": {
            "name": "Word Recognition",
            "name_si": "වචන හඳුනාගැනීම",
            "explanation_si": "ශිෂ්‍යයාට දෙන ලද වචන නිවැරදිව හඳුනාගෙන කියවීමට තවත් පුහුණුවක් අවශ්‍ය විය හැක.",
            "advice": "Practise reading common Sinhala words from the same grade and level.",
            "advice_si": "එම ශ්‍රේණියට සහ මට්ටමට අදාළ සාමාන්‍ය සිංහල වචන නැවත නැවත කියවීමට පුහුණු වන්න.",
            "activity_type": "word_recognition_practice",
        },
        "reading_fluency": {
            "name": "Reading Fluency",
            "name_si": "කියවීමේ වේගය සහ සරලත්වය",
            "explanation_si": "ශිෂ්‍යයාගේ කියවීමේ වේගය අඩු විය හැකි නිසා වාක්‍ය සරලව කියවීමට පුහුණුවක් අවශ්‍ය වේ.",
            "advice": "Practise short Sinhala sentences repeatedly to improve speed and confidence.",
            "advice_si": "කෙටි සිංහල වාක්‍ය නැවත නැවත කියවීමෙන් වේගය සහ විශ්වාසය වැඩි කරගන්න.",
            "activity_type": "fluency_practice",
        },
        "sentence_tracking": {
            "name": "Sentence Tracking and Attention",
            "name_si": "වාක්‍යය අනුගමනය කිරීම සහ අවධානය",
            "explanation_si": "ශිෂ්‍යයා වාක්‍යය කියවද්දී සමහර වචන මඟහැරීම හෝ අමතර වචන එකතු කිරීම සිදු කර ඇත.",
            "advice": "Use finger tracking or line-by-line reading practice to follow the sentence correctly.",
            "advice_si": "වාක්‍යය නිවැරදිව අනුගමනය කිරීමට ඇඟිල්ලෙන් පෙන්වමින් හෝ පේළියෙන් පේළිය කියවීමට පුහුණු වන්න.",
            "activity_type": "sentence_tracking_practice",
        },
        "visual_tracking": {
            "name": "Visual Tracking",
            "name_si": "ඇස් චලනය සහ කියවීමේ පේළි අනුගමනය කිරීම",
            "explanation_si": "ඇස් චලන දත්ත අනුව ශිෂ්‍යයාට කියවීමේදී පේළිය හෝ වචන අනුගමනය කිරීමට අපහසු විය හැක.",
            "advice": "Practise slow line-by-line reading and tracking words from left to right.",
            "advice_si": "වචන වමෙන් දකුණට සෙමින් අනුගමනය කරමින් පේළියෙන් පේළිය කියවීමට පුහුණු වන්න.",
            "activity_type": "visual_tracking_practice",
        },
        "character_discrimination": {
            "name": "Sinhala Character Discrimination",
            "name_si": "සමාන සිංහල අකුරු වෙනස් කර හඳුනාගැනීම",
            "explanation_si": "ශිෂ්‍යයා සමාන පෙනුමක් හෝ සමාන ශබ්දයක් ඇති සිංහල අකුරු අතර ව්‍යාකූල වී ඇත.",
            "advice": "Practise identifying and reading similar Sinhala letters separately.",
            "advice_si": "සමාන පෙනුමක් ඇති සිංහල අකුරු වෙන වෙනම හඳුනාගෙන කියවීමට පුහුණු වන්න.",
            "activity_type": "character_discrimination_practice",
        },
    }

    return details.get(area, details["word_recognition"])


def _build_simple_explanation(area: str) -> str:
    explanations = {
        "phonological_awareness": (
            "The student confused similar Sinhala sounds while reading. "
            "This shows that the student needs more practice in matching Sinhala letters with their correct sounds."
        ),
        "word_recognition": (
            "The student read some words differently from the given sentence. "
            "This shows that the student needs more practice in recognizing Sinhala words."
        ),
        "reading_fluency": (
            "The student read slowly or with hesitation. "
            "This shows that the student needs more practice to read smoothly."
        ),
        "sentence_tracking": (
            "The student skipped or added words while reading. "
            "This shows that the student needs more practice in following the sentence correctly."
        ),
        "visual_tracking": (
            "The student may have difficulty following the reading line or word order. "
            "This shows that visual tracking practice is needed."
        ),
        "character_discrimination": (
            "The student may have confused similar Sinhala letters. "
            "This shows that the student needs more practice in identifying Sinhala characters."
        ),
    }

    return explanations.get(area, explanations["word_recognition"])