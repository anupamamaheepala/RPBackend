# services/dyslexia_metrics.py
import re
import json
from typing import Optional, Tuple, List
from jiwer import wer as jiwer_wer
from jiwer import wer
from difflib import SequenceMatcher
import numpy as np

SINHALA_NORMALIZATION_MAP = {
    "ණ": "න",
    "ළ": "ල",
    "ශ": "ෂ",
    "ඤ": "ඥ",
    "ග": "ඟ",
    "ලු": "ළු",
    "ද්‍යා": "ද්යා",
    "ඨ": "ට",
    "න්‍ය": "න්ය",
}

def normalize_sinhala_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)

    for src, tgt in SINHALA_NORMALIZATION_MAP.items():
        text = text.replace(src, tgt)

    return text

def compact_sinhala(text: str) -> str:
    text = normalize_sinhala_text(text)
    text = re.sub(r"[^\u0D80-\u0DFF]", "", text)  # keep Sinhala block only
    return text

def extract_word_errors(reference: str, transcript: str) -> Tuple[List[str], List[str]]:
    ref = normalize_sinhala_text(reference)
    hyp = normalize_sinhala_text(transcript)

    ref_words = ref.split()
    hyp_compact = compact_sinhala(hyp)

    correct, incorrect = [], []
    pos = 0

    for w in ref_words:
        w_comp = compact_sinhala(w)
        idx = hyp_compact.find(w_comp, pos)

        if idx != -1:
            correct.append(w)
            pos = idx + len(w_comp)
        else:
            incorrect.append(w)

    return correct, incorrect

def clamp(value: float, min_value=0.0, max_value=1.0) -> float:
    return max(min_value, min(value, max_value))

def char_error_rate(reference: str, transcript: str) -> float:
    r = " ".join(list(compact_sinhala(reference)))
    h = " ".join(list(compact_sinhala(transcript)))
    return wer(r, h) * 100

def compute_metrics(reference: str, transcript: str, duration: Optional[float] = None):
    reference = normalize_sinhala_text(reference)
    transcript = normalize_sinhala_text(transcript)

    correct_words_list, incorrect_words_list = extract_word_errors(reference, transcript)

    ref_words = reference.split()
    hyp_words = transcript.split()

    total_words = len(ref_words)
    correct_words = len(correct_words_list)

    accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0.0

    true_wer = jiwer_wer(reference, transcript) * 100

    if correct_words == total_words:
        true_wer = 0

    if total_words <= 5:
        true_wer = min(true_wer, 10)

    accuracy_based_wer = (1 - (correct_words / total_words)) * 100 if total_words > 0 else 100
    if abs(true_wer - accuracy_based_wer) > 30:
        true_wer = accuracy_based_wer

    cer = char_error_rate(reference, transcript)

    speed = None
    if duration and float(duration) > 0:
        speed = round(len(hyp_words) / float(duration), 2)

    return {
        "reference": reference,
        "transcript": transcript,
        "total_words": total_words,
        "correct_words": correct_words,
        "accuracy_percent": round(accuracy, 2),
        "wer": round(true_wer, 2),
        "cer": round(cer, 2),
        "words_per_second": speed,
        # IMPORTANT: make it a LIST (frontend aggregation needs list)
        "incorrect_words": incorrect_words_list,
    }


def calculate_session_metrics(payload):
    """
    Transforms raw session data into the 10 features required by the ML model.
    """
    # 1. Phonological Risk: Derived from Error Rates (WER/CER)
    # If errors are high, risk approaches 1.0. 
    # We use 50% error as a common 'high risk' threshold for normalization.
    phonological_risk = min(1.0, (payload.avg_WER + payload.avg_CER) / 100.0)

    # 2. Fluency Risk: Derived from Words Per Second (WPS)
    # Benchmark: 3.0 WPS is fluent for Grade 3-7. 
    # Risk increases as speed drops below 3.0.
    fluency_risk = max(0.0, 1.0 - (payload.avg_words_per_second / 3.0))

    # 3. Eye Risk: Derived from Gaze Data
    # Dyslexic readers typically have higher regressions and longer fixations.
    # Normalizing: 20+ regressions = high risk; 500ms+ fixation = high risk.
    reg_score = min(1.0, payload.avg_regression_count / 20.0)
    fix_score = min(1.0, payload.avg_fixation_time / 500.0)
    eye_risk = (reg_score * 0.7) + (fix_score * 0.3)

    return {
        "grade": payload.grade,
        "level": payload.level,
        "total_words": payload.total_words,
        "overall_accuracy": payload.overall_accuracy,
        "avg_WER": payload.avg_WER,
        "avg_CER": payload.avg_CER,
        "total_time_seconds": payload.total_time_seconds,
        "dyslexia_assessment.phonological_risk": round(float(phonological_risk), 4),
        "dyslexia_assessment.fluency_risk": round(float(fluency_risk), 4),
        "dyslexia_assessment.eye_risk": round(float(eye_risk), 4)
    }

def compute_dyslexia_risk(audio_metrics: dict, eye_metrics: dict):
    accuracy = audio_metrics.get("accuracy_percent", 0)
    wer_val = audio_metrics.get("wer", 100)

    if audio_metrics.get("correct_words") == audio_metrics.get("total_words"):
        wer_val = 0

    accuracy_risk = 1 - (accuracy / 100)
    wer_risk = wer_val / 100
    phonological_risk = (accuracy_risk + wer_risk) / 2

    wps = audio_metrics.get("words_per_second", 0) or 0
    fluency_risk = clamp((2.5 - wps) / 2.5)

    avg_fixation = eye_metrics.get("avg_fixation_ms", 0)
    regression_count = eye_metrics.get("regression_count", 0)
    word_count = audio_metrics.get("total_words", 0)

    if word_count <= 5:
        eye_risk = 0.2
    else:
        fixation_risk = clamp((avg_fixation - 300) / 1200)
        regression_risk = clamp(regression_count / 5)
        eye_risk = (0.7 * fixation_risk) + (0.3 * regression_risk)
        if audio_metrics.get("accuracy_percent", 0) >= 95 and regression_count == 0:
            eye_risk = min(eye_risk, 0.3)

    final_risk = (0.35 * phonological_risk + 0.25 * fluency_risk + 0.40 * eye_risk)

    if final_risk <= 0.30:
        level = "LOW"
    elif final_risk <= 0.60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "phonological_risk": round(phonological_risk, 3),
        "fluency_risk": round(fluency_risk, 3),
        "eye_risk": round(eye_risk, 3),
        "risk_score": round(final_risk, 3),
        "risk_level": level,
    }
