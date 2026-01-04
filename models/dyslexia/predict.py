# models/dyslexia/predict.py

import pandas as pd
from .model_loader import model, scaler, label_encoder

FEATURES = [
    "Accuracy Percentage",
    "WER",
    "Words Per Second",
    "Avg Fixation ms",
    "Fixation Count",
    "Regression Count",
    "Total Words",
    "Duration",
]

def predict_dyslexia_risk_ml(audio_metrics: dict, eye_metrics: dict, duration: float):
    """
    ML-based dyslexia risk prediction
    """

    row = {
        "Accuracy Percentage": audio_metrics.get("accuracy_percent", 0),
        "WER": audio_metrics.get("wer", 100),
        "Words Per Second": audio_metrics.get("words_per_second", 0) or 0,
        "Avg Fixation ms": eye_metrics.get("avg_fixation_ms", 0),
        "Fixation Count": eye_metrics.get("fixation_count", 0),
        "Regression Count": eye_metrics.get("regression_count", 0),
        "Total Words": audio_metrics.get("total_words", 0),
        "Duration": duration or 0,
    }

    X = pd.DataFrame([row], columns=FEATURES)

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled).max()

    risk_level = label_encoder.inverse_transform(pred)[0]

    return {
        "risk_level": risk_level,
        "confidence": round(float(prob), 3),
        "method": "ML",
    }
