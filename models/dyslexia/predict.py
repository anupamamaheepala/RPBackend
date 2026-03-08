import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "dyslexia_model.pkl") 
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

FEATURES = [
    'grade', 'level', 'total_words', 'overall_accuracy', 
    'avg_WER', 'avg_CER', 'total_time_seconds',
    'dyslexia_assessment.phonological_risk', 
    'dyslexia_assessment.fluency_risk', 
    'dyslexia_assessment.eye_risk'
]

def predict_dyslexia_risk_ml(ml_features: dict):
    try:
        # 1. Create DataFrame with explicit column ordering
        X = pd.DataFrame([ml_features], columns=FEATURES)

        # 2. Predict numeric class and probabilities
        numerical_prediction = model.predict(X)
        probabilities = model.predict_proba(X)
        max_confidence = np.max(probabilities)

        # 3. Decode numeric result (0, 1, 2) -> ("LOW", "MEDIUM", "HIGH")
        risk_level = label_encoder.inverse_transform(numerical_prediction)[0]

        return {
            "risk_level": risk_level,
            "confidence": round(float(max_confidence), 3),
            "raw_scores": ml_features # Useful for debugging
        }
    except Exception as e:
        return {"error": str(e), "risk_level": "UNKNOWN"}