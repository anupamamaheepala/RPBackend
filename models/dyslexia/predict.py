# models/dyslexia/predict.py

import pandas as pd
import numpy as np
from .model_loader import model, scaler, label_encoder, FEATURES, thresholds


# -------------------------------
# Helper: Accuracy band
# -------------------------------
def get_accuracy_band(acc):
    if acc < 50:
        return 2
    elif acc < 75:
        return 1
    else:
        return 0


def predict_dyslexia_risk_ml(ml_features: dict):

    try:
        # -------------------------------
        # 🔥 1. Build ML input (Feature Engineering)
        # -------------------------------
        engineered_input = {
            "grade": ml_features.get("grade", 0),
            "level": ml_features.get("level", 0),
            "total_words": ml_features.get("total_words", 0),

            "overall_accuracy": ml_features.get("overall_accuracy", 0),
            "avg_WER": ml_features.get("avg_WER", 0),
            "avg_CER": ml_features.get("avg_CER", 0),
            "total_time_seconds": ml_features.get("total_time_seconds", 0),

            # 🔥 Data-driven thresholds (from training)
            "phonological_risk": 1 if ml_features.get("avg_CER", 0) > thresholds["cer"] else 0,
            "fluency_risk": 1 if ml_features.get("avg_words_per_second", 0) < thresholds["speed"] else 0,
            "eye_risk": 1 if ml_features.get("avg_regression_count", 0) > thresholds["regression"] else 0,

            # 🔥 Important derived feature
            "accuracy_band": get_accuracy_band(ml_features.get("overall_accuracy", 0))
        }

        # -------------------------------
        # 🔥 2. Validate required features
        # -------------------------------
        missing = [f for f in FEATURES if f not in engineered_input]

        if missing:
            return {
                "error": f"Missing ML features: {missing}",
                "method": "Machine Learning Error"
            }

        # -------------------------------
        # 🔥 3. Create DataFrame (correct order)
        # -------------------------------
        X = pd.DataFrame([{f: engineered_input.get(f, 0) for f in FEATURES}])

        # -------------------------------
        # 🔥 4. Convert to numeric safely
        # -------------------------------
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.fillna(0)

        # -------------------------------
        # 🔥 5. Scale input (same as training)
        # -------------------------------
        X_scaled = scaler.transform(X)

        # -------------------------------
        # 🔥 6. Predict
        # -------------------------------
        pred_num = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[0]

        risk_level = label_encoder.inverse_transform(pred_num)[0]
        confidence = float(np.max(probabilities))

        # -------------------------------
        # 🔥 7. Class probabilities
        # -------------------------------
        class_probabilities = {
            label_encoder.classes_[i]: round(float(probabilities[i]), 3)
            for i in range(len(label_encoder.classes_))
        }

        # -------------------------------
        # 🔥 8. Confidence interpretation
        # -------------------------------
        if confidence < 0.5:
            note = "Low confidence prediction"
        elif confidence < 0.7:
            note = "Moderate confidence prediction"
        else:
            note = "High confidence prediction"

        # -------------------------------
        # 🔥 9. Final output
        # -------------------------------
        return {
            "risk_level": risk_level,
            "confidence": round(confidence, 3),
            "class_probabilities": class_probabilities,
            "note": note,
            "method": "Machine Learning (Random Forest + Data-Driven Features)"
        }

    except Exception as e:
        print("❌ CRITICAL ML ERROR:", str(e))
        print("INPUT DATA:", ml_features)

        return {
            "error": f"Prediction failed: {str(e)}",
            "method": "Machine Learning Error"
        }