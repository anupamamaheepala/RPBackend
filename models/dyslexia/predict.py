import pandas as pd
import numpy as np
from .model_loader import model, label_encoder

# These MUST match the headers in your generated CSV/Training code exactly.
# If a single letter is different (e.g., 'Grade' vs 'grade'), the model will fail.
FEATURES = [
    'grade', 
    'level', 
    'total_words', 
    'overall_accuracy', 
    'avg_WER', 
    'avg_CER', 
    'total_time_seconds',
    'dyslexia_assessment.phonological_risk', 
    'dyslexia_assessment.fluency_risk', 
    'dyslexia_assessment.eye_risk'
]

def predict_dyslexia_risk_ml(ml_features: dict):
    """
    Core ML prediction function.
    
    Args:
        ml_features (dict): A dictionary containing all keys defined in FEATURES.
        
    Returns:
        dict: The risk level and the model's confidence score.
    """
    try:
        # 1. Convert dictionary to Pandas DataFrame (required for Scikit-Learn)
        # We specify the columns=FEATURES to ensure the order is correct.
        X = pd.DataFrame([ml_features], columns=FEATURES)

        # 2. Perform the prediction
        # No scaler needed for Random Forest as it is scale-invariant.
        numerical_prediction = model.predict(X)
        
        # 3. Get the probability/confidence score
        # This returns a list of probabilities for [LOW, MEDIUM, HIGH]
        probabilities = model.predict_proba(X)
        max_confidence = np.max(probabilities)

        # 4. Decode the numeric result (0, 1, 2) back to ("LOW", "MEDIUM", "HIGH")
        risk_level = label_encoder.inverse_transform(numerical_prediction)[0]

        return {
            "risk_level": risk_level,
            "confidence": round(float(max_confidence), 3),
            "method": "Machine Learning (Random Forest)"
        }

    except Exception as e:
        # Detailed error reporting for debugging column mismatches
        print(f"CRITICAL: ML Prediction Error: {str(e)}")
        return {
            "error": f"Prediction failed: {str(e)}",
            "method": "Error"
        }