# models/dyslexia/model_loader.py

import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "risk_model.pkl")
scaler = joblib.load(BASE_DIR / "risk_features.pkl")
label_encoder = joblib.load(BASE_DIR / "risk_label_encoder.pkl")
