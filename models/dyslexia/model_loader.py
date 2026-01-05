# models/dyslexia/model_loader.py

import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "dyslexia_stage_model.pkl")
scaler = joblib.load(BASE_DIR / "feature_list.pkl")
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
