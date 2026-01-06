# models/dyslexia/model_loader.py

import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# models/dyslexia/model_loader.py

model = joblib.load(BASE_DIR / "dyslexia_stage_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
