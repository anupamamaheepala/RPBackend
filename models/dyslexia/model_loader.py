# models/dyslexia/model_loader.py

from pathlib import Path
import joblib

# This resolves to /app/models/dyslexia in Railway
BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "risk_model.pkl")
label_encoder = joblib.load(BASE_DIR / "risk_label_encoder.pkl")
scaler = joblib.load(BASE_DIR / "risk_features.pkl")
