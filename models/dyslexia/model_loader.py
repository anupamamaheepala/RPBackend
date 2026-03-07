# models/dyslexia/model_loader.py
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load only the files you actually generated
model = joblib.load(BASE_DIR / "dyslexia_model.pkl") 
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

# Note: Scaler is removed because Random Forest handles raw features effectively.