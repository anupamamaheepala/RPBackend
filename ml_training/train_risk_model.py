# train_risk_model.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sinhala_dyslexia_dataset_5000.xlsx"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_excel(DATA_PATH)

# ✅ Use correct label column name from your dataset
LABEL_COL = "dyslexia_risk_level"  # if KeyError, change to risk_level_score_based

FEATURES = [
    "grade",
    "level",
    "accuracy_percent",
    "wer",
    "words_per_second",
    "total_words",
    "duration_seconds",
    "fixation_count",
    "avg_fixation_ms",
    "regression_count"
]

X = df[FEATURES]
y = df[LABEL_COL]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, MODEL_DIR / "risk_model.pkl")
joblib.dump(le, MODEL_DIR / "risk_label_encoder.pkl")
joblib.dump(FEATURES, MODEL_DIR / "risk_features.pkl")

print("✅ Saved model files to ml_training/models/")
