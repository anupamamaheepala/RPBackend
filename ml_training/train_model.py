# train_model.py

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sinhala_dyslexia_dataset_5000.xlsx"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_excel(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nDyslexia stage distribution:")
print(df["dyslexia_risk_level"].value_counts())

# -------------------------------
# FEATURES & LABEL
# -------------------------------
FEATURES = [
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
y = df["dyslexia_risk_level"]

# -------------------------------
# LABEL ENCODING
# -------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClasses:", list(label_encoder.classes_))

# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# -------------------------------
# SAVE ARTIFACTS
# -------------------------------
joblib.dump(model, MODEL_DIR / "dyslexia_stage_model.pkl")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
joblib.dump(FEATURES, MODEL_DIR / "feature_list.pkl")

print("\n✅ Model artifacts saved successfully")

# -------------------------------
# SAMPLE TEST
# -------------------------------
sample = pd.DataFrame([{
    "accuracy_percent": 78,
    "wer": 25,
    "words_per_second": 1.6,
    "total_words": 14,
    "duration_seconds": 12,
    "fixation_count": 35,
    "avg_fixation_ms": 720,
    "regression_count": 3
}])

pred = model.predict(sample)
print("\nSample Prediction:",
      label_encoder.inverse_transform(pred)[0])
