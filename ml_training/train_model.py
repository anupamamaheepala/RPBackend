# train_model.py

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
print("\nRisk distribution:")
print(df["dyslexia_risk_level"].value_counts())

# -------------------------------
# 🔥 RENAME COLUMNS (if needed)
# -------------------------------
df = df.rename(columns={
    "accuracy_percent": "overall_accuracy",
    "wer": "avg_WER",
    "cer": "avg_CER",
    "duration_seconds": "total_time_seconds"
})

# -------------------------------
# 🔥 DATA CLEANING
# -------------------------------
df = df.dropna()

# -------------------------------
# 🔥 DATA-DRIVEN THRESHOLDS
# -------------------------------
cer_threshold = df["avg_CER"].quantile(0.6)
speed_threshold = df["words_per_second"].quantile(0.4)
regression_threshold = df["regression_count"].quantile(0.6)

print("\nThresholds:")
print("CER:", cer_threshold)
print("Speed:", speed_threshold)
print("Regression:", regression_threshold)

# Save thresholds
joblib.dump({
    "cer": cer_threshold,
    "speed": speed_threshold,
    "regression": regression_threshold
}, MODEL_DIR / "thresholds.pkl")

# -------------------------------
# 🔥 FEATURE ENGINEERING
# -------------------------------
df["phonological_risk"] = (df["avg_CER"] > cer_threshold).astype(int)
df["fluency_risk"] = (df["words_per_second"] < speed_threshold).astype(int)
df["eye_risk"] = (df["regression_count"] > regression_threshold).astype(int)

# 🔥 Accuracy band (VERY IMPORTANT)
def get_accuracy_band(acc):
    if acc < 50:
        return 2   # HIGH risk zone
    elif acc < 75:
        return 1   # MEDIUM
    else:
        return 0   # LOW

df["accuracy_band"] = df["overall_accuracy"].apply(get_accuracy_band)

# -------------------------------
# FEATURES & LABEL
# -------------------------------
FEATURES = [
    "grade",
    "level",
    "total_words",
    "overall_accuracy",
    "avg_WER",
    "avg_CER",
    "total_time_seconds",
    "phonological_risk",
    "fluency_risk",
    "eye_risk",
    "accuracy_band"
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
# FEATURE SCALING
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = RandomForestClassifier(
    n_estimators=600,
    max_depth=20,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test_scaled)

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
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
joblib.dump(FEATURES, MODEL_DIR / "feature_list.pkl")

print("\n✅ Model artifacts saved successfully")

# -------------------------------
# SAMPLE TEST
# -------------------------------
sample = pd.DataFrame([{
    "grade": 5,
    "level": 1,
    "total_words": 14,
    "overall_accuracy": 68,
    "avg_WER": 30,
    "avg_CER": 20,
    "total_time_seconds": 12,
    "phonological_risk": 1,
    "fluency_risk": 1,
    "eye_risk": 0,
    "accuracy_band": get_accuracy_band(68)
}])

sample_scaled = scaler.transform(sample)

pred = model.predict(sample_scaled)
prob = model.predict_proba(sample_scaled).max()

print("\nSample Prediction:",
      label_encoder.inverse_transform(pred)[0])
print("Confidence:", round(float(prob), 3))