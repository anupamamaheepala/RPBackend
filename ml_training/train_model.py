# train_model.py

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -------------------------------
# STEP 6 — Load dataset
# -------------------------------
df = pd.read_excel("data/sinhala_dyslexia_synthetic_5000.xlsx")

print("Dataset shape:", df.shape)
print(df["Risk Level"].value_counts())

# -------------------------------
# STEP 7 — Select features & label
# -------------------------------
features = [
    "Accuracy Percentage",
    "WER",
    "Words Per Second",
    "Avg Fixation ms",
    "Fixation Count",
    "Regression Count",
    "Total Words",
    "Duration"
]

X = df[features]
y = df["Risk Level"]

# -------------------------------
# STEP 8 — Encode labels
# -------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)

# -------------------------------
# STEP 9 — Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -------------------------------
# STEP 10 — Feature scaling
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# STEP 11 — Train model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -------------------------------
# STEP 12 — Evaluate model
# -------------------------------
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# -------------------------------
# STEP 13 — Feature importance
# -------------------------------
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# -------------------------------
# STEP 14 — Save model artifacts
# -------------------------------
joblib.dump(model, "dyslexia_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n✅ Model, scaler, and label encoder saved")

# -------------------------------
# STEP 15 — Test with sample input
# -------------------------------
sample = [[
    78,     # Accuracy %
    25,     # WER
    1.6,    # Words Per Second
    720,    # Avg Fixation ms
    35,     # Fixation Count
    3,      # Regression Count
    14,     # Total Words
    12      # Duration
]]

sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)

print("\nSample Prediction:", label_encoder.inverse_transform(pred)[0])
