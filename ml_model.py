# ======================================
# STEP 1: IMPORT LIBRARIES
# ======================================
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib
import shap
import os

# ======================================
# STEP 2: LOAD DATA
# ======================================
df = pd.read_csv("predictive_maintenance_data.csv")

df = df[df["time"] <= 120].reset_index(drop=True)

print("🔹 Data Preview:")
print(df.head())

# ======================================
# STEP 3: FEATURE SELECTION
# ======================================
features = ["temperature", "vibration", "pressure", "rpm"]

X = df[features]
y_rul = df["RUL"]

# ======================================
# STEP 4: ANOMALY DETECTION
# ======================================
anomaly_model = IsolationForest(
    n_estimators=150,
    contamination=0.1,
    random_state=42
)

anomaly_model.fit(X)

df["anomaly"] = anomaly_model.predict(X)
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

print("\n🔹 Anomaly Detection Preview:")
print(df[["temperature", "vibration", "anomaly"]].head())

# ======================================
# STEP 5: RUL PREDICTION (PIPELINE 🔥)
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_rul, test_size=0.2, random_state=42
)

rul_model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    ))
])

rul_model.fit(X_train, y_train)

# ✅ FIXED LINE
y_pred = rul_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("\n🔹 Model Performance:")
print("MAE:", round(mae, 3))

# ======================================
# STEP 6: QUICK LIVE TEST
# ======================================
sample = X.iloc[[10]]

print("\n🔹 Live Test (Early-stage Machine)")
print("Anomaly:", anomaly_model.predict(sample))
print("Predicted RUL:", rul_model.predict(sample))

# ======================================
# STEP 7: SAVE TRAINED MODELS
# ======================================
joblib.dump(anomaly_model, "anomaly_model.pkl")
joblib.dump(rul_model, "rul_model.pkl")

print("\n✅ Models saved successfully!")
print("📂 Location:", os.getcwd())

# ======================================
# STEP 8: EXPLAINABLE AI (OPTIONAL CHECK)
# ======================================
rf_model = rul_model.named_steps["rf"]
scaler = rul_model.named_steps["scaler"]

X_test_scaled = scaler.transform(X_test)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_scaled)

# (Plot disabled during training)
# shap.summary_plot(shap_values, X_test)
