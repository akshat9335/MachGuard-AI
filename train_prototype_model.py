import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# Generate synthetic Arduino data
# ==============================

n = 2000

temperature = np.random.uniform(25, 45, n)
vibration = np.random.uniform(0, 200, n)
rpm = np.random.uniform(800, 3000, n)
pressure = np.random.uniform(4.5, 5.5, n)

data = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "rpm": rpm
})

# Remaining Useful Life (fake target)
rul = 120 - (
    (temperature - 25) * 0.8 +
    vibration * 0.05 +
    (rpm - 800) * 0.02
)

rul = np.clip(rul, 0, 120)

# ==============================
# Train RUL model
# ==============================

rul_model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=100))
])

rul_model.fit(data, rul)

# ==============================
# Train anomaly model
# ==============================

anomaly_model = IsolationForest(contamination=0.05)

anomaly_model.fit(data)

# ==============================
# Save models
# ==============================

joblib.dump(rul_model, "prototype_rul_model.pkl")
joblib.dump(anomaly_model, "prototype_anomaly_model.pkl")

print("Prototype models created successfully") 