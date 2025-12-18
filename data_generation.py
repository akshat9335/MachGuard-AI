import numpy as np
import pandas as pd

np.random.seed(42)

# ===============================
# CONFIG
# ===============================
rows = 150
max_rul = 120

# ===============================
# TIME
# ===============================
time = np.arange(rows)

# ===============================
# RUL (SMOOTH + REALISTIC)
# ===============================
RUL = max_rul - time + np.random.normal(0, 3, rows)
RUL = np.clip(RUL, 0, max_rul)

# Wear factor (0 = healthy, 1 = end-of-life)
wear = 1 - (RUL / max_rul)

# ===============================
# SENSOR SIMULATION (STAGE-AWARE)
# ===============================

# 🌡 Temperature → mid + late stage effect
temperature = (
    60
    + wear * 15
    + np.random.normal(0, 2, rows)
)

# 📳 Vibration → mainly late-stage wear
vibration = (
    0.8
    + np.where(wear > 0.4, (wear - 0.4) * 3.0, 0)
    + np.random.normal(0, 0.15, rows)
)

# 🧪 Pressure → mostly stable
pressure = (
    5
    + np.random.normal(0, 0.25, rows)
)

# ⚙ RPM → gradual efficiency loss
rpm = (
    1500
    - wear * 400
    + np.random.normal(0, 25, rows)
)

# ===============================
# FAILURE LABEL
# ===============================
failure = np.where(RUL <= 15, 1, 0)

# ===============================
# DATAFRAME
# ===============================
df = pd.DataFrame({
    "time": time,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "rpm": rpm,
    "RUL": RUL.astype(int),
    "failure": failure
})

# ===============================
# SAVE
# ===============================
df.to_csv("predictive_maintenance_data.csv", index=False)

print("✅ FINAL stage-aware predictive maintenance data generated")
print("📊 Rows:", len(df))
print("⏳ RUL range:", df['RUL'].min(), "to", df['RUL'].max())
