import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np
import random
import time
import os

from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import serial
import streamlit as st
import time

@st.cache_resource
def init_serial():
    try:
        arduino = serial.Serial('COM5', 9600, timeout=1)

        time.sleep(2)

        arduino.reset_input_buffer()

        st.success("✅ Arduino connected on COM5")

        return arduino

    except Exception as e:
        st.warning("⚠ Arduino not connected. Running in simulation mode.")
        print("Serial error:", e)
        return None


arduino = init_serial()

# ===============================
# CONSTANTS
# ===============================
MAX_LIFECYCLE = 120
FEATURES = ["temperature", "vibration", "pressure", "rpm"]

# ===============================
# SAFE OPERATING RANGES
# ===============================
SAFE_RANGES = {
    "temperature": (50, 70),
    "vibration": (0.5, 1.8),
    "pressure": (4.2, 6.0),
    "rpm": (1250, 1600)
}

SENSOR_UNITS = {
    "temperature": "°C",
    "vibration": "mm/s",
    "pressure": "bar",
    "rpm": "RPM"
}


# =====================================================
# ENV + OPENROUTER CLIENT
# =====================================================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

# =====================================================
# STYLING
# =====================================================
st.markdown("""
<style>
h1 { font-size: 40px !important; }
h2 { font-size: 30px !important; }
h3 { font-size: 24px !important; }

[data-testid="metric-container"] {
    background-color: #0f172a;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 12px;
}

.contribution-box {
    font-size: 18px;
    line-height: 1.6;
    background: #020617;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #334155;
    margin-bottom: 12px;
}

.ai-insight-box {
    font-size: 18px;
    line-height: 1.7;
    background: #020617;
    padding: 16px;
    border-radius: 10px;
    border-left: 5px solid #38bdf8;
    margin-top: 10px;
}

.contribution-card {
    background: linear-gradient(135deg, #020617, #0f172a);
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 12px;
    transition: 0.3s ease;
}

.contribution-card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 6px 20px rgba(56,189,248,0.2);
}

.sensor-title {
    font-size: 18px;
    font-weight: bold;
    color: #38bdf8;
}

.sensor-value {
    font-size: 15px;
    color: #e2e8f0;
}

.progress-bg {
    width: 100%;
    background: #1e293b;
    border-radius: 6px;
    height: 6px;
    margin-top: 8px;
}

.progress-fill {
    height: 6px;
    border-radius: 6px;
    background: linear-gradient(to right, #22c55e, #facc15, #ef4444);
}



</style>
""", unsafe_allow_html=True)

st.sidebar.image("machai logo.png", width=140)


col_logo, col_text = st.columns([3, 5])   # 👈 pehle 1 tha

with col_logo:
    st.image("machai logo.png", width=300)

with col_text:
    st.markdown(
        """
        <div style="padding-top: 70px;">
            <h1 style="margin-bottom: 9px;">MachGuard AI</h1>
            <p style="color: #94a3b8; font-size: 22px;">
                AI-Powered Predictive Maintenance & Fault Diagnosis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# CONSTANTS
# =====================================================
MAX_LIFECYCLE = 120


# =====================================================
# LOAD DATA & MODELS
# =====================================================
df = pd.read_csv("predictive_maintenance_data.csv")
df = df[df["time"] <= MAX_LIFECYCLE].reset_index(drop=True)

anomaly_model = joblib.load("anomaly_model.pkl")
rul_model = joblib.load("rul_model.pkl")

scaler = rul_model.named_steps["scaler"]
rf_model = rul_model.named_steps["rf"]
explainer = shap.TreeExplainer(rf_model)

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def smooth_health(new_health):
    if "health_history" not in st.session_state:
        st.session_state.health_history = []

    st.session_state.health_history.append(new_health)

    if len(st.session_state.health_history) > 5:
        st.session_state.health_history.pop(0)

    return int(sum(st.session_state.health_history) / len(st.session_state.health_history))

def calculate_health(status, sample):

    temp = sample["temperature"].values[0]
    vib = sample["vibration"].values[0]
    rpm = sample["rpm"].values[0]

    # base from status
    if "🔴" in status:
        health = 40
    elif "🟡" in status:
        health = 65
    else:
        health = 90

    # fine tuning
    if vib > 1.8:
        health -= 20
    elif vib > 1.2:
        health -= 10

    if temp > 75:
        health -= 15

    if rpm > 2500:
        health -= 10

    return max(0, min(100, int(health)))



def smooth_status(new_status):
    if "status_history" not in st.session_state:
        st.session_state.status_history = []

    st.session_state.status_history.append(new_status)

    # last 3 hi rakho
    if len(st.session_state.status_history) > 3:
        st.session_state.status_history.pop(0)

    # majority vote
    return max(set(st.session_state.status_history), key=st.session_state.status_history.count)



def get_sensor_status(sample):
    temp = sample["temperature"].values[0]
    vib = sample["vibration"].values[0]
    rpm = sample["rpm"].values[0]

    if vib > 1.8 or temp > 75 or rpm > 2500:
        return "🔴 At Risk"
    elif vib > 1.2 or temp > 65 or rpm > 1800:
        return "🟡 Warning"
    else:
        return "🟢 Healthy"


def dynamic_maintenance(sample):
    temp = sample["temperature"].values[0]
    vib = sample["vibration"].values[0]
    rpm = sample["rpm"].values[0]

    actions = []

    if vib > 1.8:
        actions.append("🔧 High vibration → Check bearings")
    elif vib > 1.2:
        actions.append("⚠️ Moderate vibration → Inspect alignment")

    if temp > 75:
        actions.append("🌡️ Overheating → Check cooling")

    if rpm > 2500:
        actions.append("⚙️ Overspeed → Reduce motor speed")

    if not actions:
        actions.append("✅ No maintenance needed")

    return actions




def render_sensor_table(df):
    html = "<table style='width:100%; text-align:center;'>"
    html += "<tr>"
    for col in df.columns:
        html += f"<th style='font-size:20px'>{col.upper()}</th>"
    html += "</tr><tr>"
    for v in df.iloc[0]:
        html += f"<td style='font-size:18px'>{v:.2f}</td>"
    html += "</tr></table>"
    return html


def get_llm_reason_and_fix(sensor, value, influence, history=""):
    prompt = f"""
You are a predictive maintenance AI.

Sensor: {sensor}
Current value: {value}
Impact: {influence:.1f}%

Format strictly:
REASON: <cause1>; <cause2 optional>
ACTION: <action1>; <action2 optional>
"""

    if history:
        prompt += f"\nAvoid repeating:\n{history}"

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": "Follow the format strictly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        text = response.choices[0].message.content.strip()
    except Exception:
        text = ""

    reasons, actions = [], []

    for line in text.splitlines():
        if line.upper().startswith("REASON:"):
            reasons = [r.strip() for r in line.replace("REASON:", "").split(";") if r.strip()]
        elif line.upper().startswith("ACTION:"):
            actions = [a.strip() for a in line.replace("ACTION:", "").split(";") if a.strip()]

    if not reasons:
        reasons = [f"Abnormal {sensor} behavior affecting system performance"]
    if not actions:
        actions = ["Inspect related components and monitor operating conditions"]

    return reasons, actions


def get_llm_hybrid_comment(sensor, status, rul, anomaly):
    prompt = f"""
Overall machine status: {status}
Remaining useful life: {rul}
Anomaly detected: {"Yes" if anomaly == -1 else "No"}
Key sensor: {sensor}

Write exactly two short sentences.
"""

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": "Generate concise industrial insight."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return (
            "The machine shows deviation from normal operating behavior. "
            "A key operating parameter is contributing to this condition."
        )

# =====================================================
# 🔌 REAL SENSOR INTEGRATION (FUTURE USE - COMMENTED)
# =====================================================
# When using real sensors (ESP32 / Arduino),
# data will be received via an external API (Flask server).
# Uncomment this section only during real deployment.

# from receiver import latest_sensor_data

# def get_realtime_sensor_data():
#     """
#     Fetch real-time sensor data sent by IoT device.
#     Expected JSON format:
#     {
#         "temperature": float,
#         "vibration": float,
#         "pressure": float,
#         "rpm": int
#     }
#     """
#     if not latest_sensor_data:
#         # Fallback to safe default to avoid crash
#         return {
#             "temperature": 0,
#             "vibration": 0,
#             "pressure": 0,
#             "rpm": 0
#         }
#     return latest_sensor_data


#when in real-time mode, simulate sensor data,sensor not connected,when sensor connected,replace this function with upper function
def get_realtime_sensor_data():
    return {
        "temperature": random.uniform(55, 85),
        "vibration": random.uniform(0.6, 2.5),
        "pressure": random.uniform(4.5, 6.0),
        "rpm": random.uniform(1100, 1500)
    }


#audrino mode function to read data from arduino, when in real-time mode, replace this function with upper function

def get_arduino_sensor_data():

    if arduino is None:
        return None, None

    try:
        # read latest line
        line = arduino.readline().decode(errors="ignore").strip().lower()

        if not line:
            return None, None

        parts = line.split(",")

        if len(parts) < 3:
            return None, None

        try:
            rpm = int(parts[0].split(":")[1])
            vib = float(parts[1].split(":")[1])
            temp = float(parts[2].split(":")[1])
        except:
            return None, None

        # RPM noise filter
        rpm = min(rpm, 3000)

        # OPTIONAL: flush buffer (remove old data)
        arduino.flushInput()
        #reset_input_buffer()

        raw_data = {
            "temperature": temp,
            "vibration": vib,
            "pressure": 5,
            "rpm": rpm
        }

        model_data = {
            "temperature": temp + 25,
            "vibration": vib / 50,
            "pressure": 5,
            "rpm": rpm
        }

        return raw_data, model_data

    except Exception as e:
        print("Serial error:", e)
        return None, None

# =====================================================
# SIDEBAR
# =====================================================
mode = st.sidebar.radio(
    "Select input type:",
    [
        "📊 Dataset Machine",
        "✍️ Manual Sensor Input",
        "📡 Real-Time Monitoring",
        "🔌 Arduino Live Sensors"
    ]
)

# =====================================================
# INPUT SELECTION
# =====================================================
if mode == "📊 Dataset Machine":
    row_id = st.sidebar.slider("Time Index", 0, len(df)-1, 10)
    sample = df[FEATURES].iloc[[row_id]]

elif mode == "✍️ Manual Sensor Input":
    sample = pd.DataFrame([{
        "temperature": st.sidebar.number_input("Temperature", 0.0, 200.0, 60.0),
        "vibration": st.sidebar.number_input("Vibration", 0.0, 10.0, 0.8),
        "pressure": st.sidebar.number_input("Pressure", 0.0, 20.0, 5.0),
        "rpm": st.sidebar.number_input("RPM", 0, 5000, 1500)
    }])

#starttttttttt

elif mode == "🔌 Arduino Live Sensors":

    st.subheader("🔌 Arduino Live Monitoring")

    start = st.checkbox("▶ Start")

    if not start:
        st.info("Click Start to begin monitoring")
        st.stop()   # 👈 safe stop (only inside mode)

    # ================= GET DATA =================
    raw_data, model_data = get_arduino_sensor_data()

    if raw_data is None:
        st.warning("Waiting for Arduino data...")
        time.sleep(1)
        st.rerun()

    sample_display = pd.DataFrame([raw_data])[FEATURES]
    sample_model = pd.DataFrame([model_data])[FEATURES]

    # ================= ML CONTROL =================
    if "last_pred_time" not in st.session_state:
        st.session_state.last_pred_time = 0
        st.session_state.anomaly = 1
        st.session_state.rul = 100

    if time.time() - st.session_state.last_pred_time > 5:
        anomaly = anomaly_model.predict(sample_model)[0]
        rul = int(np.clip(rul_model.predict(sample_model)[0], 0, MAX_LIFECYCLE))

        st.session_state.last_pred_time = time.time()
        st.session_state.anomaly = anomaly
        st.session_state.rul = rul
    else:
        anomaly = st.session_state.anomaly
        rul = st.session_state.rul

    # ================= SENSOR VALUES =================
    st.markdown("## 📌 Sensor Values")
    st.markdown(render_sensor_table(sample_display), unsafe_allow_html=True)

    # ================= AI SUMMARY =================
    c1, c2, c3 = st.columns(3)
    c1.metric("⏳ RUL", rul)

    # SENSOR STATUS
    sensor_status = get_sensor_status(sample_display)

    # HYBRID DECISION
    if anomaly == -1 and rul < 30:
        status = "🔴 Critical Anomalous"
    elif "🔴" in sensor_status:
        status = "🔴 At Risk"
    elif rul < 60 or "🟡" in sensor_status:
        status = "🟡 Warning"
    else:
        status = "🟢 Healthy"

    # SMOOTH STATUS
    status = smooth_status(status)

    # CLEAN TYPE
    if "🔴" in status:
        status_type = "critical"
    elif "🟡" in status:
        status_type = "warning"
    else:
        status_type = "healthy"

    # DISPLAY STATUS
    if status_type == "critical":
        c2.error(status)
    elif status_type == "warning":
        c2.warning(status)
    else:
        c2.success(status)

    # ================= MACHINE HEALTH =================
    st.markdown("## 🩺 Machine Health")

    health = calculate_health(status, sample_display)
    health = smooth_health(health)

    st.progress(health / 100)
    st.caption(f"Health Score: {health}%")

    if status_type == "healthy":
        st.success("🟢 System Healthy")
    elif status_type == "warning":
        st.warning("🟡 Needs Attention")
    else:
        st.error("🔴 Critical Condition")

    # ================= 🛠️ MAINTENANCE =================
    st.markdown("## 🛠️ Smart Maintenance Suggestions")

    actions = dynamic_maintenance(sample_display)

    for act in actions:
        if "🔧" in act or "🌡️" in act:
            st.error(act)
        elif "⚠️" in act:
            st.warning(act)
        else:
            st.success(act)

    # ================= CONTRIBUTING FACTORS (IMPROVED UI) =================
    impact = np.abs(sample_model.values[0])
    impact_pct = (impact / impact.sum()) * 100

    top_idx = np.argsort(impact_pct)[::-1][:2]
    
    st.markdown("## 🧠 Contributing Factors")

    for i in top_idx:
        percent = impact_pct[i]

        st.markdown(f"""
        <div class="contribution-card">
            <div class="sensor-title">{FEATURES[i].upper()}</div>
            <div class="sensor-value">Impact: {percent:.1f}%</div>

            <div class="progress-bg">
                <div class="progress-fill" style="width:{percent}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ================= ALERT =================
    if status_type == "critical":
        st.error("⚠ Machine needs immediate attention!")

    # ================= REFRESH =================
    time.sleep(2)
    st.rerun()
  
  #endddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
    
elif mode == "📡 Real-Time Monitoring":

    st.subheader("📡 Real-Time Monitoring")

    start = st.checkbox("▶ Start")

    if not start:
        st.info("Click Start to begin monitoring")
        st.stop()   # ✅ safe stop only inside mode

    # ================= GET DATA =================
    sample = pd.DataFrame([get_realtime_sensor_data()])

    anomaly = anomaly_model.predict(sample)[0]
    rul = int(np.clip(rul_model.predict(sample)[0], 0, MAX_LIFECYCLE))

    # ================= SENSOR VALUES =================
    st.markdown("## 📌 Sensor Values")
    st.markdown(render_sensor_table(sample), unsafe_allow_html=True)

    # ================= AI SUMMARY =================
    c1, c2, c3 = st.columns(3)
    c1.metric("⏳ RUL", rul)

    # 🔥 SENSOR STATUS (same as Arduino)
    sensor_status = get_sensor_status(sample)

    if anomaly == -1 and rul < 30:
        status = "🔴 Critical Anomalous"
    elif "🔴" in sensor_status:
        status = "🔴 At Risk"
    elif rul < 60 or "🟡" in sensor_status:
        status = "🟡 Warning"
    else:
        status = "🟢 Healthy"

    # 🔥 SMOOTH
    status = smooth_status(status)

    # 🔥 TYPE
    if "🔴" in status:
        status_type = "critical"
    elif "🟡" in status:
        status_type = "warning"
    else:
        status_type = "healthy"

    # DISPLAY
    if status_type == "critical":
        c2.error(status)
    elif status_type == "warning":
        c2.warning(status)
    else:
        c2.success(status)

    # ================= MACHINE HEALTH =================
    st.markdown("## 🩺 Machine Health")

    health = calculate_health(status, sample)
    health = smooth_health(health)

    st.progress(health / 100)
    st.caption(f"Health Score: {health}%")

    if status_type == "healthy":
        st.success("🟢 System Healthy")
    elif status_type == "warning":
        st.warning("🟡 Needs Attention")
    else:
        st.error("🔴 Critical Condition")

    # ================= MAINTENANCE =================
    st.markdown("## 🛠️ Smart Maintenance Suggestions")

    actions = dynamic_maintenance(sample)

    for act in actions:
        if "🔧" in act or "🌡️" in act:
            st.error(act)
        elif "⚠️" in act:
            st.warning(act)
        else:
            st.success(act)

    # ================= CONTRIBUTING FACTORS =================
    impact = np.abs(sample.values[0])
    impact_pct = (impact / impact.sum()) * 100

    top_idx = np.argsort(impact_pct)[::-1][:2]
    
    st.markdown("## 🧠 Contributing Factors")

    for i in top_idx:
        percent = impact_pct[i]

        st.markdown(f"""
        <div class="contribution-card">
            <div class="sensor-title">{FEATURES[i].upper()}</div>
            <div class="sensor-value">Impact: {percent:.1f}%</div>

            <div class="progress-bg">
                <div class="progress-fill" style="width:{percent}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ================= ALERT =================
    if status_type == "critical":
        st.error("⚠ Machine needs immediate attention!")

    # ================= AUTO REFRESH =================
    time.sleep(3)
    st.rerun()


# =====================================================
# COMMON DASHBOARD (DATASET + MANUAL)
# =====================================================
st.markdown("## 📌 Sensor Values")
st.markdown(render_sensor_table(sample), unsafe_allow_html=True)

anomaly = anomaly_model.predict(sample)[0]
rul = int(np.clip(rul_model.predict(sample)[0], 0, MAX_LIFECYCLE))

st.markdown("## 🤖 AI Prediction Summary")
c1, c2, c3 = st.columns(3)
c1.metric("⏳ RUL", rul)

if anomaly == -1 and rul < 30:
    status = "CRITICAL ANOMALOUS"
    c2.error("🔴 Critical Anomalous")
    
elif anomaly == -1:
    status = "EARLY ANOMALY"
    c2.warning("🟠 Early Anomaly")
    
elif rul < 30:
    status = "CRITICAL"
    c2.error("🔴 Critical")
   
elif rul < 60:
    status = "AT RISK"
    c2.warning("🟡 At Risk")
   
else:
    status = "HEALTHY"
    c2.success("🟢 Healthy")

# ================= MAINTENANCE URGENCY =================
if status in ["CRITICAL", "CRITICAL ANOMALOUS"]:
    urgency = "HIGH"
    st.error("🛠 Maintenance Urgency: HIGH")

elif status in ["AT RISK", "EARLY ANOMALY"]:
    urgency = "MEDIUM"
    st.warning("🛠 Maintenance Urgency: MEDIUM")

else:
    urgency = "LOW"
    st.success("🛠 Maintenance Urgency: LOW")
# ================= SHAP VALUES =================   

shap_vals = explainer.shap_values(scaler.transform(sample))
impact = np.abs(shap_vals[0]).flatten()
top_sensor = FEATURES[np.argmax(impact)]



st.markdown("## 🩺 Machine Health")
health = int((rul / MAX_LIFECYCLE) * 100)
st.progress(health / 100)
st.caption(f"Overall health score: {health}%")


ai_comment = get_llm_hybrid_comment(top_sensor, status, rul, anomaly)

st.markdown(
    f"""
    <div class="ai-insight-box">
        🧠 <b>AI Insight:</b><br><br>
        {ai_comment}
    </div>
    """,
    unsafe_allow_html=True
)


st.caption("AI recommendations are generated using historical data patterns and explainable feature contributions.")
# =====================================================
# EXPLAINABLE AI

st.markdown("## 🔍 Explainable AI")
fig, ax = plt.subplots(figsize=(6,3))
shap.bar_plot(shap_vals[0], feature_names=FEATURES, show=False)
st.pyplot(fig)

# =====================================================
# CONTRIBUTING FACTORS
# =====================================================
# =====================================================
# CONTRIBUTING FACTORS
# =====================================================
st.markdown("## 🧠 Contributing Factors")

# ✅ Show only when attention is required
if status == "HEALTHY" and anomaly != -1:
    st.success("✅ Machine is operating within normal limits. No maintenance action is required.")

else:
    impact_pct = (impact / impact.sum()) * 100 if impact.sum() != 0 else np.zeros(len(FEATURES))
    cols = st.columns(2)

    if "llm_history" not in st.session_state:
        st.session_state.llm_history = []

    for idx, i in enumerate(np.argsort(impact_pct)[::-1][:4]):

        reasons, actions = get_llm_reason_and_fix(
            FEATURES[i],
            sample[FEATURES[i]].values[0],
            impact_pct[i],
            "\n".join(st.session_state.llm_history[-3:])
        )

        # 🔹 Normalize output (single → list)
        if not isinstance(reasons, list):
            reasons = [reasons]

        if not isinstance(actions, list):
            actions = [actions]

        # store only first reason for repetition control
        st.session_state.llm_history.append(reasons[0])

        # 🔹 HTML formatting
        reason_html = "".join([f"• {r}<br>" for r in reasons])
        action_html = "".join([f"• {a}<br>" for a in actions])

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="contribution-box">
                    <b>{FEATURES[i].upper()}</b> → {impact_pct[i]:.1f}%<br><br>
                    <b>Reasons:</b><br>{reason_html}<br>
                    <b>Recommended Actions:</b><br>{action_html}
                </div>
                """,
                unsafe_allow_html=True
            )


# -------------------------------
import plotly.graph_objects as go

st.subheader("📈 Sensor Trend — Advanced Diagnostic View")

feature_to_plot = st.selectbox(
    "Choose sensor",
    FEATURES
)

window = st.slider(
    "Window size (data points)",
    min_value=10,
    max_value=60,
    value=25
)
row_id = 0
center = row_id
start = max(0, center - window)
end = min(len(df) - 1, center + window)

plot_df = df.iloc[start:end + 1]

low, high = SAFE_RANGES[feature_to_plot]

fig = go.Figure()

# ===============================
# SENSOR LINE (SMOOTH + PREMIUM)
# ===============================
fig.add_trace(go.Scatter(
    x=plot_df["time"],
    y=plot_df[feature_to_plot],
    mode="lines+markers",
    name=feature_to_plot.capitalize(),
    line=dict(
        width=3.5,
        shape="spline",
        color="#2563EB"   # premium blue
    ),
    marker=dict(
        size=6,
        color="#2563EB",
        line=dict(width=1.2, color="white")
    ),
    hovertemplate=
        "<b>Time:</b> %{x}<br>"
        "<b>Value:</b> %{y:.2f}<extra></extra>"
))

# ===============================
# SAFE RANGE (GRADIENT FEEL)
# ===============================
fig.add_hrect(
    y0=low,
    y1=high,
    fillcolor="rgba(34,197,94,0.18)",  # emerald green
    line_width=0,
    annotation_text="Safe operating range",
    annotation_position="top left"
)

# ===============================
# WARNING RANGE
# ===============================
fig.add_hrect(
    y0=high,
    y1=high * 1.15,
    fillcolor="rgba(245,158,11,0.22)",  # amber
    line_width=0,
    annotation_text="Warning zone",
    annotation_position="top left"
)

# ===============================
# SELECTED POINT (FOCUS MARKER)
# ===============================
fig.add_vline(
    x=df.iloc[center]["time"],
    line_dash="dot",
    line_color="rgba(220,38,38,0.7)",
    annotation_text="Focus",
    annotation_position="top"
)

# ===============================
# ANOMALY MARKERS (IF ANY)
# ===============================
if "anomaly" in df.columns:
    anomaly_df = plot_df[plot_df["anomaly"] == 1]
    if not anomaly_df.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_df["time"],
            y=anomaly_df[feature_to_plot],
            mode="markers",
            name="Anomaly",
            marker=dict(
                size=10,
                color="#DC2626",
                symbol="x"
            ),
            hovertemplate=
                "<b>Anomaly</b><br>"
                "Time: %{x}<br>"
                "Value: %{y:.2f}<extra></extra>"
        ))

# ===============================
# ULTIMATE LAYOUT POLISH
# ===============================
fig.update_layout(
    height=540,
    hovermode="x unified",
    dragmode="zoom",
    template="simple_white",

    plot_bgcolor="rgba(248,250,252,1)",
    paper_bgcolor="rgba(248,250,252,1)",

    title=dict(
        text=f"{feature_to_plot.capitalize()} — Sensor Health Trend",
        x=0.02,
        xanchor="left",
        font=dict(size=20, color="#111827")
    ),

    font=dict(
        family="Inter, Segoe UI, Arial",
        size=13,
        color="#1F2937"
    ),

    margin=dict(l=50, r=40, t=70, b=45),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    ),

    xaxis=dict(
    title=dict(
        text="Time",
        font=dict(color="#111827", size=14)
    ),
    tickfont=dict(color="#111827", size=12),
    showgrid=True,
    gridcolor="rgba(0,0,0,0.06)",
    zeroline=False,
    showline=True,
    linecolor="#CBD5E1",
    linewidth=1
),

yaxis=dict(
    title=dict(
        text=feature_to_plot.capitalize(),
        font=dict(color="#111827", size=14)
    ),
    tickfont=dict(color="#111827", size=12),
    showgrid=True,
    gridcolor="rgba(0,0,0,0.06)",
    zeroline=False,
    showline=True,
    linecolor="#CBD5E1",
    linewidth=1
)

)

st.plotly_chart(fig, use_container_width=True)