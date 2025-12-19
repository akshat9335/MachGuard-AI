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
features = ["temperature", "vibration", "pressure", "rpm"]

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

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Mode")
mode = st.sidebar.radio(
    "Select input type:",
    ["📊 Dataset Machine", "✍️ Manual Sensor Input", "📡 Real-Time Monitoring"]
)

# =====================================================
# INPUT SELECTION
# =====================================================
if mode == "📊 Dataset Machine":
    row_id = st.sidebar.slider("Time Index", 0, len(df)-1, 10)
    sample = df[features].iloc[[row_id]]

elif mode == "✍️ Manual Sensor Input":
    sample = pd.DataFrame([{
        "temperature": st.sidebar.number_input("Temperature", 0.0, 200.0, 60.0),
        "vibration": st.sidebar.number_input("Vibration", 0.0, 10.0, 0.8),
        "pressure": st.sidebar.number_input("Pressure", 0.0, 20.0, 5.0),
        "rpm": st.sidebar.number_input("RPM", 0, 5000, 1500)
    }])

else:
    st.subheader("📡 Real-Time Monitoring")
    start = st.checkbox("▶ Start")

    placeholder = st.empty()

    if start:
        while True:
            sample = pd.DataFrame([get_realtime_sensor_data()])

            anomaly = anomaly_model.predict(sample)[0]
            rul = int(np.clip(rul_model.predict(sample)[0], 0, MAX_LIFECYCLE))

            with placeholder.container():

                # ================= SENSOR VALUES =================
                st.markdown("## 📌 Sensor Values")
                st.markdown(render_sensor_table(sample), unsafe_allow_html=True)

                # ================= AI PREDICTION SUMMARY =================
                st.markdown("## 🤖 AI Prediction Summary")
                c1, c2, c3 = st.columns(3)

                c1.metric("⏳ RUL", rul)

                # ✅ SAME STATUS LOGIC AS METHOD 1 & 2
                if anomaly == -1 and rul < 30:
                    status = "CRITICAL ANOMALOUS"
                    c2.error("🔴 Critical Anomalous")
                    

                

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
                top_sensor = features[np.argmax(impact)]

                # ================= MACHINE HEALTH =================
                st.markdown("## 🩺 Machine Health")
                health = int((rul / MAX_LIFECYCLE) * 100)
                st.progress(health / 100)
                st.caption(f"Overall health score: {health}%")

                # ================= AI INSIGHT =================
                ai_comment = get_llm_hybrid_comment(
                    top_sensor, status, rul, anomaly
                )
                st.markdown(
                    f"""
                    <div style="
                        font-size:18px;
                        padding:12px;
                        border-radius:10px;
                        background-color:#020617;
                        border:1px solid #334155;
                    ">
                        🧠 <b>AI Insight:</b> {ai_comment}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    "AI insights are generated using explainable feature contributions and historical patterns."
                )

                # ================= EXPLAINABLE AI =================
                st.markdown("## 🔍 Explainable AI")
                fig, ax = plt.subplots(figsize=(6,3))
                shap.bar_plot(shap_vals[0], feature_names=features, show=False)
                st.pyplot(fig)

                # ================= CONTRIBUTING FACTORS =================
                st.markdown("## 🧠 Contributing Factors")

                # 🔒 SAME SUPPRESSION LOGIC
                if status == "HEALTHY" and anomaly != -1:
                    st.success(
                        "✅ Machine is operating within normal limits. No maintenance action required."
                    )

                else:
                    impact_pct = (
                        (impact / impact.sum()) * 100
                        if impact.sum() != 0
                        else np.zeros(len(features))
                    )

                    cols = st.columns(2)

                    if "llm_history" not in st.session_state:
                        st.session_state.llm_history = []

                    for idx, i in enumerate(np.argsort(impact_pct)[::-1][:4]):

                        reasons, actions = get_llm_reason_and_fix(
                            features[i],
                            sample[features[i]].values[0],
                            impact_pct[i],
                            "\n".join(st.session_state.llm_history[-3:])
                        )

                        # 🔹 Normalize single/multiple outputs
                        if not isinstance(reasons, list):
                            reasons = [reasons]
                        if not isinstance(actions, list):
                            actions = [actions]

                        st.session_state.llm_history.append(reasons[0])

                        reason_html = "".join([f"• {r}<br>" for r in reasons])
                        action_html = "".join([f"• {a}<br>" for a in actions])

                        with cols[idx % 2]:
                            st.markdown(
                                f"""
                                <div class="contribution-box">
                                    <b>{features[i].upper()}</b> → {impact_pct[i]:.1f}%<br><br>
                                    <b>Reasons:</b><br>{reason_html}<br>
                                    <b>Recommended Actions:</b><br>{action_html}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            time.sleep(3)

    st.stop()



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
top_sensor = features[np.argmax(impact)]



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
shap.bar_plot(shap_vals[0], feature_names=features, show=False)
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
    impact_pct = (impact / impact.sum()) * 100 if impact.sum() != 0 else np.zeros(len(features))
    cols = st.columns(2)

    if "llm_history" not in st.session_state:
        st.session_state.llm_history = []

    for idx, i in enumerate(np.argsort(impact_pct)[::-1][:4]):

        reasons, actions = get_llm_reason_and_fix(
            features[i],
            sample[features[i]].values[0],
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
                    <b>{features[i].upper()}</b> → {impact_pct[i]:.1f}%<br><br>
                    <b>Reasons:</b><br>{reason_html}<br>
                    <b>Recommended Actions:</b><br>{action_html}
                </div>
                """,
                unsafe_allow_html=True
            )


# -------------------------------
# Sensor Trend with Zoom
# -------------------------------
st.subheader("📈 Sensor Trend Analysis (Zoom at Sharp End)")

sensor_option = st.selectbox(
    "Select Sensor",
    ["temperature", "vibration", "pressure", "rpm"]
)

zoom_points = st.slider(
    "🔍 Zoom last N readings (Sharp End)",
    min_value=20,
    max_value=200,
    value=60,
    step=10
)

zoom_df = df.tail(zoom_points)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    zoom_df["time"],
    zoom_df[sensor_option],
    color="red",
    linewidth=2,
    label=sensor_option.capitalize()
)

ax.set_title(f"Zoomed Sensor Trend – {sensor_option.capitalize()}")
ax.set_xlabel("Time")
ax.set_ylabel("Sensor Value")
ax.legend()
ax.grid(True)

st.pyplot(fig)
