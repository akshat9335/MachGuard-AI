import random
import requests

# =====================================================
# 🔁 DATA SOURCE SWITCH (FUTURE READY)
# =====================================================
USE_REAL_SENSOR = False   # 🔥 FUTURE: Set True when real sensors are connected

# =====================================================
# 🌐 SENSOR API CONFIG
# =====================================================
API_URL = "http://localhost:8000/latest"  
# FUTURE:
# - Replace localhost with server IP
# - Example: http://192.168.1.50:8000/latest
# - Can be cloud endpoint as well

def get_sensor_data():
    """
    Unified sensor data interface.

    FUTURE EXTENSIONS:
    - ESP32 / PLC / SCADA integration
    - MQTT broker subscription
    - Edge gateway data
    """

    if USE_REAL_SENSOR:
        try:
            res = requests.get(API_URL, timeout=1)

            # FUTURE:
            # - Validate response schema
            # - Add checksum / auth token
            # - Handle partial sensor failure

            return res.json()

        except Exception as e:
            # FUTURE:
            # - Log error to file / database
            # - Trigger fallback mode
            # - Raise alert if sensor offline
            return None

    # =================================================
    # 🔬 SIMULATED DATA (CURRENT MODE)
    # =================================================
    return {
        "temperature": random.uniform(55, 85),
        "vibration": random.uniform(0.6, 2.5),
        "pressure": random.uniform(4.5, 6.0),
        "rpm": random.uniform(1100, 1500)

        # FUTURE:
        # - Add current, voltage, torque
        # - Add acoustic / sound level
        # - Add oil quality index
    }
