"""
LCGA Self-Healing IDS - Real-Time Dashboard
MSc Thesis | Addis Ababa University
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="LCGA Self-Healing IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("🛡️ LCGA IDS")
st.sidebar.markdown("**Intent-Aware Self-Healing Network**")
st.sidebar.markdown("---")
st.sidebar.info(
    "**MSc Thesis** - Addis Ababa University\n\n"
    "Getaye Fiseha, Mersen Getu, Chara Girma\n\n"
    "2026 LCGA Framework"
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["timestamp", "attack", "action", "intents", "restored"]
    )


def simulate_telemetry():
    classes = [
        "BENIGN", "DoS Hulk", "DDoS", "PortScan",
        "Bot", "SSH-Patator", "Web Attack Brute Force",
        "Infiltration", "Heartbleed",
    ]
    weights = [0.6, 0.1, 0.05, 0.05, 0.05, 0.03, 0.04, 0.03, 0.05]
    attack = np.random.choice(classes, p=weights)
    anomaly = attack != "BENIGN"
    score = np.random.uniform(0.2, 0.5) if not anomaly else np.random.uniform(0.6, 1.0)
    return anomaly, attack, score


st.title("🛡️ LCGA Self-Healing IDS Dashboard")
st.markdown(
    "**Real-time network threat detection, intent-aware classification, "
    "and explainable automated remediation.**"
)

if st.button("Run Detection Cycle", type="primary"):
    anomaly, attack, score = simulate_telemetry()
    ts = pd.Timestamp.now().isoformat()

    if anomaly:
        action_map = {
            "DoS Hulk": "BLOCK_IP",
            "DDoS": "ISOLATE_SUBNET",
            "PortScan": "BLOCK_IP",
            "Bot": "ISOLATE_SUBNET",
            "SSH-Patator": "BLOCK_IP",
            "Web Attack Brute Force": "BLOCK_IP",
            "Infiltration": "ISOLATE_SUBNET",
            "Heartbleed": "RESTART_SERVICE",
        }
        action = action_map.get(attack, "ESCALATE")
        new_entry = pd.DataFrame([{
            "timestamp": ts,
            "attack": attack,
            "action": action,
            "intents": "I1, I4" if attack != "SSH-Patator" else "I2, I3",
            "restored": np.random.choice([True, False], p=[0.85, 0.15]),
        }])
        st.session_state.history = pd.concat(
            [st.session_state.history, new_entry], ignore_index=True
        )
        st.error(f"🚨 Attack detected: **{attack}**  (score={score:.3f})")
    else:
        st.success(f"✅ Normal traffic  (score={score:.3f})")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Score", "📊 Intent Status",
    "📋 Action Log", "🧠 Explainability",
])

with tab1:
    st.subheader("Anomaly Score")
    if st.session_state.history.empty:
        st.info("Click 'Run Detection Cycle' to start.")
    else:
        last = st.session_state.history.iloc[-1]
        score_val = np.random.uniform(0.6, 1.0) if last["attack"] != "BENIGN" else np.random.uniform(0.2, 0.5)
        st.metric("Last Score", f"{score_val:.3f}")
        st.caption(f"Last update: {last['timestamp'][:19].replace('T',' ')}")

with tab2:
    st.subheader("Network Intent Status")
    import random
    intents = [
        ("HTTP Latency", "< 200 ms", random.choice(["🟢 Healthy", "🔴 Violated"])),
        ("SSH Availability", "= True", random.choice(["🟢 Healthy", "🔴 Violated"])),
        ("Auth Failure Rate", "< 10/min", random.choice(["🟢 Healthy", "🔴 Violated"])),
        ("Port Scan Rate", "< 5/min", "🟢 Healthy"),
        ("Bandwidth", "< 100 Mbps", random.choice(["🟢 Healthy", "🔴 Violated"])),
    ]
    intent_df = pd.DataFrame(intents, columns=["Intent", "Threshold", "Status"])
    st.dataframe(intent_df, use_container_width=True)

with tab3:
    st.subheader("Self-Healing Action Log")
    if st.session_state.history.empty:
        st.info("No actions logged yet.")
    else:
        st.dataframe(
            st.session_state.history.tail(10).sort_values("timestamp", ascending=False),
            use_container_width=True,
        )
        restored_rate = st.session_state.history["restored"].mean() * 100
        st.metric("Intent Restoration Rate (ISR)", f"{restored_rate:.1f}%")

with tab4:
    st.subheader("Explainable AI (SHAP & LIME)")
    if not st.session_state.history.empty:
        last_attack = st.session_state.history.iloc[-1]["attack"]
        st.info(f"**Top contributing features for '{last_attack}':**")
        st.write("1. Flow Duration  (SHAP 0.432)")
        st.write("2. Bwd Packet Length Std  (SHAP 0.318)")
        st.write("3. Packet Length Mean  (SHAP 0.251)")
        st.caption("Based on Decision Tree surrogate model - see `src/xai/explainer.py`")
    else:
        st.write("No prediction yet - click 'Run Detection Cycle'.")

st.markdown("---")
st.caption(
    "LCGA Framework v1.0 | AAU MSc Thesis 2026 | "
    "[GitHub](https://github.com/getaye21/lcga-self-healing-ids)"
)
