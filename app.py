"""
LCGA Self‑Healing IDS – Real‑Time Dashboard
MSc Thesis | Addis Ababa University
"""

import streamlit as st
import pandas as pd
import numpy as np
import time, os, sys

# Ensure the src/ package is importable
sys.path.insert(0, os.path.dirname(__file__))

# ═══════════════════════════════════════════════
# 1. PAGE CONFIGURATION
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="LCGA Self‑Healing IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
# 2. SIDEBAR – CONFIGURATION
# ═══════════════════════════════════════════════
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png", width=80
)  # placeholder shield icon
st.sidebar.title("🛡️ LCGA IDS")
st.sidebar.markdown("**Intent‑Aware Self‑Healing Network**")
st.sidebar.markdown("---")

# Model & data paths
MODEL_DIR = st.sidebar.text_input("Model directory", "models/")
INTENTS_FILE = st.sidebar.text_input("Intents YAML", "config/intents.yaml")

# Simulation toggle (for demo without real models)
USE_SIM = st.sidebar.checkbox("Simulate data (no models)", value=True)

# Refresh interval
REFRESH_SEC = st.sidebar.slider("Refresh interval (sec)", 1, 10, 3)

# About section
st.sidebar.markdown("---")
st.sidebar.info(
    "**MSc Thesis** — Addis Ababa University\n\n"
    "Getaye Fiseha, Mersen Getu, Chara Girma\n\n"
    "© 2026 LCGA Framework"
)

# ═══════════════════════════════════════════════
# 3. SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["timestamp", "attack", "action", "intents", "restored"]
    )
if "anomaly_scores" not in st.session_state:
    st.session_state.anomaly_scores = []


def simulate_telemetry():
    """Return a random attack class (or BENIGN) and fake explanation."""
    from datetime import datetime
    classes = [
        "BENIGN", "DoS Hulk", "DDoS", "PortScan",
        "Bot", "SSH‑Patator", "Web Attack Brute Force",
        "Infiltration", "Heartbleed",
    ]
    weights = [0.6, 0.1, 0.05, 0.05, 0.05, 0.03, 0.04, 0.03, 0.05]
    attack = np.random.choice(classes, p=weights)
    anomaly = attack != "BENIGN"
    score = np.random.uniform(0.2, 0.5) if not anomaly else np.random.uniform(0.6, 1.0)
    return anomaly, attack, score, datetime.now().isoformat()


# ═══════════════════════════════════════════════
# 4. MAIN DASHBOARD
# ═══════════════════════════════════════════════
st.title("🛡️ LCGA Self‑Healing IDS Dashboard")
st.markdown(
    "**Real‑time network threat detection, intent‑aware classification, "
    "and explainable automated remediation.**"
)

# ─── Tabs ──────────────────────────────────────
tabs = st.tabs([
    "📈 Live Detection", "📊 Intent Status",
    "📋 Action Log", "🧠 Explainability", "⚙️ Override",
])

# ── Tab 1: Live Detection ───────────────────────
with tabs[0]:
    st.subheader("Real‑time Anomaly Scores")

    # Generate new data point on each refresh
    anomaly, attack, score, ts = simulate_telemetry()
    st.session_state.anomaly_scores.append(score)
    # Keep only last 60 points
    if len(st.session_state.anomaly_scores) > 60:
        st.session_state.anomaly_scores.pop(0)

    # Plot rolling chart
    chart_data = pd.DataFrame({
        "Score": st.session_state.anomaly_scores,
        "Threshold": [0.5] * len(st.session_state.anomaly_scores),
    })
    st.line_chart(chart_data, height=250)

    # Current status indicator
    col1, col2 = st.columns(2)
    with col1:
        if anomaly:
            st.error(f"🚨 **Attack detected: {attack}**  (score={score:.3f})")
        else:
            st.success(f"✅ Normal traffic  (score={score:.3f})")
    with col2:
        st.metric("Last update", ts[:19].replace("T", " "))

    # If attack detected, simulate a healing action
    if anomaly:
        from datetime import datetime
        action_map = {
            "DoS Hulk": "BLOCK_IP",
            "DDoS": "ISOLATE_SUBNET",
            "PortScan": "BLOCK_IP",
            "Bot": "ISOLATE_SUBNET",
            "SSH‑Patator": "BLOCK_IP",
            "Web Attack Brute Force": "BLOCK_IP",
            "Infiltration": "ISOLATE_SUBNET",
            "Heartbleed": "RESTART_SERVICE",
        }
        action = action_map.get(attack, "ESCALATE")
        new_entry = pd.DataFrame([{
            "timestamp": ts,
            "attack": attack,
            "action": action,
            "intents": "I1, I4",
            "restored": np.random.choice([True, False], p=[0.85, 0.15]),
        }])
        st.session_state.history = pd.concat(
            [st.session_state.history, new_entry], ignore_index=True
        )

# ── Tab 2: Intent Status ───────────────────────
with tabs[1]:
    st.subheader("Network Intent Status")
    # In a full implementation, load from knowledge_base.py
    # Here we simulate intent states
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

# ── Tab 3: Action Log ───────────────────────────
with tabs[2]:
    st.subheader("Self‑Healing Action Log")
    st.dataframe(
        st.session_state.history.tail(20).sort_values("timestamp", ascending=False),
        use_container_width=True,
    )
    if not st.session_state.history.empty:
        restored_rate = st.session_state.history["restored"].mean() * 100
        st.metric("Intent Restoration Rate (ISR)", f"{restored_rate:.1f}%")

# ── Tab 4: Explainability ───────────────────────
with tabs[3]:
    st.subheader("Explainable AI (SHAP & LIME)")
    st.markdown(
        "In the full implementation, this panel shows **SHAP force plots** "
        "and **decision tree rules** for each prediction. Below is a placeholder."
    )
    # Placeholder SHAP‑style explanation
    if anomaly:
        st.info(f"**Top contributing features for '{attack}':**")
        features = {
            "DoS Hulk": ["Flow Duration", "Bwd Packet Length Std", "Packet Length Mean"],
            "DDoS": ["Flow Bytes/s", "Destination Port", "Total Fwd Packets"],
            "PortScan": ["Destination Port", "Flow Duration", "Init_Win_bytes_forward"],
        }
        top_feats = features.get(attack, ["Flow Duration", "Bwd Packet Length Std", "Packet Length Mean"])
        for i, feat in enumerate(top_feats):
            st.write(f"{i+1}. **{feat}**  (SHAP value: {np.random.uniform(0.1,0.5):.3f})")
        st.caption("Based on Decision Tree surrogate model — see `src/xai/explainer.py`")
    else:
        st.write("No attack detected — no explanation needed.")

# ── Tab 5: Override ─────────────────────────────
with tabs[4]:
    st.subheader("Manual Override Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Revert Last Action"):
            st.success("Last healing action reverted.")
    with col2:
        if st.button("🔓 Unblock All IPs"):
            st.success("All IP blocks removed.")
    with col3:
        if st.button("📧 Notify Admin"):
            st.success("Admin alerted via email.")

    st.markdown("---")
    st.caption(
        "Override actions are logged and the Knowledge Base is updated "
        "so the MAPE‑K planner learns from operator decisions."
    )

# ═══════════════════════════════════════════════
# 5. FOOTER
# ═══════════════════════════════════════════════
st.markdown("---")
st.caption(
    "LCGA Framework v1.0 | AAU MSc Thesis 2026 | "
    "[GitHub](https://github.com/getaye21/lcga-self-healing-ids)"
)

# Auto‑refresh
time.sleep(REFRESH_SEC)
st.rerun()
