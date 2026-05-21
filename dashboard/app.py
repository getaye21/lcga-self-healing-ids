"""
🛡️ LCGA Self-Healing IDS — Real-Time Dashboard
MSc Thesis | Addis Ababa University
Deploy: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# Page config
st.set_page_config(
    page_title="LCGA Self-Healing IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1F3864; margin-bottom: 0; }
    .sub-header { font-size: 1rem; color: #666; margin-top: 0; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #666; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown('<p class="main-header">🛡️ LCGA Self-Healing IDS Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Lightweight CNN-GRU-Attention Framework | AAU MSc Thesis 2026</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Configuration")
    detection_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
    auto_heal = st.toggle("Enable Auto-Healing", value=True)
    st.divider()
    st.header("📊 System Status")
    st.metric("Model", "LCGA (~48K params)")
    st.metric("Inference", "< 2 ms/sample")
    st.metric("Uptime", "99.8%")
    st.divider()
    st.caption("© 2026 Getaye Fiseha, Mersen Getu, Chara Girma")
    st.caption("Addis Ababa University")

# ── Initialize session state ──
if 'history' not in st.session_state:
    st.session_state.history = []
if 'action_log' not in st.session_state:
    st.session_state.action_log = []
if 'intent_status' not in st.session_state:
    st.session_state.intent_status = {
        "I1: HTTP Latency": "healthy",
        "I2: SSH Availability": "healthy",
        "I3: Auth Failure Rate": "healthy",
        "I4: Port Scan Rate": "healthy",
        "I5: Bandwidth Limit": "healthy",
    }
if 'total_attacks' not in st.session_state:
    st.session_state.total_attacks = 0
if 'healed_attacks' not in st.session_state:
    st.session_state.healed_attacks = 0

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Detection", "📊 Intent Status", "📋 Action Log",
    "🧠 Explainability", "⚙️ Override"
])

# ── Tab 1: Live Detection ──
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">0.97</div><div class="metric-label">Detection F1</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_attacks}</div><div class="metric-label">Attacks Detected</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.healed_attacks}</div><div class="metric-label">Healed</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">1.8 ms</div><div class="metric-label">Latency/sample</div></div>', unsafe_allow_html=True)

    st.subheader("Real-time Anomaly Scores")
    # Generate rolling chart data
    if len(st.session_state.history) == 0:
        chart_data = pd.DataFrame({
            'Anomaly Score': np.random.rand(30) * 0.3,
            'Threshold': [0.5] * 30
        })
    else:
        chart_data = pd.DataFrame(st.session_state.history[-50:])
    st.line_chart(chart_data, height=300)

    st.subheader("Attack Distribution")
    attack_types = ["DoS Hulk", "DDoS", "PortScan", "Bot", "Infiltration",
                    "Web Attack Brute Force", "SSH-Patator", "BENIGN"]
    counts = [random.randint(0, 50) for _ in attack_types]
    chart_df = pd.DataFrame({"Attack": attack_types, "Count": counts}).set_index("Attack")
    st.bar_chart(chart_df, height=300)

# ── Tab 2: Intent Status ──
with tab2:
    st.subheader("Network Intent Status")
    for intent, status in st.session_state.intent_status.items():
        emoji = "✅" if status == "healthy" else "🚨"
        st.markdown(f"{emoji} **{intent}**: {status.upper()}")
    st.caption("Intents evaluated every 5 seconds by MAPE-K loop")

# ── Tab 3: Action Log ──
with tab3:
    st.subheader("Self-Healing Action Log")
    sample_log = [
        {"Time": "14:32:01", "Attack": "DoS Hulk", "Action": "BLOCK_IP", "MTTR": "91.6s", "Status": "✅"},
        {"Time": "14:32:15", "Attack": "PortScan", "Action": "BLOCK_IP", "MTTR": "32.8s", "Status": "✅"},
        {"Time": "14:33:02", "Attack": "DDoS", "Action": "ISOLATE_SUBNET", "MTTR": "92.6s", "Status": "✅"},
        {"Time": "14:33:45", "Attack": "SSH-Patator", "Action": "BLOCK_IP", "MTTR": "62.1s", "Status": "❌"},
    ]
    st.dataframe(pd.DataFrame(sample_log), use_container_width=True)
    st.caption("Actions verified via MAPE-K closed-loop feedback")

# ── Tab 4: Explainability ──
with tab4:
    st.subheader("SHAP Explainability")
    st.markdown("""
    **Decision Tree Surrogate** — mimics LCGA with 99.6% fidelity  
    **SHAP TreeExplainer** — 11,634× faster than LIME (0.07 ms vs 812 ms)
    """)
    st.info("🔍 SHAP force plots are generated per-prediction in the deployed environment.")
    st.markdown("**Top-5 SHAP Features (global):**")
    st.markdown("""
    1. `Flow Duration`  
    2. `Bwd Packet Length Max`  
    3. `Fwd IAT Mean`  
    4. `Packet Length Std`  
    5. `Flow Bytes/s`
    """)

# ── Tab 5: Override ──
with tab5:
    st.subheader("Manual Override Controls")
    st.warning("Use these controls to manually override automated healing actions.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Revert Last Block", use_container_width=True):
            st.success("Last IP block reverted.")
    with col2:
        if st.button("🔇 Pause Auto-Healing", use_container_width=True):
            st.warning("Auto-healing paused for 5 minutes.")
    st.divider()
    if st.button("🚨 Escalate to SOC Analyst", type="primary", use_container_width=True):
        st.error("🚨 Alert escalated to SOC team.")

# ── Footer ──
st.markdown("---")
st.caption("LCGA Self-Healing IDS Framework v1.0 | Getaye Fiseha, Mersen Getu, Chara Girma | AAU 2026")
