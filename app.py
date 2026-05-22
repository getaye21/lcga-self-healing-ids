"""
LCGA Self-Healing IDS — Scientific Dashboard
MSc Thesis | Addis Ababa University
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(
    page_title="LCGA IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helper to load results ----------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_json(path):
    import json
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

# ---------- Sidebar ----------
st.sidebar.title("🛡️ LCGA Self‑Healing IDS")
st.sidebar.markdown("**Intent‑Aware Autonomous Network Defense**")
st.sidebar.markdown("---")
st.sidebar.info(
    "MSc Thesis — Addis Ababa University\n\n"
    "Getaye Fiseha, Mersen Getu, Chara Girma\n\n"
    "© 2026 LCGA Framework"
)
st.sidebar.markdown("---")

# ---------- Main ----------
st.title("🛡️ LCGA: Lightweight Self‑Healing Intrusion Detection System")
st.markdown("""
### A Hybrid Deep Learning Framework for Real‑Time Threat Detection and Intent‑Aware Automated Remediation
""")

# Tabs for structured navigation
tabs = st.tabs([
    "📖 Overview", "⚙️ Methodology", "📊 Key Results",
    "🧠 Explainability", "🩺 Self‑Healing Demo", "📋 Conclusions"
])

# ================== TAB 1: Overview ==================
with tabs[0]:
    st.header("Problem & Motivation")
    st.markdown("""
    Modern enterprise networks face a rapidly evolving threat landscape.
    Existing intrusion detection systems (IDS) either:
    - Rely on **manual** investigation and remediation (high MTTR),
    - Operate as **black boxes** without explaining their decisions,
    - Lack **intent alignment** — they cannot map attacks to business‑level goals.
    
    **Our solution**: A lightweight, explainable deep‑learning framework that
    autonomously detects attacks, classifies them, maps them to violated network
    intents, and executes appropriate self‑healing actions — all with real‑time
    explanations for operators.
    """)
    
    st.header("Key Contributions")
    st.markdown("""
    1. **LCGA architecture** — A hybrid CNN‑GRU‑Attention model with only **41k parameters**,
       achieving **99.67% accuracy** and **87.6% intent satisfaction rate**.
    2. **Explainable AI via DT surrogate** — SHAP TreeExplainer on a distilled Decision Tree
       delivers explanations **11,635× faster** than LIME.
    3. **MAPE‑K closed‑loop orchestrator** — Adaptive verification windows reduce MTTR by
       **87%** compared to open‑loop systems.
    4. **Fully reproducible** — All code, data, and experiments are open‑source.
    """)

# ================== TAB 2: Methodology ==================
with tabs[1]:
    st.header("LCGA Architecture")
    st.markdown("""
    The LCGA model processes raw network flows through:
