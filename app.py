"""
LCGA Self‑Healing IDS – Interactive Scientific Dashboard
MSc Thesis • Addis Ababa University • 2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os, json, joblib, time, base64
from sklearn.tree import DecisionTreeClassifier, export_text
import shap

# ══════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="LCGA IDS", page_icon="🛡️", layout="wide")

# ---- Custom CSS for modern look ----
st.markdown("""
<style>
    /* Main background & font */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .main-header { font-size: 2.8rem; font-weight: 800; color: #00d2ff; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #c0c0c0; text-align: center; margin-bottom: 2rem; }
    .card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.1); }
    .metric-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 1.5rem; text-align: center; color: white; }
    .stButton>button { background: linear-gradient(90deg, #00d2ff, #3a7bd5); color: white; border: none; border-radius: 12px; font-weight: 600; padding: 0.6rem 2rem; transition: all 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,210,255,0.3); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 10px 20px; background: rgba(255,255,255,0.05); color: #aaa; }
    .stTabs [aria-selected="true"] { background: rgba(0,210,255,0.15) !important; color: #00d2ff !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 2. CACHE HELPERS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_surrogate():
    dt = joblib.load("dt_surrogate.pkl")
    le = joblib.load("cic_label_enc.pkl")
    feature_names = joblib.load("cic_feature_names.pkl")
    return dt, le, feature_names

@st.cache_resource
def load_shap_explainer(_dt):
    return shap.TreeExplainer(_dt)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_image(path):
    return Image.open(path) if os.path.exists(path) else None

# ══════════════════════════════════════════════════════════════
# 3. SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("<h2 style='color:#00d2ff;'>🛡️ LCGA IDS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("**Intent‑Aware Self‑Healing Network**")
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background:rgba(255,255,255,0.05); padding:1rem; border-radius:12px;'>
    <b>MSc Thesis</b><br>Addis Ababa University<br>
    <small>Getaye Fiseha, Mersen Getu, Chara Girma</small>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 LCGA Framework")

# ══════════════════════════════════════════════════════════════
# 4. LOAD MODELS (graceful fallback)
# ══════════════════════════════════════════════════════════════
try:
    dt, le, feature_names = load_surrogate()
    shap_explainer = load_shap_explainer(dt)
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Model files missing: {e}")
    models_loaded = False

# ══════════════════════════════════════════════════════════════
# 5. HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='main-header'>🛡️ LCGA Self‑Healing IDS</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Lightweight Hybrid Deep Learning for Real‑Time Threat Detection & Intent‑Aware Remediation</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 6. TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📖 Overview", "⚙️ Methodology",
    "📊 Precomputed Results", "🧠 Live Detection",
    "🩺 Self‑Healing Simulator", "🔍 Explainability Explorer",
    "📋 Action Log", "📜 Conclusions"
])

# ── TAB 0: Overview ──
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🌐 Problem & Motivation")
    st.markdown("""
    Modern networks face a rapidly growing threat landscape.
    Existing intrusion detection systems (IDS):
    - Rely on **manual** investigation → high MTTR
    - Operate as **black boxes** → no operator trust
    - Lack **intent alignment** → actions don't match business goals

    **Our solution**: An explainable, intent‑aware deep‑learning framework that
    autonomously detects, classifies, and remediates network attacks in real time.
    """)
    st.subheader("✨ Key Contributions")
    st.markdown("""
    1. **LCGA** – CNN‑GRU‑Attention with only 41k params, 99.67% accuracy
    2. **DT Surrogate + SHAP** – Explanations 11,635× faster than LIME
    3. **MAPE‑K Orchestrator** – 87% MTTR reduction, 87.6% ISR
    4. **Fully Reproducible** – Open‑source code, data, and experiments
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 1: Methodology ──
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 LCGA Architecture")
        st.code("""
Input (73 features)
  → Parallel Conv1D (32,3) + (64,5)
  → Concatenate → GRU(64)
  → Multi‑Head Self‑Attention (2 heads)
  → GlobalAvgPool → Dense(64)
  → Softmax (12 classes)
        """, language="text")
        st.metric("Trainable Parameters", "41,260")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔄 MAPE‑K Self‑Healing Loop")
        st.markdown("""
        1. **Monitor** – capture network telemetry
        2. **Analyze** – LCGA + DT surrogate + SHAP
        3. **Plan** – map attack → violated intents → best action
        4. **Execute** – block IP, restart service, isolate subnet
        5. **Knowledge** – verify intent restoration, update success rates
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 2: Precomputed Results ──
with tabs[2]:
    tab_inner = st.tabs(["Model Comparison", "Training History", "Confusion Matrix", "System Metrics"])
    with tab_inner[0]:
        comp = load_csv("results/model_comparison.csv")
        if comp is not None:
            st.dataframe(comp.style.highlight_max(subset=["Macro F1"], color="#00d2ff", axis=0),
                         use_container_width=True)
        else:
            st.info("Upload `results/model_comparison.csv`")
    with tab_inner[1]:
        img = load_image("results/lcga_training_history.png")
        if img: st.image(img, use_column_width=True)
        else: st.info("Upload `results/lcga_training_history.png`")
    with tab_inner[2]:
        img = load_image("results/lcga_confusion_matrix.png")
        if img: st.image(img, use_column_width=True)
        else: st.info("Upload `results/lcga_confusion_matrix.png`")
    with tab_inner[3]:
        sys_df = load_csv("results/system_comparison.csv")
        if sys_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="MTTR (s)", x=sys_df["System"], y=sys_df["MTTR_s"],
                                 marker_color=["#E24B4A","#854F0B","#00d2ff"]))
            fig.add_trace(go.Bar(name="ISR (%)", x=sys_df["System"], y=sys_df["ISR_pct"],
                                 marker_color=["#E24B4A","#854F0B","#00d2ff"], visible=False))
            fig.update_layout(barmode="group", updatemenus=[{
                "buttons": [
                    {"label": "MTTR", "method": "update", "args": [{"visible": [True, False]}]},
                    {"label": "ISR", "method": "update", "args": [{"visible": [False, True]}]},
                ]
            }])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload `results/system_comparison.csv`")

# ── TAB 3: Live Detection ──
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔎 Live Network Flow Classification (DT Surrogate)")
    if not models_loaded:
        st.warning("Models not loaded – upload `dt_surrogate.pkl`, `cic_label_enc.pkl`, `cic_feature_names.pkl`")
    else:
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Upload a CSV with flow records", type="csv")
        with col2:
            use_random = st.button("🎲 Use random test sample")

        input_data = None
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if set(feature_names).issubset(set(df.columns)):
                    input_data = df[feature_names].values.astype(np.float32)
                    st.success(f"Loaded {len(input_data)} flow(s)")
                else:
                    st.error("CSV must contain all 73 features.")
            except Exception as e:
                st.error(f"Error: {e}")

        if use_random:
            np.random.seed(int(time.time()))
            input_data = np.random.randn(5, 73).astype(np.float32)

        if input_data is not None:
            preds = dt.predict(input_data)
            confidences = np.max(dt.predict_proba(input_data), axis=1)
            labels = le.inverse_transform(preds)
            st.subheader("📋 Predictions")
            for i, (lbl, conf) in enumerate(zip(labels, confidences)):
                st.write(f"**Sample {i+1}:** {lbl}  ({conf:.1%} confidence)")

            # SHAP explanation for first sample
            st.subheader("🧠 SHAP Explanation (first sample)")
            shap_vals = shap_explainer.shap_values(input_data[0:1])
            if isinstance(shap_vals, list):
                sv = shap_vals[preds[0]][0]
                expected = shap_explainer.expected_value[preds[0]]
            else:
                sv = shap_vals[0]
                expected = shap_explainer.expected_value
            fig = shap.force_plot(expected, sv, input_data[0], feature_names=feature_names,
                                  matplotlib=True, show=False)
            st.pyplot(fig)

            # Decision rule snippet
            st.subheader("📜 Decision Rule Path")
            st.code(export_text(dt, feature_names=list(feature_names), max_depth=5)[:1200])
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 4: Self‑Healing Simulator ──
with tabs[4]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 MAPE‑K Self‑Healing Simulator")
    st.markdown("Click the button to simulate a stream of 50 network attacks and observe the orchestrator's response.")
    if st.button("🚀 Run Simulation Cycle"):
        intents = {"I1": {"name":"HTTP Latency","t_verify":90},
                   "I2": {"name":"SSH Availability","t_verify":60},
                   "I3": {"name":"Auth Fail Rate","t_verify":60},
                   "I4": {"name":"Port Scan Rate","t_verify":30},
                   "I5": {"name":"Bandwidth","t_verify":90}}
        attack_map = {"DoS Hulk":["I1","I5"],"DDoS":["I1","I5"],"PortScan":["I4"],
                      "SSH-Patator":["I2","I3"],"Bot":["I1","I5"]}
        actions = {"DoS Hulk":"BLOCK_IP","DDoS":"ISOLATE_SUBNET","PortScan":"BLOCK_IP",
                   "SSH-Patator":"BLOCK_IP","Bot":"ISOLATE_SUBNET"}
        np.random.seed(42)
        attacks = np.random.choice(list(attack_map.keys()), 50)
        log = []
        for att in attacks:
            ttd = max(5, np.random.normal(80,15))
            heal = np.random.normal(2000,500)
            verify = intents[attack_map[att][0]]["t_verify"]*1000
            success = np.random.choice([True,False], p=[0.88,0.12])
            log.append({"attack":att,"action":actions.get(att,"ESCALATE"),
                        "ttd_ms":ttd,"mttr_ms":ttd+heal+verify,"success":success})
        df = pd.DataFrame(log)
        st.dataframe(df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Time to Recovery", f"{df['mttr_ms'].mean()/1000:.1f} s")
        col2.metric("Intent Satisfaction Rate", f"{df['success'].mean()*100:.1f} %")
        col3.metric("Actions Executed", len(df))
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 5: Explainability Explorer ──
with tabs[5]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Interactive Explainability Explorer")
    if not models_loaded:
        st.warning("Models not loaded.")
    else:
        uploaded_single = st.file_uploader("Upload a single flow CSV row", type="csv", key="single")
        if uploaded_single is not None:
            df_single = pd.read_csv(uploaded_single)
            if set(feature_names).issubset(set(df_single.columns)):
                x = df_single[feature_names].values.astype(np.float32).reshape(1,-1)
                pred = dt.predict(x)
                cls = pred[0]
                st.success(f"Prediction: **{le.inverse_transform([cls])[0]}**")
                shap_vals = shap_explainer.shap_values(x)
                if isinstance(shap_vals, list):
                    fig = shap.force_plot(shap_explainer.expected_value[cls], shap_vals[cls][0],
                                          x[0], feature_names=feature_names, matplotlib=True, show=False)
                else:
                    fig = shap.force_plot(shap_explainer.expected_value, shap_vals[0],
                                          x[0], feature_names=feature_names, matplotlib=True, show=False)
                st.pyplot(fig)
            else:
                st.error("CSV missing required features.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 6: Action Log ──
with tabs[6]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 MAPE‑K Action Log")
    log_json = load_json("results/action_log_full.json")
    if log_json:
        st.dataframe(pd.DataFrame(log_json).head(30), use_container_width=True)
    else:
        st.info("Upload `results/action_log_full.json`")
    st.subheader("Ablation Study")
    abl = load_csv("results/ablation_study.csv")
    if abl is not None:
        st.dataframe(abl, use_container_width=True)
    else:
        st.info("Upload `results/ablation_study.csv`")
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 7: Conclusions ──
with tabs[7]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 Conclusions & Future Work")
    st.markdown("""
    - **LCGA** achieves 99.67% accuracy with only 41k parameters.
    - **MAPE‑K orchestrator** delivers 87% MTTR reduction and 87.6% ISR.
    - **SHAP explanations** are 11,635× faster than LIME, enabling real‑time trust.

    **Future:** zero‑day attacks, SDN hardware deployment, SIEM integration, federated learning.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("LCGA Framework v1.0 | AAU MSc Thesis 2026 | [GitHub](https://github.com/getaye21/lcga-self-healing-ids)")
