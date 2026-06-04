"""
LCGA Self-Healing IDS - Scientific Dashboard with Live Inference (DT Surrogate)
MSc ML Thesis | Addis Ababa University
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import os, json, time
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
import shap

st.set_page_config(page_title="LCGA IDS", page_icon="🛡️", layout="wide")

# ========== AAU Color Theme ==========
AAU_BLUE = "#1f4e79"
AAU_YELLOW = "#ffcd00"
AAU_WHITE = "#ffffff"
AAU_LIGHT_BG = "#f0f4f8"

st.markdown(f"""
<style>
    /* Main background light */
    .stApp {{
        background: {AAU_LIGHT_BG};
    }}
    /* Sidebar styling - AAU blue gradient */
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, {AAU_BLUE} 0%, #2c6e9e 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: {AAU_WHITE} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stInfo,
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] .stButton>button {{
        color: {AAU_WHITE} !important;
    }}
    /* Sidebar button hover */
    [data-testid="stSidebar"] .stButton>button:hover {{
        background-color: {AAU_YELLOW} !important;
        color: {AAU_BLUE} !important;
        border: none;
    }}
    /* Main content cards */
    .stApp h1, .stApp h2, .stApp h3, .stApp .stMarkdown {{
        color: {AAU_BLUE};
    }}
    .card {{
        background: {AAU_WHITE};
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border: 1px solid #d0d5dd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        color: {AAU_BLUE};
    }}
    .card-blue {{ border-left: 6px solid {AAU_BLUE}; }}
    .card-yellow {{ border-left: 6px solid {AAU_YELLOW}; }}
    .main-header {{
        font-size: 2.8rem;
        font-weight: 800;
        color: {AAU_BLUE};
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: {AAU_BLUE};
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background: #eef2f5;
        color: {AAU_BLUE};
    }}
    .stTabs [aria-selected="true"] {{
        background: {AAU_YELLOW} !important;
        color: {AAU_BLUE} !important;
        font-weight: 600;
    }}
    .stMetric {{
        background: {AAU_WHITE};
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #d0d5dd;
    }}
    .stButton>button {{
        background: {AAU_BLUE};
        color: {AAU_WHITE};
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.8rem;
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        background: {AAU_YELLOW};
        color: {AAU_BLUE};
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(31,78,121,0.3);
    }}
    .shap-force-plot {{
        background: {AAU_WHITE} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ---------- Define the EXACT 73 features your DT expects ----------
EXPECTED_FEATURES_73 = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "avg_packet_size",
    "fwd_pkt_range", "bwd_pkt_range"
]

# ---------- Helper: align any DataFrame to expected 73 features ----------
def align_to_73_features(df):
    """Keep only expected columns, add missing with 0, reorder."""
    df.columns = df.columns.str.strip()
    missing = [col for col in EXPECTED_FEATURES_73 if col not in df.columns]
    if missing:
        st.warning(f"Missing {len(missing)} feature(s). Filling with 0.")
        for col in missing:
            df[col] = 0.0
    df = df[[col for col in EXPECTED_FEATURES_73 if col in df.columns]]
    df = df[EXPECTED_FEATURES_73]
    return df

# ---------- Cache helpers ----------
@st.cache_resource
def load_surrogate():
    try:
        dt = joblib.load("models/dt_surrogate.pkl")
        le = joblib.load("models/cic_label_enc.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None
    return dt, le, EXPECTED_FEATURES_73

@st.cache_resource
def load_shap_explainer(_dt):
    return shap.TreeExplainer(_dt)

@st.cache_data
def load_csv(path):
    try:
        if os.path.exists(path): return pd.read_csv(path)
    except Exception: return None

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path) as f: return json.load(f)
        except Exception: return None
    return None

@st.cache_data
def load_image(path):
    try:
        if os.path.exists(path): return Image.open(path)
    except Exception: return None

# ---------- Load models ----------
dt, le, feature_names = load_surrogate()
models_loaded = (dt is not None)
if models_loaded:
    shap_explainer = load_shap_explainer(dt)
    n_features = len(feature_names)   # 73
else:
    st.warning("Models not loaded. Place joblib files in `models/`.")
    n_features = 73

# ---------- Sidebar (AAU themed) ----------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/3/33/Addis_Ababa_University_logo.png/150px-Addis_Ababa_University_logo.png", width=80)
    st.markdown(f"<h2 style='color:{AAU_YELLOW}; margin-top:-10px;'>LCGA IDS</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{AAU_WHITE};'><b>Intent-Aware Self-Healing Network</b></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<p style='color:{AAU_WHITE}; font-size:0.9rem;'><b>Researchers:</b><br>Getaye Fiseha<br>Mersen Getu<br>Chara Girma</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{AAU_WHITE}; font-size:0.9rem;'><b>Advisor:</b><br>Dr. Yaregal A.</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<p style='color:{AAU_WHITE}; font-size:0.8rem;'>© 2026 LCGA Framework<br>Addis Ababa University</p>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<div class='main-header'>🛡️ LCGA Self-Healing IDS</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Lightweight Hybrid Deep Learning for Real-Time Threat Detection &amp; Intent-Aware Remediation</div>", unsafe_allow_html=True)

# ---------- Tabs ----------
tabs = st.tabs([
    "📖 Overview", "⚙️ Methodology", "📊 Results",
    "🧠 Live Detection", "🩺 Simulator",
    "🔍 Explainability", "📋 Action Log", "📜 Conclusions"
])

# --- TAB 0: Overview ---
with tabs[0]:
    st.markdown("<div class='card card-blue'>", unsafe_allow_html=True)
    st.subheader("Problem & Motivation")
    st.markdown("""Modern networks face a rapidly growing threat landscape. Existing IDS either:
- Rely on **manual** investigation → high MTTR
- Operate as **black boxes** → no operator trust
- Lack **intent alignment** → actions don't match business goals

**Our solution**: An explainable, intent-aware deep-learning framework that autonomously detects, classifies, and remediates network attacks in real time.""")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card card-yellow'>", unsafe_allow_html=True)
    st.subheader("Key Contributions")
    st.markdown("""1. **LCGA** – CNN‑GRU‑Attention with 41k params, 99.67% accuracy
2. **DT Surrogate + SHAP** – Explanations 11,635× faster than LIME
3. **MAPE‑K Orchestrator** – 87% MTTR reduction, 87.6% ISR
4. **Fully Reproducible** – Open‑source code, data, experiments""")
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 1: Methodology ---
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card card-blue'>", unsafe_allow_html=True)
        st.subheader("LCGA Architecture")
        st.code("Input (73 features)\n  → Parallel Conv1D (32×3) + (64×5)\n  → Concatenate → GRU(64)\n  → Multi‑Head Self‑Attention (2 heads)\n  → GlobalAvgPool → Dense(64)\n  → Softmax (12 classes)", language="text")
        st.metric("Trainable Parameters", "41,260")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card card-yellow'>", unsafe_allow_html=True)
        st.subheader("MAPE‑K Self‑Healing Loop")
        st.markdown("""1. **Monitor** – capture network telemetry
2. **Analyze** – LCGA + DT surrogate + SHAP
3. **Plan** – map attack to violated intents, select best action
4. **Execute** – block IP, restart service, isolate subnet
5. **Knowledge** – verify intent restoration, update success rates""")
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: Results ---
with tabs[2]:
    st.markdown("<div class='card card-blue'>", unsafe_allow_html=True)
    st.subheader("Model Comparison")
    comp = load_csv("results/model_comparison.csv")
    if comp is not None:
        f1_cols = [c for c in comp.columns if "macro" in c.lower() and "f1" in c.lower()]
        st.dataframe(comp.style.highlight_max(subset=f1_cols[:1] if f1_cols else [], color=AAU_BLUE, axis=0) if f1_cols else comp, use_container_width=True)
    else: st.info("Upload `results/model_comparison.csv`")
    st.markdown("</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='card card-yellow'>", unsafe_allow_html=True)
        st.subheader("Training History")
        img = load_image("results/lcga_training_history.png")
        if img: st.image(img, use_column_width=True)
        else: st.info("Upload `results/lcga_training_history.png`")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<div class='card card-yellow'>", unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        img = load_image("results/lcga_confusion_matrix.png")
        if img: st.image(img, use_column_width=True)
        else: st.info("Upload `results/lcga_confusion_matrix.png`")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card card-blue'>", unsafe_allow_html=True)
    st.subheader("System Comparison")
    sys_df = load_csv("results/system_comparison.csv")
    if sys_df is not None:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="MTTR (s)", x=sys_df["System"], y=sys_df["MTTR_s"], marker_color=[AAU_BLUE, "#854F0B", AAU_YELLOW]))
        fig.add_trace(go.Bar(name="ISR (%)", x=sys_df["System"], y=sys_df["ISR_pct"], marker_color=[AAU_BLUE, "#854F0B", AAU_YELLOW], visible=False))
        fig.update_layout(barmode="group", updatemenus=[{"buttons":[{"label":"MTTR","method":"update","args":[{"visible":[True,False]}]},{"label":"ISR","method":"update","args":[{"visible":[False,True]}]}]}])
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Upload `results/system_comparison.csv`")
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3: Live Detection (AUTO‑ALIGNED to 73 features) ---
with tabs[3]:
    st.markdown("<div class='card card-blue'>", unsafe_allow_html=True)
    st.subheader("Live Network Flow Classification (DT Surrogate)")
    if not models_loaded:
        st.warning("Models not loaded.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
        with col2:
            use_random = st.button("Use random test sample")

        input_data = None
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                if 'Label' in df_raw.columns:
                    df_raw = df_raw.drop('Label', axis=1)
                df_aligned = align_to_73_features(df_raw)
                st.success(f"Aligned shape: {df_aligned.shape[1]} features (expected 73)")
                input_data = df_aligned.values.astype(np.float32)
                st.info(f"Loaded {len(input_data)} flow(s) | Features: {input_data.shape[1]}")
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

        if use_random:
            np.random.seed(int(time.time()))
            input_data = np.random.randn(5, n_features).astype(np.float32)

        if input_data is not None:
            preds = dt.predict(input_data)
            confidences = np.max(dt.predict_proba(input_data), axis=1)
            labels = le.inverse_transform(preds)

            st.subheader("Predictions")
            for i, (lbl, conf) in enumerate(zip(labels, confidences)):
                st.write(f"**Sample {i+1}:** {lbl}  ({conf:.1%} confidence)")

            # SHAP explanation for first sample (fixed API)
            st.subheader("SHAP Explanation (first sample)")
            shap_vals = shap_explainer.shap_values(input_data[0:1])
            cls_idx = preds[0]
            if isinstance(shap_vals, list):
                sv = shap_vals[cls_idx][0]
                expected = shap_explainer.expected_value[cls_idx]
            else:
                sv = shap_vals[0]
                expected = shap_explainer.expected_value

            fig = shap.plots.force(expected, sv, input_data[0],
                                   feature_names=feature_names,
                                   matplotlib=True, show=False)
            st.pyplot(fig)

            st.subheader("Decision Tree Rule Path (first 5 levels)")
            st.code(export_text(dt, feature_names=list(feature_names), max_depth=5)[:1200])

    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 4: Self-Healing Simulator ---
with tabs[4]:
    st.markdown("<div class='card card-yellow'>", unsafe_allow_html=True)
    st.subheader("MAPE‑K Self‑Healing Simulator")
    if st.button("Run Simulation Cycle"):
        intents = {"I1":{"name":"HTTP Latency","t_verify":90},"I2":{"name":"SSH Availability","t_verify":60},"I3":{"name":"Auth Fail Rate","t_verify":60},"I4":{"name":"Port Scan Rate","t_verify":30},"I5":{"name":"Bandwidth","t_verify":90}}
        attack_map = {"DoS Hulk":["I1","I5"],"DDoS":["I1","I5"],"PortScan":["I4"],"SSH-Patator":["I2","I3"],"Bot":["I1","I5"]}
        actions = {"DoS Hulk":"BLOCK_IP","DDoS":"ISOLATE_SUBNET","PortScan":"BLOCK_IP","SSH-Patator":"BLOCK_IP","Bot":"ISOLATE_SUBNET"}
        np.random.seed(42)
        attacks = np.random.choice(list(attack_map.keys()), 50)
        log = []
        for att in attacks:
            ttd = max(5, np.random.normal(80,15))
            heal = np.random.normal(2000,500)
            verify = intents[attack_map[att][0]]["t_verify"]*1000
            success = np.random.choice([True,False], p=[0.88,0.12])
            log.append({"attack":att,"action":actions.get(att,"ESCALATE"),"ttd_ms":ttd,"mttr_ms":ttd+heal+verify,"success":success})
        df = pd.DataFrame(log)
        st.dataframe(df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Time to Recovery", f"{df['mttr_ms'].mean()/1000:.1f} s")
        col2.metric("Intent Satisfaction Rate", f"{df['success'].mean()*100:.1f} %")
        col3.metric("Actions Executed", len(df))
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 5: Explainability Explorer ---
with tabs[5]:
    st.markdown("<div class='card card-blue'>", unsafe_allow_html=True)
    st.subheader("Interactive Explainability Explorer")
    if not models_loaded:
        st.warning("Models not loaded.")
    else:
        uploaded_single = st.file_uploader("Upload single flow CSV", type="csv", key="single")
        if uploaded_single is not None:
            df_single = pd.read_csv(uploaded_single)
            if 'Label' in df_single.columns:
                df_single = df_single.drop('Label', axis=1)
            df_single.columns = df_single.columns.str.strip()
            df_aligned = align_to_73_features(df_single)
            x = df_aligned.values.astype(np.float32).reshape(1, -1)
            pred = dt.predict(x)
            cls = pred[0]
            st.success(f"Prediction: **{le.inverse_transform([cls])[0]}**")
            shap_vals = shap_explainer.shap_values(x)
            if isinstance(shap_vals, list):
                fig = shap.plots.force(shap_explainer.expected_value[cls], shap_vals[cls][0], x[0],
                                       feature_names=feature_names, matplotlib=True, show=False)
            else:
                fig = shap.plots.force(shap_explainer.expected_value, shap_vals[0], x[0],
                                       feature_names=feature_names, matplotlib=True, show=False)
            st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 6: Action Log ---
with tabs[6]:
    st.markdown("<div class='card card-blue'>", unsafe_allow_html=True)
    st.subheader("MAPE‑K Action Log")
    log_json = load_json("results/action_log_full.json")
    if log_json: st.dataframe(pd.DataFrame(log_json).head(30), use_container_width=True)
    else: st.info("Upload `results/action_log_full.json`")
    st.subheader("Ablation Study")
    abl = load_csv("results/ablation_study.csv")
    if abl is not None: st.dataframe(abl, use_container_width=True)
    else: st.info("Upload `results/ablation_study.csv`")
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 7: Conclusions ---
with tabs[7]:
    st.markdown("<div class='card card-yellow'>", unsafe_allow_html=True)
    st.subheader("Conclusions & Future Work")
    st.markdown("""- **LCGA** achieves 99.67% accuracy with only 41k parameters.
- **MAPE‑K orchestrator** delivers 87% MTTR reduction and 87.6% ISR.
- **SHAP explanations** are 11,635× faster than LIME, enabling real‑time trust.

**Future:** zero‑day attacks, SDN hardware deployment, SIEM integration, federated learning.""")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("LCGA Framework v1.0 | AAU MSc Thesis 2026 | [GitHub](https://github.com/getaye21/lcga-self-healing-ids)")
