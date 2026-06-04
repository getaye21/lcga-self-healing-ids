"""
LCGA Self-Healing IDS — Real-Time Scientific Dashboard  v3.0
MSc ML Thesis | Addis Ababa University
Fixes:
  - class labels shown as integers → mapped to CICIDS2017 names
  - SHAP NotImplementedError   → use shap.Explainer (universal) with fallback
  - ambiguous array truth test → explicit len() / shape checks
  - restored all thesis tabs: Overview, Methodology, Results, Self-Healing,
    Live Detection, Explainability Explorer, Action Log, Conclusion
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, io
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LCGA Self-Healing IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: dark headers, coloured tabs, upload area ─────────────────────────────
st.markdown("""
<style>
h1,h2,h3,h4,h5 { color:#1a2a4a !important; }
[data-testid="stMetricLabel"] p { color:#1a2a4a !important; font-weight:700; }
button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span { color:#1a2a4a !important; font-weight:600; }
[data-testid="stFileUploaderDropzone"] {
    border:2px dashed #4472c4 !important;
    background:#f0f4ff !important;
}
.stDataFrame thead th {
    background:#1f3864 !important; color:white !important;
}
div[data-testid="stMetricValue"] { color:#1f3864 !important; font-weight:800; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# Canonical CICIDS2017 class names in sklearn label-encoder order
# (alphabetical, which is what LabelEncoder produces by default)
CICIDS_CLASSES = [
    "BENIGN",           # 0
    "Bot",              # 1
    "DDoS",             # 2
    "DoS GoldenEye",    # 3
    "DoS Hulk",         # 4
    "DoS Slowhttptest", # 5
    "DoS slowloris",    # 6
    "FTP-Patator",      # 7
    "Heartbleed",       # 8
    "Infiltration",     # 9
    "PortScan",         # 10
    "SSH-Patator",      # 11
]

def idx_to_label(idx):
    """Convert integer class index → human-readable label."""
    try:
        i = int(idx)
        if 0 <= i < len(CICIDS_CLASSES):
            return CICIDS_CLASSES[i]
    except (ValueError, TypeError):
        pass
    return str(idx)   # already a string label

ACTION_MAP = {
    "DoS Hulk":        "BLOCK_IP",
    "DoS GoldenEye":   "BLOCK_IP",
    "DoS Slowhttptest":"RATE_LIMIT",
    "DoS slowloris":   "RATE_LIMIT",
    "DDoS":            "ISOLATE_SUBNET",
    "PortScan":        "BLOCK_IP",
    "Bot":             "ISOLATE_SUBNET",
    "SSH-Patator":     "BLOCK_IP",
    "FTP-Patator":     "BLOCK_IP",
    "Heartbleed":      "RESTART_SERVICE",
    "Infiltration":    "ISOLATE_SUBNET",
    "BENIGN":          "NONE",
}

INTENT_VIOLATIONS = {
    "DoS Hulk":        ["I1 - HTTP Latency", "I5 - Bandwidth"],
    "DoS GoldenEye":   ["I1 - HTTP Latency", "I5 - Bandwidth"],
    "DoS Slowhttptest":["I1 - HTTP Latency"],
    "DoS slowloris":   ["I1 - HTTP Latency"],
    "DDoS":            ["I1 - HTTP Latency", "I5 - Bandwidth"],
    "PortScan":        ["I4 - Port Scan Rate"],
    "Bot":             ["I3 - Auth Failure Rate"],
    "SSH-Patator":     ["I2 - SSH Availability", "I3 - Auth Failure Rate"],
    "FTP-Patator":     ["I3 - Auth Failure Rate"],
    "Heartbleed":      ["I2 - SSH Availability"],
    "Infiltration":    ["I1 - HTTP Latency", "I2 - SSH Availability"],
    "BENIGN":          [],
}

SHAP_PROFILES = {
    "DoS Hulk":        {"Flow Duration":0.48,"Bwd Packet Length Std":0.41,"Fwd Packet Length Max":0.38,"Total Fwd Packets":0.32,"Packet Length Mean":-0.12},
    "DDoS":            {"Total Length of Fwd Packets":0.52,"Destination Port":0.44,"Total Fwd Packets":0.39,"Flow Duration":-0.28,"Bwd Packets/s":0.22},
    "PortScan":        {"Destination Port":0.61,"Flow Duration":0.45,"Total Fwd Packets":0.38,"Fwd IAT Total":-0.21,"Init_Win_bytes_forward":0.15},
    "SSH-Patator":     {"Destination Port":0.55,"Flow Duration":0.47,"Fwd Packet Length Std":0.31,"Total Fwd Packets":0.28,"Bwd Packet Length Mean":-0.18},
    "FTP-Patator":     {"Destination Port":0.58,"Flow Duration":0.44,"Total Fwd Packets":0.33,"Fwd Packet Length Mean":0.25,"Flow Bytes/s":-0.16},
    "Bot":             {"Flow Duration":0.44,"Packet Length Std":0.37,"Average Packet Size":0.29,"Fwd IAT Mean":0.21,"Idle Mean":-0.14},
    "Heartbleed":      {"Total Fwd Packets":0.53,"Fwd Packet Length Max":0.48,"Destination Port":0.42,"Flow Duration":-0.31,"Bwd Packet Length Max":0.19},
    "Infiltration":    {"Flow Duration":0.46,"Fwd Packet Length Mean":0.38,"Total Length of Fwd Packets":0.34,"Idle Mean":0.22,"Packet Length Variance":-0.15},
    "DoS GoldenEye":   {"Flow Duration":0.43,"Bwd IAT Total":0.39,"Fwd Packets/s":0.34,"Packet Length Mean":-0.25,"Total Fwd Packets":0.21},
    "DoS Slowhttptest":{"Flow Duration":0.57,"Fwd IAT Total":0.45,"Total Fwd Packets":0.28,"Fwd Packet Length Mean":-0.19,"Active Mean":0.14},
    "DoS slowloris":   {"Flow Duration":0.59,"Fwd IAT Mean":0.44,"Total Fwd Packets":0.27,"Fwd Packet Length Std":-0.17,"Active Mean":0.13},
}

# ── Load DT surrogate ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import joblib
        m  = joblib.load("models/dt_surrogate.pkl")  if os.path.exists("models/dt_surrogate.pkl")  else None
        sc = joblib.load("models/scaler.pkl")         if os.path.exists("models/scaler.pkl")         else None
        fn = joblib.load("models/feature_names.pkl")  if os.path.exists("models/feature_names.pkl")  else None
        return m, sc, fn, m is not None
    except Exception:
        return None, None, None, False

dt_model, scaler, saved_features, model_loaded = load_model()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["timestamp","attack","confidence","action","intents","restored"])

# ─────────────────────────────────────────────────────────────────────────────
# SHAP helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extract_shap(model, X_row2d):
    """
    Safely compute SHAP values for a single row using the most compatible
    API available. Returns (sv_1d, base_val, class_name_str).

    Handles:
      - shap.TreeExplainer  (preferred for sklearn DT)
      - shap.Explainer      (universal fallback)
      - NotImplementedError if model type unsupported
      - ambiguous-array errors from bad truth tests
    """
    import shap

    sv_1d, base_val, class_idx = None, 0.0, 0

    # ── Try TreeExplainer first ───────────────────────────────────────────────
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row2d)  # list or ndarray

        ev = explainer.expected_value

        if isinstance(shap_values, list) and len(shap_values) > 0:
            # Multi-class: list of [n_samples × n_features]
            # Pick class with largest mean |SHAP|
            means = [np.mean(np.abs(sv[0])) for sv in shap_values]
            class_idx = int(np.argmax(means))
            sv_1d     = shap_values[class_idx][0]
            base_val  = float(ev[class_idx]) if hasattr(ev, "__len__") else float(ev)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (n_samples, n_features, n_classes)
            class_idx = int(np.argmax(np.mean(np.abs(shap_values[0]), axis=0)))
            sv_1d     = shap_values[0, :, class_idx]
            base_val  = float(ev[class_idx]) if hasattr(ev, "__len__") else float(ev)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            sv_1d    = shap_values[0]
            base_val = float(ev[1]) if hasattr(ev, "__len__") and len(ev) > 1 else float(ev)
        else:
            raise ValueError("Unexpected shap_values shape")

        return sv_1d, base_val, class_idx

    except (NotImplementedError, Exception) as e1:
        pass  # fall through to universal explainer

    # ── Universal Explainer fallback ─────────────────────────────────────────
    try:
        explainer = shap.Explainer(model, algorithm="auto")
        sv_obj    = explainer(X_row2d)

        vals = sv_obj.values  # shape: (1, n_features) or (1, n_features, n_classes)
        if vals.ndim == 3:
            class_idx = int(np.argmax(np.mean(np.abs(vals[0]), axis=0)))
            sv_1d     = vals[0, :, class_idx]
            base_val  = float(sv_obj.base_values[0, class_idx]) \
                        if sv_obj.base_values.ndim > 1 else float(sv_obj.base_values[0])
        else:
            sv_1d    = vals[0]
            base_val = float(sv_obj.base_values[0]) \
                       if not hasattr(sv_obj.base_values[0], "__len__") \
                       else float(sv_obj.base_values[0][1])

        return sv_1d, base_val, class_idx

    except Exception as e2:
        return None, 0.0, 0


def shap_bar_chart(sv_1d, feature_names, class_name, top_n=15):
    """Horizontal bar chart of top-N SHAP contributions."""
    n = min(top_n, len(sv_1d))
    idx    = np.argsort(np.abs(sv_1d))[-n:]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sv_1d[idx]]
    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.32)))
    ax.barh([feature_names[i] for i in idx], sv_1d[idx], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value  (red = increases risk, blue = decreases)")
    ax.set_title(f"SHAP Feature Contributions — Predicted: {class_name}",
                 fontsize=11, color="#1a2a4a", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


def shap_force_fig(sv_1d, base_val, feature_names, X_row_1d, class_name):
    """
    shap.force_plot with the corrected >= 0.20 signature.
    Falls back to bar chart if matplotlib rendering fails.
    """
    import shap
    try:
        fig = shap.force_plot(
            float(base_val),
            sv_1d,
            X_row_1d,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        return fig, "force"
    except Exception:
        return shap_bar_chart(sv_1d, feature_names, class_name), "bar"


# ── Simulation helper ─────────────────────────────────────────────────────────
def simulate_telemetry():
    weights = [0.55,0.05,0.07,0.04,0.08,0.03,0.03,0.04,0.02,0.02,0.04,0.03]
    attack  = np.random.choice(CICIDS_CLASSES, p=weights)
    anomaly = attack != "BENIGN"
    score   = round(np.random.uniform(0.6,0.99) if anomaly else np.random.uniform(0.1,0.45), 4)
    return anomaly, attack, score


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/AAU_Logo.png/200px-AAU_Logo.png",
                 use_column_width=True) if False else None   # logo placeholder
st.sidebar.title("🛡️ LCGA IDS")
st.sidebar.markdown("**Intent-Aware Self-Healing Network**")
st.sidebar.markdown("---")
st.sidebar.info(
    "**MSc ML Thesis** — Addis Ababa University\n\n"
    "Getaye Fiseha · Mersen Getu · Chara Girma\n\n"
    "Advisor: Dr. Yaregal A.\n\n"
    "June 2026"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Key Results**")
st.sidebar.metric("Accuracy",       "99.67%")
st.sidebar.metric("MTTR Reduction", "87%")
st.sidebar.metric("ISR",            "87.6%")
st.sidebar.metric("SHAP Speedup",   "11,635×")
st.sidebar.markdown("---")
st.sidebar.markdown("[📂 GitHub](https://github.com/getaye21/lcga-self-healing-ids)")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🛡️ LCGA Self-Healing IDS — Scientific Dashboard")
st.markdown(
    "**Lightweight CNN-GRU-Attention · SHAP Explainability · MAPE-K Closed-Loop Remediation**  \n"
    "*MSc ML Thesis, Department of Computer Science, Addis Ababa University, June 2026*"
)

# ── 8 tabs matching the original structure ────────────────────────────────────
tabs = st.tabs([
    "📖 Overview",
    "⚙️ Methodology",
    "📊 Results",
    "🔧 Self-Healing",
    "🔬 Live Detection",
    "🧠 Explainability",
    "📋 Action Log",
    "✅ Conclusion",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("📖 Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
### Problem
Modern enterprise networks face an ever-growing volume and sophistication of cyber threats.
Existing IDS either sacrifice latency for accuracy, lack automated remediation, or operate
as black boxes that undermine operator trust.

### Proposed Framework
This thesis presents an **Optimised Hybrid Deep Learning Framework** integrating:
- 🧠 **LCGA model** (41,260 params) — CNN-GRU-Attention for 12-class classification
- 🌳 **Decision Tree Surrogate** — distilled from LCGA for real-time SHAP explanations
- 🔄 **MAPE-K Orchestrator** — maps attacks to violated intents and executes healing
- 📊 **Live Dashboard** — SHAP force plots, intent status, self-healing simulator

### Datasets
| Dataset | Samples | Classes | Task |
|---------|---------|---------|------|
| CICIDS2017 | ~2.52M | 12 | Multi-class attack classification |
| NSL-KDD | 148,517 | 2 | Binary anomaly detection |
        """)
    with col2:
        st.markdown("### Key Results")
        st.metric("Accuracy (CICIDS2017)", "99.67%", delta="+0.14% vs RF")
        st.metric("Inference Latency",     "1.85 ms/flow", delta="-3.3× vs GRU")
        st.metric("SHAP Speed",            "0.05 ms",      delta="11,635× faster than LIME")
        st.metric("ISR",                   "87.6%",        delta="+23.4pp vs rule-based")
        st.metric("MTTR Reduction",        "87%",          delta="vs open-loop")
        st.metric("Parameters",            "41,260",       delta="5-10× smaller than SOTA")

    st.markdown("---")
    st.markdown("""
### Framework Architecture
```
Network Traffic
      │
      ▼
┌─────────────────────────────────────────┐
│  Data Preprocessing Layer               │
│  (Cleaning · RFE · RobustScaler · SMOTE)│
└─────────────────────┬───────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐    ┌─────────────────────┐
│  ANN-GRU Stack  │    │    LCGA Model       │
│  (NSL-KDD)      │    │  CNN→GRU→MH-Attn   │
│  Binary detect  │    │  12-class classify  │
└─────────────────┘    └─────────┬───────────┘
                                  │ soft labels
                                  ▼
                       ┌─────────────────────┐
                       │  DT Surrogate       │
                       │  (knowledge distil) │
                       │  + SHAP TreeExp.    │
                       └─────────┬───────────┘
                                  │ class + explanation
                                  ▼
                       ┌─────────────────────┐
                       │  MAPE-K Orchestrator│
                       │  Monitor→Analyse    │
                       │  →Plan→Execute      │
                       │  →Verify→KB update  │
                       └─────────────────────┘
```
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Methodology
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("⚙️ Methodology")

    with st.expander("🏗️ LCGA Architecture (41,260 parameters)", expanded=True):
        st.markdown("""
| Block | Layer | Config | Output Shape |
|-------|-------|--------|-------------|
| 1 | Conv1D Branch A | 32 filters, kernel 3, ReLU, BN, Dropout(0.2), MaxPool | (36, 32) |
| 1 | Conv1D Branch B | 64 filters, kernel 5, ReLU, BN, Dropout(0.2), MaxPool | (36, 64) |
| 1 | Concatenate | — | (36, 96) |
| 2 | GRU | 64 units, return_sequences=True, Dropout(0.2) | (36, 64) |
| 3 | MultiHeadAttention | 2 heads, key_dim=16 + residual + LayerNorm | (36, 32) |
| 4 | GlobalAvgPool1D | — | (32,) |
| 4 | Dense + Dropout | 64 units, ReLU, Dropout(0.3) | (64,) |
| 4 | Output | Softmax(12) | (12,) |

**Training:** Adam (lr=0.001) · Sparse categorical CE · EarlyStopping(patience=10) · ReduceLROnPlateau · max 60 epochs
        """)

    with st.expander("🌳 DT Surrogate + SHAP", expanded=False):
        st.markdown("""
**Knowledge distillation:** A `DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=5)` is trained on the LCGA model's soft probability vectors on the test set.

**Fidelity:** 99.64% agreement with LCGA predictions on the test set.

**SHAP:** `shap.TreeExplainer` applied to the surrogate gives explanations in 0.05 ms/sample — **11,635× faster than LIME** (812 ms/sample), with deterministic outputs (LIME top-3 consistency: 30%).
        """)

    with st.expander("🔄 MAPE-K Orchestrator + Intent Formalisation", expanded=False):
        st.markdown("""
**5 formalised network intents:**

| ID | Metric | Threshold | Cooldown |
|----|--------|-----------|---------|
| I1 | HTTP Latency | < 200 ms | 90 s |
| I2 | SSH Availability | = True | 60 s |
| I3 | Auth Failure Rate | < 10/min | 60 s |
| I4 | Port Scan Rate | < 5/min | 30 s |
| I5 | Bandwidth | < 100 Mbps | 90 s |

**MAPE-K loop:**
1. **Monitor** — telemetry sampled every 5 s
2. **Analyse** — LCGA classifies; DT surrogate generates SHAP; violated intents identified
3. **Plan** — action selected by historical success rate for that attack–intent pair
4. **Execute** — action applied and logged
5. **Verify** — after adaptive cooldown, intent re-evaluated; 3 consecutive failures → escalation
        """)

    with st.expander("📦 Data Preprocessing", expanded=False):
        st.markdown("""
**CICIDS2017:** 8 CSV files → clean (drop NaN/Inf) → Pearson r > 0.85 correlation filter → 73 features retained → stratified 60/20/20 split → RobustScaler (fit on train) → SMOTE on minority classes (<5k samples).

**NSL-KDD:** 41 features → binarise labels → RFE (Random Forest, top 20) → 60/20/20 split → StandardScaler → SMOTE → sliding window (length 10) for BiLSTM input.
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Results
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("📊 Experimental Results")

    st.subheader("Model Comparison — CICIDS2017 Test Set")
    model_df = pd.DataFrame({
        "Model":        ["Random Forest", "CNN Baseline", "GRU Baseline", "LCGA (Ours)"],
        "Accuracy":     ["99.53%", "97.65%", "98.30%", "99.67% ✓"],
        "Macro F1":     ["0.9527", "0.9420", "0.9490", "0.8170 *"],
        "Weighted F1":  ["0.9953", "0.9760", "0.9825", "0.9967 ✓"],
        "MCC":          ["0.9945", "0.9745", "0.9820", "0.9945 ✓"],
        "Parameters":   ["100 trees", "5,196", "~20,000", "41,260"],
        "Inference ms": ["0.10", "5.17", "4.80", "1.85 ✓"],
    })
    st.dataframe(model_df, use_container_width=True)
    st.caption("✓ = best or matching best.  * Macro F1 pulled down by Heartbleed (11 samples) and Infiltration (36 samples).")

    st.markdown("---")
    st.subheader("XAI Comparison — SHAP vs LIME")
    xai_df = pd.DataFrame({
        "Metric":               ["Compute time (ms/sample)", "Speedup", "Top-3 consistency", "Surrogate fidelity"],
        "SHAP (DT Surrogate)":  ["0.05", "11,635×", "Deterministic", "99.64%"],
        "LIME":                 ["812", "—", "30.0%", "N/A"],
    })
    st.dataframe(xai_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Self-Healing System Comparison")
    heal_df = pd.DataFrame({
        "System":         ["Open-loop (no healing)", "Rule-based (fixed 60s)", "LCGA + MAPE-K (Ours)"],
        "MTTR (s)":       [598.5, 65.1, 78.4],
        "ISR (%)":        [0.0, 64.2, 87.6],
        "MTTR Reduction": ["—", "89.1%", "86.9%"],
    })
    st.dataframe(heal_df, use_container_width=True)

    # Bar charts
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        systems = ["Open-loop", "Rule-based", "LCGA+MAPE-K"]
        mttr    = [598.5, 65.1, 78.4]
        colors  = ["#d62728", "#ff7f0e", "#2ca02c"]
        ax.bar(systems, mttr, color=colors)
        ax.set_ylabel("MTTR (seconds)")
        ax.set_title("Mean Time to Recovery", color="#1a2a4a", fontweight="bold")
        for i, v in enumerate(mttr):
            ax.text(i, v + 5, str(v), ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        isr = [0.0, 64.2, 87.6]
        ax.bar(systems, isr, color=colors)
        ax.set_ylabel("ISR (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Intent Satisfaction Rate", color="#1a2a4a", fontweight="bold")
        for i, v in enumerate(isr):
            ax.text(i, v + 1, f"{v}%", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close()

    st.markdown("---")
    st.subheader("Ablation Study")
    abl_df = pd.DataFrame({
        "Config":       ["A: Full LCGA+MAPE-K", "B: No KB Feedback (fixed 15s)", "C: Open-loop", "D: No DT Surrogate"],
        "Accuracy":     ["99.67%"]*4,
        "Macro F1":     ["0.8170"]*4,
        "ISR (%)":      ["87.6", "72.4", "0.0", "87.6"],
        "MTTR (s)":     ["78.4", "17.1", "598.5", "78.4*"],
    })
    st.dataframe(abl_df, use_container_width=True)
    st.caption("* D: healing unchanged; only explanation latency increases 3–10×.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Self-Healing Simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("🔧 MAPE-K Self-Healing Simulator")
    st.markdown("Run detection cycles to trigger the MAPE-K loop and observe intent restoration.")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("▶ Run Detection Cycle", type="primary", key="run_heal"):
            anomaly, attack, score = simulate_telemetry()
            ts = pd.Timestamp.now().isoformat()
            if anomaly:
                action   = ACTION_MAP.get(attack, "ESCALATE")
                intents  = ", ".join(INTENT_VIOLATIONS.get(attack, ["—"]))
                restored = bool(np.random.choice([True, False], p=[0.876, 0.124]))
                new_row  = pd.DataFrame([{
                    "timestamp": ts, "attack": attack, "confidence": score,
                    "action": action, "intents": intents, "restored": restored,
                }])
                st.session_state.history = pd.concat(
                    [st.session_state.history, new_row], ignore_index=True)
                st.error(f"🚨 **{attack}** | Action: **{action}**")
            else:
                st.success(f"✅ Normal traffic (score={score:.3f})")

        if not st.session_state.history.empty:
            isr = st.session_state.history["restored"].mean() * 100
            st.metric("Session ISR", f"{isr:.1f}%")
            st.metric("Attacks detected",
                      int((st.session_state.history["attack"] != "BENIGN").sum()))

        if st.button("🗑️ Clear Log", key="clear_heal"):
            st.session_state.history = pd.DataFrame(
                columns=["timestamp","attack","confidence","action","intents","restored"])
            st.rerun()

    with col2:
        st.subheader("Network Intent Status")
        recent = st.session_state.history["attack"].tolist()[-3:] \
                 if not st.session_state.history.empty else []

        def istatus(key):
            for a in recent:
                if any(key in v for v in INTENT_VIOLATIONS.get(a, [])):
                    return "🔴 Violated"
            return "🟢 Satisfied"

        intent_data = [
            ("I1","HTTP Latency < 200 ms",   istatus("I1"), "90 s"),
            ("I2","SSH Availability = True",  istatus("I2"), "60 s"),
            ("I3","Auth Failure Rate < 10/m", istatus("I3"), "60 s"),
            ("I4","Port Scan Rate < 5/m",     istatus("I4"), "30 s"),
            ("I5","Bandwidth < 100 Mbps",     istatus("I5"), "90 s"),
        ]
        idf = pd.DataFrame(intent_data, columns=["ID","Intent","Status","Cooldown"])
        st.dataframe(idf, use_container_width=True)

        violated = [r for r in intent_data if "Violated" in r[2]]
        if violated:
            st.warning(f"⚠️ {len(violated)} intent(s) violated: {', '.join(r[0] for r in violated)}")
        else:
            st.success("✅ All intents satisfied")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Live Detection (CSV upload)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🔬 Live Network Flow Classification (DT Surrogate)")
    st.markdown(
        "Upload a CSV of network flows. Columns are auto-aligned to the 73 CICIDS2017 "
        "features. Each row is classified and the first sample receives a SHAP explanation."
    )

    uploaded = st.file_uploader(
        "Upload CSV (any size)",
        type=["csv"],
        help="No per-file size limit. For HuggingFace Spaces set maxUploadSize in .streamlit/config.toml"
    )

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        df_raw.columns = [c.strip() for c in df_raw.columns]

        # Drop label column if present
        for cand in ["Label","label","Class","class"]:
            if cand in df_raw.columns:
                df_raw = df_raw.drop(columns=[cand])
                break

        df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_raw.fillna(0, inplace=True)

        # Align to expected 73 features
        expected = saved_features if (saved_features is not None and len(saved_features)==73) \
                   else list(df_raw.columns)[:73]

        if df_raw.shape[1] != 73:
            for col in expected:
                if col not in df_raw.columns:
                    df_raw[col] = 0.0
            df_raw = df_raw[expected]
            st.info(f"Aligned shape: {df_raw.shape[1]} features (expected 73)")
        else:
            expected = list(df_raw.columns)
            st.success(f"Shape matches: {df_raw.shape[1]} features ✓")

        X = df_raw.values.astype(np.float32)
        feature_names = list(expected)
        st.success(f"Loaded **{len(X)} flow(s)**  |  Features: {X.shape[1]}")

        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception:
                pass

        # ── Predict ──────────────────────────────────────────────────────────
        st.markdown("#### 🎯 Predictions")
        if model_loaded and dt_model is not None:
            raw_preds = dt_model.predict(X)
            probas    = dt_model.predict_proba(X)
            # ── FIX: map integer indices to label strings ─────────────────────
            preds = [idx_to_label(p) for p in raw_preds]
        else:
            st.warning("⚠️ No trained model found at `models/dt_surrogate.pkl`. Showing mock predictions.")
            preds  = [np.random.choice(CICIDS_CLASSES) for _ in range(len(X))]
            probas = np.zeros((len(X), len(CICIDS_CLASSES)))
            for i, p in enumerate(preds):
                probas[i, CICIDS_CLASSES.index(p)] = np.random.uniform(0.85, 1.0)

        pred_df = pd.DataFrame({
            "Sample":          [f"Sample {i+1}" for i in range(len(X))],
            "Predicted Class": preds,
            "Confidence":      [f"{probas[i].max():.1%}" for i in range(len(X))],
            "Action":          [ACTION_MAP.get(p, "ESCALATE") for p in preds],
        })
        st.dataframe(pred_df, use_container_width=True)

        for i in range(min(len(preds), 5)):
            conf = probas[i].max()
            icon = "🚨" if preds[i] != "BENIGN" else "✅"
            st.write(f"{icon} **Sample {i+1}:** {preds[i]} ({conf:.1%} confidence)")

        # ── SHAP Explanation ─────────────────────────────────────────────────
        st.markdown("#### 🧠 SHAP Explanation (first sample)")

        try:
            import shap

            if model_loaded and dt_model is not None:
                X_row2d = X[:1]   # shape (1, n_features)
                sv_1d, base_val, class_idx = _extract_shap(dt_model, X_row2d)

                if sv_1d is not None:
                    pred_label = idx_to_label(class_idx)

                    # Force plot
                    force_fig, kind = shap_force_fig(
                        sv_1d, base_val, feature_names, X_row2d[0], pred_label)
                    st.pyplot(force_fig, clear_figure=True)
                    plt.close("all")

                    if kind == "bar":
                        st.caption("ℹ️ Force plot unavailable for this model — showing bar chart.")

                    # Always also show bar chart for clarity
                    bar_fig = shap_bar_chart(sv_1d, feature_names, pred_label)
                    st.pyplot(bar_fig, clear_figure=True)
                    plt.close("all")

                    # Top-5 table
                    top_idx = np.argsort(np.abs(sv_1d))[-5:][::-1]
                    top_df = pd.DataFrame({
                        "Feature":    [feature_names[i] for i in top_idx],
                        "SHAP Value": [f"{sv_1d[i]:+.4f}" for i in top_idx],
                        "Direction":  ["↑ Increases risk" if sv_1d[i]>0 else "↓ Reduces risk"
                                       for i in top_idx],
                    })
                    st.markdown("**Top 5 most influential features:**")
                    st.dataframe(top_df, use_container_width=True)
                else:
                    st.warning("Could not compute SHAP values for this model type.")
                    # Precomputed fallback from profile dict
                    profile = SHAP_PROFILES.get(preds[0], {})
                    if profile:
                        sv_mock = np.array([profile.get(f, 0.0) for f in feature_names])
                        st.pyplot(shap_bar_chart(sv_mock, feature_names, preds[0]),
                                  clear_figure=True)
                        plt.close("all")
            else:
                # Mock bar chart
                sv_mock = np.random.randn(len(feature_names)) * 0.3
                st.pyplot(shap_bar_chart(sv_mock, feature_names, preds[0]),
                          clear_figure=True)
                plt.close("all")
                st.caption("Mock SHAP — load real model for true explanations.")

        except ImportError:
            st.error("SHAP not installed. Add `shap` to requirements.txt.")
        except Exception as e:
            st.error(f"Unexpected error in SHAP: {e}")
            import traceback
            st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explainability Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("🧠 Explainability Explorer")
    st.markdown("Select an attack class to explore which features drive the DT surrogate's decision.")

    sel = st.selectbox("Select attack class:", CICIDS_CLASSES[1:], key="exp_sel")
    profile = SHAP_PROFILES.get(sel, {"Flow Duration":0.4,"Destination Port":0.35,
                                       "Total Fwd Packets":0.28,"Packet Length Mean":0.2,
                                       "Bwd Packet Length Std":0.15})
    feats  = list(profile.keys())
    vals   = list(profile.values())
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(feats, vals, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Feature Impact — {sel}", fontsize=12, color="#1a2a4a", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Violated intents:** {', '.join(INTENT_VIOLATIONS.get(sel, ['None']))}")
        st.markdown(f"**Recommended action:** `{ACTION_MAP.get(sel, 'ESCALATE')}`")
    with col2:
        st.markdown("**Top feature explanation:**")
        top_feat = max(profile, key=lambda k: abs(profile[k]))
        st.info(f"**{top_feat}** (SHAP={profile[top_feat]:+.2f}) is the strongest driver "
                f"for detecting `{sel}` traffic.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Action Log
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("📋 MAPE-K Action Log")

    if st.session_state.history.empty:
        st.info("No actions logged yet. Go to the Self-Healing tab and run detection cycles.")
    else:
        disp = st.session_state.history.tail(30).sort_values("timestamp", ascending=False).copy()
        disp["timestamp"] = disp["timestamp"].str[:19].str.replace("T", " ")
        disp["restored"]  = disp["restored"].map({True: "✅ Yes", False: "❌ No"})
        st.dataframe(disp, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        isr   = st.session_state.history["restored"].mean() * 100
        total = len(st.session_state.history)
        n_att = int((st.session_state.history["attack"] != "BENIGN").sum())
        n_ok  = int(st.session_state.history["restored"].sum())
        c1.metric("Cycles Run",      total)
        c2.metric("Attacks Detected",n_att)
        c3.metric("Intents Restored",n_ok)
        c4.metric("ISR",             f"{isr:.1f}%")

        # Action distribution
        if n_att > 0:
            act_counts = (
                st.session_state.history[st.session_state.history["attack"] != "BENIGN"]["action"]
                .value_counts()
            )
            fig, ax = plt.subplots(figsize=(6, 3))
            act_counts.plot.bar(ax=ax, color="#4472c4")
            ax.set_ylabel("Count")
            ax.set_title("Healing Actions Executed", color="#1a2a4a", fontweight="bold")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close()

        if st.button("🗑️ Clear Log", key="clear_log"):
            st.session_state.history = pd.DataFrame(
                columns=["timestamp","attack","confidence","action","intents","restored"])
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Conclusion
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("✅ Conclusion and Future Work")

    st.markdown("""
### Summary of Contributions

This thesis presented the **LCGA Framework** — the first integrated system that simultaneously
addresses real-time inference, knowledge-distilled explainability, and intent-aware closed-loop
self-healing for network security.

| Contribution | Detail |
|---|---|
| Lightweight LCGA model | 41,260 params, 99.67% accuracy, 1.85 ms CPU inference |
| DT Surrogate + SHAP | 99.64% fidelity, 11,635× faster than LIME, deterministic |
| MAPE-K + IBN integration | First closed-loop framework linking DL classification to intent violation and adaptive healing |
| ISR improvement | 87.6% vs 0% (open-loop), 64.2% (rule-based) |
| MTTR reduction | 78.4 s vs 598.5 s baseline (87% reduction) |
| Open-source | Full code, models, datasets, live dashboard |

### Limitations
- Evaluated on public benchmarks (CICIDS2017, NSL-KDD), not live production traffic
- Healing actions are simulated, not deployed on real SDN hardware
- Macro F1 on extremely rare classes (Heartbleed: 11 samples) is low due to class imbalance
- No zero-day detection capability in the current version

### Future Work
1. **Zero-day detection** — Online learning / anomaly head for unseen attack types
2. **Real SDN deployment** — ONOS + Mininet end-to-end latency validation
3. **RL-based healing** — Replace success-rate planning with deep Q-learning
4. **User trust study** — Controlled SOC analyst experiment using Trust in Automation Scale (TAS)
5. **Federated learning** — Multi-site training without sharing raw traffic data
6. **Rare-class improvement** — Few-shot learning or GAN-based data augmentation

---
*LCGA Framework v3.0 · Addis Ababa University · MSc Computer Science · June 2026*
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Authors:**
- Getaye Fiseha (GSE/6132/18)
- Mersen Getu
- Chara Girma

**Advisor:** Dr. Yaregal A.
        """)
    with col2:
        st.markdown("""
**Links:**
- 📂 [GitHub Repository](https://github.com/getaye21/lcga-self-healing-ids)
- 📄 [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- 📄 [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "LCGA Self-Healing IDS v3.0 · MSc ML Thesis, Addis Ababa University, June 2026 · "
    "[GitHub](https://github.com/getaye21/lcga-self-healing-ids)"
)
