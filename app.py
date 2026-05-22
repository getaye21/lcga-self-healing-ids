"""
LCGA Self-Healing IDS — Scientific Dashboard
MSc Thesis | Addis Ababa University
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os, json

st.set_page_config(
    page_title="LCGA IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- helpers ----------
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

# ---------- sidebar ----------
st.sidebar.title("🛡️ LCGA Self‑Healing IDS")
st.sidebar.markdown("**Intent‑Aware Autonomous Network Defense**")
st.sidebar.markdown("---")
st.sidebar.info(
    "MSc Thesis — Addis Ababa University\n\n"
    "Getaye Fiseha, Mersen Getu, Chara Girma\n\n"
    "© 2026 LCGA Framework"
)

# ---------- main ----------
st.title("🛡️ LCGA: Lightweight Self‑Healing Intrusion Detection System")
st.markdown(
    "### A Hybrid Deep Learning Framework for Real‑Time Threat Detection "
    "and Intent‑Aware Automated Remediation"
)

tabs = st.tabs([
    "📖 Overview", "⚙️ Methodology", "📊 Key Results",
    "🧠 Explainability", "🩺 Self‑Healing Demo", "📋 Conclusions"
])

# ================== TAB 1: Overview ==================
with tabs[0]:
    st.header("Problem & Motivation")
    st.markdown(
        "Modern enterprise networks face a rapidly evolving threat landscape. "
        "Existing intrusion detection systems (IDS) either:\n"
        "- Rely on **manual** investigation and remediation (high MTTR),\n"
        "- Operate as **black boxes** without explaining their decisions,\n"
        "- Lack **intent alignment** — they cannot map attacks to business‑level goals.\n\n"
        "**Our solution**: A lightweight, explainable deep‑learning framework that "
        "autonomously detects attacks, classifies them, maps them to violated network "
        "intents, and executes appropriate self‑healing actions — all with real‑time "
        "explanations for operators."
    )
    st.header("Key Contributions")
    st.markdown(
        "1. **LCGA architecture** — A hybrid CNN‑GRU‑Attention model with only **41k parameters**, "
        "achieving **99.67% accuracy** and **87.6% intent satisfaction rate**.\n"
        "2. **Explainable AI via DT surrogate** — SHAP TreeExplainer on a distilled Decision Tree "
        "delivers explanations **11,635× faster** than LIME.\n"
        "3. **MAPE‑K closed‑loop orchestrator** — Adaptive verification windows reduce MTTR by "
        "**87%** compared to open‑loop systems.\n"
        "4. **Fully reproducible** — All code, data, and experiments are open‑source."
    )

# ================== TAB 2: Methodology ==================
with tabs[1]:
    st.header("LCGA Architecture")
    st.markdown(
        "The LCGA model processes raw network flows through:\n"
        "```\n"
        "Input (73 features)\n"
        "  → Parallel Conv1D (32×3) + Conv1D (64×5) → Concatenate\n"
        "  → GRU (64 units, return sequences)\n"
        "  → Multi‑Head Self‑Attention (2 heads, residual connection)\n"
        "  → Global Average Pooling → Dense(64) → Softmax(12)\n"
        "```\n"
        "**Total trainable parameters: 41,260** — suitable for real‑time CPU inference."
    )
    st.header("Self‑Healing Loop (MAPE‑K)")
    st.markdown(
        "The orchestrator runs a continuous closed‑loop:\n"
        "1. **Monitor** – capture network telemetry\n"
        "2. **Analyze** – LCGA detects anomaly → DT surrogate classifies + SHAP explains\n"
        "3. **Plan** – map attack to violated intents, select best healing action from history\n"
        "4. **Execute** – block IP, restart service, isolate subnet, throttle bandwidth\n"
        "5. **Knowledge** – verify intent restoration, update action success rates"
    )

# ================== TAB 3: Key Results ==================
with tabs[2]:
    st.header("Detection & Classification Performance")
    comp = load_csv("results/model_comparison.csv")
    if comp is not None:
        st.dataframe(comp.style.highlight_max(subset=["Macro F1"], color="lightgreen", axis=0),
                     use_container_width=True)
    else:
        st.info("Upload `results/model_comparison.csv` to display the baseline comparison.")

    hist_img = load_image("results/lcga_training_history.png")
    if hist_img:
        st.image(hist_img, caption="LCGA Training History", use_column_width=True)

    f1_img = load_image("results/lcga_per_class_f1.png")
    if f1_img:
        st.image(f1_img, caption="Per‑Class F1 Scores", use_column_width=True)

    cm_img = load_image("results/lcga_confusion_matrix.png")
    if cm_img:
        st.image(cm_img, caption="Confusion Matrix (row‑normalised)", use_column_width=True)

    st.header("Self‑Healing Performance")
    sys_df = load_csv("results/system_comparison.csv")
    if sys_df is not None:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="MTTR (s)", x=sys_df["System"], y=sys_df["MTTR_s"],
                             marker_color=["#E24B4A", "#854F0B", "#1D9E75"]))
        fig.add_trace(go.Bar(name="ISR (%)", x=sys_df["System"], y=sys_df["ISR_pct"],
                             marker_color=["#E24B4A", "#854F0B", "#1D9E75"], visible=False))
        fig.update_layout(barmode="group", updatemenus=[{
            "buttons": [
                {"label": "MTTR", "method": "update", "args": [{"visible": [True, False]}]},
                {"label": "ISR", "method": "update", "args": [{"visible": [False, True]}]},
            ],
            "direction": "right", "x": 0.1, "xanchor": "left"
        }])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload `results/system_comparison.csv`.")

    sys_img = load_image("results/system_comparison.png")
    if sys_img:
        st.image(sys_img, caption="System Comparison", use_column_width=True)

# ================== TAB 4: Explainability ==================
with tabs[3]:
    st.header("Explainable AI (SHAP & LIME)")
    shap_img = load_image("results/shap_global_importance.png")
    if shap_img:
        st.image(shap_img, caption="Global SHAP Feature Importance", use_column_width=True)

    force_img = load_image("results/shap_force_dos_hulk.png")
    if force_img:
        st.image(force_img, caption="SHAP Force Plot for DoS Hulk Sample", use_column_width=True)

    lime_img = load_image("results/lime_dos_hulk.png")
    if lime_img:
        st.image(lime_img, caption="LIME Explanation (slower, less stable)", use_column_width=True)

    xai = load_json("results/xai_comparison.json")
    if xai:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("SHAP (ms)", f"{xai['shap_ms']:.2f}")
        col2.metric("LIME (ms)", f"{xai['lime_ms']:.1f}")
        col3.metric("Speedup", f"{xai['lime_ms']/xai['shap_ms']:.0f}×")
        col4.metric("Surrogate Fidelity", f"{xai['surrogate_fidelity']:.4f}")
    else:
        st.info("Upload `results/xai_comparison.json`.")

# ================== TAB 5: Self‑Healing Demo ==================
with tabs[4]:
    st.header("MAPE‑K Action Log (real test‑set execution)")
    log_json = load_json("results/action_log_full.json")
    if log_json:
        log_df = pd.DataFrame(log_json)
        st.dataframe(log_df.head(20), use_container_width=True)
    else:
        st.info("Upload `results/action_log_full.json`.")

    ablation = load_csv("results/ablation_study.csv")
    if ablation is not None:
        st.subheader("Ablation Study")
        st.dataframe(ablation, use_container_width=True)
    else:
        st.info("Upload `results/ablation_study.csv`.")

# ================== TAB 6: Conclusions ==================
with tabs[5]:
    st.header("Conclusions & Future Work")
    st.markdown(
        "- The **LCGA framework** successfully demonstrates autonomous threat detection, "
        "intent‑aware classification, and closed‑loop self‑healing with **real‑time explanations**.\n"
        "- **87% MTTR reduction** and **87.6% ISR** validate the effectiveness of the MAPE‑K orchestrator.\n"
        "- SHAP‑based explanations are **four orders of magnitude faster** than LIME, "
        "enabling real‑time operator trust.\n\n"
        "**Future directions:**\n"
        "- Extend to zero‑day attacks via online learning.\n"
        "- Deploy on actual SDN hardware (e.g., ONOS, Mininet).\n"
        "- Integrate with SIEM systems and federated learning for multi‑site defense."
    )
    st.markdown("---")
    st.caption(
        "LCGA Framework v1.0 | AAU MSc Thesis 2026 | "
        "[GitHub](https://github.com/getaye21/lcga-self-healing-ids)"
    )
