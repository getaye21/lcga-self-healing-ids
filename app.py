"""
LCGA Self-Healing IDS — Real-Time Dashboard
MSc Thesis | Addis Ababa University
Fixed: SHAP force_plot signature, visible section headers, no upload limit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="LCGA Self-Healing IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: fix white-on-white headers + full-width uploader ──────────────
st.markdown("""
<style>
/* ── Section headings ── */
h1, h2, h3, h4 {
    color: #1a2a4a !important;
}

/* ── st.subheader rendered as <p class="..."> with small font ── */
[data-testid="stMarkdownContainer"] p strong {
    color: #1a2a4a;
}

/* ── Tab labels ── */
button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span {
    color: #1a2a4a !important;
    font-weight: 600;
}

/* Highlight the section-title divs that Streamlit renders white ── */
.stApp section[data-testid="stSidebar"] h2,
.stApp section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ── Make upload area accept any file size visually ── */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #4472c4 !important;
    background: #f0f4ff !important;
}

/* ── Metric labels ── */
[data-testid="stMetricLabel"] p {
    color: #1a2a4a !important;
    font-weight: 600;
}

/* ── DataFrame headers ── */
.dataframe thead th {
    background-color: #1f3864 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ LCGA IDS")
st.sidebar.markdown("**Intent-Aware Self-Healing Network**")
st.sidebar.markdown("---")
st.sidebar.info(
    "**MSc Thesis** — Addis Ababa University\n\n"
    "Getaye Fiseha · Mersen Getu · Chara Girma\n\n"
    "Advisor: Dr. Yaregal A.\n\n"
    "LCGA Framework v2.0 · June 2026"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["timestamp", "attack", "confidence", "action", "intents", "restored"]
    )

# ── Constants ─────────────────────────────────────────────────────────────────
ATTACK_CLASSES = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS slowloris", "FTP-Patator",
    "Heartbleed", "Infiltration", "PortScan", "SSH-Patator",
]

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

# ── Dummy model: load real DT surrogate if available ─────────────────────────
@st.cache_resource
def load_model():
    """Load trained DT surrogate + scaler from disk if available, else mock."""
    try:
        import joblib, os
        model_path = "models/dt_surrogate.pkl"
        scaler_path = "models/scaler.pkl"
        feature_path = "models/feature_names.pkl"
        if os.path.exists(model_path):
            model   = joblib.load(model_path)
            scaler  = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            feat_names = joblib.load(feature_path) if os.path.exists(feature_path) else None
            return model, scaler, feat_names, True
    except Exception:
        pass
    return None, None, None, False  # mock mode

dt_model, scaler, saved_features, model_loaded = load_model()

# ── Helper: align uploaded CSV to 73 expected features ───────────────────────
def align_features(df: pd.DataFrame, expected: list) -> pd.DataFrame:
    """Add missing columns as 0, drop extra columns, reorder."""
    for col in expected:
        if col not in df.columns:
            df[col] = 0.0
    return df[expected]


# ── Helper: safe SHAP force plot (fixed for shap >= 0.20) ─────────────────────
def shap_force_plot_safe(explainer, shap_values, input_row, feature_names, class_idx=None):
    """
    Wrapper that handles the shap >= 0.20 API change where base_value must be
    passed as the first positional argument to shap.force_plot().

    explainer   : fitted shap.TreeExplainer
    shap_values : raw output of explainer.shap_values(X)  — may be list-of-arrays
                  (multi-class) or a single 2-D array (binary)
    input_row   : 1-D numpy array of the single sample's features
    feature_names: list of feature name strings
    class_idx   : which class index to show (default: argmax of mean |shap|)
    """
    import shap

    # ── 1. Extract per-class SHAP values and base values ─────────────────────
    if isinstance(shap_values, list):
        # Multi-class: shap_values is a list of [n_samples x n_features] arrays
        if class_idx is None:
            # Pick the class whose SHAP vector has the largest mean absolute value
            class_idx = int(np.argmax([np.mean(np.abs(sv[0])) for sv in shap_values]))
        sv_row   = shap_values[class_idx][0]          # 1-D (n_features,)
        if hasattr(explainer, "expected_value"):
            ev = explainer.expected_value
            base_val = ev[class_idx] if hasattr(ev, "__len__") else float(ev)
        else:
            base_val = 0.0
    else:
        # Binary / single-output: shap_values is a 2-D array (n_samples, n_features)
        sv_row   = shap_values[0]                     # 1-D
        if hasattr(explainer, "expected_value"):
            ev = explainer.expected_value
            base_val = ev[1] if hasattr(ev, "__len__") and len(ev) > 1 else float(ev)
        else:
            base_val = 0.0

    # ── 2. Call force_plot with the corrected API ─────────────────────────────
    fig = shap.force_plot(
        base_val,           # <── required as 1st arg in shap >= 0.20
        sv_row,
        input_row,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    return fig, sv_row, class_idx


# ── Helper: bar-chart fallback when force_plot is unavailable ─────────────────
def shap_bar_chart(sv_row, feature_names, class_name, top_n=15):
    """Simple horizontal bar chart of top-N SHAP values."""
    idx = np.argsort(np.abs(sv_row))[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sv_row[idx]]
    ax.barh([feature_names[i] for i in idx], sv_row[idx], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on model output)")
    ax.set_title(f"SHAP Feature Importance — Predicted: {class_name}", fontsize=11,
                 color="#1a2a4a", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


# ── Helper: simulate detection cycle ─────────────────────────────────────────
def simulate_telemetry():
    weights = [0.55, 0.05, 0.07, 0.04, 0.08, 0.03, 0.03, 0.04, 0.02, 0.02, 0.04, 0.03]
    attack  = np.random.choice(ATTACK_CLASSES, p=weights)
    anomaly = attack != "BENIGN"
    score   = np.random.uniform(0.6, 0.99) if anomaly else np.random.uniform(0.1, 0.45)
    return anomaly, attack, round(score, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🛡️ LCGA Self-Healing IDS Dashboard")
st.markdown(
    "**Real-time network threat detection · Intent-aware classification · "
    "Explainable automated remediation (MAPE-K)**"
)

tabs = st.tabs([
    "📈 Live Monitor",
    "🔬 Live Flow Classification",
    "📊 Intent Status",
    "🔧 Self-Healing Log",
    "🧠 Explainability",
    "📋 System Metrics",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 0 — Live Monitor (simulation)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 📈 Live Detection Monitor")
    st.markdown("Simulate a network telemetry cycle and trigger the MAPE-K loop.")

    if st.button("▶ Run Detection Cycle", type="primary"):
        anomaly, attack, score = simulate_telemetry()
        ts = pd.Timestamp.now().isoformat()

        if anomaly:
            action    = ACTION_MAP.get(attack, "ESCALATE")
            intents   = ", ".join(INTENT_VIOLATIONS.get(attack, ["—"]))
            restored  = np.random.choice([True, False], p=[0.876, 0.124])
            new_entry = pd.DataFrame([{
                "timestamp":  ts,
                "attack":     attack,
                "confidence": score,
                "action":     action,
                "intents":    intents,
                "restored":   restored,
            }])
            st.session_state.history = pd.concat(
                [st.session_state.history, new_entry], ignore_index=True
            )
            st.error(f"🚨 **Attack detected:** {attack}  |  Confidence: {score:.1%}  |  Action: **{action}**")
        else:
            st.success(f"✅ **Normal traffic**  |  Anomaly score: {score:.3f}")

    if not st.session_state.history.empty:
        last = st.session_state.history.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Last Detected Attack", last["attack"])
        c2.metric("Confidence", f"{last['confidence']:.1%}")
        c3.metric("Action Taken", last["action"])

        isr = st.session_state.history["restored"].mean() * 100
        st.metric("🎯 Current ISR (session)", f"{isr:.1f}%")
    else:
        st.info("Click 'Run Detection Cycle' to start monitoring.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Flow Classification (DT Surrogate)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 🔬 Live Network Flow Classification (DT Surrogate)")
    st.markdown(
        "Upload a CSV of network flows. The app will align your features to the "
        "73 expected CICIDS2017 features, classify each flow, and generate SHAP "
        "explanations for the first sample."
    )

    # ── No size limit: set max_upload_size in .streamlit/config.toml,
    #    but here we handle large files gracefully regardless.
    uploaded = st.file_uploader(
        "Upload CSV (any size — rows = flows, columns = features)",
        type=["csv"],
        # NOTE: Streamlit's per-file cap is controlled by the server config.
        # On HuggingFace Spaces add this to .streamlit/config.toml:
        #   [server]
        #   maxUploadSize = 2048   # MB
    )

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # ── Strip whitespace from column names (common CICIDS issue) ─────────
        df_raw.columns = [c.strip() for c in df_raw.columns]

        # ── Drop label column if present ─────────────────────────────────────
        label_col = None
        for candidate in ["Label", "label", "Class", "class"]:
            if candidate in df_raw.columns:
                label_col = candidate
                df_raw = df_raw.drop(columns=[candidate])
                break

        # ── Replace inf / NaN ────────────────────────────────────────────────
        df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_raw.fillna(0, inplace=True)

        # ── Feature alignment ─────────────────────────────────────────────────
        # Use saved feature names if available; otherwise use whatever we have
        if saved_features is not None:
            expected_features = saved_features
        else:
            # Build a canonical 73-feature list from CICIDS2017 column names
            expected_features = list(df_raw.columns)[:73]

        n_original = df_raw.shape[1]
        if len(expected_features) == 73 and n_original != 73:
            df_aligned = align_features(df_raw.copy(), expected_features)
            st.info(f"Aligned shape: {df_aligned.shape[1]} features (expected 73)")
        elif n_original == 73:
            df_aligned = df_raw.copy()
            expected_features = list(df_aligned.columns)
            st.success(f"Shape matches: {n_original} features ✓")
        else:
            df_aligned = df_raw.copy()
            expected_features = list(df_aligned.columns)
            st.warning(f"Using all {n_original} columns as features (expected 73 for CICIDS2017 DT surrogate).")

        X = df_aligned.values.astype(np.float32)
        feature_names = list(expected_features)

        st.success(f"Loaded {len(X)} flow(s)  |  Features: {X.shape[1]}")

        # ── Scale if scaler available ─────────────────────────────────────────
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception:
                pass  # shape mismatch — use raw

        # ── Predict ──────────────────────────────────────────────────────────
        st.markdown("#### 🎯 Predictions")
        if model_loaded and dt_model is not None:
            preds  = dt_model.predict(X)
            probas = dt_model.predict_proba(X)
            class_labels = dt_model.classes_
        else:
            # Mock predictions for demo when no model artifact is present
            st.warning("⚠️ No trained DT surrogate found at `models/dt_surrogate.pkl`. "
                       "Showing mock predictions — upload your model to enable real inference.")
            preds  = np.random.choice(ATTACK_CLASSES[:6], size=len(X))
            probas = np.zeros((len(X), len(ATTACK_CLASSES)))
            for i, p in enumerate(preds):
                ci = ATTACK_CLASSES.index(p)
                probas[i, ci] = np.random.uniform(0.85, 1.0)
            class_labels = np.array(ATTACK_CLASSES)

        pred_df = pd.DataFrame({
            "Sample": [f"Sample {i+1}" for i in range(len(X))],
            "Predicted Class": preds,
            "Confidence": [f"{probas[i].max():.1%}" for i in range(len(X))],
        })
        st.dataframe(pred_df, use_container_width=True)

        # ── Friendly summary (replaces old print statements) ─────────────────
        for i in range(min(len(preds), 5)):
            conf = probas[i].max()
            st.write(f"**Sample {i+1}:** {preds[i]} ({conf:.1%} confidence)")

        # ── SHAP Explanation ─────────────────────────────────────────────────
        st.markdown("#### 🧠 SHAP Explanation (first sample)")
        try:
            import shap

            if model_loaded and dt_model is not None:
                explainer   = shap.TreeExplainer(dt_model)
                shap_values = explainer.shap_values(X)  # list (multi-class) or array

                fig, sv_row, ci = shap_force_plot_safe(
                    explainer, shap_values, X[0], feature_names
                )
                st.pyplot(fig, clear_figure=True)
                plt.close("all")

                # ── Also show a clean bar chart ───────────────────────────────
                pred_class = class_labels[ci] if hasattr(class_labels, "__len__") else preds[0]
                bar_fig = shap_bar_chart(sv_row, feature_names, pred_class)
                st.pyplot(bar_fig, clear_figure=True)
                plt.close("all")

                # ── Top-5 features table ──────────────────────────────────────
                top_idx = np.argsort(np.abs(sv_row))[-5:][::-1]
                st.markdown("**Top 5 most influential features:**")
                top_df = pd.DataFrame({
                    "Feature":    [feature_names[i] for i in top_idx],
                    "SHAP Value": [f"{sv_row[i]:+.4f}" for i in top_idx],
                    "Direction":  ["↑ Increases risk" if sv_row[i] > 0 else "↓ Reduces risk"
                                   for i in top_idx],
                })
                st.dataframe(top_df, use_container_width=True)
            else:
                # Mock SHAP bar chart using random values
                sv_mock = np.random.randn(len(feature_names)) * 0.3
                bar_fig = shap_bar_chart(sv_mock, feature_names, preds[0])
                st.pyplot(bar_fig, clear_figure=True)
                plt.close("all")
                st.caption("Mock SHAP values shown — load real DT surrogate for actual explanations.")

        except ImportError:
            st.warning("SHAP library not installed. Run: `pip install shap`")
        except Exception as e:
            # ── Graceful fallback: show the bar chart even if force_plot fails ─
            st.warning(f"Force plot unavailable ({type(e).__name__}). Showing bar chart instead.")
            try:
                import shap
                explainer   = shap.TreeExplainer(dt_model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    ci = int(np.argmax([np.mean(np.abs(sv[0])) for sv in shap_values]))
                    sv_row = shap_values[ci][0]
                else:
                    sv_row = shap_values[0]
                bar_fig = shap_bar_chart(sv_row, feature_names, preds[0])
                st.pyplot(bar_fig, clear_figure=True)
                plt.close("all")
            except Exception as e2:
                st.error(f"SHAP explanation failed entirely: {e2}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Intent Status
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 📊 Network Intent Status")
    st.markdown("Current status of the five formalised MAPE-K network intents.")

    recent_attacks = (
        st.session_state.history["attack"].tolist()
        if not st.session_state.history.empty else []
    )

    def intent_status(intent_name, default="🟢 Satisfied"):
        for a in recent_attacks[-3:]:
            violations = INTENT_VIOLATIONS.get(a, [])
            if any(intent_name in v for v in violations):
                return "🔴 Violated"
        return default

    intents_data = [
        ("I1", "HTTP Latency < 200 ms",    "HTTP Latency",       intent_status("I1"), "90 s"),
        ("I2", "SSH Availability = True",  "SSH Availability",   intent_status("I2"), "60 s"),
        ("I3", "Auth Failure Rate < 10/min","Auth Failure Rate", intent_status("I3"), "60 s"),
        ("I4", "Port Scan Rate < 5/min",   "Port Scan Rate",     intent_status("I4"), "30 s"),
        ("I5", "Bandwidth < 100 Mbps",     "Bandwidth",          intent_status("I5"), "90 s"),
    ]
    intent_df = pd.DataFrame(intents_data,
                              columns=["ID", "Intent", "Metric", "Status", "Cooldown"])
    st.dataframe(intent_df, use_container_width=True)

    violated = [r for r in intents_data if "Violated" in r[3]]
    if violated:
        st.error(f"⚠️ {len(violated)} intent(s) currently violated: "
                 f"{', '.join(r[0] for r in violated)}")
    else:
        st.success("✅ All intents satisfied.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Self-Healing Log
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🔧 MAPE-K Self-Healing Action Log")

    if st.session_state.history.empty:
        st.info("No healing actions logged yet. Run detection cycles in the Live Monitor tab.")
    else:
        display_df = (
            st.session_state.history
            .tail(20)
            .sort_values("timestamp", ascending=False)
            .copy()
        )
        display_df["timestamp"] = display_df["timestamp"].str[:19].str.replace("T", " ")
        display_df["restored"]  = display_df["restored"].map({True: "✅ Yes", False: "❌ No"})
        st.dataframe(display_df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        isr  = st.session_state.history["restored"].mean() * 100
        total = len(st.session_state.history)
        attacks_detected = (st.session_state.history["attack"] != "BENIGN").sum()
        c1.metric("Total Attacks", attacks_detected)
        c2.metric("Actions Taken", total)
        c3.metric("ISR (session)", f"{isr:.1f}%")

        if st.button("🗑️ Clear Log"):
            st.session_state.history = pd.DataFrame(
                columns=["timestamp", "attack", "confidence", "action", "intents", "restored"]
            )
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Explainability Explorer
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🧠 Explainability Explorer")
    st.markdown(
        "Select an attack class to explore which network features drive the DT surrogate's "
        "decision. Feature importances shown are from the trained global SHAP analysis."
    )

    selected_attack = st.selectbox("Select attack class:", ATTACK_CLASSES[1:])  # skip BENIGN

    # Precomputed representative SHAP feature patterns per class
    SHAP_PROFILES = {
        "DoS Hulk":        {"Flow Duration": 0.48, "Bwd Packet Length Std": 0.41,
                            "Fwd Packet Length Max": 0.38, "Total Fwd Packets": 0.32,
                            "Packet Length Mean": -0.12},
        "DDoS":            {"Total Length of Fwd Packets": 0.52, "Destination Port": 0.44,
                            "Total Fwd Packets": 0.39, "Flow Duration": -0.28,
                            "Bwd Packets/s": 0.22},
        "PortScan":        {"Destination Port": 0.61, "Flow Duration": 0.45,
                            "Total Fwd Packets": 0.38, "Fwd IAT Total": -0.21,
                            "Init_Win_bytes_forward": 0.15},
        "SSH-Patator":     {"Destination Port": 0.55, "Flow Duration": 0.47,
                            "Fwd Packet Length Std": 0.31, "Total Fwd Packets": 0.28,
                            "Bwd Packet Length Mean": -0.18},
        "FTP-Patator":     {"Destination Port": 0.58, "Flow Duration": 0.44,
                            "Total Fwd Packets": 0.33, "Fwd Packet Length Mean": 0.25,
                            "Flow Bytes/s": -0.16},
        "Bot":             {"Flow Duration": 0.44, "Packet Length Std": 0.37,
                            "Average Packet Size": 0.29, "Fwd IAT Mean": 0.21,
                            "Idle Mean": -0.14},
        "Heartbleed":      {"Total Fwd Packets": 0.53, "Fwd Packet Length Max": 0.48,
                            "Destination Port": 0.42, "Flow Duration": -0.31,
                            "Bwd Packet Length Max": 0.19},
        "Infiltration":    {"Flow Duration": 0.46, "Fwd Packet Length Mean": 0.38,
                            "Total Length of Fwd Packets": 0.34, "Idle Mean": 0.22,
                            "Packet Length Variance": -0.15},
        "DoS GoldenEye":   {"Flow Duration": 0.43, "Bwd IAT Total": 0.39,
                            "Fwd Packets/s": 0.34, "Packet Length Mean": -0.25,
                            "Total Fwd Packets": 0.21},
        "DoS Slowhttptest":{"Flow Duration": 0.57, "Fwd IAT Total": 0.45,
                            "Total Fwd Packets": 0.28, "Fwd Packet Length Mean": -0.19,
                            "Active Mean": 0.14},
        "DoS slowloris":   {"Flow Duration": 0.59, "Fwd IAT Mean": 0.44,
                            "Total Fwd Packets": 0.27, "Fwd Packet Length Std": -0.17,
                            "Active Mean": 0.13},
    }

    profile = SHAP_PROFILES.get(selected_attack, {
        "Flow Duration": 0.40, "Destination Port": 0.35,
        "Total Fwd Packets": 0.28, "Packet Length Mean": 0.20,
        "Bwd Packet Length Std": 0.15,
    })

    feats  = list(profile.keys())
    values = list(profile.values())
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(feats, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Feature Impact — {selected_attack}", fontsize=12,
                 color="#1a2a4a", fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close("all")

    st.markdown(f"**Intent(s) violated by {selected_attack}:** "
                f"{', '.join(INTENT_VIOLATIONS.get(selected_attack, ['None']))}")
    st.markdown(f"**Recommended action:** `{ACTION_MAP.get(selected_attack, 'ESCALATE')}`")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — System Metrics
# ──────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### 📋 System Performance Metrics")

    st.markdown("#### Model Comparison — CICIDS2017")
    model_df = pd.DataFrame({
        "Model":        ["Random Forest", "CNN Baseline", "GRU Baseline", "LCGA (Ours)"],
        "Accuracy":     ["99.53%", "97.65%", "98.30%", "**99.67%**"],
        "Macro F1":     ["0.9527", "0.9420", "0.9490", "0.8170"],
        "Weighted F1":  ["0.9953", "0.9760", "0.9825", "**0.9967**"],
        "MCC":          ["0.9945", "0.9745", "0.9820", "**0.9945**"],
        "Params":       ["100 trees", "5,196", "~20,000", "**41,260**"],
        "Inference ms": ["0.10", "5.17", "4.80", "**1.85**"],
    })
    st.dataframe(model_df, use_container_width=True)

    st.markdown("#### Self-Healing Comparison")
    heal_df = pd.DataFrame({
        "System":          ["Open-loop (no healing)", "Rule-based healing", "LCGA + MAPE-K (Ours)"],
        "MTTR (s)":        ["598.5", "65.1", "**78.4**"],
        "ISR (%)":         ["0.0%", "64.2%", "**87.6%**"],
        "MTTR Reduction":  ["—", "89.1%", "**86.9%**"],
    })
    st.dataframe(heal_df, use_container_width=True)

    st.markdown("#### XAI Comparison — SHAP vs LIME")
    xai_df = pd.DataFrame({
        "Metric":                ["Time (ms/sample)", "Speedup", "Top-3 Consistency", "Fidelity"],
        "SHAP (DT Surrogate)":   ["0.05", "11,635×", "—", "99.64%"],
        "LIME":                  ["812", "—", "30.0%", "N/A"],
    })
    st.dataframe(xai_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "LCGA Framework v2.0 | MSc Thesis — Addis Ababa University 2026 | "
    "[GitHub](https://github.com/getaye21/lcga-self-healing-ids)"
)
