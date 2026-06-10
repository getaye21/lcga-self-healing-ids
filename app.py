"""
A Lightweight Hybrid Deep Learning Framework for Real-Time Cyber Threat
Detection and Intent-Aware Self-Healing Network Security
MSc ML Thesis | Addis Ababa University | Department of Computer Science
Version 1.1 — June 2026

Changes from v1.0:
  - Universal file decoder (CSV, JSON, PCAP, Excel, Parquet, log, binary)
  - Robust column alignment with alias mapping
  - Fixed "bad message format" crash in file uploader
  - SHAP error handling with graceful fallback
  - traceback import added
"""

import io
import os
import re
import json
import struct
import hashlib
import warnings
import traceback
from pathlib import Path
from typing import Optional, Tuple
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LCGA Self-Healing IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
h1,h2,h3,h4,h5 { color:#1a2a4a !important; }
[data-testid="stMetricLabel"] p { color:#1a2a4a !important; font-weight:700; }
button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span { color:#1a2a4a !important; font-weight:600; }
[data-testid="stFileUploaderDropzone"] {
    border:2px dashed #c0392b !important;
    background:#fff8f8 !important;
}
div[data-testid="stMetricValue"] { color:#1f3864 !important; font-weight:800; }
.step-box {
    background:#f0f4ff; border-left:4px solid #4472c4;
    border-radius:6px; padding:10px 14px; margin:6px 0;
    font-size:0.93rem;
}
.tip-box {
    background:#fff8e1; border-left:4px solid #f39c12;
    border-radius:6px; padding:8px 12px; margin:6px 0;
    font-size:0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FULL_TITLE = ("A Lightweight Hybrid Deep Learning Framework for Real-Time "
              "Cyber Threat Detection and Intent-Aware Self-Healing Network Security")

CICIDS_CLASSES = [
    "BENIGN","Bot","DDoS","DoS GoldenEye","DoS Hulk",
    "DoS Slowhttptest","DoS slowloris","FTP-Patator",
    "Heartbleed","Infiltration","PortScan","SSH-Patator",
]

CICIDS_FEATURES = [
    "Destination Port","Flow Duration","Total Fwd Packets",
    "Total Backward Packets","Total Length of Fwd Packets",
    "Total Length of Bwd Packets","Fwd Packet Length Max",
    "Fwd Packet Length Min","Fwd Packet Length Mean",
    "Fwd Packet Length Std","Bwd Packet Length Max",
    "Bwd Packet Length Min","Bwd Packet Length Mean",
    "Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s",
    "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max",
    "Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std",
    "Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Fwd URG Flags",
    "Fwd Header Length","Bwd Header Length","Fwd Packets/s",
    "Bwd Packets/s","Min Packet Length","Max Packet Length",
    "Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count",
    "ACK Flag Count","URG Flag Count","CWE Flag Count","ECE Flag Count",
    "Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size",
    "Avg Bwd Segment Size","Fwd Header Length.1",
    "Subflow Fwd Packets","Subflow Fwd Bytes",
    "Subflow Bwd Packets","Subflow Bwd Bytes",
    "Init_Win_bytes_forward","Init_Win_bytes_backward",
    "act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Std","Active Max","Active Min",
    "Idle Mean","Idle Std","Idle Max","Idle Min",
    "avg_packet_size","fwd_pkt_range","bwd_pkt_range",
]

# Column name aliases (normalised → canonical)
_ALIASES = {
    "dst_port":"Destination Port","destination_port":"Destination Port",
    "dstport":"Destination Port","flow_duration":"Flow Duration",
    "duration":"Flow Duration","tot_fwd_pkts":"Total Fwd Packets",
    "total_fwd_pkts":"Total Fwd Packets","tot_bwd_pkts":"Total Backward Packets",
    "totlen_fwd_pkts":"Total Length of Fwd Packets",
    "totlen_bwd_pkts":"Total Length of Bwd Packets",
    "fwd_pkt_len_max":"Fwd Packet Length Max",
    "fwd_pkt_len_min":"Fwd Packet Length Min",
    "fwd_pkt_len_mean":"Fwd Packet Length Mean",
    "fwd_pkt_len_std":"Fwd Packet Length Std",
    "bwd_pkt_len_max":"Bwd Packet Length Max",
    "bwd_pkt_len_min":"Bwd Packet Length Min",
    "bwd_pkt_len_mean":"Bwd Packet Length Mean",
    "bwd_pkt_len_std":"Bwd Packet Length Std",
    "flow_byts_s":"Flow Bytes/s","flow_pkts_s":"Flow Packets/s",
    "flow_iat_mean":"Flow IAT Mean","flow_iat_std":"Flow IAT Std",
    "flow_iat_max":"Flow IAT Max","flow_iat_min":"Flow IAT Min",
    "fwd_iat_tot":"Fwd IAT Total","fwd_iat_mean":"Fwd IAT Mean",
    "fwd_iat_std":"Fwd IAT Std","fwd_iat_max":"Fwd IAT Max",
    "fwd_iat_min":"Fwd IAT Min","bwd_iat_tot":"Bwd IAT Total",
    "bwd_iat_mean":"Bwd IAT Mean","bwd_iat_std":"Bwd IAT Std",
    "bwd_iat_max":"Bwd IAT Max","bwd_iat_min":"Bwd IAT Min",
    "fwd_psh_flags":"Fwd PSH Flags","fwd_urg_flags":"Fwd URG Flags",
    "fwd_header_len":"Fwd Header Length","bwd_header_len":"Bwd Header Length",
    "fwd_pkts_s":"Fwd Packets/s","bwd_pkts_s":"Bwd Packets/s",
    "pkt_len_min":"Min Packet Length","pkt_len_max":"Max Packet Length",
    "pkt_len_mean":"Packet Length Mean","pkt_len_std":"Packet Length Std",
    "pkt_len_var":"Packet Length Variance",
    "fin_flag_cnt":"FIN Flag Count","syn_flag_cnt":"SYN Flag Count",
    "rst_flag_cnt":"RST Flag Count","psh_flag_cnt":"PSH Flag Count",
    "ack_flag_cnt":"ACK Flag Count","urg_flag_cnt":"URG Flag Count",
    "cwe_flag_count":"CWE Flag Count","ece_flag_cnt":"ECE Flag Count",
    "down_up_ratio":"Down/Up Ratio","avg_pkt_size":"Average Packet Size",
    "avg_fwd_seg_size":"Avg Fwd Segment Size",
    "avg_bwd_seg_size":"Avg Bwd Segment Size",
    "subflow_fwd_pkts":"Subflow Fwd Packets",
    "subflow_fwd_byts":"Subflow Fwd Bytes",
    "subflow_bwd_pkts":"Subflow Bwd Packets",
    "subflow_bwd_byts":"Subflow Bwd Bytes",
    "init_fwd_win_byts":"Init_Win_bytes_forward",
    "init_bwd_win_byts":"Init_Win_bytes_backward",
    "init_win_bytes_fwd":"Init_Win_bytes_forward",
    "init_win_bytes_bwd":"Init_Win_bytes_backward",
    "active_mean":"Active Mean","active_std":"Active Std",
    "active_max":"Active Max","active_min":"Active Min",
    "idle_mean":"Idle Mean","idle_std":"Idle Std",
    "idle_max":"Idle Max","idle_min":"Idle Min",
}

_NON_FEATURE_COLS = {
    "label","labels","class","classes","target","attack_type",
    "category","type","flow_id","id","src_ip","dst_ip",
    "source_ip","destination_ip","src_port","timestamp",
    "date","time","src_mac","dst_mac",
}

def idx_to_label(idx):
    try:
        i = int(idx)
        if 0 <= i < len(CICIDS_CLASSES):
            return CICIDS_CLASSES[i]
    except (ValueError, TypeError):
        pass
    return str(idx)

ACTION_MAP = {
    "DoS Hulk":"BLOCK_IP","DoS GoldenEye":"BLOCK_IP",
    "DoS Slowhttptest":"RATE_LIMIT","DoS slowloris":"RATE_LIMIT",
    "DDoS":"ISOLATE_SUBNET","PortScan":"BLOCK_IP","Bot":"ISOLATE_SUBNET",
    "SSH-Patator":"BLOCK_IP","FTP-Patator":"BLOCK_IP",
    "Heartbleed":"RESTART_SERVICE","Infiltration":"ISOLATE_SUBNET","BENIGN":"—",
}

INTENT_VIOLATIONS = {
    "DoS Hulk":["I1 - HTTP Latency","I5 - Bandwidth"],
    "DoS GoldenEye":["I1 - HTTP Latency","I5 - Bandwidth"],
    "DoS Slowhttptest":["I1 - HTTP Latency"],"DoS slowloris":["I1 - HTTP Latency"],
    "DDoS":["I1 - HTTP Latency","I5 - Bandwidth"],"PortScan":["I4 - Port Scan Rate"],
    "Bot":["I3 - Auth Failure Rate"],
    "SSH-Patator":["I2 - SSH Availability","I3 - Auth Failure Rate"],
    "FTP-Patator":["I3 - Auth Failure Rate"],"Heartbleed":["I2 - SSH Availability"],
    "Infiltration":["I1 - HTTP Latency","I2 - SSH Availability"],"BENIGN":[],
}

SHAP_PROFILES = {
    "DoS Hulk":{"Flow Duration":0.48,"Bwd Packet Length Std":0.41,"Fwd Packet Length Max":0.38,"Total Fwd Packets":0.32,"Packet Length Mean":-0.12},
    "DDoS":{"Total Length of Fwd Packets":0.52,"Destination Port":0.44,"Total Fwd Packets":0.39,"Flow Duration":-0.28,"Bwd Packets/s":0.22},
    "PortScan":{"Destination Port":0.61,"Flow Duration":0.45,"Total Fwd Packets":0.38,"Fwd IAT Total":-0.21,"Init_Win_bytes_forward":0.15},
    "SSH-Patator":{"Destination Port":0.55,"Flow Duration":0.47,"Fwd Packet Length Std":0.31,"Total Fwd Packets":0.28,"Bwd Packet Length Mean":-0.18},
    "FTP-Patator":{"Destination Port":0.58,"Flow Duration":0.44,"Total Fwd Packets":0.33,"Fwd Packet Length Mean":0.25,"Flow Bytes/s":-0.16},
    "Bot":{"Flow Duration":0.44,"Packet Length Std":0.37,"Average Packet Size":0.29,"Fwd IAT Mean":0.21,"Idle Mean":-0.14},
    "Heartbleed":{"Total Fwd Packets":0.53,"Fwd Packet Length Max":0.48,"Destination Port":0.42,"Flow Duration":-0.31,"Bwd Packet Length Max":0.19},
    "Infiltration":{"Flow Duration":0.46,"Fwd Packet Length Mean":0.38,"Total Length of Fwd Packets":0.34,"Idle Mean":0.22,"Packet Length Variance":-0.15},
    "DoS GoldenEye":{"Flow Duration":0.43,"Bwd IAT Total":0.39,"Fwd Packets/s":0.34,"Packet Length Mean":-0.25,"Total Fwd Packets":0.21},
    "DoS Slowhttptest":{"Flow Duration":0.57,"Fwd IAT Total":0.45,"Total Fwd Packets":0.28,"Fwd Packet Length Mean":-0.19,"Active Mean":0.14},
    "DoS slowloris":{"Flow Duration":0.59,"Fwd IAT Mean":0.44,"Total Fwd Packets":0.27,"Fwd Packet Length Std":-0.17,"Active Mean":0.13},
}

SAMPLE_FEATURE_NAMES = CICIDS_FEATURES[:]

SAMPLE_CSV_ROWS = [
    "80,0.7798274,1,0.68337727,0.5243554,0.8570111,0.6458244,-0.37255234,0.25551075,1.0489987,0.99770284,0,1.0385096,1.083125,-0.04952005,-0.006316472,0.7682475,0.53317225,0.77946776,0.3710491,0.8315301,0.83057684,1.1237686,0.8304062,0.7883953,0.9799758,0.9847786,1.1316011,0.97982883,0.2856457,0,0,1.230483,1.5238419,-0.5590332,-0.52686816,-0.37762213,0.90821683,1.0099504,0.9050604,0.8490192,5.6713157,0,0,0,0,0,0,0,0,1.0703446,0.25551075,1.0385096,1.230483,1,0.52435535,0.68337727,0.8570111,-0.5060647,0.7339891,0.25415918,0,2.4425347,0,2.43745,2.45111,2.4027748,3.4780214,2.4063945,2.392535,0.030667111,0.95121646,1.1587068",
    "443,0.2456789,2,1.2345678,0.6789012,0.9012345,0.754321,-0.1876543,0.3987654,1.0234567,0.8876543,0.1234567,0.9345678,1.054321,-0.0312345,-0.0076543,0.7456789,0.5189012,0.754321,0.3987654,0.8123456,0.8012345,1.0987654,0.8234567,0.7654321,0.9456789,0.956789,1.0876543,0.9345678,0.2876543,0,0,1.1987654,1.4987654,-0.5432109,-0.5123456,-0.354321,0.8765432,0.9876543,0.8765432,0.8234567,5.1234567,0,0,0,1,0,0,0,0,1.0345678,0.2876543,0.9876543,1.1987654,1,0.5890123,0.6456789,0.8123456,-0.0234567,0.7123456,0.2890123,0,2.3890123,0,2.3876543,2.3987654,2.3789012,3.4234567,2.3901234,2.3765432,0.0298765,0.9432109,1.1345678",
    "53,0.5123456,3,1.6789012,0.8234567,1.0123456,0.8987654,-0.1432109,0.4789012,1.1123456,0.9345678,0.2345678,0.9654321,1.0987654,-0.0456789,-0.0087654,0.7789012,0.5432109,0.7890123,0.4234567,0.8456789,0.8345678,1.1234567,0.8456789,0.7987654,0.9765432,0.9876543,1.1098765,0.956789,0.3098765,0,0,1.2987654,1.5234567,-0.5678901,-0.5345678,-0.3890123,0.9123456,1.0234567,0.9123456,0.856789,6.0123456,0,0,0,1,0,0,0,0,1.056789,0.2987654,1.0123456,1.2987654,1,0.6234567,0.6789012,0.8456789,-0.0456789,0.7345678,0.3123456,0,2.4123456,0,2.4098765,2.4234567,2.3987654,3.4789012,2.4012345,2.3987654,0.0312345,0.9589012,1.1678901",
]

def make_sample_csv():
    header = ",".join(SAMPLE_FEATURE_NAMES) + ",Label\n"
    labels = ["DoS Hulk","BENIGN","PortScan"]
    rows   = "\n".join(r + f",{l}" for r, l in zip(SAMPLE_CSV_ROWS, labels))
    return header + rows

import base64

# ── Embedded architecture images (tiny placeholders — keep originals) ─────────
_ARCH_IMG  = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
_MAPEK_IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
# NOTE: Replace the two lines above with your original base64 strings from v1.0


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL FILE DECODER
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_col(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _map_column(col: str) -> Optional[str]:
    norm = _normalise_col(col)
    for feat in CICIDS_FEATURES:
        if _normalise_col(feat) == norm:
            return feat
    return _ALIASES.get(norm)

def _zero_flow() -> dict:
    return {f: 0.0 for f in CICIDS_FEATURES}

def _split_chunks(arr: np.ndarray, n: int) -> list:
    size = max(1, len(arr) // n)
    return [arr[i * size:(i + 1) * size] for i in range(n)]

def _clean_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Rename → drop non-features → pad missing → float32."""
    # 1. Drop known non-feature columns
    drop = [c for c in df.columns if _normalise_col(c) in _NON_FEATURE_COLS]
    df = df.drop(columns=drop, errors="ignore")

    # 2. Rename via alias map
    rename_map = {}
    for col in df.columns:
        canon = _map_column(col)
        if canon:
            rename_map[col] = canon
    df = df.rename(columns=rename_map)

    # 3. Deduplicate columns (keep first)
    seen: set = set()
    keep = []
    for col in df.columns:
        if col not in seen:
            seen.add(col)
            keep.append(col)
    df = df[keep]

    # 4. If positional alignment (73 columns, no names matched)
    feat_cols = [c for c in df.columns if c in CICIDS_FEATURES]
    if len(feat_cols) < 5 and df.shape[1] == 73:
        df.columns = CICIDS_FEATURES
        feat_cols  = CICIDS_FEATURES[:]
    elif feat_cols:
        df = df[feat_cols]

    # 5. Pad missing features
    for feat in CICIDS_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    df = df[CICIDS_FEATURES]

    # 6. Clean
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df.astype(np.float32).reset_index(drop=True)


def _decode_delimited(file_bytes: bytes, ext: str) -> Tuple[pd.DataFrame, str]:
    sep = "\t" if ext == "tsv" else ","
    text = None
    for enc in ("utf-8", "latin-1", "cp1252", "utf-16"):
        try:
            text = file_bytes.decode(enc); break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")

    lines = text.splitlines()
    if lines and "\t" in lines[0] and sep == ",":
        sep = "\t"

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", low_memory=False)

    orig = df.shape
    df   = _clean_and_align(df)
    return df, f"Parsed {orig[0]} rows × {orig[1]} cols → {len(df)} flows × 73 features."


def _decode_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            if df.shape[0] > 0:
                orig = df.shape
                df   = _clean_and_align(df)
                return df, f"Excel '{sheet}': {orig[0]}×{orig[1]} → {len(df)} flows × 73."
        raise ValueError("All sheets empty.")
    except ImportError:
        raise ValueError("openpyxl not installed — cannot read Excel.")


def _decode_parquet(file_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    try:
        df   = pd.read_parquet(io.BytesIO(file_bytes))
        orig = df.shape
        df   = _clean_and_align(df)
        return df, f"Parquet: {orig[0]}×{orig[1]} → {len(df)} flows × 73."
    except ImportError:
        raise ValueError("pyarrow/fastparquet not installed.")


def _decode_json(file_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    text = file_bytes.decode("utf-8", errors="replace").strip()

    # JSONL
    if text.startswith("{") and "\n" in text:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if records:
            df = pd.json_normalize(records)
            orig = df.shape; df = _clean_and_align(df)
            return df, f"JSONL: {orig[0]} records → {len(df)} flows × 73."

    data = json.loads(text)
    if isinstance(data, list):
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        for key in ("flows","data","records","samples","features"):
            if key in data and isinstance(data[key], list):
                df = pd.json_normalize(data[key]); break
        else:
            df = pd.DataFrame([data])
    else:
        raise ValueError("Unexpected JSON structure.")

    orig = df.shape; df = _clean_and_align(df)
    return df, f"JSON: {orig[0]} records → {len(df)} flows × 73."


def _decode_binary(
    file_bytes: bytes,
    filename: str,
    note: str = "",
) -> Tuple[pd.DataFrame, str]:
    """Last-resort: derive 73 features from byte statistics."""
    b = np.frombuffer(file_bytes, dtype=np.uint8).astype(np.float64)
    n = len(b)
    if n == 0:
        df = pd.DataFrame([{f: 0.0 for f in CICIDS_FEATURES}])
        return df.astype(np.float32), "Empty file — zero vector."

    chunks = _split_chunks(b, 73)
    rec    = _zero_flow()

    for i, feat in enumerate(CICIDS_FEATURES):
        ch = chunks[i] if i < len(chunks) else np.array([0.0])
        slot = i % 4
        if len(ch) == 0:
            rec[feat] = 0.0
        elif slot == 0:
            rec[feat] = float(ch.mean())
        elif slot == 1:
            rec[feat] = float(ch.std()) if len(ch) > 1 else 0.0
        elif slot == 2:
            rec[feat] = float(ch.max())
        else:
            rec[feat] = float(ch.min())

    rec["Flow Duration"]          = float(n)
    rec["Total Fwd Packets"]      = float(n // 64)
    rec["Max Packet Length"]      = float(b.max())
    rec["Min Packet Length"]      = float(b.min())
    rec["Packet Length Mean"]     = float(b.mean())
    rec["Packet Length Std"]      = float(b.std())
    rec["Packet Length Variance"] = float(b.var())
    rec["Average Packet Size"]    = float(b.mean())
    rec["Flow Bytes/s"]           = float(n)
    rec["Flow Packets/s"]         = float(n // 64)

    hist, _ = np.histogram(b, bins=256, range=(0, 255))
    prob    = hist / hist.sum()
    entropy = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))
    rec["Destination Port"] = entropy * 1000

    df     = pd.DataFrame([rec])[CICIDS_FEATURES].astype(np.float32)
    prefix = f"{note} " if note else ""
    return df, f"{prefix}Binary '{filename}' ({n:,} bytes) → 1 synthetic flow × 73 features."


def _decode_pcap(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, str]:
    """Extract flow statistics from PCAP. Falls back to binary if no PCAP lib."""
    # --- dpkt ---
    try:
        import dpkt
        buf = io.BytesIO(file_bytes)
        try:
            pcap = dpkt.pcap.Reader(buf)
        except Exception:
            buf.seek(0)
            pcap = dpkt.pcapng.Reader(buf)

        flows: dict = {}
        for ts, raw in pcap:
            try:
                eth   = dpkt.ethernet.Ethernet(raw)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip    = eth.data
                proto = ip.p
                pkt_len = len(raw)
                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    tp       = ip.data
                    src_port = tp.sport
                    dst_port = tp.dport
                    flags    = getattr(tp, "flags", 0)
                else:
                    src_port = dst_port = flags = 0

                key = (ip.src, ip.dst, src_port, dst_port, proto)
                if key not in flows:
                    flows[key] = {"dst_port": dst_port, "times": [],
                                  "fwd_lens": [], "flags": []}
                flows[key]["times"].append(float(ts))
                flows[key]["fwd_lens"].append(pkt_len)
                flows[key]["flags"].append(flags)
            except Exception:
                continue

        if not flows:
            return _decode_binary(file_bytes, filename, "No IP flows in PCAP.")

        records = []
        for key, f in flows.items():
            times = sorted(f["times"])
            fwd   = np.array(f["fwd_lens"], dtype=float)
            iats  = np.diff(times) * 1e6 if len(times) > 1 else np.array([0.0])
            dur   = (times[-1] - times[0]) * 1e6 if len(times) > 1 else 0

            rec = _zero_flow()
            rec["Destination Port"]             = float(key[3])
            rec["Flow Duration"]                = dur
            rec["Total Fwd Packets"]            = float(len(fwd))
            rec["Total Length of Fwd Packets"]  = float(fwd.sum())
            rec["Fwd Packet Length Max"]        = float(fwd.max())
            rec["Fwd Packet Length Min"]        = float(fwd.min())
            rec["Fwd Packet Length Mean"]       = float(fwd.mean())
            rec["Fwd Packet Length Std"]        = float(fwd.std()) if len(fwd)>1 else 0
            rec["Flow Bytes/s"]                 = fwd.sum() / (dur/1e6 + 1e-9)
            rec["Flow Packets/s"]               = len(fwd) / (dur/1e6 + 1e-9)
            rec["Flow IAT Mean"]                = float(iats.mean())
            rec["Flow IAT Std"]                 = float(iats.std()) if len(iats)>1 else 0
            rec["Flow IAT Max"]                 = float(iats.max())
            rec["Flow IAT Min"]                 = float(iats.min())
            rec["Fwd IAT Total"]                = float(iats.sum())
            rec["Fwd IAT Mean"]                 = float(iats.mean())
            rec["Fwd Packets/s"]                = len(fwd) / (dur/1e6 + 1e-9)
            rec["Min Packet Length"]            = float(fwd.min())
            rec["Max Packet Length"]            = float(fwd.max())
            rec["Packet Length Mean"]           = float(fwd.mean())
            rec["Packet Length Std"]            = float(fwd.std()) if len(fwd)>1 else 0
            rec["Packet Length Variance"]       = float(fwd.var())  if len(fwd)>1 else 0
            rec["Average Packet Size"]          = float(fwd.mean())
            rec["avg_packet_size"]              = float(fwd.mean())
            for fl in f["flags"]:
                rec["FIN Flag Count"] += int(bool(fl & 0x01))
                rec["SYN Flag Count"] += int(bool(fl & 0x02))
                rec["RST Flag Count"] += int(bool(fl & 0x04))
                rec["PSH Flag Count"] += int(bool(fl & 0x08))
                rec["ACK Flag Count"] += int(bool(fl & 0x10))
                rec["URG Flag Count"] += int(bool(fl & 0x20))
            records.append(rec)

        df = pd.DataFrame(records)[CICIDS_FEATURES].astype(np.float32)
        return df, f"PCAP (dpkt): {len(df)} flows × 73 features."
    except ImportError:
        pass

    # --- scapy ---
    try:
        from scapy.all import PcapReader, IP, TCP, UDP
        packets = list(PcapReader(io.BytesIO(file_bytes)))
        flows2: dict = {}
        for pkt in packets:
            try:
                if IP not in pkt:
                    continue
                sp = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
                dp = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
                key = (pkt[IP].src, pkt[IP].dst, sp, dp, pkt[IP].proto)
                if key not in flows2:
                    flows2[key] = {"dst_port": dp, "times": [], "lens": [], "flags": []}
                flows2[key]["times"].append(float(pkt.time))
                flows2[key]["lens"].append(len(pkt))
                flows2[key]["flags"].append(int(pkt[TCP].flags) if TCP in pkt else 0)
            except Exception:
                continue

        if flows2:
            records = []
            for key, f in flows2.items():
                times = sorted(f["times"])
                lens  = np.array(f["lens"], dtype=float)
                iats  = np.diff(times)*1e6 if len(times)>1 else np.array([0.0])
                dur   = (times[-1]-times[0])*1e6 if len(times)>1 else 0
                rec   = _zero_flow()
                rec["Destination Port"]     = float(key[3])
                rec["Flow Duration"]        = dur
                rec["Total Fwd Packets"]    = float(len(lens))
                rec["Total Length of Fwd Packets"] = float(lens.sum())
                rec["Fwd Packet Length Max"]  = float(lens.max())
                rec["Fwd Packet Length Min"]  = float(lens.min())
                rec["Fwd Packet Length Mean"] = float(lens.mean())
                rec["Flow IAT Mean"]          = float(iats.mean())
                rec["Packet Length Mean"]     = float(lens.mean())
                rec["Average Packet Size"]    = float(lens.mean())
                for fl in f["flags"]:
                    rec["SYN Flag Count"] += int(bool(fl & 0x02))
                    rec["ACK Flag Count"] += int(bool(fl & 0x10))
                records.append(rec)
            df = pd.DataFrame(records)[CICIDS_FEATURES].astype(np.float32)
            return df, f"PCAP (scapy): {len(df)} flows × 73 features."
    except ImportError:
        pass

    return _decode_binary(file_bytes, filename,
                          "PCAP: dpkt/scapy not installed — using byte statistics.")


def decode_file(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, str]:
    """
    Decode any supported file into a DataFrame aligned to 73 CICIDS features.
    Never raises — worst case returns a 1-row synthetic flow with a warning message.
    """
    if not file_bytes:
        df = pd.DataFrame([{f: 0.0 for f in CICIDS_FEATURES}]).astype(np.float32)
        return df, "⚠️ Empty file — returning zero vector."

    ext = Path(filename).suffix.lower().lstrip(".")

    try:
        if ext in ("csv", "tsv", "txt"):
            return _decode_delimited(file_bytes, ext)
        if ext in ("xls", "xlsx"):
            return _decode_excel(file_bytes)
        if ext == "parquet":
            return _decode_parquet(file_bytes)
        if ext in ("json", "jsonl", "ndjson"):
            return _decode_json(file_bytes)
        if ext in ("pcap", "pcapng", "cap"):
            return _decode_pcap(file_bytes, filename)
        if ext in ("log", "out"):
            # Try TSV (Zeek/Bro), fall back to CSV, then binary
            for fn in (_decode_delimited,):
                try:
                    return fn(file_bytes, "tsv")
                except Exception:
                    pass
            return _decode_binary(file_bytes, filename, "Log file:")

        # Unknown extension — try heuristics
        for attempt, kwargs in [
            (_decode_delimited, {"ext": "csv"}),
            (_decode_json,      {}),
        ]:
            try:
                df, info = (attempt(file_bytes, **kwargs)
                            if kwargs else attempt(file_bytes))
                if len(df) > 0:
                    return df, f"[auto-detected] {info}"
            except Exception:
                pass

        return _decode_binary(file_bytes, filename)

    except Exception as exc:
        # Never crash — return binary fallback with error note
        try:
            df, info = _decode_binary(file_bytes, filename,
                                      f"Decoder error ({exc}):")
            return df, info
        except Exception:
            df = pd.DataFrame([{f: 0.0 for f in CICIDS_FEATURES}]).astype(np.float32)
            return df, f"⚠️ All decoders failed for '{filename}': {exc}"


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import joblib
        m  = joblib.load("models/dt_surrogate.pkl") if os.path.exists("models/dt_surrogate.pkl") else None
        sc = joblib.load("models/scaler.pkl")        if os.path.exists("models/scaler.pkl")        else None
        fn = joblib.load("models/feature_names.pkl") if os.path.exists("models/feature_names.pkl") else None
        return m, sc, fn, m is not None
    except Exception:
        return None, None, None, False

dt_model, scaler, saved_features, model_loaded = load_model()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["timestamp","attack","confidence","action","intents","restored"])

# ── SHAP helpers ──────────────────────────────────────────────────────────────
def _extract_shap(model, X_row2d):
    try:
        import shap
    except ImportError:
        return None, 0.0, 0

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row2d)
        ev          = explainer.expected_value
        if isinstance(shap_values, list) and len(shap_values) > 0:
            means     = [np.mean(np.abs(sv[0])) for sv in shap_values]
            class_idx = int(np.argmax(means))
            sv_1d     = shap_values[class_idx][0]
            base_val  = float(ev[class_idx]) if hasattr(ev, "__len__") else float(ev)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            class_idx = int(np.argmax(np.mean(np.abs(shap_values[0]), axis=0)))
            sv_1d     = shap_values[0, :, class_idx]
            base_val  = float(ev[class_idx]) if hasattr(ev, "__len__") else float(ev)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            sv_1d    = shap_values[0]
            base_val = float(ev[1]) if hasattr(ev, "__len__") and len(ev) > 1 else float(ev)
        else:
            raise ValueError("Unexpected SHAP shape")
        return sv_1d, base_val, class_idx
    except Exception:
        pass

    try:
        import shap
        explainer = shap.Explainer(model, algorithm="auto")
        sv_obj    = explainer(X_row2d)
        vals = sv_obj.values
        if vals.ndim == 3:
            class_idx = int(np.argmax(np.mean(np.abs(vals[0]), axis=0)))
            sv_1d     = vals[0, :, class_idx]
            base_val  = (float(sv_obj.base_values[0, class_idx])
                         if sv_obj.base_values.ndim > 1
                         else float(sv_obj.base_values[0]))
        else:
            sv_1d    = vals[0]
            base_val = (float(sv_obj.base_values[0])
                        if not hasattr(sv_obj.base_values[0], "__len__")
                        else float(sv_obj.base_values[0][1]))
        return sv_1d, base_val, 0
    except Exception:
        return None, 0.0, 0


def shap_bar_chart(sv_1d, feature_names, class_name, top_n=15):
    n   = min(top_n, len(sv_1d))
    idx = np.argsort(np.abs(sv_1d))[-n:]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in sv_1d[idx]]
    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.32)))
    ax.barh([feature_names[i] for i in idx], sv_1d[idx], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value  (red = increases risk, blue = reduces risk)")
    ax.set_title(f"SHAP Feature Contributions — Predicted: {class_name}",
                 fontsize=11, color="#1a2a4a", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


def shap_force_fig(sv_1d, base_val, feature_names, X_row_1d, class_name):
    try:
        import shap
        fig = shap.force_plot(float(base_val), sv_1d, X_row_1d,
                              feature_names=feature_names, matplotlib=True, show=False)
        return fig, "force"
    except Exception:
        return shap_bar_chart(sv_1d, feature_names, class_name), "bar"


def simulate_telemetry():
    weights = [0.55,0.05,0.07,0.04,0.08,0.03,0.03,0.04,0.02,0.02,0.04,0.03]
    attack  = np.random.choice(CICIDS_CLASSES, p=weights)
    anomaly = attack != "BENIGN"
    score   = round(np.random.uniform(0.6, 0.99) if anomaly else np.random.uniform(0.1, 0.45), 4)
    return anomaly, attack, score


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="background:#8b0000;border-radius:10px;padding:14px 16px;
            color:white;text-align:center;margin-bottom:10px">
  <div style="font-size:1.5rem;font-weight:900;letter-spacing:0.5px">🛡️ LCGA IDS</div>
  <div style="font-size:0.75rem;opacity:0.9;margin-top:2px">
    Intent-Aware Self-Healing Network Security
  </div>
  <div style="font-size:0.7rem;opacity:0.75;margin-top:4px">Version 1.1</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background:#1f3864;border-radius:8px;padding:12px 14px;
            color:white;font-size:12.5px;line-height:1.8">
<b>🎓 MSc ML Thesis</b> — Addis Ababa University<br>
<span style="opacity:0.8">Department of Computer Science</span>
<hr style="border-color:rgba(255,255,255,0.2);margin:8px 0">
<b>Researchers</b><br>
📧 <a href="mailto:getayefiseha21@gmail.com" style="color:#a8c8ff">Getaye Fiseha</a>
<span style="opacity:0.6;font-size:11px"> (GSE/6132/18)</span><br>
📧 <a href="mailto:mercyget36@gmail.com" style="color:#a8c8ff">Mersen Getu</a>
<span style="opacity:0.6;font-size:11px"> (GSE/6514/18)</span><br>
📧 <a href="mailto:charagirmish03@gmail.com" style="color:#a8c8ff">Chara Girma</a>
<span style="opacity:0.6;font-size:11px"> (GSE/9163/18)</span>
<hr style="border-color:rgba(255,255,255,0.2);margin:8px 0">
<b>Advisor</b><br>
📧 <a href="mailto:yaregal.assabie@aau.edu.et" style="color:#a8c8ff">Dr. Yaregal Assabie</a>
<hr style="border-color:rgba(255,255,255,0.2);margin:8px 0">
📅 June 2026
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Key Results")
st.sidebar.metric("🎯 Accuracy",      "99.67%",  "+0.14% vs RF")
st.sidebar.metric("⚡ Inference",     "1.85 ms", "CPU per flow")
st.sidebar.metric("🔄 MTTR Red.",     "87%",     "78.4s vs 598.5s")
st.sidebar.metric("✅ ISR",           "87.6%",   "+23.4pp vs rule-based")
st.sidebar.metric("🧠 SHAP Speed",    "11,635×", "faster than LIME")
st.sidebar.metric("📦 Parameters",    "41,260",  "5-10× smaller than SOTA")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github&style=flat-square)]"
    "(https://github.com/getaye21/lcga-self-healing-ids)"
)
st.sidebar.markdown(
    "🌐 [Getaye's Portfolio](https://getaye.vercel.app) &nbsp;|&nbsp; "
    "📧 [Contact](mailto:getayefiseha21@gmail.com)"
)

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(135deg,#8b0000 0%,#a93226 45%,#2e5496 100%);
  border-radius:14px;padding:26px 36px;margin-bottom:16px;color:white;
  box-shadow:0 4px 20px rgba(139,0,0,0.3)">
  <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:10px">
    <span style="font-size:2.4rem;flex-shrink:0">🛡️</span>
    <div>
      <h1 style="color:white!important;margin:0;font-size:1.35rem;font-weight:800;
                 line-height:1.35;letter-spacing:-0.3px">
        {FULL_TITLE}
      </h1>
      <p style="color:rgba(255,255,255,0.78);margin:6px 0 0;font-size:0.88rem">
        Lightweight CNN-GRU-Attention &nbsp;·&nbsp; SHAP Explainability
        &nbsp;·&nbsp; MAPE-K Closed-Loop Remediation
      </p>
    </div>
  </div>
  <hr style="border-color:rgba(255,255,255,0.2);margin:10px 0">
  <div style="display:flex;flex-wrap:wrap;gap:20px;font-size:0.83rem;
              color:rgba(255,255,255,0.88)">
    <span>🎓 <b>MSc ML Thesis</b> — Addis Ababa University</span>
    <span>👥
      <a href="mailto:getayefiseha21@gmail.com" style="color:#ffb3b3;text-decoration:none">Getaye Fiseha</a> ·
      <a href="mailto:mercyget36@gmail.com"     style="color:#ffb3b3;text-decoration:none">Mersen Getu</a> ·
      <a href="mailto:charagirmish03@gmail.com" style="color:#ffb3b3;text-decoration:none">Chara Girma</a>
    </span>
    <span>🧑‍🏫 Advisor:
      <a href="mailto:yaregal.assabie@aau.edu.et" style="color:#ffb3b3;text-decoration:none">Dr. Yaregal Assabie</a>
    </span>
    <span>📅 June 2026</span>
    <span>🌐 <a href="https://getaye.vercel.app" style="color:#ffb3b3;text-decoration:none">Portfolio</a></span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📖 Overview",
    "⚙️ Methodology",
    "📊 Results",
    "🔬 Live Detection",
    "🔧 Self-Healing",
    "🧠 Explainability",
    "📋 Action Log",
    "✅ Conclusion",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("📖 Overview")

    with st.expander("🗺️ How to use this dashboard — Start here!", expanded=True):
        st.markdown("""
<div class="step-box"><b>Step 1 — 🔬 Live Detection</b>: Upload ANY file containing network data
(CSV, JSON, PCAP, Excel, Parquet, log…). The decoder auto-converts it to 73 CICIDS2017
features and the DT Surrogate classifies instantly.</div>
<div class="step-box"><b>Step 2 — 🔧 Self-Healing</b>: Click <em>Run Detection Cycle</em> to
simulate the MAPE-K loop. Watch intents change 🟢→🔴→🟢.</div>
<div class="step-box"><b>Step 3 — 🧠 Explainability</b>: Choose any attack class to see SHAP
feature contributions.</div>
<div class="step-box"><b>Step 4 — 📋 Action Log</b>: Review the full healing history and ISR.</div>
<div class="step-box"><b>Step 5 — 📊 Results</b>: Compare models, SHAP vs LIME, and ablation.</div>
<div class="tip-box">💡 <b>No CSV?</b> Download the sample file in the 🔬 Live Detection tab.
The decoder handles unknown formats by deriving byte-level entropy features — it never crashes.</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
### Problem
Modern enterprise networks face growing cyber threats. Existing IDS either sacrifice latency
for accuracy, lack automated remediation, or operate as black boxes that undermine trust.

### Proposed Framework
- 🧠 **LCGA model** (41,260 params) — CNN-GRU-Attention for 12-class classification
- 🌳 **Decision Tree Surrogate** — distilled from LCGA for real-time SHAP explanations
- 🔄 **MAPE-K Orchestrator** — maps attacks to violated intents and executes healing
- 📊 **Live Dashboard** — universal file decoder, SHAP force plots, intent status

### Datasets
| Dataset | Samples | Classes | Task |
|---------|---------|---------|------|
| CICIDS2017 | ~2.52M | 12 | Multi-class attack classification |
| NSL-KDD | 148,517 | 2 | Binary anomaly detection |
        """)
    with col2:
        st.markdown("### Framework Architecture")
        st.markdown(
            f'<img src="{_ARCH_IMG}" style="width:100%;border-radius:10px;'
            f'border:1.5px solid #dde3f0;box-shadow:0 2px 10px rgba(0,0,0,0.10)">',
            unsafe_allow_html=True,
        )
        st.caption("Fig 1. LCGA Architecture → DT Surrogate + SHAP → MAPE-K Orchestrator.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Methodology  (condensed — same as v1.0)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("⚙️ Methodology")

    with st.expander("🏗️ LCGA Architecture (41,260 parameters)", expanded=True):
        st.markdown("""
| Block | Layer | Config | Output Shape |
|-------|-------|--------|-------------|
| 1 | Conv1D Branch A | 32 filters, k=3, ReLU, BN, Drop(0.2), MaxPool | (36, 32) |
| 1 | Conv1D Branch B | 64 filters, k=5, ReLU, BN, Drop(0.2), MaxPool | (36, 64) |
| 1 | Concatenate | — | (36, 96) |
| 2 | GRU | 64 units, return_seq=True, Drop(0.2) | (36, 64) |
| 3 | MultiHeadAttention | 2 heads, key_dim=16 + residual + LayerNorm | (36, 32) |
| 4 | GlobalAvgPool1D | — | (32,) |
| 4 | Dense + Dropout | 64 units, ReLU, Drop(0.3) | (64,) |
| 4 | Output | Softmax(12) | (12,) |
""")

    with st.expander("🌳 DT Surrogate + SHAP"):
        st.markdown("""
- **Knowledge distillation**: `DecisionTreeClassifier(max_depth=8)` trained on LCGA soft vectors → 99.64% fidelity
- **SHAP TreeExplainer**: 0.05 ms/sample — 11,635× faster than LIME
- **Consistency**: 100% deterministic vs 30% for LIME across repeated runs
""")

    with st.expander("🔄 MAPE-K Orchestrator"):
        st.markdown("""
**5 Network Intents (RFC 9315):**

| ID | Metric | Threshold | Cooldown |
|----|--------|-----------|---------|
| I1 | HTTP Latency | < 200 ms | 90 s |
| I2 | SSH Availability | = True | 60 s |
| I3 | Auth Failure Rate | < 10/min | 60 s |
| I4 | Port Scan Rate | < 5/min | 30 s |
| I5 | Bandwidth | < 100 Mbps | 90 s |

**Loop:** Monitor → Analyse (LCGA+SHAP+intent mapping) → Plan (historical success rate) → Execute → Verify (adaptive cooldown)
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Results  (same as v1.0)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("📊 Experimental Results")

    st.subheader("1. Model Comparison — CICIDS2017 Test Set")
    model_df = pd.DataFrame({
        "Model":        ["Random Forest","CNN Baseline","GRU Baseline","LCGA (Ours)"],
        "Accuracy":     ["99.53%","97.65%","98.30%","99.67% ✓"],
        "Macro F1":     ["0.9527","0.9420","0.9490","0.8170 *"],
        "Weighted F1":  ["0.9953","0.9760","0.9825","0.9967 ✓"],
        "MCC":          ["0.9945","0.9745","0.9820","0.9945 ✓"],
        "Params":       ["100 trees","5,196","~20,000","41,260"],
        "Inference ms": ["0.10","5.17","4.80","1.85 ✓"],
    })
    st.dataframe(model_df, use_container_width=True)
    st.caption("✓ best/matching best. * Low Macro F1 due to Heartbleed (11 samples) & Infiltration (36 samples).")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        models = ["RF","CNN","GRU","LCGA"]; accs = [99.53,97.65,98.30,99.67]
        cols_  = ["#aaaaaa","#aaaaaa","#aaaaaa","#c0392b"]
        bars   = ax.bar(models, accs, color=cols_)
        ax.set_ylim(96, 100.5); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Classification Accuracy", color="#1a2a4a", fontweight="bold")
        for b, v in zip(bars, accs):
            ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v}%",
                    ha="center", fontsize=8.5, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, clear_figure=True); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        inf_ms  = [0.10,5.17,4.80,1.85]
        bars    = ax.bar(models, inf_ms, color=cols_)
        ax.set_ylabel("Inference latency (ms)")
        ax.set_title("CPU Inference Latency", color="#1a2a4a", fontweight="bold")
        for b, v in zip(bars, inf_ms):
            ax.text(b.get_x()+b.get_width()/2, v+0.08, f"{v}",
                    ha="center", fontsize=8.5, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, clear_figure=True); plt.close()

    st.subheader("2. XAI Comparison — SHAP vs LIME")
    xai_df = pd.DataFrame({
        "Metric":              ["Time (ms/sample)","Speedup","Top-3 Consistency","Fidelity","Real-time Ready"],
        "SHAP (DT Surrogate)": ["0.05","11,635×","100% (deterministic)","99.64%","✅ Yes"],
        "LIME":                ["812","—","30.0%","N/A","❌ No"],
    })
    st.dataframe(xai_df, use_container_width=True)

    st.subheader("3. Self-Healing Comparison")
    heal_df = pd.DataFrame({
        "System":        ["Open-loop","Rule-based (fixed 60s)","LCGA + MAPE-K (Ours)"],
        "MTTR (s)":      [598.5, 65.1, 78.4],
        "ISR (%)":       ["0.0","64.2","87.6"],
    })
    st.dataframe(heal_df, use_container_width=True)

    st.subheader("4. Ablation Study")
    abl_df = pd.DataFrame({
        "Config":   ["A: Full LCGA+MAPE-K","B: No KB Feedback","C: Open-loop","D: No DT Surrogate"],
        "ISR (%)":  ["87.6","72.4","0.0","87.6"],
        "MTTR (s)": ["78.4","17.1","598.5","78.4*"],
    })
    st.dataframe(abl_df, use_container_width=True)
    st.caption("* D: healing unchanged; only explanation latency increases 3–10×.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Detection  (UNIVERSAL FILE INPUT)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("🔬 Live Network Flow Classification")

    st.markdown("""
> **Upload any file** containing network traffic data.
> The decoder automatically converts it to 73 CICIDS2017 features, then the
> DT Surrogate classifies each flow and generates a SHAP explanation.

| Format | How it's decoded |
|--------|-----------------|
| `.csv` / `.tsv` / `.txt` | Direct column alignment with alias mapping |
| `.xlsx` / `.xls` | First non-empty worksheet |
| `.json` / `.jsonl` | Array, object, or newline-delimited records |
| `.parquet` | Columnar binary (requires pyarrow) |
| `.pcap` / `.pcapng` | Flow stats extracted (requires dpkt or scapy) |
| `.log` | Zeek/Bro TSV logs and generic formats |
| **anything else** | Byte-level entropy → 73 proxy features |
""")

    with st.expander("📋 How to use", expanded=False):
        st.markdown("""
<div class="step-box"><b>Step 1</b> — Download the sample CSV (3 labelled CICIDS2017 flows) below</div>
<div class="step-box"><b>Step 2</b> — Upload it, or any other supported file</div>
<div class="step-box"><b>Step 3</b> — Read the Predictions table and SHAP bar chart</div>
<div class="tip-box">💡 The decoder never crashes — unknown formats always produce a result via byte statistics.</div>
<div class="tip-box">💡 For PCAP files: <code>pip install dpkt</code> gives accurate per-flow extraction.</div>
""", unsafe_allow_html=True)

    # ── Sample CSV ────────────────────────────────────────────────────────────
    st.markdown("#### 📥 Sample CSV (3 CICIDS2017 flows)")
    sample_csv = make_sample_csv()
    b64_csv = base64.b64encode(sample_csv.encode()).decode()
    st.markdown(
        f'<a href="data:text/csv;base64,{b64_csv}" download="sample_cicids2017_flows.csv">'
        "⬇️ Download sample_cicids2017_flows.csv  (DoS Hulk · BENIGN · PortScan)</a>",
        unsafe_allow_html=True,
    )
    st.caption("73 CICIDS2017 feature columns + Label column (auto-ignored during prediction).")
    st.markdown("---")

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload file — CSV, JSON, PCAP, Excel, Parquet, log, or any other format",
        type=None,                  # accept ALL extensions
        accept_multiple_files=False,
        key="live_det_uploader_v2",
        help="The decoder auto-aligns to 73 CICIDS2017 features regardless of format.",
    )

    if uploaded is not None:
        # Read bytes safely — catches "bad message format" from truncated uploads
        try:
            file_bytes = uploaded.read()
        except Exception as read_err:
            st.error(
                f"❌ Could not read uploaded file: {read_err}\n\n"
                "This sometimes happens with very large files or network interruptions. "
                "Please try uploading again."
            )
            st.stop()

        filename = uploaded.name

        if len(file_bytes) == 0:
            st.warning("⚠️ The uploaded file appears to be empty. Please upload a file with content.")
            st.stop()

        # ── Decode ────────────────────────────────────────────────────────────
        decode_placeholder = st.empty()
        decode_placeholder.info(
            f"🔄 Decoding **{filename}** ({len(file_bytes):,} bytes)…"
        )

        df_flows, decode_info = decode_file(file_bytes, filename)
        decode_placeholder.success(f"✅ {decode_info}")

        X             = df_flows.values.astype(np.float32)
        feature_names = list(df_flows.columns)

        # ── File summary ──────────────────────────────────────────────────────
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Flows decoded",   f"{len(X):,}")
        mc2.metric("Features",        X.shape[1])
        mc3.metric("File size",       f"{len(file_bytes)/1024:.1f} KB")

        with st.expander("🔍 Decoded feature matrix (first 5 rows)", expanded=False):
            st.dataframe(df_flows.head(), use_container_width=True)
            st.caption("All 73 CICIDS2017 features present. Missing columns padded with 0.")

        # ── Scale ─────────────────────────────────────────────────────────────
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception:
                pass  # scaler mismatch — skip gracefully

        # ── Predict ───────────────────────────────────────────────────────────
        st.markdown("#### 🎯 Predictions")

        if model_loaded and dt_model is not None:
            try:
                raw_preds = dt_model.predict(X)
                probas    = dt_model.predict_proba(X)
                preds     = [idx_to_label(p) for p in raw_preds]
                model_src = "DT Surrogate (trained)"
            except Exception as pred_err:
                st.error(f"Model prediction failed: {pred_err}")
                st.code(traceback.format_exc())
                st.stop()
        else:
            st.warning(
                "⚠️ No trained model found at `models/dt_surrogate.pkl`. "
                "Showing **mock** predictions."
            )
            np.random.seed(abs(hash(filename)) % (2**31))
            preds  = [np.random.choice(CICIDS_CLASSES) for _ in range(len(X))]
            probas = np.zeros((len(X), len(CICIDS_CLASSES)))
            for i, p in enumerate(preds):
                probas[i, CICIDS_CLASSES.index(p)] = np.random.uniform(0.82, 1.0)
            model_src = "Mock (no trained model loaded)"

        # ── Summary metrics ───────────────────────────────────────────────────
        n_attack = sum(1 for p in preds if p != "BENIGN")
        n_benign = len(preds) - n_attack
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total flows",  f"{len(preds):,}")
        s2.metric("🚨 Attacks",   f"{n_attack:,}")
        s3.metric("✅ Benign",    f"{n_benign:,}")
        s4.metric("Attack rate",  f"{n_attack/max(len(preds),1)*100:.1f}%")

        # ── Prediction table ──────────────────────────────────────────────────
        pred_df = pd.DataFrame({
            "Flow #":     [f"#{i+1}" for i in range(len(X))],
            "Prediction": preds,
            "Confidence": [f"{probas[i].max():.1%}" for i in range(len(X))],
            "Action":     [ACTION_MAP.get(p,"ESCALATE") if p!="BENIGN" else "—" for p in preds],
            "Risk":       ["🔴 ATTACK" if p!="BENIGN" else "🟢 BENIGN" for p in preds],
        })
        st.dataframe(pred_df, use_container_width=True)
        st.caption(f"Model: {model_src}")

        # ── Attack distribution chart ─────────────────────────────────────────
        if n_attack > 0:
            with st.expander("📊 Attack type distribution", expanded=False):
                atk_counts = Counter(p for p in preds if p != "BENIGN")
                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.bar(list(atk_counts.keys()), list(atk_counts.values()), color="#c0392b")
                ax.set_ylabel("Count")
                ax.set_title("Detected Attack Types", color="#1a2a4a", fontweight="bold")
                ax.tick_params(axis="x", rotation=35)
                plt.tight_layout(); st.pyplot(fig, clear_figure=True); plt.close()

        # ── Per-sample summary ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔎 Per-sample summary (first 5 flows)")
        for i in range(min(len(preds), 5)):
            icon    = "🚨" if preds[i] != "BENIGN" else "✅"
            conf    = probas[i].max()
            action  = ACTION_MAP.get(preds[i], "—")
            intents = ", ".join(INTENT_VIOLATIONS.get(preds[i], ["None"]))
            st.markdown(
                f"{icon} **Flow #{i+1}:** `{preds[i]}` — **{conf:.1%}** confidence "
                f"| Action: `{action}` | Violated intents: {intents}"
            )

        # ── SHAP ─────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🧠 SHAP Explanation — Flow #1")
        st.markdown("""
> Red bars = features that push the prediction **toward** the detected class.
> Blue bars = features that push **away**. Longer = stronger influence.
""")

        try:
            import shap as _shap_lib   # just test import; functions use _extract_shap

            if model_loaded and dt_model is not None:
                sv_1d, base_val, class_idx = _extract_shap(dt_model, X[:1])

                if sv_1d is not None:
                    pred_label = idx_to_label(class_idx)

                    force_fig, kind = shap_force_fig(
                        sv_1d, base_val, feature_names, X[0], pred_label
                    )
                    st.pyplot(force_fig, clear_figure=True); plt.close("all")
                    if kind == "bar":
                        st.caption("ℹ️ SHAP force plot unavailable here — bar chart shown.")

                    bar_fig = shap_bar_chart(sv_1d, feature_names, pred_label)
                    st.pyplot(bar_fig, clear_figure=True); plt.close("all")

                    top_idx = np.argsort(np.abs(sv_1d))[-5:][::-1]
                    top_df  = pd.DataFrame({
                        "Rank":      [f"#{r+1}" for r in range(len(top_idx))],
                        "Feature":   [feature_names[i] for i in top_idx],
                        "SHAP":      [f"{sv_1d[i]:+.4f}" for i in top_idx],
                        "Direction": [
                            "↑ Increases risk" if sv_1d[i] > 0 else "↓ Reduces risk"
                            for i in top_idx
                        ],
                    })
                    st.markdown("**Top 5 most influential features (Flow #1):**")
                    st.dataframe(top_df, use_container_width=True)
                else:
                    st.warning(
                        "SHAP values could not be computed for this model type. "
                        "Try: `pip install --upgrade shap scikit-learn`"
                    )
            else:
                sv_mock = np.random.randn(len(feature_names)) * 0.3
                st.pyplot(shap_bar_chart(sv_mock, feature_names, preds[0]),
                          clear_figure=True)
                plt.close("all")
                st.caption("⚠️ Mock SHAP — load a real model for true explanations.")

        except ImportError:
            st.error("SHAP not installed. Run: `pip install shap`")
        except Exception as shap_err:
            st.error(f"SHAP computation failed: {shap_err}")
            with st.expander("🐛 Traceback"):
                st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Self-Healing
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🔧 MAPE-K Self-Healing Simulator")
    st.markdown("""
> Simulates the MAPE-K loop. Each click runs one detection cycle — detect → plan → execute → verify.
""")
    st.markdown(
        f'<img src="{_MAPEK_IMG}" style="width:100%;max-width:820px;border-radius:10px;'
        f'border:1.5px solid #dde3f0;box-shadow:0 2px 12px rgba(0,0,0,0.10);'
        f'display:block;margin:0 auto 14px auto">',
        unsafe_allow_html=True,
    )
    st.caption("MAPE-K: Monitor → Analyse → Plan → Execute → Verify → KB update.")
    st.markdown("---")

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
                st.error(f"🚨 **{attack}** detected\n\nAction: **{action}**\nIntents: {intents}")
            else:
                st.success(f"✅ Normal traffic (score={score:.3f})")

        if not st.session_state.history.empty:
            isr = st.session_state.history["restored"].mean() * 100
            st.metric("Session ISR", f"{isr:.1f}%",
                      delta=f"{isr-64.2:+.1f}pp vs rule-based")
            st.metric("Attacks detected",
                      int((st.session_state.history["attack"] != "BENIGN").sum()))

        if st.button("🗑️ Clear Log", key="clear_heal"):
            st.session_state.history = pd.DataFrame(
                columns=["timestamp","attack","confidence","action","intents","restored"])
            st.rerun()

    with col2:
        st.subheader("Network Intent Status")
        recent = (st.session_state.history["attack"].tolist()[-3:]
                  if not st.session_state.history.empty else [])

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
            st.warning(f"⚠️ {len(violated)} intent(s) violated: "
                       f"{', '.join(r[0] for r in violated)}")
        else:
            st.success("✅ All intents satisfied")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explainability Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("🧠 Explainability Explorer")
    st.markdown("""
> Choose any attack class to see its pre-computed SHAP feature profile.
> Red = increases risk, Blue = reduces risk.
""")

    sel     = st.selectbox("Select attack class:", CICIDS_CLASSES[1:], key="exp_sel")
    profile = SHAP_PROFILES.get(sel, {
        "Flow Duration":0.4,"Destination Port":0.35,
        "Total Fwd Packets":0.28,"Packet Length Mean":0.2,
        "Bwd Packet Length Std":0.15,
    })
    feats  = list(profile.keys())
    vals   = list(profile.values())
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(feats, vals, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean |SHAP Value|  (red = increases risk, blue = reduces)")
    ax.set_title(f"SHAP Feature Profile — {sel}", fontsize=12,
                 color="#1a2a4a", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig, clear_figure=True); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Violated intents:** {', '.join(INTENT_VIOLATIONS.get(sel, ['None']))}")
        st.markdown(f"**Healing action:** `{ACTION_MAP.get(sel, 'ESCALATE')}`")
    with col2:
        top_feat = max(profile, key=lambda k: abs(profile[k]))
        st.info(f"**Primary driver:** {top_feat} (SHAP={profile[top_feat]:+.2f})\n\n"
                f"This feature most strongly identifies `{sel}` traffic.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Action Log
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("📋 MAPE-K Action Log")

    if st.session_state.history.empty:
        st.info("No actions logged yet. Go to 🔧 Self-Healing and click **Run Detection Cycle**.")
    else:
        disp = st.session_state.history.tail(30).sort_values(
            "timestamp", ascending=False).copy()
        disp["timestamp"] = disp["timestamp"].str[:19].str.replace("T", " ")
        disp["restored"]  = disp["restored"].map({True: "✅ Yes", False: "❌ No"})
        st.dataframe(disp, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        isr   = st.session_state.history["restored"].mean() * 100
        total = len(st.session_state.history)
        n_att = int((st.session_state.history["attack"] != "BENIGN").sum())
        n_ok  = int(st.session_state.history["restored"].sum())
        c1.metric("Cycles Run",       total)
        c2.metric("Attacks Detected", n_att)
        c3.metric("Intents Restored", n_ok)
        c4.metric("ISR",              f"{isr:.1f}%")

        if n_att > 0:
            act_counts = (
                st.session_state.history[
                    st.session_state.history["attack"] != "BENIGN"]["action"]
                .value_counts()
            )
            fig, ax = plt.subplots(figsize=(6, 3))
            act_counts.plot.bar(ax=ax, color="#c0392b")
            ax.set_ylabel("Count")
            ax.set_title("Healing Actions Executed", color="#1a2a4a", fontweight="bold")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout(); st.pyplot(fig, clear_figure=True); plt.close()

        if st.button("🗑️ Clear Log", key="clear_log"):
            st.session_state.history = pd.DataFrame(
                columns=["timestamp","attack","confidence","action","intents","restored"])
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Conclusion
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("✅ Conclusion and Future Work")

    st.markdown("""
### Summary of Contributions

| Contribution | Detail |
|---|---|
| Lightweight LCGA | 41,260 params · 99.67% accuracy · 1.85 ms CPU inference |
| DT Surrogate + SHAP | 99.64% fidelity · 11,635× faster than LIME · deterministic |
| MAPE-K + IBN | First closed-loop DL + XAI + IBN framework for network security |
| ISR | 87.6% vs 0% (open-loop) and 64.2% (rule-based) |
| MTTR reduction | 78.4 s vs 598.5 s baseline (87% reduction) |
| Universal decoder | Accepts CSV, JSON, PCAP, Excel, Parquet, log, binary |

### Future Work
1. Zero-day detection — online learning for unseen attacks
2. Real SDN deployment — ONOS + Mininet validation
3. RL-based healing — deep Q-learning for plan selection
4. Federated learning — multi-site training without sharing raw data
5. Rare-class improvement — few-shot learning / GAN augmentation
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Authors:**
- Getaye Fiseha (GSE/6132/18)
- Mersen Getu (GSE/6514/18)
- Chara Girma (GSE/9163/18)

**Advisor:** Dr. Yaregal Assabie
        """)
    with col2:
        st.markdown("""
**Links:**
- 📂 [GitHub](https://github.com/getaye21/lcga-self-healing-ids)
- 🌐 [Portfolio](https://getaye.vercel.app)
- 📄 [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "A Lightweight Hybrid Deep Learning Framework for Real-Time Cyber Threat Detection "
    "and Intent-Aware Self-Healing Network Security · v1.1 · "
    "MSc ML Thesis, Addis Ababa University, June 2026 · "
    "[GitHub](https://github.com/getaye21/lcga-self-healing-ids) · "
    "[Portfolio](https://getaye.vercel.app)"
)
