---
title: LCGA Self-Healing IDS
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
---
lcga-self-healing-ids
An Optimized Hybrid Deep Learning Framework for Real-Time Cyber Threat Detection and Self-Healing Network Security

# 🛡️ LCGA — Lightweight Self‑Healing Intrusion Detection System

**MSc Thesis — Addis Ababa University, Department of Computer Science**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/Getaye/lcga-self-healing-ids)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/getaye21/lcga-self-healing-ids?style=social)](https://github.com/getaye21/lcga-self-healing-ids)

> A lightweight, explainable hybrid deep‑learning framework for real‑time cyber threat detection and intent‑aware self‑healing network security.

---

## 📊 Highlights

| Metric | Value |
|--------|-------|
| **LCGA Accuracy** | 99.67% |
| **Model Parameters** | 41,260 |
| **MTTR Reduction** | 87% vs open‑loop |
| **Intent Satisfaction Rate (ISR)** | 87.6% |
| **SHAP Speedup vs LIME** | ~11,635× |
| **DT Surrogate Fidelity** | 99.64% |

---

## 🏗️ Architecture
Network Traffic
│
▼
┌─────────────────────────────────┐
│ LCGA Detector (~41K params) │
│ Conv1D → GRU → Attention │
│ Multi-class (12 CICIDS2017) │
└─────────────┬───────────────────┘
│ attack_class + confidence
▼
┌─────────────────────────────────┐
│ DT Surrogate + SHAP │
│ Real-time explanations │
└─────────────┬───────────────────┘
│ violated intents + rule path
▼
┌─────────────────────────────────┐
│ MAPE-K Orchestrator │
│ Monitor → Analyze → Plan │
│ Execute → Verify (closed loop)│
└─────────────┬───────────────────┘
│
▼
Healing Action
(BLOCK_IP / RATE_LIMIT / ...)

---

## 🧪 Key Contributions

1. **LCGA**: A lightweight CNN‑GRU‑Attention model with only 41 k parameters, achieving 99.67 % accuracy on CICIDS2017.
2. **DT Surrogate + SHAP**: Real‑time explanations via a distilled Decision Tree surrogate, 11 635× faster than LIME.
3. **MAPE‑K Closed‑Loop Self‑Healing**: Adaptive verification windows reduce MTTR by 87 % while achieving 87.6 % ISR.
4. **Fully Reproducible**: All code, data, experiments, and the live demo are open‑source.

---

## 🚀 Live Demo

Try the interactive dashboard: [lcga-self-healing-ids on Hugging Face Spaces](https://huggingface.co/spaces/Getaye/lcga-self-healing-ids)

---

## 📁 Repository Structure
lcga-self-healing-ids/
├── app.py ← HF Spaces / Streamlit entry
├── train.py ← Full training pipeline
├── requirements.txt
├── config/
│ ├── config.yaml ← Hyperparameters
│ └── intents.yaml ← Intent KB + action library
├── src/
│ ├── preprocessing/ ← NSL-KDD & CICIDS2017 preprocessors
│ ├── models/ ← LCGA, ANN, GRU, Ensemble
│ ├── xai/ ← DT Surrogate + SHAP + LIME
│ ├── mape_k/ ← Knowledge Base + Orchestrator
│ ├── evaluation/ ← Metrics + tables + stats
│ └── utils/ ← Visualization utilities
├── dashboard/
│ └── app.py ← Streamlit dashboard (5 panels)
├── notebooks/ ← Kaggle notebooks (00-05)
├── models/ ← Trained model artifacts
└── results/ ← Plots, tables, logs

---

## 📦 Kaggle Notebooks

| # | Notebook | Link |
|---|----------|------|
| 00 | Full EDA | [Kaggle](https://www.kaggle.com/code/getayefiseha/eda-for-nlskdd-cic-ids-ipynb) |
| 01 | Preprocessing | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-01-preprocessing-for-model-training) |
| 02 | Baseline Models | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-02-baseline-models) |
| 03 | LCGA Training | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-03-lcga-training) |
| 04 | DT Surrogate + SHAP + LIME | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-04-dt-surrogate-shap-lime-explain) |
| 05 | MAPE‑K Orchestrator + System Evaluation | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-05-mape-k-orchestrator-system-evaluat) |

---

## 🛠️ Quick Start

```bash
git clone https://github.com/getaye21/lcga-self-healing-ids.git
cd lcga-self-healing-ids
pip install -r requirements.txt
streamlit run app.py
👥 Authors
Name	ID	Role
Getaye Fiseha	GSE/6132/18	Lead
Mersen Getu	—	Co‑investigator
Chara Girma	—	Co‑investigator
Advisor: Dr. Yaregal A.
Institution: Addis Ababa University, 2026

📄 Citation
@mastersthesis{fiseha2026lcga,
  title   = {A Lightweight Hybrid Deep Learning Framework for Real-Time
             Cyber Threat Detection and Intent-Aware Self-Healing Network Security},
  author  = {Fiseha, Getaye and Getu, Mersen and Girma, Chara},
  school  = {Addis Ababa University},
  year    = {2026},
  type    = {MSc Thesis}
}

🏗️ Framework Overview
Network Traffic → LCGA Detector (~48K params) → DT Surrogate + SHAP → MAPE-K Orchestrator → Healing Action

Key Components
LCGA: Lightweight CNN-GRU-Attention classifier (Conv1D → GRU → Multi-Head Self-Attention)
DT Surrogate + SHAP: Explainable AI via Decision Tree distillation and SHAP TreeExplainer
MAPE-K Loop: Monitor → Analyze → Plan → Execute → Knowledge feedback
Real-Time Dashboard: Streamlit app with live monitoring, intent status, SHAP explanations
📊 Datasets
Dataset	Samples	Classes	Use
CICIDS2017	~2.5M	15	Multi-class attack classification
NSL-KDD	148,517	2 (binary)	Binary anomaly detection
🏆 Results
Model	Macro F1	Params	Inference (ms)
Random Forest	0.953	100 trees	—
CNN Baseline	0.942	5,196	5.2
GRU Baseline	0.949	~20K	—
LCGA (ours)	0.968	~48K	< 2
MTTR Reduction: 63% vs open-loop baseline
Intent Satisfaction Rate (ISR): 88%
User Trust (TAS): 4.7/5.0 with SHAP explanations

📦 Kaggle Notebooks

#	Notebook	Link
00	Full EDA	Kaggle
01	Preprocessing	Kaggle
02	Baseline Models	Kaggle
03	LCGA Training	Kaggle
04	DT Surrogate + SHAP	Kaggle
05	MAPE-K Evaluation	Kaggle

📜 License
MIT License
