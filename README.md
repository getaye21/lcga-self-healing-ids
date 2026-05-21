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

🛡️ LCGA — Lightweight Self-Healing Intrusion Detection System
MSc Thesis — Addis Ababa University, Department of Computer Science

A Lightweight Hybrid Deep Learning Framework for Real-Time Cyber Threat Detection and Intent-Aware Self-Healing Network Security

👥 Authors
Name	ID	Role
Getaye Fiseha	GSE/6132/18	Lead
Mersen Getu	—	Co-investigator
Chara Girma	—	Co-investigator
Advisor: [Advisor Name], PhD

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
🚀 Quick Start
git clone https://github.com/getaye21/lcga-self-healing-ids.git
cd lcga-self-healing-ids
pip install -r requirements.txt
streamlit run app.py
Live Demo 🔗 https://lcga-self-healing-ids.streamlit.app

📁 Repository Structure lcga-self-healing-ids/ ├── app.py ← Streamlit dashboard ├── src/ │ ├── preprocessing/ ← NSL-KDD & CICIDS2017 preprocessors │ ├── models/ ← LCGA, ANN, GRU, Ensemble │ ├── xai/ ← DT Surrogate + SHAP + LIME │ ├── mape_k/ ← Knowledge Base + Orchestrator │ ├── evaluation/ ← Metrics + comparison tables │ └── utils/ ← Visualization utilities ├── config/ │ ├── config.yaml ← Hyperparameters │ └── intents.yaml ← Intent KB + action library ├── notebooks/ ← Kaggle notebooks (00-05) └── models/ ← Trained model artifacts 📦 Kaggle Notebooks

Notebook Link
00 Full EDA Kaggle 01 Preprocessing Kaggle 02 Baseline Models Kaggle 03 LCGA Training Kaggle 04 DT Surrogate + SHAP Kaggle 05 MAPE-K Evaluation Kaggle 📄 Citation @mastersthesis{fiseha2026lcga, title = {An Optimized Hybrid Deep Learning Framework for Real-Time Cyber Threat Detection and Self-Healing Network Security}, author = {Fiseha, Getaye and Getu, Mersen and Girma, Chara}, school = {Addis Ababa University}, year = {2026}, type = {MSc Thesis} } 📜 License MIT License

