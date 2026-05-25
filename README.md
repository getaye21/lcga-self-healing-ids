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

# LCGA Self-Healing IDS

**An Optimized Hybrid Deep Learning Framework for Real-Time Cyber Threat Detection and Self-Healing Network Security**

**MSc Thesis — Addis Ababa University, Department of Computer Science**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/Getaye/lcga-self-healing-ids)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/getaye21/lcga-self-healing-ids?style=social)](https://github.com/getaye21/lcga-self-healing-ids)

> A lightweight, explainable hybrid deep-learning framework for real-time cyber threat detection and intent-aware self-healing network security.

## Table of Contents
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Key Contributions](#key-contributions)
- [Live Demo](#live-demo)
- [Repository Structure](#repository-structure)
- [Kaggle Notebooks](#kaggle-notebooks)
- [Quick Start](#quick-start)
- [Authors](#authors)
- [Citation](#citation)
- [License](#license)

## Highlights

| Metric | Value |
|--------|-------|
| LCGA Accuracy | 99.67% |
| Model Parameters | 41,260 |
| MTTR Reduction | 87% vs open-loop |
| Intent Satisfaction Rate (ISR) | 87.6% |
| SHAP Speedup vs LIME | ~11,635× |
| DT Surrogate Fidelity | 99.64% |

## Architecture

```text
Network Traffic
   ↓
LCGA Detector (~41K params)
Conv1D → GRU → Attention
Multi-class classification (12 CICIDS2017 classes)
   ↓
DT Surrogate + SHAP
Real-time explanations
   ↓
MAPE-K Orchestrator
Monitor → Analyze → Plan → Execute → Verify
   ↓
Healing Action
BLOCK_IP / RATE_LIMIT / ...
```

## Key Contributions

1. **LCGA**: A lightweight CNN-GRU-Attention model with 41K parameters, achieving 99.67% accuracy on CICIDS2017.
2. **DT Surrogate + SHAP**: Real-time explanations via a distilled Decision Tree surrogate, with major speed gains over LIME.
3. **MAPE-K Closed Loop**: Adaptive verification windows reduce MTTR while improving self-healing behavior.
4. **Reproducibility**: Code, experiments, and demo are organized for easy reuse.

## Live Demo

Try the interactive dashboard on Hugging Face Spaces: [lcga-self-healing-ids](https://huggingface.co/spaces/Getaye/lcga-self-healing-ids)

## Repository Structure

```text
lcga-self-healing-ids/
├── app.py
├── train.py
├── requirements.txt
├── config/
│   ├── config.yaml
│   └── intents.yaml
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── xai/
│   ├── mape_k/
│   ├── evaluation/
│   └── utils/
├── dashboard/
│   └── app.py
├── notebooks/
├── models/
└── results/
```

## Kaggle Notebooks

| # | Notebook | Link |
|---|----------|------|
| 00 | Full EDA | [Kaggle](https://www.kaggle.com/code/getayefiseha/eda-for-nlskdd-cic-ids-ipynb) |
| 01 | Preprocessing | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-01-preprocessing-for-model-training) |
| 02 | Baseline Models | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-02-baseline-models) |
| 03 | LCGA Training | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-03-lcga-training) |
| 04 | DT Surrogate + SHAP + LIME | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-04-dt-surrogate-shap-lime-explain) |
| 05 | MAPE-K Orchestrator + Evaluation | [Kaggle](https://www.kaggle.com/code/getayefiseha/notebook-05-mape-k-orchestrator-system-evaluat) |

## Quick Start

```bash
git clone https://github.com/getaye21/lcga-self-healing-ids.git
cd lcga-self-healing-ids
pip install -r requirements.txt
streamlit run app.py
```

## Authors

- **Getaye Fiseha** — Lead
- **Mersen Getu** — Co-investigator
- **Chara Girma** — Co-investigator
- **Advisor:** Dr. Yaregal A.
- **Institution:** Addis Ababa University, 2026

## Citation

```bibtex
@mastersthesis{fiseha2026lcga,
  title   = {A Lightweight Hybrid Deep Learning Framework for Real-Time Cyber Threat Detection and Intent-Aware Self-Healing Network Security},
  author  = {Fiseha, Getaye and Getu, Mersen and Girma, Chara},
  school  = {Addis Ababa University},
  year    = {2026},
  type    = {MSc Thesis}
}
```

## License

MIT License. See the [LICENSE](LICENSE) file.
