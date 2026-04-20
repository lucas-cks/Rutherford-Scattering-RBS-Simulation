# Rutherford Backscattering Spectrometry (RBS) Pipeline
### From First-Principles Monte Carlo to Deep Learning Surrogate Models

This repository contains a comprehensive three-stage computational framework for simulating Rutherford scattering and Rutherford Backscattering Spectrometry (RBS). Developed for **PHYS3061** at **The Chinese University of Hong Kong (CUHK)**, this project demonstrates the integration of high-performance physics simulations with modern machine learning techniques.

---

## 🚀 Overview

The project is structured into three distinct stages, following the evolution of experimental nuclear physics to real-time predictive modeling:

### Stage 1: Classical Monte Carlo Simulation
A C-based engine that reproduces the historic **Geiger–Marsden experiment**.
* **Physics Core:** Implements uniform impact parameter sampling, **Bethe-Bloch stopping power**, and **Gaussian energy straggling**.
* **Validation:** Successfully validated against Rutherford’s theoretical predictions for Platinum ($Z=78$) and Gold ($Z=79$) targets.

### Stage 2: High-Performance RBS Simulation
Extended simulation optimized for rare backscattering events.
* **Importance Sampling:** Utilizes quadratic biasing of the impact parameter to increase sampling efficiency by **1000x**.
* **Parallel Computing:** Powered by **OpenMP**, enabling the generation of **2.1 billion particles** across 2,100 independent runs in approximately 10 hours.
* **Data Generation:** Produces a high-fidelity dataset for energy loss, backscatter probability, and spectral distribution.

### Stage 3: Neural Network Surrogate Modeling
Transitions from slow simulations to millisecond-latency AI predictions.
* **Architecture:** Deep Neural Networks (DNN) trained on the Stage 2 dataset.
* **Performance:** Achieved a test accuracy of **$R^2 > 0.96$** for scalar predictions.
* **Applications:** Includes a **Periodic-Table Scanner** and a standalone Windows executable for real-time RBS analysis.

---

## 🛠️ Tech Stack

- **Simulation:** C (Standard C11)
- **Parallelization:** OpenMP
- **Analysis & AI:** Python 3, NumPy, PyTorch / TensorFlow
- **Visualization:** Matplotlib

---

## 📊 Key Results

| Parameter | Simulation Result | Theoretical Value |
|-----------|-------------------|-------------------|
| Pt Backscatter Prob. | $9.8 \times 10^{-5}$ | $9.9 \times 10^{-5}$ |
| Au Backscatter Prob. | $3.0 \times 10^{-5}$ | $3.1 \times 10^{-5}$ |
| NN Prediction Accuracy | $R^2 = 0.96+$ | N/A |

> **Note:** The simulation successfully captures the $1/\sin^4(\theta/2)$ angular dependence and energy degradation through thin metallic foils.

---

## 📂 Directory Structure

```text
├── src/
│   ├── rutherford_core.c       # Pure MC simulation
│   ├── rbs_importance.c        # RBS with Importance Sampling & OpenMP
│   └── nn_surrogate.py         # Neural Network training & inference
├── docs/
│   └── 3061_Final_Report.pdf   # Full academic report
├── data/                       # Generated datasets (sample)
└── results/                    # Validation plots and NN performance metrics

## References
* [1] Geiger & Marsden (1909) - Historic experiment context.
* [2] Rutherford (1911) - Theoretical scattering model.
* [3] Chu et al. (1978) - RBS methodology and physics.
* [4] Heinzelmann et al. - Specific physical data/methodology.

## Image Credits
* All other diagrams and illustrations used in the report are credited as per the list in `docs/3061_Final_Report.pdf`.
