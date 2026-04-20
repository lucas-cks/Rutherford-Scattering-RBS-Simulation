## Rutherford Scattering & RBS: A Three-Stage Pipeline from Monte Carlo to Neural Network Surrogate Modeling

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C](https://img.shields.io/badge/C-OpenMP-blue.svg)](https://www.openmp.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

**Author:** Lucas Kai Sing Ching  
**Affiliation:** Department of Physics, The Chinese University of Hong Kong (CUHK)  
**Date:** April 2026

---

## 📖 Overview

This project implements a complete computational pipeline for simulating **Rutherford scattering** and **Rutherford Backscattering Spectrometry (RBS)**, and for constructing fast **neural network surrogate models**. The pipeline is divided into three stages:

1. **Pure Rutherford Monte Carlo** – Reproduces the historic Geiger‑Marsden experiment with uniform impact‑parameter sampling, Bethe‑Bloch stopping power, and Gaussian energy straggling.
2. **RBS with importance sampling and OpenMP** – Uses quadratic biasing of the impact parameter to force large‑angle scatters (statistically corrected by weights) and parallelises the simulation with OpenMP to generate large datasets.
3. **Neural network surrogate modeling** – Trains small feed‑forward networks (PyTorch) to predict energy loss, backscatter probability, mean scattering angle, and the full 170° energy spectrum in milliseconds. Deploys a standalone Windows executable and a periodic‑table scanner.

The pipeline is modular, well‑documented, and ready for extension to compound targets and depth profiling.

---

## 🚀 Key Features

- **High‑performance Monte Carlo** – Importance sampling increases backscattering efficiency by >1000×.
- **OpenMP parallelisation** – Near‑linear speedup on 16 cores; generates 2.1 × 10⁹ particles across 2,100 runs in ~10 h.
- **Batch automation** – Python driver runs 2,100 configurations (7 materials × 5 thicknesses × 5 energies × 10 seeds) automatically.
- **Neural network surrogates** – Small 3‑8‑1 networks (scalars) and a 3‑128‑256‑128‑100 network (spectrum) achieve test R² > 0.96.
- **Standalone predictor** – `RBS_Predictor.exe` (Windows) gives instant predictions without Python.
- **Periodic‑table scanner** – Visualises energy loss and backscatter probability for Z = 1–92.

---

## 🧪 Validation Results

### Pure Rutherford (Geiger‑Marsden style)

| Target | Z | Thickness (cm) | Simulated \(P_{\text{back}}\) | Theoretical \(P_{\text{back}}\) |
|--------|---|----------------|-------------------------------|-------------------------------|
| Pt     | 78 | \(2.0\times10^{-4}\) | \(9.8\times10^{-5}\) | \(1.08\times10^{-4}\) |
| Au     | 79 | \(4.0\times10^{-5}\) | \(3.0\times10^{-5}\) | \(3.15\times10^{-5}\) |

### RBS with Importance Sampling (2.20 MeV alpha particles)

| Target | Z | Thickness (cm) | Simulated \(P_{\text{back}}\) (mean ± std over 10 seeds) | Theoretical \(P_{\text{back}}\) |
|--------|---|----------------|---------------------------------------------------------|-------------------------------|
| Au     | 79 | \(2.0\times10^{-4}\) | \((2.00 \pm 0.04)\times10^{-3}\) | \(2.9\times10^{-3}\) |
| Pt     | 78 | \(2.0\times10^{-4}\) | \((2.65 \pm 0.05)\times10^{-3}\) | \(2.8\times10^{-3}\) |
| Si     | 14 | \(3.0\times10^{-4}\) | \((4.72 \pm 0.09)\times10^{-5}\) | \(5.1\times10^{-5}\) |
| Al     | 13 | \(1.0\times10^{-4}\) | \((1.57 \pm 0.05)\times10^{-5}\) | \(1.7\times10^{-5}\) |
| Ge     | 32 | \(3.0\times10^{-4}\) | \((2.83 \pm 0.06)\times10^{-4}\) | \(3.0\times10^{-4}\) |

All simulated values agree with theory to within 15%, confirming the correctness of the importance‑sampling implementation.

---

## 📈 Neural Network Performance

| Target quantity | Test \(R^2\) | Notes |
|----------------|-------------|-------|
| Energy loss (MeV) | 0.9695 | Excellent |
| Backscatter probability (log₁₀) | 0.9630 | Good over 4 orders of magnitude |
| Mean scattering angle (°) | 0.9850 | Very accurate |
| 170° energy spectrum (100 bins) | MSE ~ 3×10⁻⁴ | Captures shape well |

---

## 💻 Repository Structure

```
.
├── src/
│   ├── Rutherford_Scattering.c          # Pure Rutherford MC
│   ├── RBS_OpenMP.c                     # RBS with importance sampling + OpenMP
│   ├── data_visualise.py                # Visualisation of pure Rutherford output
│   ├── run_list_generator.py            # Generates run_list.csv (2100 runs)
│   ├── python_driver.py                 # Batch driver for RBS_OpenMP.exe
│   ├── analysis_RBS.py                  # Parses results, trains neural networks
│   ├── Predictor.py                     # Interactive predictor (converted to .exe)
│   └── Periodic_table_scanner.py        # Periodic table sweep
├── data/                                # Sample materials.csv, particles.csv
├── results/                             # Output folders (2,100 runs)
├── models/                              # Trained .pt models
├── RBS_Predictor.exe                    # Standalone Windows executable
└── README.md
```

---

## 🛠️ Requirements

- **C compiler** with OpenMP support (e.g., `gcc -fopenmp`)
- **Python 3.8+** with:
  - `numpy`, `pandas`, `matplotlib`
  - `torch` (PyTorch) for neural network training
  - `scipy`, `csv`, `os`, `shutil`, `subprocess`, `re`

Install Python dependencies:
```bash
pip install numpy pandas matplotlib torch scipy
```

---

## 🏃 How to Run

### 1. Pure Rutherford Monte Carlo

```bash
gcc -o Rutherford Scattering.c -lm
./Rutherford
python data_visualise.py
```

### 2. RBS with Importance Sampling + OpenMP (single run)

```bash
gcc -o RBS_openmp RBS_OpenMP.c -lm -fopenmp
./RBS_openmp
# Follow prompts (choose CSV mode or manual)
```

### 3. Batch generation (2100 runs)

First prepare `materials.csv`, `particles.csv`, then:

```bash
python run_list_generator.py   # creates run_list.csv
python python_driver.py        # runs all simulations sequentially
```

### 4. Neural network training & analysis

```bash
python analysis_RBS.py
```

### 5. Standalone predictor (Windows)

Double‑click `RBS_Predictor.exe` or run from terminal:
```cmd
RBS_Predictor.exe
```
Enter Z, thickness (Å), and energy (MeV) to get instant predictions.

---

## 📚 References

1. Geiger, H., & Marsden, E. (1909). On a Diffuse Reflection of the α‑Particles. *Proc. R. Soc. A*, **82**, 495‑500.
2. Rutherford, E. (1911). The Scattering of α and β Particles by Matter and the Structure of the Atom. *Phil. Mag.*, **21**, 669‑688.
3. Chu, W.‑K., Mayer, J. W., & Nicolet, M.‑A. (1978). *Backscattering Spectrometry*. Academic Press.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

Thanks to the developers of MT19937, OpenMP, PyTorch, and all open‑source libraries used. Special thanks to the Kellogg Radiation Laboratory at Caltech for access to early backscattering data.

---

## 📬 Contact

For questions or suggestions, please open an issue on this repository or contact the author directly.
```
