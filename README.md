# Monte Carlo Simulation of Rutherford Scattering & RBS
## A Three‑Stage Pipeline from Monte Carlo to Neural Network Predictions

**Author:** Lucas Kai Sing Ching  

[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=lucas-cks&layout=compact&theme=vision-friendly-dark)](https://github.com/anuraghazra/github-readme-stats)

![C](https://img.shields.io/badge/Language-C-blue?logo=c)
![Python](https://img.shields.io/badge/Language-Python-yellow?logo=python)

![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green?logo=openmp)

![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-ffffff?logo=matplotlib&logoColor=black)
![PyTorch](https://img.shields.io/badge/Library-PyTorch-EE4C2C?logo=pytorch&logoColor=white)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Table of Contents
1. [Overview](#overview)
2. [Physics Background](#physics-background)
3. [Implementation Details](#implementation-details)
4. [Repository Structure](#repository-structure)
5. [Code Structure](#code-structure)
6. [Installation & Usage](#installation--usage)
7. [Key Results & Validation](#key-results--validation)
8. [Neural Network Surrogate Modeling](#neural-network-surrogate-modeling)
9. [Performance & Limitations](#performance--limitations)
10. [License](#license)
11. [References](#references)
12. [Contact](#contact)

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Overview
This repository implements a validated, modular three‑stage computational pipeline to simulate Rutherford scattering and Rutherford backscattering spectrometry (RBS), generate large, physics‑grounded datasets, and train/ship fast neural‑network surrogates.

**Pipeline stages:**

1. **Pure Rutherford Monte Carlo (C)** – Reproduces the Geiger‑Marsden experiment with uniform impact‑parameter sampling, Bethe‑Bloch stopping, and Gaussian straggling. Validated against theory.
2. **Importance‑sampled RBS (C + OpenMP)** – Quadratic bias \(b = b_{\max}u^2\) with weight \(w = 4u^3\) to efficiently sample rare backscattering events. Batch automation produced 2,100 runs (total \(2.1\times10^9\) weighted particles) in ≈10 h on 16 cores.
3. **Neural‑network surrogates (Python/PyTorch)** – Small scalar models (3‑8‑1) and a spectrum model (3‑128‑256‑128‑100) to predict energy loss, backscatter probability, mean scattering angle, and full 170° spectra. Deployed as a standalone Windows executable (`RBS_Predictor.exe`) and a periodic‑table scanner.

Designed for reproducibility, extensibility (compound targets, depth profiling), and rapid inference in research or educational settings.

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Physics Background
Implemented physics and modelling choices (following Chu, Mayer, Nicolet 1978):

- **Rutherford elastic Coulomb scattering** – differential cross‑section \(d\sigma/d\Omega \propto 1/\sin^4(\theta/2)\) in the heavy‑target limit.
- **Kinematic factor \(K\)** – lab‑frame energy transfer: \(E_1 = K E_0\).
- **Bethe‑Bloch stopping power** – continuous in‑layer energy loss; Bragg’s rule for compounds.
- **Gaussian energy straggling** – Bohr variance \(\Omega_{\mathrm{B}}^2 = 4\pi (Z_1 e^2)^2 Z_2 N t\).
- **Depth → energy conversion** – \(\Delta E = [\epsilon] N t\) for areal density estimation.
- **Thin‑target Rutherford backscatter probability** – \(P_{\mathrm{back}} \approx N t \sigma_{\mathrm{back}}(E_{\mathrm{eff}})\) with energy‑loss evaluated along the path.

Primary phenomena reproduced:
- Angular distributions consistent with Rutherford \(\csc^4(\theta/2)\) scaling.
- Backscatter probabilities consistent with theoretical cross‑sections when energy loss is accounted for.
- Spectral broadening from energy straggling and depth distributions.

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Implementation Details
- **Languages:** C for high‑performance simulation cores and importance‑sampled RBS; Python for drivers, analysis, dataset aggregation, and ML.
- **RNG:** Per‑thread MT19937 for OpenMP runs (seeded `master_seed + thread_id`).
- **Importance sampling:** sample \(b = b_{\max} \cdot u^2\), weight \(w = 4 u^3\). Weighted tallies preserve unbiased estimates while vastly increasing large‑angle statistics.
- **Parallelisation:** OpenMP with private thread accumulators, dynamic scheduling, and final reduction to global results.
- **Output formats:** Per‑run CSV files with weighted histograms (energy bins, angular bins), scalar summaries, and run metadata.
- **Batch automation:** Python driver generates run lists and dispatches simulator instances; supports resume and multi‑worker execution.

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Repository Structure
```
.
├── docs/
│   ├── Backscattering-Spectrometry-Wei-Kan-Chu-James-W-Mayer-and-Marc-A-Nicolet-Academic-Press-1978.pdf
│   ├── Geiger H. & Marsden E. (1909).pdf
│   ├── Rutherford E. (1911).pdf
│   └── report.pdf
├── stage_1_rutherford_scattering/
│   ├── rutherford_scattering.c
│   ├── rutherford_scattering.exe
│   ├── data_visualise.py
│   ├── materials.csv
│   ├── particles.csv
│   ├── Result_Z78_T2.00e-04_N1000000_E6.96MeV/
│   │   ├── angle_distribution.png
│   │   ├── energy_spectrum.png
│   │   ├── heatmap.png
│   │   ├── histogram.csv
│   │   └── simulation_results.txt
│   └── Result_Z79_T4.00e-05_N1000000_E5.52MeV/
├── stage_2_RBS/
│   ├── RBS_OpenMP.c
│   ├── RBS_openmp.exe
│   ├── energy_spectrum_170.csv
│   ├── histogram.csv
│   ├── materials.csv
│   ├── particles.csv
│   └── simulation_results.txt
├── stage_3_data_factory/
│   ├── RBS_openmp.exe
│   ├── python_driver.py
│   ├── run_list_generator.py
│   ├── run_list.csv
│   ├── materials.csv
│   ├── particles.csv
│   └── Results.zip
├── stage_4_neural_network/
│   ├── Analysis_Results/                     # 100+ plots and CSV aggregates
│   ├── Periodic_table_scanner.py
│   ├── Predictor.py
│   ├── analyse_RBS.py
│   ├── Z_sweep_validation.png
│   └── ...
├── LICENSE
└── README.md
```

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Code Structure
High‑level flow of the C simulation:

```text
main()
├── parse_args_and_config()
├── init_simulation_parameters()
├── init_rng_per_thread()
├── # parallel OpenMP region
│   └── for each particle:
│       ├── sample_impact_parameter()   # biased or uniform
│       ├── propagate_through_layers()  # stopping, straggling
│       ├── attempt_scatter()           # Rutherford kinematics if event occurs
│       ├── assign_weight_if_biased()
│       └── accumulate_private_tallies()
└── reduction_and_write_output()          # weighted sums, histograms, scalars
```

Auxiliary Python workflow:

- `run_list_generator.py` – builds full sweep of (Z, thickness, energy, seeds)
- `python_driver.py` – parallel dispatcher, collects outputs
- `analyse_RBS.py` – parses results, trains neural networks, generates plots
- `Predictor.py` – interactive predictor (converted to `.exe`)
- `Periodic_table_scanner.py` – periodic table sweep

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Installation & Usage

### Prerequisites
- C compiler with OpenMP support (gcc/clang)
- Python 3.8+
- Python packages: `numpy`, `pandas`, `matplotlib`, `torch`, `scipy`

### Clone
```bash
git clone https://github.com/lucas-cks/Rutherford-Scattering-RBS-Simulation-Surrogate-Modelling.git
cd Rutherford-Scattering-RBS-Simulation-Surrogate-Modelling
```

### Python environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib torch scipy
```

### Compile simulators
```bash
cd stage_1_rutherford_scattering
gcc -O3 -o rutherford_scattering rutherford_scattering.c -lm

cd ../stage_2_RBS
gcc -O3 -fopenmp -o RBS_openmp RBS_OpenMP.c -lm
```

### Run pure Rutherford (Stage 1)
```bash
cd stage_1_rutherford_scattering
./rutherford_scattering
# Enter parameters (or use CSV mode)
python data_visualise.py
```

### Run importance‑sampled RBS (Stage 2)
```bash
cd ../stage_2_RBS
./RBS_openmp
# Follow prompts (CSV mode recommended)
```

### Batch generation (Stage 3)
```bash
cd ../stage_3_data_factory
python run_list_generator.py      # creates run_list.csv
python python_driver.py           # runs all simulations sequentially
```

### Neural network training & analysis (Stage 4)
```bash
cd ../stage_4_neural_network
python analyse_RBS.py
```
This will parse all result folders (from `stage_3_data_factory/Results/`), generate plots, train the surrogate models, and save them as `.pt` files.

### Standalone predictor
- Use `Predictor.py` interactively:
  ```bash
  python Predictor.py
  ```
- Or run the pre‑built `RBS_Predictor.exe` (Windows) after moving the `.pt` models into a `models/` subfolder.

### Periodic table scanner
```bash
python Periodic_table_scanner.py
```
Produces `Z_sweep_validation.png` showing energy loss and backscatter probability for Z = 1–92.

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Key Results & Validation

### Pure Rutherford Monte Carlo validation (Geiger‑Marsden style)

**Platinum (Z = 78; E₀ = 6.96 MeV; t = 2.0×10⁻⁴ cm; 1×10⁶ particles)**

| Quantity | Value |
| :--- | :---: |
| Mean final energy (MeV) | 5.5041 |
| Energy straggling σ (MeV) | 0.0238 |
| Mean scattering angle (°) | 3.8944 |
| Backscatter count (sim) | 98 |
| Simulated \(P_{\text{back}}\) | \(9.8\times10^{-5}\) |
| Theoretical \(P_{\text{back}}\) | \(1.08\times10^{-4}\) |

**Gold (Z = 79; E₀ = 5.52 MeV; t = 4.0×10⁻⁵ cm; 1×10⁶ particles)**

| Quantity | Value |
| :--- | :---: |
| Mean final energy (MeV) | 5.2452 |
| Energy straggling σ (MeV) | 0.0100 |
| Mean scattering angle (°) | 1.8326 |
| Backscatter count (sim) | 30 |
| Simulated \(P_{\text{back}}\) | \(3.0\times10^{-5}\) |
| Theoretical \(P_{\text{back}}\) | \(3.15\times10^{-5}\) |

Agreement: simulated backscatter probabilities agree with Rutherford‑theory‑based calculations within ≈10% for validation cases above.

### Importance‑sampled RBS batch
- **Sweep:** Z ∈ {79,78,13,14,6,8,32}; energies = {1.54,2.20,2.72,3.30,4.00} MeV; six thicknesses; seeds 1–10.
- **Runs:** 2,100 independent runs (10 seeds each); weighted particles simulated ≈ 2.1×10⁹.
- **Wall time:** ≈10 hours on a 16‑core machine.
- **Validation:** Simulated \(P_{\text{back}}\) matches theoretical \(P_{\text{back}}(E_{\text{eff}})\) within ≲15% across tested targets and thicknesses (energy‑loss corrections integrated numerically).

**Example validation table (2.20 MeV alpha particles, θ = 170°):**

| Target | Z | Thickness (cm) | Simulated \(P_{\text{back}}\) (mean ± std) | Theoretical \(P_{\text{back}}\) |
|--------|---|----------------|-------------------------------------------|-------------------------------|
| Au     | 79 | \(2.0\times10^{-4}\) | \((2.00 \pm 0.04)\times10^{-3}\) | \(2.9\times10^{-3}\) |
| Pt     | 78 | \(2.0\times10^{-4}\) | \((2.65 \pm 0.05)\times10^{-3}\) | \(2.8\times10^{-3}\) |
| Si     | 14 | \(3.0\times10^{-4}\) | \((4.72 \pm 0.09)\times10^{-5}\) | \(5.1\times10^{-5}\) |
| Al     | 13 | \(1.0\times10^{-4}\) | \((1.57 \pm 0.05)\times10^{-5}\) | \(1.7\times10^{-5}\) |
| Ge     | 32 | \(3.0\times10^{-4}\) | \((2.83 \pm 0.06)\times10^{-4}\) | \(3.0\times10^{-4}\) |

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Neural Network Surrogate Modeling

### Data & preprocessing
- **Aggregated dataset:** scalar targets (energy loss, backscatter probability, mean scattering angle) and 170° energy spectra (100 bins).
- **Inputs:** Z, thickness (Å), incident energy (MeV). Inputs normalised (zero mean, unit variance); backscatter probability \(\log_{10}\)-transformed before normalisation.
- **Scalar training:** uses all per‑seed samples (≈2,100 examples) to preserve variability.
- **Spectrum model:** trained on averaged spectra per configuration (~210 samples).

### Architectures

**Scalar networks (SmallNN):**
- Input: 3
- Hidden: 8 neurons (ReLU)
- Output: 1

**Spectrum network (SpectrumNN):**
- Input: 3
- Hidden: 128 → 256 → 128 (ReLU)
- Output: 100 (linear)

### Training
- Optimiser: Adam (lr = 1e‑3)
- Loss: MSE (future: Poisson deviance for spectra)
- Split: 80/20 train/test
- Epochs: 500–800
- Batch size: 32

### Performance
- **Scalars:** test \(R^2\) ≈ 0.97–0.99 (energy loss, \(\log_{10} P_{\text{back}}\), mean scattering angle).
- **Spectrum:** captures overall 170° energy shape; larger MSE due to scarcity of exact backscatter events – training on more per‑seed spectra or using a different loss could improve fidelity.

### Deployment
- Models and normalisers saved in `Analysis_Results/` as `.pt` + metadata.
- `RBS_Predictor.exe` (PyInstaller) bundles scalar predictors for interactive use.
- `Periodic_table_scanner.py` demonstrates learned monotonic trends across Z = 1–92 while shading the training domain Z = 6–79.

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Performance & Limitations

### Performance
- **Importance sampling** increases effective sampling for large‑angle scatters by >1000×.
- **OpenMP** delivers near‑linear scaling on multi‑core CPUs; reported batch completed in ≈10 h on 16 cores.
- **Surrogates** provide millisecond inference; training benefits from GPU but small models are fast on CPU.

### Known limitations
- Rutherford theory assumes point‑like Coulomb scattering; screening, nuclear reactions, and high‑velocity corrections are not included.
- Bethe‑Bloch stopping may break down at very low velocities (< 0.1 MeV/u).
- Surrogates are only valid within training ranges: Z = 6–79, thickness 4,000–50,000 Å, energy 1.54–4.00 MeV – do not extrapolate without validation.
- No uncertainty quantification (point estimates only).
- Spectrum network trained on averaged spectra; per‑seed training or tailored loss functions could improve spectral fidelity.

### Future work / extensions
- Add **compound targets** (stoichiometric inputs), multi‑layer profiles.
- Integrate with **Geant4** or GPU‑accelerated Monte Carlo for complex geometries and speed‑ups.
- Add **Bayesian neural nets** or MC‑dropout for uncertainty quantification.
- Use **Poisson deviance** or count‑aware losses for spectrum learning.

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## References
1. Geiger, H., & Marsden, E. (1909). On a Diffuse Reflection of the α‑Particles. *Proc. R. Soc. Lond. A*, **82**, 495‑500. (PDF available in `docs/`)
2. Rutherford, E. (1911). The Scattering of α and β Particles by Matter and the Structure of the Atom. *Phil. Mag.*, **21**, 669‑688. (PDF available in `docs/`)
3. Chu, W.‑K., Mayer, J. W., & Nicolet, M.‑A. (1978). *Backscattering Spectrometry*. Academic Press. (PDF available in `docs/`)

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)

---

## Contact
For questions, feature requests, or collaboration, please open an issue or contact:

**Ching Kai Sing, Lucas**  
Department of Physics, The Chinese University of Hong Kong (CUHK)  
Repository: [https://github.com/lucas-cks/Rutherford-Scattering-RBS-Simulation-Surrogate-Modelling](https://github.com/lucas-cks/Rutherford-Scattering-RBS-Simulation-Surrogate-Modelling)

[Back to Top](#monte-carlo-simulation-of-rutherford-scattering--rbs)
```
