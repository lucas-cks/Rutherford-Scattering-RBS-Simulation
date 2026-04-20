# Monte Carlo Simulation of Rutherford Scattering & RBS
Based on Geiger‑Marsden (1909), Rutherford (1911), and Chu, Mayer, Nicolet (1978)

**Author:** Lucas Kai Sing Ching

[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=lucas-cks&layout=compact&theme=vision-friendly-dark)](https://github.com/anuraghazra/github-readme-stats)

![C](https://img.shields.io/badge/Language-C-blue?logo=c)
![OpenMP](https://img.shields.io/badge/OpenMP-2A6F6F?logo=openmp&logoColor=white)
![Python](https://img.shields.io/badge/Language-Python-yellow?logo=python)

![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-ffffff?logo=matplotlib&logoColor=black)
![PyTorch](https://img.shields.io/badge/Library-PyTorch-EE4C2C?logo=pytorch&logoColor=white)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 1. Overview
This project implements a complete three‑stage computational pipeline for simulating **Rutherford scattering** and **Rutherford Backscattering Spectrometry (RBS)** and for building **fast neural network surrogate models**. The first stage reproduces the historic Geiger‑Marsden experiment using uniform impact‑parameter sampling, Bethe‑Bloch stopping power, and Gaussian straggling. The second stage introduces importance sampling (quadratic bias \(b = b_{\max}u^2\) with weight \(w = 4u^3\)) and OpenMP parallelisation to efficiently simulate rare backscattering events. The third stage trains small neural networks (PyTorch) to predict energy loss, backscatter probability, mean scattering angle, and the full 170° energy spectrum in milliseconds. The trained models are deployed as a standalone Windows executable and a periodic‑table scanner.

[Back to Top](#readme-top)

## 2. Physics Background
The simulation models the elastic scattering of alpha particles (⁴He²⁺) by target nuclei, including continuous energy loss and straggling.

* **Elastic Scattering:** Rutherford cross‑section, kinematic factor \(K\).
* **Energy Loss:** Bethe‑Bloch stopping power (or polynomial fits from SRIM).
* **Straggling:** Gaussian fluctuations (Bohr theory).
* **Importance Sampling:** Quadratic biasing of the impact parameter to force large‑angle scatters; statistical weight \(w = 4u^3\).
* **Parallelisation:** OpenMP (per‑thread RNG, private accumulators, dynamic scheduling).
* **Batch Automation:** 2,100 independent runs (7 materials, 5 thicknesses, 5 energies, 10 seeds).
* **Neural Networks:** Small feed‑forward networks (3‑8‑1 for scalars, 3‑128‑256‑128‑100 for spectrum).

[Back to Top](#readme-top)

## 3. Implementation Details
* **Languages:** C (simulation, OpenMP) and Python (analysis, visualisation, ML).
* **Scale:** 2.1 × 10⁹ particles simulated across 2,100 configurations.
* **RNG:** Mersenne Twister (MT19937) with per‑thread seeds.
* **Validation:** Pure Rutherford (Pt, Au) and importance‑sampled RBS (Au, Pt, Si, Al, Ge) agree with theory within 15%.
* **Neural Network Training:** 80/20 train/test split, Adam optimiser, MSE loss, 500‑800 epochs.
* **Deployment:** PyInstaller standalone `.exe` and periodic‑table scanner.

[Back to Top](#readme-top)

## 4. Repository Structure
```
.
├── src/
│   ├── Rutherford_Scattering.c          # Pure Rutherford MC
│   ├── RBS_OpenMP.c                     # RBS with importance sampling + OpenMP
│   ├── data_visualise.py                # Visualisation for pure Rutherford
│   ├── run_list_generator.py            # Creates run_list.csv (2100 runs)
│   ├── python_driver.py                 # Batch driver for RBS_OpenMP.exe
│   ├── analysis_RBS.py                  # Parses results, trains neural networks
│   ├── Predictor.py                     # Interactive predictor (→ .exe)
│   └── Periodic_table_scanner.py        # Periodic table sweep
├── data/                                # Example materials.csv, particles.csv
├── results/                             # Output folders for 2,100 runs
├── models/                              # Trained .pt models
├── RBS_Predictor.exe                    # Standalone Windows executable
├── README.md
└── LICENSE
```

[Back to Top](#readme-top)

## 5. Code Structure
```text
main() [C]
├── input_parameters()              # CSV mode or manual
├── main_loop() [OpenMP parallel]
│   └── simulate_one_particle()
│       ├── scattering_determine()  # uses b_max
│       ├── yes_scatter: sample b_actual = b_max * u^2, weight *= 4u^3
│       ├── compute θ (Rutherford)
│       ├── rotate_direction()
│       ├── energy_loss()           # Bethe‑Bloch + straggling
│       └── tally weighted histograms
└── output_results()

Python (analysis_RBS.py)
├── parse all result folders (regex)
├── aggregate scalar data & spectra
├── plot: energy loss vs thickness, backscatter prob vs energy, spectra
├── train neural networks (PyTorch)
│   ├── SmallNN (3‑8‑1) for scalars
│   └── SpectrumNN (3‑128‑256‑128‑100) for spectrum
└── save models (.pt) and prediction plots
```

[Back to Top](#readme-top)

## 6. Usage
### Prerequisites
- C compiler with OpenMP (`gcc -fopenmp`)
- Python 3.8+ with `numpy`, `pandas`, `matplotlib`, `torch`, `scipy`

### Installation
```bash
git clone https://github.com/your_username/RBS_surrogate.git
cd RBS_surrogate
pip install numpy pandas matplotlib torch scipy
```

### Compilation
```bash
gcc -o Rutherford src/Rutherford_Scattering.c -lm
gcc -o RBS_openmp src/RBS_OpenMP.c -lm -fopenmp
```

### Execution

**1. Pure Rutherford simulation**
```bash
./Rutherford
python src/data_visualise.py
```

**2. Single RBS run (importance sampling)**
```bash
./RBS_openmp
# Choose CSV mode or manual input
```

**3. Batch generation (2,100 runs)**
```bash
python src/run_list_generator.py
python src/python_driver.py
```

**4. Neural network training & analysis**
```bash
python src/analysis_RBS.py
```

**5. Standalone predictor (Windows)**
```cmd
RBS_Predictor.exe
```
Enter Z, thickness (Å), energy (MeV) → instant predictions.

**6. Periodic table scanner**
```bash
python src/Periodic_table_scanner.py
```

[Back to Top](#readme-top)

## 7. Key Results & Validation

### Pure Rutherford (Geiger‑Marsden style)

| Target | Z | Thickness (cm) | Simulated \(P_{\text{back}}\) | Theoretical \(P_{\text{back}}\) |
|--------|---|----------------|-------------------------------|-------------------------------|
| Pt     | 78 | \(2.0\times10^{-4}\) | \(9.8\times10^{-5}\) | \(1.08\times10^{-4}\) |
| Au     | 79 | \(4.0\times10^{-5}\) | \(3.0\times10^{-5}\) | \(3.15\times10^{-5}\) |

### RBS with Importance Sampling (2.20 MeV alpha particles)

| Target | Z | Thickness (cm) | Simulated \(P_{\text{back}}\) (mean ± std) | Theoretical \(P_{\text{back}}\) |
|--------|---|----------------|-------------------------------------------|-------------------------------|
| Au     | 79 | \(2.0\times10^{-4}\) | \((2.00 \pm 0.04)\times10^{-3}\) | \(2.9\times10^{-3}\) |
| Pt     | 78 | \(2.0\times10^{-4}\) | \((2.65 \pm 0.05)\times10^{-3}\) | \(2.8\times10^{-3}\) |
| Si     | 14 | \(3.0\times10^{-4}\) | \((4.72 \pm 0.09)\times10^{-5}\) | \(5.1\times10^{-5}\) |
| Al     | 13 | \(1.0\times10^{-4}\) | \((1.57 \pm 0.05)\times10^{-5}\) | \(1.7\times10^{-5}\) |
| Ge     | 32 | \(3.0\times10^{-4}\) | \((2.83 \pm 0.06)\times10^{-4}\) | \(3.0\times10^{-4}\) |

### Neural Network Performance

| Target quantity | Test \(R^2\) |
|----------------|-------------|
| Energy loss (MeV) | 0.9695 |
| Backscatter probability (log₁₀) | 0.9630 |
| Mean scattering angle (°) | 0.9850 |
| 170° energy spectrum (MSE) | ~3×10⁻⁴ |

All results are fully reproducible using the provided scripts.

[Back to Top](#readme-top)

## 8. License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 9. References
1. Geiger, H., & Marsden, E. (1909). On a Diffuse Reflection of the α‑Particles. *Proc. R. Soc. A*, **82**, 495‑500.
2. Rutherford, E. (1911). The Scattering of α and β Particles by Matter and the Structure of the Atom. *Phil. Mag.*, **21**, 669‑688.
3. Chu, W.‑K., Mayer, J. W., & Nicolet, M.‑A. (1978). *Backscattering Spectrometry*. Academic Press.

## 10. Contact

For questions or suggestions, please open an issue on this repository or contact the author directly.

**Lucas Kai Sing Ching**  
Department of Physics, The Chinese University of Hong Kong  
*Project Link:* [https://github.com/your_username/RBS_surrogate](https://github.com/your_username/RBS_surrogate)

[Back to Top](#readme-top)
```
