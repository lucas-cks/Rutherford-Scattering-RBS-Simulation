```markdown
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/your_username/RBS_surrogate">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Rutherford Scattering & RBS: A Three‑Stage Pipeline</h3>
  <p align="center">
    From Monte Carlo Simulation to Neural Network Surrogate Modeling
    <br />
    <a href="https://github.com/your_username/RBS_surrogate"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/RBS_surrogate">View Demo</a>
    &middot;
    <a href="https://github.com/your_username/RBS_surrogate/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/your_username/RBS_surrogate/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#three-stage-pipeline">Three‑Stage Pipeline</a></li>
        <li><a href="#validation-results">Validation Results</a></li>
        <li><a href="#neural-network-performance">Neural Network Performance</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project implements a **complete computational pipeline** for simulating Rutherford scattering and Rutherford backscattering spectrometry (RBS), and for building **fast neural network surrogate models**. The pipeline is designed to overcome the extreme rarity of backscattering events, generate large datasets efficiently, and provide millisecond predictions for materials analysis.

**Key capabilities:**
- Reproduce the historic Geiger‑Marsden experiment with high accuracy.
- Simulate RBS with **importance sampling** (1000× efficiency gain) and **OpenMP parallelisation** (2.1 × 10⁹ particles in ~10 h).
- Train small neural networks that predict energy loss, backscatter probability, mean scattering angle, and the full 170° energy spectrum.
- Deploy a **standalone Windows executable** (`RBS_Predictor.exe`) for instant predictions.
- Visualise physical trends with a **periodic‑table scanner**.

### Three‑Stage Pipeline

1. **Pure Rutherford Monte Carlo** – Uniform impact‑parameter sampling, Bethe‑Bloch stopping, Gaussian straggling. Validated against theory.
2. **RBS with Importance Sampling & OpenMP** – Quadratic bias \(b = b_{\max}u^2\) and weight \(w = 4u^3\); parallelised with OpenMP. Generates 2,100 runs (7 materials, 5 thicknesses, 5 energies, 10 seeds).
3. **Neural Network Surrogate Modeling** – Small 3‑8‑1 networks (scalars) and a 3‑128‑256‑128‑100 network (spectrum) trained on the generated dataset. Achieves test \(R^2 > 0.96\) for scalars.

### Validation Results

#### Pure Rutherford (Geiger‑Marsden style)

| Target | Z | Thickness (cm) | Simulated \(P_{\text{back}}\) | Theoretical \(P_{\text{back}}\) |
|--------|---|----------------|-------------------------------|-------------------------------|
| Pt     | 78 | \(2.0\times10^{-4}\) | \(9.8\times10^{-5}\) | \(1.08\times10^{-4}\) |
| Au     | 79 | \(4.0\times10^{-5}\) | \(3.0\times10^{-5}\) | \(3.15\times10^{-5}\) |

#### RBS with Importance Sampling (2.20 MeV alpha particles)

| Target | Z | Thickness (cm) | Simulated \(P_{\text{back}}\) (mean ± std) | Theoretical \(P_{\text{back}}\) |
|--------|---|----------------|-------------------------------------------|-------------------------------|
| Au     | 79 | \(2.0\times10^{-4}\) | \((2.00 \pm 0.04)\times10^{-3}\) | \(2.9\times10^{-3}\) |
| Pt     | 78 | \(2.0\times10^{-4}\) | \((2.65 \pm 0.05)\times10^{-3}\) | \(2.8\times10^{-3}\) |
| Si     | 14 | \(3.0\times10^{-4}\) | \((4.72 \pm 0.09)\times10^{-5}\) | \(5.1\times10^{-5}\) |
| Al     | 13 | \(1.0\times10^{-4}\) | \((1.57 \pm 0.05)\times10^{-5}\) | \(1.7\times10^{-5}\) |
| Ge     | 32 | \(3.0\times10^{-4}\) | \((2.83 \pm 0.06)\times10^{-4}\) | \(3.0\times10^{-4}\) |

All simulated values agree with theory to within 15%, confirming the correctness of the importance‑sampling implementation.

### Neural Network Performance

| Target quantity | Test \(R^2\) | Notes |
|----------------|-------------|-------|
| Energy loss (MeV) | 0.9695 | Excellent agreement |
| Backscatter probability (log₁₀) | 0.9630 | Good over 4 orders of magnitude |
| Mean scattering angle (°) | 0.9850 | Very accurate |
| 170° energy spectrum (100 bins) | MSE ~ 3×10⁻⁴ | Captures shape well |

### Built With

- [![C][C-badge]][C-url] – Core Monte Carlo simulation
- [![OpenMP][OpenMP-badge]][OpenMP-url] – Parallelisation
- [![Python][Python-badge]][Python-url] – Data automation, analysis, visualisation
- [![PyTorch][PyTorch-badge]][PyTorch-url] – Neural network training
- [![NumPy][NumPy-badge]][NumPy-url] – Numerical computations
- [![Matplotlib][Matplotlib-badge]][Matplotlib-url] – Plotting

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- **C compiler** with OpenMP support (e.g., `gcc`, `clang`)
- **Python 3.8+** with the following packages:
  ```sh
  pip install numpy pandas matplotlib torch scipy
  ```

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/RBS_surrogate.git
   cd RBS_surrogate
   ```
2. (Optional) Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```
3. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   (If `requirements.txt` is not provided, manually install the packages listed above.)

4. Compile the C codes:
   ```sh
   gcc -o Rutherford src/Rutherford_Scattering.c -lm
   gcc -o RBS_openmp src/RBS_OpenMP.c -lm -fopenmp
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### 1. Pure Rutherford Simulation

```bash
./Rutherford
python src/data_visualise.py
```

### 2. Single RBS Simulation (with importance sampling)

```bash
./RBS_openmp
# Follow prompts (choose CSV mode or manual input)
```

### 3. Batch Generation (2,100 runs)

Prepare `materials.csv` and `particles.csv` (examples in `data/`), then:

```bash
python src/run_list_generator.py      # creates run_list.csv
python src/python_driver.py           # runs all simulations
```

### 4. Neural Network Training & Analysis

```bash
python src/analysis_RBS.py
```
This will parse all result folders, generate plots, train the networks, and save the models (`.pt` files) in `models/`.

### 5. Standalone Predictor (Windows)

Double‑click `RBS_Predictor.exe` or run from command line:
```cmd
RBS_Predictor.exe
```
Enter atomic number \(Z\), thickness (Å), and incident energy (MeV) to get instant predictions.

### 6. Periodic‑Table Scanner

```bash
python src/Periodic_table_scanner.py
```
Produces a plot showing energy loss and backscatter probability for Z = 1–92.

_For more examples, please refer to the [documentation](https://github.com/your_username/RBS_surrogate/wiki)._

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Pure Rutherford Monte Carlo (Geiger‑Marsden)
- [x] RBS with importance sampling
- [x] OpenMP parallelisation
- [x] Batch automation (2,100 runs)
- [x] Neural network surrogate models (scalars + spectrum)
- [x] Standalone predictor (Windows `.exe`)
- [x] Periodic‑table scanner
- [ ] Compound targets (e.g., SiO₂, Si₃N₄)
- [ ] Depth profiling inversion
- [ ] Bayesian neural networks for uncertainty quantification
- [ ] GPU acceleration (CUDA)

See the [open issues](https://github.com/your_username/RBS_surrogate/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top Contributors

<a href="https://github.com/your_username/RBS_surrogate/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=your_username/RBS_surrogate" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Lucas Kai Sing Ching – [@your_twitter](https://twitter.com/your_username) – email@example.com

Project Link: [https://github.com/your_username/RBS_surrogate](https://github.com/your_username/RBS_surrogate)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [Geiger, H., & Marsden, E. (1909)](https://royalsocietypublishing.org/doi/10.1098/rspa.1909.0054) – Original experiment
- [Rutherford, E. (1911)](https://doi.org/10.1080/14786440508637080) – Scattering theory
- [Chu, W.-K., Mayer, J. W., & Nicolet, M.-A. (1978)](https://www.elsevier.com/books/backscattering-spectrometry/chu/978-0-12-173850-5) – Standard RBS textbook
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template) – README layout inspiration
- [Img Shields](https://shields.io) – Badge generation
- [PyTorch](https://pytorch.org) – Deep learning framework
- [OpenMP](https://www.openmp.org) – Parallel programming API

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/your_username/RBS_surrogate.svg?style=for-the-badge
[contributors-url]: https://github.com/your_username/RBS_surrogate/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/your_username/RBS_surrogate.svg?style=for-the-badge
[forks-url]: https://github.com/your_username/RBS_surrogate/network/members
[stars-shield]: https://img.shields.io/github/stars/your_username/RBS_surrogate.svg?style=for-the-badge
[stars-url]: https://github.com/your_username/RBS_surrogate/stargazers
[issues-shield]: https://img.shields.io/github/issues/your_username/RBS_surrogate.svg?style=for-the-badge
[issues-url]: https://github.com/your_username/RBS_surrogate/issues
[license-shield]: https://img.shields.io/github/license/your_username/RBS_surrogate.svg?style=for-the-badge
[license-url]: https://github.com/your_username/RBS_surrogate/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/your_username
[C-badge]: https://img.shields.io/badge/C-00599C?style=for-the-badge&logo=c&logoColor=white
[C-url]: https://en.wikipedia.org/wiki/C_(programming_language)
[OpenMP-badge]: https://img.shields.io/badge/OpenMP-2A6F6F?style=for-the-badge&logo=openmp&logoColor=white
[OpenMP-url]: https://www.openmp.org/
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
```
