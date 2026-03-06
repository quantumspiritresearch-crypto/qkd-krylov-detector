[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18889224.svg)](https://doi.org/10.5281/zenodo.18889224)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-17%2F17%20passed-brightgreen.svg)](#tests)

# QKD Krylov Detector

A three-layer eavesdropper detection framework for Quantum Key Distribution (QKD) based on **Krylov complexity** and **sidereal filtering**.

This package implements the methods from a series of seven research papers by Daniel Süß, providing a complete pipeline from raw QBER time series to Eve detection with ROC analysis.

---

## How It Works — At a Glance

<p align="center">
  <img src="docs/quickstart_plot.png" alt="Visual Quickstart: Clean vs. Attack detection scores, score distributions, and ROC curve" width="100%">
</p>

The left panel shows the **Krylov detection score** over time. The green line (clean channel) stays flat and low, while the red line (Eve's iid attack) spikes sharply during the attack window (shaded region). The middle panel shows the resulting score distributions — clean and Eve are well separated. The right panel confirms this with the ROC curve. Over many trials with realistic hardware noise, the detector achieves **AUC = 0.9989** and **12.13σ separation** (see [Published Results](#published-results)).

---

## Overview

The detector operates in three layers:

| Layer | Module | Function | Paper |
|-------|--------|----------|-------|
| **1** | `sidereal_filter` | Remove 23.93h sidereal and 24.0h diurnal periodicities | [1], [2], [3] |
| **2** | `lanczos_extractor` | Compute Krylov b_n coefficients via Lanczos algorithm | [4], [6] |
| **3** | `template_detector` | Match QBER autocorrelation against Gaussian template | [5], [6] |

---

## Installation

```bash
pip install qkd-krylov-detector
```

For NANOGrav pulsar timing validation:

```bash
pip install qkd-krylov-detector[pulsar]
```

For development:

```bash
git clone https://github.com/danielsuess/qkd-krylov-detector.git
cd qkd-krylov-detector
pip install -e ".[dev]"
```

---

## Quickstart

```python
import numpy as np
from qkd_krylov_detector import (
    build_hamiltonian, compute_lanczos, sidereal_filter,
    make_clean_qber, make_eve_qber, krylov_dynamic_detector,
    compute_roc, compute_auc,
)

# Build Hamiltonian and extract Krylov template
H = build_hamiltonian()
b_n = compute_lanczos(H)

# Simulate clean and Eve-compromised channels
t = np.linspace(0, 400, 400)
clean = sidereal_filter(make_clean_qber(t, seed=0), t)
eve = sidereal_filter(make_eve_qber(t, eve_type="iid", seed=1), t)

# Run detector
_, scores_clean = krylov_dynamic_detector(clean, b_n, t)
_, scores_eve = krylov_dynamic_detector(eve, b_n, t)

# Evaluate
fpr, tpr = compute_roc(scores_clean, scores_eve)
auc = compute_auc(fpr, tpr)
print(f"AUC = {auc:.4f}")
```

---

## Package Structure

```
qkd_krylov_detector/
├── __init__.py              # Public API exports
├── hamiltonian.py           # 8-qubit Heisenberg chain construction
├── lanczos_extractor.py     # Layer 2: Lanczos algorithm for b_n
├── sidereal_filter.py       # Layer 1: FFT notch filter + irregular LS fit
├── template_detector.py     # Layer 3: Gaussian template matching + ROC
├── qber_simulator.py        # QBER generation (idealized + realistic noise)
└── spectral_analysis.py     # ⟨r⟩ ratio and regime classification
```

---

## Modules

### `hamiltonian` — System Hamiltonian

Constructs the symmetry-broken Heisenberg chain used throughout all papers:

```
H = J·(XX + YY + ZZ)₀₁ + g·Σᵢ XXᵢ,ᵢ₊₁ + κ·ZZ₁₂ + Σᵢ(hz·Zᵢ + hx·Xᵢ)
```

Default parameters (Regime II, crossover ⟨r⟩ ≈ 0.366):

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 8 | Number of qubits |
| J | 1.0 | Heisenberg coupling (qubits 0–1) |
| g | 0.5 | XX chain coupling (qubits 2–7) |
| κ | 0.45 | ZZ coupling (qubits 1–2) |
| hz | 0.12 | Longitudinal field |
| hx | 0.08 | Transverse field |

**Important:** The system is in the crossover regime, NOT full GOE chaos. See Paper [4], Section II.

### `sidereal_filter` — Layer 1

The module provides two methods. `sidereal_filter()` is an FFT-based notch filter for uniformly sampled data, while `sidereal_filter_irregular()` uses a least-squares sinusoidal fit for irregularly sampled data such as pulsar timing observations. The irregular method was validated on the NANOGrav 15-year dataset (59,389 TOAs, PSR J1713+0747). See Paper [3].

### `lanczos_extractor` — Layer 2

Computes Lanczos coefficients b_n via the recursive Krylov algorithm:

```
O_{n+1} = i[H, Oₙ] - b_{n-1}·O_{n-1}
b_n = ‖O_{n+1}‖
```

Linear growth of b_n implies Gaussian decay of the operator autocorrelation:

```
C(t) = exp(-½·(slope·t)²)    where slope = mean(Δb_n)
```

This provides the theoretical template for Layer 3.

### `template_detector` — Layer 3

The detector performs sliding-window template matching: for each window of the QBER residuum, it computes the empirical autocorrelation, compares it against the Gaussian template derived from b_n, and returns the RMSE as a detection score. Clean channels match the template (low score), while Eve-compromised channels deviate (high score).

Additional functions include `krylov_proxy()` for higher-moment analysis (kurtosis + skewness), `compute_roc()` and `compute_auc()` for ROC analysis, and `compute_separation()` for σ-separation between score distributions.

### `qber_simulator` — QBER Generation

Two noise models are provided:

| Model | Components | Source |
|-------|-----------|--------|
| **Idealized** | Gaussian white noise + sidereal/diurnal drift | krylov_dynamic_detector.ipynb |
| **Realistic** | AR(1) + 1/f + afterpulsing + burst noise | krylov_robustness_test.ipynb |

Three Eve attack models are implemented:

| Attack | Description | Source |
|--------|-------------|--------|
| `iid` | Classical intercept-resend | Paper [5], Part I |
| `exponential` | Asymmetric exponential distribution | eve_detection_master_v3.ipynb |
| `hamiltonian` | Coupling perturbation γ·σx(1)σx(2) | eve_detection_master_v3.ipynb |

### `spectral_analysis` — Spectral Statistics

Computes the mean ratio of consecutive level spacings ⟨r⟩:

| Regime | ⟨r⟩ | Statistics |
|--------|-----|-----------|
| Integrable | 0.386 | Poisson |
| **This system** | **0.366** | **Crossover** |
| Fully chaotic | 0.536 | GOE |

---

## Examples

The `examples/` directory contains three scripts that demonstrate the framework at different levels of detail. `quickstart.py` is a minimal 20-line detection demo. `full_pipeline.py` runs the complete three-layer pipeline with ROC analysis and generates publication-quality plots. `nanograv_validation.py` demonstrates the sidereal filter validation on synthetic pulsar timing data modeled after the NANOGrav 15-year dataset.

---

## Tests

```bash
pytest tests/ -v
```

The test suite (17 tests) validates Hamiltonian construction (dimensions, Hermiticity, Eve perturbation), Lanczos coefficients (positivity, linear growth, Eve deviation), theoretical autocorrelation (Gaussian shape, normalization), sidereal filter (drift removal, signal preservation), QBER simulator (all noise models and Eve types), template detector (score computation, Eve discrimination), ROC/AUC computation, and spectral analysis (⟨r⟩ ≈ 0.366, crossover classification).

---

## Published Results

Results reproduced by this package (from Paper [5] and [6]):

| Metric | Idealized Noise | Realistic Noise |
|--------|----------------|-----------------|
| AUC | 0.9899 | 0.9989 |
| Separation | 22.25σ | 12.13σ |

Quantum attacker results (from Paper [6], Section IV):

| Attacker | Qubits (m) | AUC |
|----------|-----------|-----|
| Quantum Eve | 1 | ≥ 0.999 |
| Quantum Eve | 2 | ≥ 0.999 |

Scaling law (from Paper [4]):

```
τ_rec ~ (1/ε)^(2^N - 1)
```

Critical crossover at ⟨r⟩ = 0.366 (GOE statistics threshold).

---

## Paper References

| # | Title | DOI |
|---|-------|-----|
| [1] | Deconvolution of Sidereal and Diurnal Periodicities in QKD | [10.5281/zenodo.18701222](https://doi.org/10.5281/zenodo.18701222) |
| [2] | Dual-Layer Sidereal Detection Framework v2 | [10.5281/zenodo.18768750](https://doi.org/10.5281/zenodo.18768750) |
| [3] | Real-Data Validation on NANOGrav 15-Year Dataset | [10.5281/zenodo.18792775](https://doi.org/10.5281/zenodo.18792775) |
| [4] | Scrambling vs. Recurrence: Microscopic Origin of the Quantum Arrow of Time | [10.5281/zenodo.18813710](https://doi.org/10.5281/zenodo.18813710) |
| [5] | QKD Eve Detector: A Unified Framework — Parts I–III | [10.5281/zenodo.18873824](https://doi.org/10.5281/zenodo.18873824) |
| [6] | Quantum Scrambling as a Cryptographic Resource | [10.5281/zenodo.18889224](https://doi.org/10.5281/zenodo.18889224) |

Additional reference:

> Parker, D. E., Cao, X., Avdoshkin, A., Scaffidi, T., & Altman, E. (2019). "A Universal Operator Growth Hypothesis." *Physical Review X*, 9(4), 041017.

---

## Notebook Correspondence

Each function in this package traces back to a specific notebook cell:

| Function | Notebook | Cell |
|----------|----------|------|
| `build_hamiltonian()` | krylov_dynamic_detector.ipynb | Cell 2 |
| `compute_lanczos()` | krylov_dynamic_detector.ipynb | Cell 2 |
| `sidereal_filter()` | krylov_dynamic_detector.ipynb | Cell 2 |
| `sidereal_filter_irregular()` | pulsar_sidereal_colab-1.ipynb | Cells 5–6 |
| `krylov_dynamic_detector()` | krylov_dynamic_detector.ipynb | Cell 3 |
| `krylov_proxy()` | eve_detection_master_v3.ipynb | Cell 6 |
| `make_clean_qber()` | krylov_dynamic_detector.ipynb | Cell 2 |
| `make_realistic_clean_qber()` | krylov_robustness_test.ipynb | Cell 4 |
| `compute_r_ratio()` | eve_detection_master_v3.ipynb | Cell 4 |

---

## License

MIT License. Copyright (c) 2026 Daniel Süß.

---

## Citation

If you use this package, please cite it using the provided [`CITATION.cff`](CITATION.cff) file, or use the following BibTeX entry:

```bibtex
@software{suess2026qkd,
  author    = {Süß, Daniel},
  title     = {QKD Krylov Detector: Three-Layer Eavesdropper Detection
               for Quantum Key Distribution},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18889224},
  url       = {https://doi.org/10.5281/zenodo.18889224}
}
```

For the synthesis paper:

```bibtex
@article{suess2026scrambling,
  author  = {Süß, Daniel},
  title   = {Quantum Scrambling as a Cryptographic Resource:
             From Krylov Complexity to Eavesdropper Detection},
  year    = {2026},
  journal = {Zenodo},
  doi     = {10.5281/zenodo.18889224}
}
```
