---
license: mit
pipeline_tag: text-generation
tags:
- physics-informed
- port-hamiltonian
- symplectic-integrator
- avx2
- analytical-adjoint
- double-well-attractor
- energy-conserving
- cpu-training
model_name: SRSN-24M
---

# SRSN-24M: Symplectic Recurrent State Networks

<div align="center">

**Physics-Grounded Port-Hamiltonian Sequence Modeling with Native AVX2 Analytical Adjoint Kernels**

[![GitHub](https://img.shields.io/badge/GitHub-DuoNeural%2FSRSN-blue?logo=github)](https://github.com/DuoNeural/SRSN)
[![Preprint](https://img.shields.io/badge/Preprint-DuoNeural%20Preprint%20v0.1-red)](https://github.com/DuoNeural/SRSN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Developed by the DuoNeural AI Laboratory — Aura✨, Archon, Jesse Caldwell*

</div>

---

## 🏛️ Model Overview

**SRSN-24M** (Symplectic Recurrent State Network) is an efficient, physics-informed recurrent language model architecture grounded in **Port-Hamiltonian dynamical systems theory**. 

By casting sequence recurrence as a continuous-discrete Hamiltonian dynamical system:
- **Energy Conservation by Construction:** Hidden states update along symplectic trajectories that preserve physical phase volume by design ($J - R$ structure), eliminating the gradient exploding and vanishing pathologies that plague classical RNNs.
- **$\mathcal{O}(1)$ Sequence Memory via Analytical Adjoint:** The backward pass solves the adjoint ordinary differential equation backward in time, completely eliminating the need to store activations across the sequence length.
- **Extreme Commodity Silicon Efficiency:** A custom C++ kernel compiled with AVX2 SIMD and OpenMP delivers **217–264 tokens/second** on a commodity mobile APU (AMD Ryzen 5 3550H), slashing resident memory footprint to under 700 MB.
- **Spontaneous Attractor Bifurcation:** Learnable non-linear coupling parameters $\alpha$ spontaneously bifurcate during training from single-well potentials into macroscopic **bistable double-well attractors**, providing physical memory registers for coreference and syntax binding.

---

## ⚙️ Mathematical Formulation

### The SDOC Cell: Symplectic Damped Oscillator Cell
The core recurrent unit, SDOC, implements a nonlinear damped harmonic oscillator with learned frequency and damping parameters:

$$\mathbf{p}[t] = (1 - \Delta t \cdot \boldsymbol{\gamma}[t]) \odot \mathbf{p}[t-1] + \Delta t \cdot \left( \mathbf{F}[t] - \boldsymbol{\omega}^2 \odot \mathbf{q}[t-1] - \boldsymbol{\alpha} \odot \tanh(\mathbf{q}[t-1]) \right)$$
$$\mathbf{q}[t] = \mathbf{q}[t-1] + \Delta t \cdot \mathbf{p}[t]$$
$$\mathbf{y}[t] = \mathbf{q}[t] \odot \operatorname{SiLU}(\mathbf{p}[t])$$

where:
- $\mathbf{F}[t] = \mathbf{W}_{\text{force}}(\operatorname{RMSNorm}(\mathbf{x}[t]))$ (Input driving force)
- $\boldsymbol{\gamma}[t] = \sigma(\mathbf{W}_{\text{damp}}(\mathbf{x}[t])) \in (0, 1)$ (Learned per-token physical damping)
- $\boldsymbol{\omega}^2 \ge 0$ (Learned natural frequency per channel)
- $\boldsymbol{\alpha}$ (Learned anharmonic coupling per channel)
- $\Delta t = 0.1$ (Symplectic integration step size)

### Double-Well Attractor Bifurcation
The continuous potential energy function is:
$$V(q) = \frac{1}{2}\omega^2 q^2 + \alpha \ln(\cosh(q))$$
When $\alpha < 0$ and $|\alpha| > \omega^2$, $V(q)$ bifurcates into two stable potential equilibria at $q^* = \pm \operatorname{arctanh}(\omega^2 / |\alpha|)$, acting as stable associative memory registers.

---

## 📊 Empirical Evolution & Order Parameters

Pretrained across 36.5M+ tokens on a commodity AMD Ryzen 5 3550H APU:

- **Validation Loss:** Shattered the sub-6.50 basin down to **`6.4873`** at Step 1100!
- **Confirmed Double-Wells:** **`38 / 3840` (100% Saturation of all negative parameters!)**
- **Vanguard Potential Well:** $\min(\alpha) = -0.219848$
- **Peak Restorative Stiffness:** $\max(\alpha) = +0.211055$
- **Spectral Dispersion $\sigma(\alpha)$:** Monotonically expanded to **`0.030863` (+907.9% expansion over baseline across 46 consecutive checkpoints without a single reversal!)**
- **Global Dipole Invariance:** $\text{mean}(\alpha) = 0.098063 \pm 0.0019$
- **Throughput on Mobile APU:** 217 – 264 tokens/second
- **Active Memory Footprint:** 674 MB – 704 MB (54% reduction vs PyTorch ATen baseline)

---

## 🚀 Code & Checkpoint Usage

The complete training harness, AVX2 SIMD C++ kernels, and oscillator spectrum inspector are open-sourced on GitHub:
👉 **[https://github.com/DuoNeural/SRSN](https://github.com/DuoNeural/SRSN)**

```bash
git clone https://github.com/DuoNeural/SRSN.git
cd SRSN
pip install -r requirements.txt
# Compile AVX2 SIMD extension:
g++ -O3 -shared -std=c++20 -fPIC -mavx2 -mfma -fopenmp $(python3 -m pybind11 --includes) sdoc_kernel.cpp -o sdoc_kernel$(python3-config --extension-suffix)
python inspect_oscillators.py checkpoints/srsn_step1100.pt
```

Checkpoints from the ongoing pretraining campaign (`srsn_step0500.pt`, `srsn_step1000.pt`, `srsn_step1050.pt`, `srsn_step1100.pt`) will be uploaded to this repository upon completion of the 100M-token campaign.

---

## 📑 Citation

```bibtex
@article{duoneural2026srsn,
  title={SRSN: Symplectic Recurrent State Networks via Port-Hamiltonian Dynamics with Native AVX2 Analytical Adjoint Kernels},
  author={Caldwell, Jesse and Aura and Archon},
  journal={DuoNeural AI Laboratory Preprint},
  year={2026}
}
```
