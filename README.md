# PhyDiCT
# <div align="center">PhyDiCT: Plug-and-Play CT Reconstruction from Sparse X-Rays via Differentiable Rendering and Strong Priors</div>

<p align="center">
<b>Weicheng Dai</b><sup>1</sup>,
Shantanu Ghosh<sup>1</sup>,
Kayhan Batmanghelich<sup>1</sup>  
<br/>
<sup>1</sup>Department of Electrical and Computer Engineering, Boston University  
</p>

<p align="center">
<i>MICCAI 2026</i>
</p>

<p align="center">
<!-- Activate upon release -->
<!-- <a href=""><img src="https://img.shields.io/badge/Paper-MICCAI%202026-blue"/></a> -->
</p>

---

## 🧠 Overview

**PhyDiCT** is a **training-free, plug-and-play framework** for reconstructing **3D lung CT volumes from sparse X-ray projections**.

The method integrates:
- A **differentiable physics-based forward model** grounded in the Beer–Lambert law
- A **frozen, text-conditioned diffusion model** as a strong 3D CT prior
- **Split Gibbs sampling** to jointly enforce projection fidelity and prior consistency

Without any task-specific training or fine-tuning, PhyDiCT **outperforms fully trained CT reconstruction methods**, achieving up to **+7.5% SSIM** on public 3D CT benchmarks.

---

## ✨ Key Features

- 🔧 **Training-free inference** (no paired X-ray / CT supervision)
- 🔬 **Explicit physics modeling** via differentiable X-ray rendering
- 🧠 **Strong generative priors** from pretrained diffusion models
- 🔁 **Plug-and-play sampling** compatible with sparse-view settings
- 🧩 **Test-time refinement** for enhanced realism and anatomical coherence

---

## 🧩 Method at a Glance

<p align="center">
<img src="assets/phydict_overview.pdf" width="90%"/>
</p>

PhyDiCT performs inference by alternating between:
1. **Physics consistency** — matching rendered projections to observed X-rays
2. **Prior realism** — denoising with a frozen diffusion model

This formulation allows flexible, stable reconstruction without retraining the prior.

---

## 📊 Results Summary

We evaluate PhyDiCT on **public 3D lung CT datasets** using:
- Perceptual metrics (SSIM, PSNR)
- Semantic and structural consistency metrics
- Slice-wise and volumetric evaluations

**Key finding:**  
Combining a strong generative prior with explicit imaging physics substantially improves reconstruction quality compared to both plug-and-play diffusion baselines and fully trained models.

---

## 📦 Code Status

🚧 **Code will be released upon decision.**

The full release will include:
- Plug-and-play diffusion sampling code
- Reproducibility instructions

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dai2026phydict,
  title     = {PhyDiCT: Plug-and-Play CT Reconstruction from Sparse X-Rays via Differentiable Rendering and Strong Priors},
  author    = {Dai, Weicheng and Ghosh, Shantanu and Batmanghelich, Kayhan},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}