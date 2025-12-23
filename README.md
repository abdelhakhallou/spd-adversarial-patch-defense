
# Statistical Patch Defense (SPD) 🛡️

**A lightweight, training-free defense against physical adversarial patches**

### Overview
SPD is a fully unsupervised hybrid defense that detects and removes physical adversarial patches using hand-crafted local statistical features (intensity variance, gradient magnitude, color saturation, high-frequency DCT energy), followed by classical morphological refinement and Telea’s fast marching inpainting.

**Key results on the ImageNet-Patch benchmark (pretrained AlexNet):**
- Top-1 accuracy gain: **+41.8 percentage points** (31.4% → 73.2%, standard protocol)
- Recovery rate: **45.4%**
- True-class confidence gain: **+0.31**
- Top-5 recall after defense: **97.8%**
- Average inference time: **~120 ms/image** on CPU

Even under conservative manual labeling (clean accuracy 64%), SPD achieves a substantial **+30 percentage points** gain.

Paper in preparation (December 2025)  
**Authors**: Anass Hameddine, Hajar Chiker, Abdelhak Hallou – [AISHII](https://aishii.tech)

### Credits & Attribution
This work uses the adversarial patches and patch application utilities from the official **ImageNet-Patch** benchmark:  
- Repository: https://github.com/pralab/ImageNet-Patch  
- Paper: Maura Pintor et al., "ImageNet-Patch: A Dataset for Benchmarking Machine Learning Robustness against Adversarial Patches", Pattern Recognition, 2023  

The proposed defense pipeline, statistical feature design, evaluation metrics, and all analyses are our original contributions.

> If you use this code, please cite both the original benchmark and our upcoming paper.

### Usage
1. Open `Statistical_Patch_Defense.ipynb` in Google Colab or Jupyter.
2. Run all cells sequentially.
3. Results (figures, tables, qualitative examples) will be generated in the `evaluations/` directory.

### Repository Structure
- `assets/` — adversarial patches, class labels, and 50 clean validation images.
- `pipeline/` — complete SPD defense implementation (plug & play).
- `detection/` — statistical heatmap generation.
- `transforms/` — patch application utilities.
- `utils/` — helper functions and visualization tools.
- `evaluations/` — **generated at runtime** (figures, CSV, examples) — not committed.

### License
GPL-3.0
