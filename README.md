
# Statistical Patch Defense (SPD) 🛡️

**A lightweight, training-free defense against physical adversarial patches**

### Overview
Fully unsupervised hybrid defense that detects and removes adversarial patches using hand-crafted local statistical features (intensity variance, gradient magnitude, color saturation, high-frequency DCT energy) followed by Telea’s fast marching inpainting.

**Results on the ImageNet-Patch benchmark (pretrained AlexNet):**
- Average gain: **+30.6 percentage points** top-1 accuracy  
- Defended top-5 recall: **98.8%**  
- Average inference time: **~117 ms/image** on CPU

Paper in preparation (December 2025)  
**Authors**: Anass Hameddine, Hajar Chiker, Abdelhak Hallou – [AISHII](https://aishii.tech)

### Credits & Base Work
This work builds upon the **ImageNet-Patch** benchmark:  
- Repository: https://github.com/pralab/ImageNet-Patch  
- Paper: Maura Pintor et al., "ImageNet-Patch: A Dataset for Benchmarking Machine Learning Robustness against Adversarial Patches", Pattern Recognition, 2023 (arXiv:2203.04412)  
- Original license: **GPL-3.0**

The patches, application utilities, and evaluation setup are reused and modified from the original work.

> Thank you to the original authors! If you use this code, please cite both the original benchmark and our upcoming paper.

### Usage
Open `Statistical_Patch_Defense.ipynb` in Google Colab or Jupyter Notebook.

### Results
See the `figures/` folder for qualitative examples, heatmaps, ablation studies, and evaluation charts.

### License
GPL-3.0 (same as the base ImageNet-Patch repository)
