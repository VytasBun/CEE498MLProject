# CEE 498-ML Project

# Pavement Crack Segmentation — Method Comparison
 
A school project comparing classical image processing and deep learning approaches for pavement crack detection using the **CRACK500** dataset.
 
---
 
## Overview
 
This project benchmarks five different segmentation methods — ranging from simple pixel thresholding to transformer-based deep learning — to evaluate their effectiveness at detecting cracks in pavement images.
 
---
 
## Dataset
 
**CRACK500** — A pavement crack detection dataset containing images of road surfaces with corresponding binary segmentation masks.
 
- Images are split into `train.txt` and `test.txt` list files, each line containing an image path and its corresponding mask path.
- Input images are RGB; masks are grayscale (crack = white, background = black).
---
 
## Methods
 
### 1. Naive Threshold (`Threshold.py`)
The simplest baseline. Converts the image to grayscale and applies a fixed pixel intensity threshold (default: 35). Pixels darker than the threshold are classified as cracks.
 
- **Pros:** Extremely fast, zero dependencies
- **Cons:** Highly sensitive to lighting conditions, no spatial reasoning
### 2. Otsu + CLAHE (`Otsu_CLAHE.py`)
An adaptive classical pipeline:
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) enhances local contrast
2. A power-curve transform darkens bright regions to emphasize cracks
3. **Otsu's method** automatically finds the optimal threshold
4. Small connected components below a minimum area are removed as noise
- **Pros:** Adaptive thresholding, no training required
- **Cons:** Still relies on hand-tuned preprocessing parameters
### 3. Frangi Filter (`Frangi.py`)
A vesselness / tubular-structure detector applied to crack detection. Uses a differentiable **Soft Frangi Filter** at multiple scales (sigmas) to highlight elongated crack-like structures, followed by hysteresis thresholding to produce a binary mask.
 
- **Pros:** Specifically designed to detect thin elongated structures
- **Cons:** Computationally heavier than threshold methods, requires careful sigma tuning
### 4. U-Net (`U_Net_Model.py`)
A classic encoder-decoder CNN trained end-to-end on the CRACK500 dataset.
 
- Architecture: 3-level encoder, bottleneck, 3-level decoder with skip connections
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr = 1e-4)
- Training: Early stopping (patience = 3), up to 40 epochs
- Input size: 360 × 640
- **Pros:** Strong spatial feature learning, well-established for segmentation
- **Cons:** Requires GPU training, large dataset to generalize well
### 5. SegFormer (`SegFormer_Model.py`)
A transformer-based segmentation model fine-tuned from `nvidia/mit-b0` (pretrained on ImageNet).
 
- 2-class output (background / crack)
- Optimizer: AdamW (lr = 5e-5)
- Training: Early stopping (patience = 5), up to 40 epochs
- Logits are upsampled back to the original mask resolution
- **Pros:** State-of-the-art architecture, leverages pretrained representations
- **Cons:** Highest compute requirements, slowest inference
---