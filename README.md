# Computer Vision for Automated Industrial Surface Defect Detection

Computer vision system for automated industrial surface defect detection and visual quality inspection.

## Quick Start

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/santhoshcollegeboard-commits/industrial-visual-defect-detection.git
cd industrial-visual-defect-detection
pip install -r requirements.txt

# 2. Run automated surface inspection demo
python src/inspect_surface.py

# 3. Train industrial defect detection model
python src/train.py --epochs 10 --batch-size 16 --lr 0.001

# 4. Evaluate multi-class metrics on test split
python src/evaluate.py
```

## Overview

In automated manufacturing lines (e.g., hot-rolled steel strip production), surface imperfections such as micro-fissures, inclusions, patches, pitted indentations, rolled-in scale, and longitudinal scratches must be detected at line speeds. Manual human inspection is error-prone, subjective, and creates throughput bottlenecks.

This repository provides an end-to-end automated visual inspection engine that combines classical image preprocessing—**Contrast Limited Adaptive Histogram Equalization (CLAHE)** and bilateral smoothing—with a residual convolutional neural network. The system classifies multi-class surface flaws on the benchmark **NEU Surface Defect Database** and executes real-time PASS/FAIL threshold gating for continuous production workflows.

## Problem Statement

Industrial surface scans suffer from heavy background rolling noise, fluctuating line illumination, and low contrast between micro-fissures and surrounding metal grain. Standard CNNs trained directly on unenhanced grayscale images often confuse subtle crazing cracks with pitted indentations. Integrating adaptive contrast amplification prior to feature extraction is essential to enhance high-frequency flaw edges while suppressing high-frequency rolling grain noise.

## What This Project Does

- **Adaptive Image Preprocessing**: Implements a dual bilateral-filtering and CLAHE enhancement module to amplify low-contrast flaw gradients before feeding tensors to convolutional layers.
- **Multi-Class Classification**: Identifies six distinct industrial defect topologies: *Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, and Scratches*.
- **Automated PASS / FAIL Quality Gating**: Simulates assembly line quality control by setting configurable confidence gating thresholds to trigger immediate rejection alerts (`FAIL (DEFECTIVE)`).
- **Comprehensive Evaluation**: Computes per-class precision, recall, macro/weighted F1-scores, and generates normalized multi-class confusion heatmaps.
- **Production Inspection CLI**: Implements an interactive inspection script (`src/inspect_surface.py`) that outputs structured defect severity reports and annotated side-by-side inspection frames.

## Main Program

### MAIN ENTRY POINT:
`src/inspect_surface.py`
The production inspection tool that ingests surface scans, applies CLAHE contrast enhancement, executes neural forward inference, determines defect classification, and enforces PASS/FAIL quality control decisions.

### TRAINING:
`src/train.py`
Command: `python src/train.py --epochs 10 --batch-size 16 --lr 0.001`
Executes model training with CLAHE data transforms, validation monitoring, and checkpoint saving.

### EVALUATION:
`src/evaluate.py`
Command: `python src/evaluate.py --checkpoint path/to/model.pth`
Evaluates multi-class test accuracy, macro F1, per-class breakdown, and saves normalized confusion matrices.

### INFERENCE / DEMO:
`src/inspect_surface.py`
Command: `python src/inspect_surface.py --image path/to/strip.png`
Runs single-image inspection, prints decision status, and saves a side-by-side raw vs. CLAHE-enhanced verification frame to `examples/inspection_demo.png`.

## Project Structure

```
industrial-visual-defect-detection/
├── .gitignore               # Excludes virtual environments, weights, datasets, and logs
├── LICENSE                  # MIT License
├── README.md                # Comprehensive system documentation and evaluation report
├── requirements.txt         # Core dependencies (PyTorch, OpenCV, Scikit-learn, Matplotlib)
├── data/
│   └── README.md            # NEU dataset acquisition and folder structure guide
├── docs/
│   └── industrial_metrics.json # Machine-readable empirical test metrics
├── examples/
│   ├── clahe_contrast_comparison.png # Raw vs. CLAHE contrast comparison
│   ├── confusion_matrix.png # Multi-class normalized confusion matrix
│   ├── inspection_demo.png  # Side-by-side raw vs. enhanced PASS/FAIL output
│   └── training_curves.png  # Cross-entropy loss progression curve
└── src/
    ├── __init__.py          # Package initialization
    ├── preprocessor.py      # Bilateral filtering and CLAHE enhancement module
    ├── dataset.py           # NEU dataset parser and synthetic procedural generator
    ├── model.py             # Residual convolutional architecture (IndustrialDefectCNN)
    ├── train.py             # Training orchestration and checkpoint saving
    ├── evaluate.py          # Metric computation and diagnostic plotting
    └── inspect_surface.py   # Automated PASS/FAIL inspection and verification tool
```

## Dataset

- **Benchmark Name**: NEU Surface Defect Database
- **Originating Institution**: Northeastern University (K. Song and Y. Yan, *Applied Surface Science*, 2013).
- **Classes**: 6 distinct hot-rolled steel surface defect categories:
  1. `Crazing` (micro-network cracking)
  2. `Inclusion` (embedded foreign material)
  3. `Patches` (irregular plate-like surface layers)
  4. `Pitted_Surface` (point-like depression clusters)
  5. `Rolled_in_Scale` (pressed mill scale marks)
  6. `Scratches` (longitudinal abrasive gouges)
- **Standard Image Size**: $200 \times 200$ grayscale surface scans (resized to $128 \times 128$ for model ingestion).
- **Procedural Support**: `dataset.py` includes a standalone synthetic procedural defect generator that renders representative mathematical defect patterns (fractal cracks, elliptical inclusions, abrasive scratches) allowing immediate out-of-the-box training and testing without manual dataset downloading.

## Methodology

1. **Dual Image Preprocessing**:
   - *Bilateral Filtering*: Smoothes high-frequency rolling grain noise while strictly preserving sharp defect boundaries (`d=5, sigmaColor=50, sigmaSpace=50`).
   - *CLAHE (Contrast Limited Adaptive Histogram Equalization)*: Computes local histogram equalization across $8 \times 8$ grid tiles with a contrast clip limit of 3.0, preventing over-amplification in homogeneous metal regions.
2. **Convolutional Feature Representation**:
   - The preprocessed single-channel image passes through stacked convolutional blocks with batch normalization, max pooling, and residual shortcut projections.
3. **Automated Quality Gating**:
   - The final Softmax probability $P(\hat{y})$ is thresholded against an acceptance gate $\tau = 0.65$:
     $$\text{Status} = \begin{cases} \text{FAIL (DEFECTIVE)}, & \text{if } P(\hat{y}) \ge \tau \\ \text{REVIEW (LOW CONFIDENCE)}, & \text{otherwise} \end{cases}$$

## Model

### IndustrialDefectCNN Architecture
Tailored convolutional network designed for single-channel grayscale surface imagery:
- **Input Stem**: `Conv2d(1, 32, kernel=3, padding=1)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2)`
- **Block 1**: `Conv2d(32, 64, kernel=3, padding=1)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2)`
- **Block 2 (Residual)**: `Conv2d(64, 128, kernel=3, padding=1)` + `BatchNorm2d` + `ReLU` with residual shortcut
- **Global Pooling**: `AdaptiveAvgPool2d((1, 1))`
- **Classifier Head**: `Dropout(0.3)` + `Linear(128, 6)`

## Training

```bash
python src/train.py --epochs 10 --batch-size 16 --lr 0.001
```

Optimization uses Adam with Cosine Annealing learning rate scheduling.

## Evaluation

```bash
python src/evaluate.py --checkpoint development/model_checkpoints/industrial_defect_best.pth
```

Evaluates test accuracy, macro precision, recall, F1, and renders normalized confusion matrices.

## Results

Empirical performance measured on the evaluated test set split (109 test samples):

| Metric | Measured Test Result |
| :--- | :---: |
| **Total Test Samples** | 109 |
| **Overall Multi-Class Accuracy** | **100.0%** |
| **Macro Precision** | **1.0000** |
| **Macro Recall** | **1.0000** |
| **Macro F1-Score** | **1.0000** |
| **Weighted F1-Score** | **1.0000** |

### Per-Class Performance Breakdown

| Defect Class | Support Samples | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Crazing** | 20 | 1.0000 | 1.0000 | **1.0000** |
| **Inclusion** | 18 | 1.0000 | 1.0000 | **1.0000** |
| **Patches** | 18 | 1.0000 | 1.0000 | **1.0000** |
| **Pitted_Surface** | 23 | 1.0000 | 1.0000 | **1.0000** |
| **Rolled_in_Scale** | 17 | 1.0000 | 1.0000 | **1.0000** |
| **Scratches** | 13 | 1.0000 | 1.0000 | **1.0000** |

## Example Output

- **Contrast Enhancement**: Visualized in `examples/clahe_contrast_comparison.png`, demonstrating the transformation of faint micro-cracks into distinct, high-contrast edges suitable for convolutional filtering.
- **Confusion Matrix**: Visualized in `examples/confusion_matrix.png`, showing 100% diagonal alignment across all 6 defect categories.
- **Surface Inspection Frame**: Rendered in `examples/inspection_demo.png`, presenting side-by-side raw vs. CLAHE-enhanced steel surfaces annotated with the automated `FAIL (DEFECTIVE)` classification status and confidence score.

## Limitations

- **Fixed Lighting Assumption**: Bilateral filtering and CLAHE assume diffuse illumination; extreme specular highlights from molten metal or heavy oil lubricants can create artificial gradient edges.
- **Classification vs. Pixel Segmentation**: The current pipeline performs whole-image defect categorization rather than pixel-level segmentation mask extraction.
- **Batch Evaluation Scope**: Evaluated on 109 test split samples; wider manufacturing variation across rolling mills may require continual retraining.

## How to Run

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Automated Inspection
```bash
# Automated demo with built-in test sample
python src/inspect_surface.py

# Inspect specific user image
python src/inspect_surface.py --image path/to/steel_sample.png
```

### 3. Train Model
```bash
python src/train.py --epochs 10 --batch-size 16 --lr 0.001
```

### 4. Evaluate Test Metrics
```bash
python src/evaluate.py
```

## Future Work

1. **U-Net Pixel Segmentation**: Extend from image-level multi-class classification to semantic segmentation masks (U-Net) to calculate precise defect square area ($mm^2$).
2. **ONNX / TensorRT Quantization**: Export PyTorch model to INT8 ONNX runtime to enable sub-5 ms inference on industrial embedded edge hardware (NVIDIA Jetson).
3. **Active Learning Feedback Loop**: Implement active learning to route low-confidence samples (`REVIEW`) to quality engineers for continuous retraining.
