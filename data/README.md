# Dataset Guide: NEU Surface Defect Database

## Benchmark Overview
- **Name**: NEU Surface Defect Database (Northeastern University, Shenyang, China)
- **Creators**: K. Song and Y. Yan, *"A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defect detection"*, **Applied Surface Science**, 285, 858-864, 2013.
- **Official Source**: Northeastern University Research Repository / Kaggle NEU Metal Surface Defects Data.
- **License**: Open research benchmark for academic study.

## Dataset Specifications
- **Modality**: High-resolution industrial optical inspection grayscale images of hot-rolled steel strips.
- **Image Resolution**: 200x200 pixels (standardized in this pipeline to 128x128 for efficient edge inference).
- **Defect Taxonomy**: 6 standard industrial surface defect classes (300 samples per class, total 1,800 images):
  1. **Crazing (Cr)**: Spider-web network of micro-fissures caused by thermal fatigue and rolling stress.
  2. **Inclusion (In)**: Non-metallic slag/oxide clusters embedded into the steel matrix.
  3. **Patches (Pa)**: Irregular localized oxidized surface layer scaling.
  4. **Pitted Surface (Ps)**: Periodic or localized mechanical cavities and gouges.
  5. **Rolled-in Scale (Rs)**: Iron oxide scale pressed into the strip during heavy rolling passes.
  6. **Scratches (Sc)**: High-contrast longitudinal abrasions caused by mechanical guide friction.

## Visual Challenges in Industrial Inspection
- **Low Contrast**: Surface flaws frequently exhibit subtle grayscale gradients against varying steel grain backgrounds.
- **Illumination Shifts**: Varying mill lighting, oil films, and oxide patches introduce intra-class variance.
- **Solution**: This repository integrates Contrast Limited Adaptive Histogram Equalization (CLAHE) to amplify high-frequency defect textures prior to CNN feature extraction.

## Partitioning Strategy
- **Train Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%
