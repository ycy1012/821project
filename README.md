# 821project

# ChestX-FeatLib

**ChestX-FeatLib** is a Python toolkit for extracting and analyzing features from chest X-ray images. It is designed to streamline medical image processing workflows and enable downstream applications such as disease prediction, visualization, and clustering. The toolkit is built around the NIH Chest X-ray14 dataset.

## Project Goal

The primary goal of ChestX-FeatLib is to support efficient and reproducible feature extraction from large-scale medical images. The toolkit will provide traditional image features (e.g., texture, edge-based descriptors) as well as deep-learning-derived embeddings. These features can be used for diagnostic support, image clustering, or visualization tasks.

## Architecture Overview

### Components

- `preprocessing/`
  - Image normalization and enhancement
  - Resize, grayscale conversion, and CLAHE
- `features/`
  - Traditional features (GLCM, edge histograms)
  - Deep features from pretrained CNNs
- `visualization/`
  - Feature map overlays
  - t-SNE / PCA 2D projections
- `cli/`
  - Command-line scripts for batch processing
 
### Inputs

- Chest X-ray images (e.g., NIH ChestX-ray14 PNG files)

### Outputs

- `.csv` files with per-image feature vectors
- `.png` visualizations (e.g., CAM overlays, t-SNE plots)

## Team Members & Roles

| Member Name | Role |
|-------------|------|
| Chenyao Yu  | Feature extraction modules, deep learning, CLI |
| Binqian Chai| Preprocessing, traditional features, visualization |
| Together    | Testing, GitHub workflow, documentation |

## Dataset

We use the NIH Chest X-ray14 dataset available on Kaggle:
https://www.kaggle.com/datasets/nih-chest-xrays/data

---

