# 821project

# ChestX-FeatLib

**ChestX-FeatLib** is a Python toolkit for extracting and analyzing features from chest X-ray images. It is designed to streamline medical image processing workflows and enable downstream applications such as disease prediction, visualization, and clustering. The toolkit is built around the NIH Chest X-ray14 dataset.

## Project Goal

The primary goal of ChestX-FeatLib is to support **efficient, reproducible extraction and visualization of medical image features**, without requiring deep learning model training. Specifically, we extract:

- Traditional handcrafted features (e.g., texture, edges)
- Deep visual embeddings from **pretrained CNNs** (e.g., ResNet)

These features can be used for:
- Unsupervised visualization (e.g., t-SNE, PCA)
- Exploratory disease pattern comparison
- Preparing inputs for downstream tasks (clustering, ML, etc.)

## Architecture Overview

### Components

- `preprocessing/`  
  - Image resizing, grayscale conversion, normalization  
  - CLAHE for contrast enhancement  

- `features/`  
  - Texture features: GLCM, edge histograms  
  - Deep features: embeddings from pretrained CNNs (no model training)  

- `visualization/`  
  - Feature maps, t-SNE plots, cluster heatmaps  

- `cli/`  
  - Command-line scripts for batch processing and feature extraction  
 
### Inputs

- Chest X-ray images (e.g., NIH ChestX-ray14 PNG files)

### Outputs

- `.csv` files with per-image feature vectors
- `.png` visualizations (e.g., PCA/t-SNE plots)

## Team Members & Roles

| Member Name | Role |
|-------------|------|
| Chenyao Yu  | Feature extraction modules, CLI |
| Binqian Chai| Preprocessing, traditional features, visualization |
| Together    | Testing, GitHub workflow, documentation |

## Dataset

We use the NIH Chest X-ray14 dataset available on Kaggle:
https://www.kaggle.com/datasets/nih-chest-xrays/data

---

