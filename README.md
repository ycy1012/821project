# BIOSTATS 821 Final Project

# ChestX-FeatLib

**ChestX-FeatLib** is a Python toolkit for extracting and analyzing features from chest X-ray images. The toolkit is built around the NIH Chest X-ray14 dataset. Designed for the NIH ChestX-ray14 dataset, this library provides reproducible pipelines for medical image preprocessing and feature extraction using both traditional and deep learning-based methods — without the need to train new models. It is designed to enable downstream applications such as disease prediction, visualization, and clustering.

---

## Project Goal

The primary goal of ChestX-FeatLib is to support **efficient, reproducible extraction of medical image features**, without requiring deep learning model training. Specifically, we extract meaningful features from grayscale chest X-rays using:

- Traditional handcrafted features (e.g., GLCM texture descriptors and Sobel edge statistics)
- Deep visual embeddings from **pretrained CNNs** (e.g., embeddings from pretrained ResNet-18)

These features can be used for:
- Unsupervised visualization (e.g., t-SNE, PCA)
- Exploratory disease pattern comparison
- Preparing inputs for downstream tasks (clustering, ML, etc.)

---

## Architecture Overview

### Project Structure

``` bash
├── cli/                       # Command-line interface for pipeline execution
│   └── run_pipline.py
├── features/                  # Feature extraction modules
│   ├── handcrafted.py
│   └── deep.py
├── preprocessing/             # Image preprocessing utilities
│   └── preprocess.py
├── input_images/              # Chest X-ray input images (subset of NIH dataset)
├── output/                    # CSVs and visualization examples generated from features
│   ├── features_handcrafted.csv
│   ├── features_deep.csv
│   ├── tsne_hc.png            # Example t-SNE of handcrafted features
│   └── tsne_deep.png          # Example t-SNE of deep features
├── tests/                     # Unit tests for all modules
├── selected_metadata.csv      # Metadata associated with selected images
├── data.py                    # Data loading utilities
├── .github/workflows/         # CI/CD GitHub Actions
│   ├── checks.yml             # Mypy + Ruff static checks
│   └── tests.yml              # Pytest + coverage + reporting
└── README.md
```

### Component Descriptions

- `preprocessing/`  
  - Image resizing, grayscale conversion, normalization  
  - CLAHE for contrast enhancement  

- `features/`  
  - Texture features: GLCM texture, Sobel edge statistics
  - Deep features: Embeddings from pretrained ResNet-18 (no training required)

- `cli/`  
  - Command-line scripts for batch processing and feature extraction   


### Inputs

- Grayscale chest X-ray images (e.g., NIH ChestX-ray14 `.png` files)
- Associated metadata (`selected_metadata.csv`)

### Outputs

- `.csv` feature files with per-image feature vectors
- `.png` t-SNE visualization examples
  - `tsne_hc.png` for handcrafted features (e.g., edge/texture) & `tsne_deep.png` for deep features (ResNet-based embeddings)
  - Note: These visuals demonstrate examples for how extracted features could be used for clustering, exploration, or ML tasks.

---

## For End Users

### Installation & Setup

1. **Clone the repository**:

```bash
git clone https://github.com/ycy1012/821project.git
cd 821project
```

2. **Install requirements**:

```bash
pip install -r requirements-test.txt
```

> You will also need `input_images/` to contain `.png` grayscale chest X-ray files (224x224 preferred), and an optional `selected_metadata.csv` file.

---

### Expected Input Format

- **Images**: grayscale `.png` files inside `input_images/`
  - Size: auto-resized to `(224, 224)`
  - Format: single-channel grayscale expected
- **Metadata (optional)**: `selected_metadata.csv`
  - Used for organizing or linking clinical labels to visualizations (not required for feature extraction)

---

### Example Usage

Run the pipeline from the root directory:

```bash
python -m cli.run_pipline --input input_images --output output --features all
```

- `--features handcrafted`: texture + edge features (7D)
- `--features deep`: 512D ResNet embeddings
- `--features all`: generate both (two CSVs)

---

### Example Outputs

- `.csv` files:
  - `output/features_handcrafted.csv`
  - `output/features_deep.csv`

- `.png` visualizations (generated separately):
  - `output/tsne_hc.png`: t-SNE projection of handcrafted features
  - `output/tsne_deep.png`: t-SNE projection of deep features

> These illustrate possible downstream applications. These plots demonstrate how extracted features can be projected into 2D space for exploration. 
> For example, similar X-rays may cluster together, and trends or separability among pathological classes may become visible when color-coded by diagnosis.

---

## For Contributors

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/821project.git
cd 821project
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

---

### Local Testing

Run all tests:

```bash
pytest tests/
```

Run with coverage:

```bash
coverage run --source=. -m pytest
coverage report -m
```

Check linting and type safety:

```bash
ruff check .
mypy .
```

---

## Team Members & Roles

| Member Name | Role |
|-------------|------|
| Chenyao Yu  | Preprocessing, Feature extraction modules, Visualization Examples |
| Binqian Chai| CLI, Documentation |
| Together    | Testing, GitHub workflow, documentation |

## Dataset Reference

This project uses a subset of the NIH ChestX-ray14 dataset:
https://www.kaggle.com/datasets/nih-chest-xrays/data

---
