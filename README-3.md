# TrashNet Waste Classification with Deep Learning

A deep learning image-classification project comparing **MobileNetV2**, **ResNet50**, and **VGG16** for multi-class waste classification using the **TrashNet** dataset.

The goal is to classify waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Project Overview

This project uses transfer learning with ImageNet-pretrained convolutional neural networks (CNNs) to evaluate different model architectures for waste classification.

The notebooks cover:

- Image preprocessing and augmentation
- Stratified train/validation/test splitting
- Transfer learning with ImageNet weights
- Two-stage training: frozen feature extractor followed by fine-tuning
- Early stopping and learning-rate reduction
- Accuracy, precision, recall, and F1-score evaluation
- Confusion matrices and class-level performance
- Model/inference benchmarking where implemented

## Models

| Model | Main Characteristic | Notebook |
|---|---|---|
| MobileNetV2 | Lightweight and deployment-friendly | `mobilenet_95percent.ipynb` |
| ResNet50 | Residual CNN architecture | `cse465-resnet50-modified.ipynb` |
| VGG16 | Larger classic CNN architecture | `465-vgg16with-80-freeeze.ipynb` |

## Current Results

The following results are taken from the outputs currently stored in the notebooks:

| Model | Recorded Accuracy |
|---|---:|
| MobileNetV2 | **90.00%** |
| ResNet50 | **89.47%** |
| VGG16 | **88.42%** |

> **Note:** These are the results recorded in the current notebook runs. MobileNetV2 reports test accuracy directly, while the ResNet50 and VGG16 notebooks report global model accuracy calculated from the test predictions. Training details also vary slightly between notebooks, so the table should be interpreted as a summary of the current experiments rather than a perfectly controlled benchmark.

## Dataset

This project uses the **TrashNet** waste-image dataset.

Dataset source:  
[TrashNet on Kaggle](https://www.kaggle.com/datasets/feyzazkefe/trashnet)

The dataset contains images from six waste categories: cardboard, glass, metal, paper, plastic, and trash.

The dataset itself is **not included in this repository**.

## Training Pipeline

Across the notebooks, the general workflow is:

1. Load image paths and labels into a Pandas DataFrame.
2. Split the dataset into approximately:
   - 70% training
   - 15% validation
   - 15% testing
3. Resize images to **224 × 224**.
4. Apply model-specific preprocessing.
5. Apply image augmentation to the training data.
6. Load an ImageNet-pretrained CNN without its original classification head.
7. Add a custom classification head.
8. Train with the pretrained base initially frozen.
9. Fine-tune selected upper layers using a lower learning rate.
10. Evaluate the model on the held-out test set.

The ResNet50 and VGG16 notebooks also oversample the minority **trash** class in the training set.

## Techniques Used

- Transfer Learning
- Fine-Tuning
- Data Augmentation
- Stratified Data Splitting
- Batch Normalization
- Dropout
- Early Stopping
- ReduceLROnPlateau
- Multi-Class Classification
- Confusion Matrix Analysis
- Precision / Recall / F1-Score Evaluation

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- psutil
- Jupyter Notebook
- Google Colab / Kaggle

## Repository Structure

```text
trashnet-deep-learning-comparison/
│
├── README.md
├── requirements.txt
│
├── mobilenet_95percent.ipynb
├── cse465-resnet50-modified.ipynb
└── 465-vgg16with-80-freeeze.ipynb
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/trashnet-deep-learning-comparison.git
cd trashnet-deep-learning-comparison
```

Create a virtual environment if desired:

```bash
python -m venv .venv
```

Activate it and install the dependencies:

```bash
pip install -r requirements.txt
```

Then open Jupyter:

```bash
jupyter notebook
```

You can also run the notebooks in **Google Colab** or **Kaggle Notebooks**.

## Dataset Setup

Download the TrashNet dataset from Kaggle and extract it locally.

The expected directory should contain one folder for each class, for example:

```text
dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Before running a notebook, update its `DATA_DIR` variable to point to your dataset.

Example:

```python
DATA_DIR = "/path/to/dataset-resized"
```

The current notebooks were developed in different notebook environments, so some paths point to Google Drive while others point to Kaggle storage.

## Important Notes

- The dataset is not stored in this repository.
- Large trained model files should generally be excluded from GitHub and stored separately if needed.
- GPU acceleration is recommended for training.
- Results may vary depending on hardware, TensorFlow version, random initialization, and training environment.

## Contributors

- Shahriar Kamal
- Umme Jahara Simki
- Mezbah Uddin Ahmed
