# Mahalanobis OOD Detector

A Streamlit web application that detects whether an uploaded image belongs to the CIFAR-10 distribution (In-Distribution) or if it is Out-of-Distribution (OOD) using the Mahalanobis Distance metric.

## Overview

This project uses a pretrained **ResNet18** model as a feature extractor. When an image is uploaded, it passes through the ResNet18 network to produce a 512-dimensional feature vector. The Mahalanobis distance is then computed between this feature vector and the precomputed class centroids (means) of the CIFAR-10 dataset using a shared covariance matrix. If the minimum distance to any class centroid exceeds a predefined threshold, the image is classified as Out-Of-Distribution (OOD).

## Features

- **Upload Image**: Supports JPG, JPEG, and PNG formats.
- **Inference**: Fast and efficient feature extraction using PyTorch and ResNet18.
- **OOD Detection**: Robust detection of Out-Of-Distribution samples using Mahalanobis distance.
- **Class Prediction**: Displays the top 3 closest CIFAR-10 classes along with their respective distances.
- **Technical Details**: View model parameters, feature dimension, and detection thresholds in an expandable section.

## Installation

1. Clone the repository or download the source code.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application by running:

```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501` (usually opens automatically).
3. Upload an image to see the prediction and whether it is considered In-Distribution (CIFAR-10) or Out-Of-Distribution.

## Project Structure

- `app.py`: The main Streamlit application script containing the UI and inference workflow.
- `requirements.txt`: Required Python packages and dependencies.
- `models/`: Contains the feature extractor model logic.
- `detector/`: Contains the OOD detector (`ood_detector.py`) and Mahalanobis scoring logic (`mahalanobis.py`).

## Dependencies

- `torch`, `torchvision`
- `numpy`, `scipy`, `scikit-learn`
- `matplotlib`, `pillow`
- `streamlit`
