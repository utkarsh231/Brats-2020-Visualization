# BraTS20 Visualization

## Description
This Python script provides visualization utilities for MRI brain scans from the BraTS 2020 dataset, primarily focusing on imaging data (FLAIR, T1, T1ce, T2) and segmentation masks for tumor detection and analysis. The script allows for visualization of individual slices, montages, segmentation overlays, and generation of animated GIFs and videos from 3D MRI data.

## Features
- Visualization of different MRI modalities (FLAIR, T1, T1ce, T2).
- Segmentation mask visualization, highlighting different tumor regions.
- Montage creation to visualize multiple MRI slices simultaneously.
- Animated GIF and video generation to visualize 3D volumes.
- Color-coded visualization to distinguish between background, foreground, tumor area, and surrounding regions.

## Dependencies
- NumPy
- Pandas
- Matplotlib
- Nilearn
- NiBabel
- OpenCV
- ImageIO
- SciPy
- TensorFlow
- Seaborn
- Scikit-learn

Install additional dependencies for GIF conversion:
```bash
pip install git+https://github.com/miykael/gif_your_nifti
```

## Dataset
Ensure that the BraTS2020 training and validation datasets are downloaded and structured as follows:
```
../input/brats20-dataset-training-validation/
├── BraTS2020_TrainingData/
│   └── MICCAI_BraTS2020_TrainingData/
└── BraTS2020_ValidationData/
    └── MICCAI_BraTS2020_ValidationData
```

## Usage
- Adjust file paths and parameters (such as slice indices) in the script as needed.
- Run the script in a Python environment (e.g., Kaggle notebook, Jupyter notebook, or local Python IDE).

## Visualization Examples
The script generates visualizations including:
- Individual slice images.
- Montage images of MRI slices.
- Segmentation masks overlayed on MRI scans.
- Animated GIFs and videos showing MRI data in 3D.

## Author
Adapted for BraTS 2020 visualization tasks.

