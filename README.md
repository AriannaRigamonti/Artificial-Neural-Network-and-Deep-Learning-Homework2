# AN2DL - Homework 2: Semantic Segmentation of Mars Terrain

## Team: Bio.log(y)
**Authors**: Luca Lepore, Arianna Rigamonti, Michele Sala, Jacopo Libero Tettamanti  
**Submission Date**: December 14, 2024

## Introduction

This project focuses on semantic segmentation of Mars terrain images, where the goal is to develop a robust deep learning model to assign each pixel to its respective class. The dataset consists of grayscale images of Mars terrain with pixel-wise masks representing five classes: **Background, Soil, Bedrock, Sand, and Big Rock**. Given the significant class imbalance, particularly for **Big Rock (<1%)**, various strategies were explored to enhance model performance.

The approach began with a baseline model and progressively incorporated **data augmentation, advanced architectures, and different loss functions** to improve segmentation accuracy. The final evaluation metric is **Mean Intersection over Union (Mean IoU)**.

---

## Problem Analysis

- **Dataset**: 2,615 grayscale images (64x128 pixels) with ground-truth segmentation masks.
- **Challenges**:
  - Class imbalance, with **Big Rock** being highly underrepresented.
  - Small dataset, requiring augmentation and efficient model architectures.
  - Need for distinguishing fine-grained texture, intensity, and structural patterns.

---

## Methods

### 1. Data Preprocessing

- **Duplicate Removal**: Identical images with different alien positions but the same masks were detected and removed, reducing the dataset size to **2,505 images**.
- **Dataset Splitting**: The dataset was manually split into **training, validation, and internal test sets**, as the **official test set masks were unavailable**.
- **Normalization**: Pixel intensity values were scaled to **[0, 1]** for better convergence during training.

### 2. Data Augmentation

To improve class balance and increase the number of training samples, two augmentation setups were tested:

1. **Light Augmentation**:
   - Flipping, rotation, shifting, zooming.
   - Quadrupled images containing **Big Rock** and doubled others.

2. **Advanced Augmentation**:
   - Cropped **Big Rock** segments were transferred to **80% of images without class 4**.
   - Albumentations library applied transformations including flipping, rotation, elastic deformations.
   - Final step: **Quadrupling** images containing **Big Rock** using shifting, scaling, zooming, and rotation.

**Outcome**: **Original dataset performed better** than augmented versions, suggesting high similarity between training and test images.

---

## 3. Models

Several models were implemented to progressively improve segmentation accuracy:

### 3.1 Baseline CNNs
- **Single-Layer CNN**: Simple **1x1 convolutional** layer with softmax activation.
- **3-Layer CNN**: Three **3x3 convolutional layers** (64, 128, 256 filters) with **ReLU activation**.

### 3.2 U-Net Architectures
- **U-Net Base**: Classic encoder-decoder structure with skip connections.
- **U-Net TC**: Deeper variant with **three downsampling blocks (32, 64, 128 filters)**.

### 3.3 Residual and Attention-based U-Nets
- **U-Net RA (Residual-Attention)**: Introduced **residual blocks** and **attention gates**.
- **U-Net MSRA (Multi-Scale Residual-Attention)**: Added **multi-scale** parallel dilated convolutions.

### 3.4 Double U-Net Architecture
- Combined **two parallel U-Net paths** to process **local and global features**.

### 3.5 Advanced Architectures
- **DeepLabV3**: Used **Atrous Spatial Pyramid Pooling (ASPP)** for multi-scale feature extraction.
- **U-Net++**: Implemented **dense skip connections** and **deep supervision**.
- **TMS U-Net++ (Transformer Multi-Scale U-Net++)**: Combined **transformer-based feature processing** with **adaptive fusion**.

---

## 4. Loss Functions

Different loss functions were tested to handle **class imbalance**:
- **Focal Loss**: Focuses on hard-to-classify pixels.
- **Dice Loss**: Measures segmentation overlap.
- **Generalized Dice Loss**: Weighs each class based on representation.

**Best loss function**: **Focal Loss (0.8) + Dice Loss (0.2)**.

---

## 5. Experiments

### Model Performance Comparison (Mean IoU)

| Model               | Validation IoU (%) | Internal Test IoU (%) | Kaggle Test IoU (%) |
|---------------------|------------------|----------------------|----------------------|
| Single-layer CNN   | 15.12            | 13.68                | 13.84                |
| 3-Layer CNN        | 38.48            | 38.64                | 38.65                |
| U-Net Base         | 51.80            | 50.16                | 52.93                |
| U-Net TC           | 59.45            | 58.91                | 59.29                |
| U-Net RA           | **65.09**        | **65.80**            | **65.73**            |
| U-Net MSRA         | 62.00            | 62.19                | 62.35                |
| Double U-Net       | 59.49            | 59.51                | 59.49                |
| DeepLabV3          | 54.39            | 56.37                | 56.18                |
| U-Net++           | 60.45            | 62.88                | 62.56                |
| TMS U-Net++        | 59.96            | 61.89                | 60.07                |

**Best Model**: **U-Net RA**, achieving **65.80% IoU** on the internal test set.

---

## 6. Discussion

- **Augmentation**: Did not improve performance due to high training-test similarity.
- **Loss Function**: **Focal Loss (0.8) + Dice Loss (0.2)** was optimal for class imbalance.
- **Best Model**: **U-Net RA** significantly outperformed **baseline CNNs (13.68% IoU)**.
- **Challenges**:
  - **Extreme class imbalance**: **Big Rock** remains difficult to segment (**3.64% IoU**).
  - **Limited dataset**: Small training size constrains model generalization.
  - **No pretrained models**: Required training from scratch.

---

## 7. Conclusions & Future Work

### Key Findings:
- **U-Net RA** provided the best **overall performance** (65.73% IoU on Kaggle).
- **Big Rock** segmentation remains a major challenge.

### Future Improvements:
- **Hyperparameter tuning**: Further optimize **loss function weighting**.
- **Two-phase training**: Adjust **loss function weights dynamically**.
- **GAN-based augmentation**: Generate realistic **Big Rock** samples.
- **Ensemble models**: Combine **U-Net RA** and **TMS U-Net++** for better **Big Rock segmentation**.
- **Pretrained models**: Explore **transfer learning** to improve generalization.

---

## References

1. Ronneberger O. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* arXiv:1505.04597, 2015.
2. Chen X. *Residual Attention U-Net for Multi-Class Segmentation.* arXiv:2004.05645, 2020.
3. Jiang Y. *Multi-Scale Residual Attention Network for Retinal Vessel Segmentation.* Symmetry, 2021.
4. Bhatnagar V. *Double U-Net for Disease Diagnosis.* Springer, 2022.
5. Hamamoto K. *DeepLabV3+ for Single Image Reflection Removal.* Springer, 2024.
6. Zhou Z. *U-Net++: Nested U-Net for Medical Image Segmentation.* Springer, 2018.
7. Jadon S. *Survey of Loss Functions for Semantic Segmentation.* IEEE CIBCB, 2020.

---

## Additional Materials

All **code, notebooks, and augmented datasets** are available at:  
[Google Drive Link](https://drive.google.com/drive/folders/1KEJqVXhDvvg8zYORY3ezNl42Knewh__r)
