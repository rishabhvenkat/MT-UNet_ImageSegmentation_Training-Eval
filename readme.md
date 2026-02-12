
# MT-UNet for Multi-Organ Medical Image Segmentation (Synapse)

## Overview

This repository contains a complete implementation, training pipeline, and evaluation setup for **MT-UNet** applied to the Synapse multi-organ abdominal CT segmentation dataset.

The project includes:

* Full MT-UNet architecture implementation
* End-to-end training pipeline
* GPU-enabled training using CUDA
* Evaluation using Dice Score and HD95
* NIfTI prediction export for volumetric segmentation

The model was trained from scratch and evaluated on the Synapse test set.

---

## Repository Structure

```
MT-UNet_ImageSegmentation_Training-Eval/
│
├── model/                      # MT-UNet architecture implementation
├── utils/                      # Loss functions and inference utilities
├── train_mtunet_Synapse.py     # Training and evaluation script (Synapse)
├── train_mtunet_ACDC.py
├── requirements.txt
├── LICENSE
└── README.md
```

The following directories are excluded from version control:

* dataset/
* checkpoint/
* predictions/
* venv/

---

## Dataset

Dataset: Synapse Multi-Organ Segmentation

Download from:
[https://zenodo.org/record/7860267](https://zenodo.org/record/7860267)

Expected directory structure:

```
dataset/
└── Synapse/
    ├── train/
    ├── test/
    └── lists_Synapse/
```

The dataset is not included in this repository due to size constraints.

---

## Environment Setup

1. Create and activate a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Install CUDA-enabled PyTorch (if using GPU):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Training Configuration

Training was performed with the following configuration:

* Epochs: 100
* Batch size: 12
* Image size: 224 × 224
* Optimizer: Adam
* Initial learning rate: 1e-4 (polynomial decay schedule)
* Loss function: 0.5 × Dice Loss + 0.5 × CrossEntropy Loss
* Number of classes: 9

Hardware used:

* NVIDIA GeForce RTX 4060
* CUDA 11.8
* Windows OS

Training time for 100 epochs: approximately 13 hours.

---

## Training

To train the model:

```
python train_mtunet_Synapse.py
```

Model checkpoints are saved in:

```
checkpoint/Synapse/mtunet/
```

---

## Evaluation

To evaluate a trained checkpoint:

```
python train_mtunet_Synapse.py --checkpoint checkpoint/Synapse/mtunet/epoch=99_lr=1.4439276885216547e-08.pth
```

Predictions are saved in:

```
predictions/
```

---

## Final Results (Synapse Test Set)

After 100 epochs of training:

* Mean Dice Score: 0.7589
* Mean HD95: 31.77

These results were obtained without extensive hyperparameter tuning and represent a reproduction-level implementation of MT-UNet on the Synapse dataset.

---

## Evaluation Metrics

* Dice Score: Measures overlap between predicted segmentation and ground truth.
* HD95 (Hausdorff Distance 95%): Measures boundary alignment robustness.

---

## Implementation Notes

During implementation and reproduction, the following engineering considerations were addressed:

* Removal of hardcoded CUDA calls to ensure device-agnostic execution.
* Proper device management for tensor creation to prevent CPU/GPU mismatches.
* Safe checkpoint saving.
* Windows-compatible NIfTI export handling.
* GPU verification via CUDA-enabled PyTorch installation.

---

## Reproducibility

To reproduce results:

1. Download and structure the Synapse dataset correctly.
2. Install dependencies and CUDA-enabled PyTorch.
3. Train for 100 epochs.
4. Evaluate using the provided checkpoint command.

---

## Author

Rishabh Venkataramanan
Electronics and Communication Engineering
Deep Learning and Medical Image Segmentation Implementation

```




This is the official implementation for our ICASSP2022 paper *MIXED TRANSFORMER UNET FOR MEDICAL IMAGE SEGMENTATION*

The entire code will be released upon paper publication.
