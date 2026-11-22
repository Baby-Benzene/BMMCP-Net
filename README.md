# BMMCP-Net: A Bitemporal Multi-scale Multi-direction Collaborative Perception Network for VHR Remote Sensing Change Detection

PyTorch implementation of **BMMCP-Net** (*"BMMCP-Net: A Bitemporal Multi-scale Multi-direction Collaborative Perception Network for VHR Remote Sensing Change Detection"*).

## 🌟 Overview

**BMMCP-Net** is a geometry-prior-driven change detection framework tailored for Very High Resolution (VHR) remote sensing imagery. It addresses three core challenges in Earth surface change monitoring—erosion of slender linear targets, boundary adhesion in dense urban settlements, and seasonal/shadow-induced pseudo-changes—through four synergistic components:

1. **Strip Pooling Feature Aggregation (SPFA)**: Captures orthogonal horizontal and vertical long-range context to preserve narrow roads, rivers, and coastlines without isotropic spatial smoothing.

2. **Twin-domain Filtering and Alignment Module (TFAM)**: Suppresses environmental and radiometric pseudo-changes via cascaded dilated differencing ($d \in \{7, 5, 3, 1\}$) and channel-spatial dual-domain filtering.

3. **Directional Shuffle Fusion Module (DSFM)**: Explicitly decouples and reassembles multi-directional ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) polygonal contours via directional 1D convolutions and parameter-free coordinate shifting.

4. **Content-Aware Progressive Decoder**: Integrates CARAFE upsampling, multi-scale Supervised Attention Modules (SAM), and a deep-semantic dual-domain gate (DualGate) to restore crisp, well-separated change boundaries.

## 🛠️ 1. Environment Setup

### Prerequisites

* Linux or Windows

* NVIDIA GPU with $\ge$ 8 GB VRAM

* Python $\ge$ 3.8

* PyTorch $\ge$ 1.12.0 (with CUDA support)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/Baby-Benzene/BMMCP-Net.git
cd BMMCP-Net
```

2. Create a conda environment and install dependencies:

```bash
conda create -n bmmcp python=3.8 -y
conda activate bmmcp
pip install torch torchvision opencv-python numpy matplotlib einops timm
```

3. Prepare ImageNet-1K pretrained weights for the **PVT-v2-B1** backbone (`pvt_v2_b1.pth`) and place it under `./pretrained/`.

## 📂 2. Repository Structure

```text
BMMCP-Net/
├── models/
│   ├── __init__.py
│   └── model_pvt_v2_b1.py      # Core BMMCP-Net (BaseNet) architecture with PVT-v2-B1, SPFA, TFAM, DSFM, SAM, and DualGate
├── tools/
│   ├── train.py                # Main script for training, validation, and automatic test-set evaluation
│   ├── test.py                 # Standalone inference and color-coded change map visualization script
│   └── torchutils.py           # PyTorch helper functions and training utilities
├── dataset.py                  # Bitemporal remote sensing dataset loader (LEVIR-CD, BCDD, SYSU-CD)
├── metric_tool.py              # Confusion matrix computation and metrics (Precision, Recall, F1, IoU, Kappa)
├── Transforms.py               # Joint bitemporal data augmentation and normalization transforms
├── utils.py                    # Visualization grid generation and logging helper utilities
└── README.md                   # Project documentation and usage instructions
```

## 🛰️ 3. Dataset Preparation

Please download the public benchmark datasets (**LEVIR-CD**, **BCDD**, and **SYSU-CD**), crop the bitemporal images into non-overlapping `256 × 256` patches, and organize the directory as follows:

```text
data/
├── LEVIR/
│   ├── train/
│   │   ├── A/               # Pre-change images (T1)
│   │   ├── B/               # Post-change images (T2)
│   │   └── label/           # Binary ground truth change masks
│   ├── val/
│   │   ├── A/
│   │   ├── B/
│   │   └── label/
│   └── test/
│       ├── A/
│       ├── B/
│       └── label/
├── BCDD/
│   └── ...
└── SYSU/
    └── ...
```

## 🚀 4. Training & Evaluation

### Training (with Automatic Test Evaluation)

`tools/train.py` executes end-to-end training with deep supervision (`1.0 * L_main + 0.4 * L_m2 + 0.2 * L_m3 + 0.1 * L_m4`), saves the best checkpoint (`best_model.pth`) based on validation F1-score, and automatically evaluates the best model on the test set upon completion.

*(Note: Before running, please ensure the dataset directory mapping in `tools/train.py` points to your local `./data/` path.)*

* **Train on LEVIR-CD**:

```bash
python tools/train.py --file_root LEVIR --inWidth 256 --inHeight 256 --batch_size 8 --max_epochs 200 --lr 3e-4 --lr_mode poly --savedir ./results --onGPU True
```

* **Train on BCDD**:

```bash
python tools/train.py --file_root BCDD --inWidth 256 --inHeight 256 --batch_size 8 --max_epochs 200 --lr 3e-4 --lr_mode poly --savedir ./results --onGPU True
```

* **Train on SYSU-CD**:

```bash
python tools/train.py --file_root SYSU --inWidth 256 --inHeight 256 --batch_size 8 --max_epochs 200 --lr 3e-4 --lr_mode poly --savedir ./results --onGPU True
```

### Standalone Testing & Visualization

To run standalone inference on the test set using a saved checkpoint and export color-coded prediction maps (TP: white, TN: black, FP: red, FN: blue):

```bash
python tools/test.py --file_root LEVIR --savedir ./results
```


## 📊 5. Quantitative Benchmark Results

Performance comparison on three public VHR change detection benchmarks (`%`):

| Dataset | Precision | Recall | F1-score | IoU |
| :--- | :---: | :---: | :---: | :---: |
| **LEVIR-CD** | 93.12 | 90.26 | **91.67** | **84.62** |
| **BCDD** | 94.76 | 90.36 | **92.51** | **86.06** |
| **SYSU-CD** | 86.85 | 80.77 | **83.70** | **71.97** |