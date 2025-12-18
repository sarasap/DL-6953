# ViT vs DeiT on ImageNet-100

This repository contains the implementation and experiments for training and evaluating **Vision Transformer (ViT-Tiny)** and **Data-efficient Image Transformer (DeiT-Tiny)** models on the **ImageNet-100** dataset.  
The project compares a baseline ViT model against a DeiT model trained with knowledge distillation and provides both quantitative and qualitative evaluations.

---

## Repository Structure

```text

├── train_deit_imagenet100.py      # Main training script (ViT + DeiT)
├── viz_random_10_vit_vs_deit.py   # Qualitative comparison script
├── requirements.txt               # Python dependencies
├── baseline_vit_tiny_best.pth/
├── Labels.json                   # WNID → human-readable label mapping
│── deit_tiny_distill_best.pth
└── README.md
`````
## Environment Setup

All required dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
`````
**Tested environment**
- Python 3.9+
- PyTorch with CUDA support
- NYU HPC Greene cluster

## Dataset and Data Splits

### Dataset Organization

- The dataset follows an **ImageNet-style directory structure** with class-wise folders.
- **Training data is split across 4 shard folders** to handle large-scale data efficiently:
  - `train.X1/`
  - `train.X2/`
  - `train.X3/`
  - `train.X4/`
- Each training shard contains:
  - Subfolders named by **WordNet IDs (WNIDs)**, one per class
  - All images belonging to that class within the shard

- **Validation data** is stored separately in a single folder:
  - `val.X/`
  - Uses the same WNID-based class folder structure as training

### Label Mapping

- A shared `Labels.json` file defines the mapping from **WNID → class index**
- Class indices are:
  - Created once
  - **Sorted and fixed across all shards**
  - Consistent between training and validation splits

### Training Split

- All four training shards are **combined logically at runtime**
- The training dataset is formed by aggregating images from:
  - `train.X1 + train.X2 + train.X3 + train.X4`
- Shuffling is enabled during training to ensure:
  - Mixing of samples across shards
  - Better stochastic optimization

### Validation Split

- Validation data is kept **fully disjoint** from training data
- No images from `val.X/` are used during training
- Validation is used only for:
  - Model evaluation
  - Selecting the best checkpoint based on Top-1 accuracy

### Summary

- **Train**: 4 shards merged into a single training set  (130,000 images, 1300 per class, 100 classes)
- **Validation**: 1 separate split  (5000 images, 50 images per class, 100 classes)
- **Class labels**: Stable and shared via `Labels.json`  
- This setup ensures **scalable data loading**, **reproducibility**, and **clean train/validation separation**


**Training Experiments**

**Train ViT-Tiny (Baseline)**

```bash
python train_deit_imagenet100.py \
  --data-path ./ \
  --model vit_tiny \
  --epochs 5 \
  --batch-size 128 \
  --lr 5e-4 \
  --run-name baseline_vit_tiny
```

**Train DeiT-Tiny (with Knowledge Distillation)**

```bash
python train_deit_imagenet100.py \
  --data-path ./ \
  --model deit_tiny_distill \
  --teacher resnet18 \
  --epochs 5 \
  --batch-size 128 \
  --lr 5e-4 \
  --run-name deit_tiny_distill
```
- Uses a ResNet-18 teacher
- Includes a distillation token
- Combines cross-entropy loss with soft distillation loss

## Evaluation and Qualitative Visualization

### ViT vs DeiT Qualitative Comparison

To qualitatively compare the predictions of **ViT-Tiny** and **DeiT-Tiny**, we use the following script:

```bash
python viz_random_10_vit_vs_deit.py
```

This script randomly samples 10 validation images, runs both ViT and DeiT on the same images and saves a comparison image (vit_vs_deit_random10.png). 
Displays for each image: Ground Truth (GT) ViT prediction DeiT prediction


## Pretrained Models and Logged Results

### Saved Checkpoints

The best-performing model checkpoints are stored in the baseline_vit_tiny_best.pth and deit_tiny_distill_best.pth.


### Checkpoint Contents

Each saved checkpoint includes:
- **Model weights**
- **Best validation Top-1 accuracy**
- **Epoch number** at which the best performance was achieved
- **Training configuration** (model type, hyperparameters, etc.)

### Logged Metrics

During training, the following metrics are tracked and logged:
- **Training loss**
- **Training Top-1 accuracy**
- **Validation loss**
- **Validation Top-1 accuracy**

Metrics are **logged at every epoch** and displayed in real time using **`tqdm` progress bars**, enabling easy monitoring of training and validation performance.





