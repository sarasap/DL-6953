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

**Training Experiments**

**Train ViT-Tiny (Baseline)**

```bash
python train_deit_imagenet100.py \
  --data-path /path/to/imagenet100 \
  --model vit_tiny \
  --epochs 5 \
  --batch-size 128 \
  --lr 5e-4 \
  --run-name baseline_vit_tiny
```

**Train DeiT-Tiny (with Knowledge Distillation)**

```bash
python train_deit_imagenet100.py \
  --data-path /path/to/imagenet100 \
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

