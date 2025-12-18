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
