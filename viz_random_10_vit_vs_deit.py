import os
import random
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # HPC-safe
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import Subset

# =========================
# CONFIG
# =========================
VAL_ROOT = "/home/sp8049/code/DL_project/val.X"
LABELS_JSON = "/home/sp8049/code/DL_project/Labels.json"

VIT_CKPT = "/home/sp8049/code/DL_project/checkpoints/baseline_vit_tiny_best.pth"
DEIT_CKPT = "/home/sp8049/code/DL_project/checkpoints/deit_tiny_distill_best.pth"

OUT_PNG = "vit_vs_deit_random10_3.png"
NUM_IMAGES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS (MATCH TRAINING)
# =========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# =========================
# DATASET
# =========================
dataset = datasets.ImageFolder(VAL_ROOT, transform=transform)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

with open(LABELS_JSON, "r") as f:
    wnid_to_name = json.load(f)

# =========================
# SAMPLE 10 RANDOM IMAGES
# =========================
all_indices = list(range(len(dataset)))
chosen_indices = random.sample(all_indices, NUM_IMAGES)
subset = Subset(dataset, chosen_indices)

# =========================
# MODEL
# =========================
from train_deit_imagenet100 import VisionTransformerDeiT


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state"]

    has_distill = any(k.startswith("dist_token") for k in state.keys())

    model = VisionTransformerDeiT(
        img_size=224,
        patch_size=16,
        num_classes=len(dataset.classes),
        embed_dim=192,
        depth=12,
        num_heads=3,
        use_distillation=has_distill,
    ).to(DEVICE)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


vit_model = load_model(VIT_CKPT)
deit_model = load_model(DEIT_CKPT)

# =========================
# PLOT: 10 ROWS × 1 COLUMN
# =========================
fig, axes = plt.subplots(NUM_IMAGES, 1, figsize=(6, 2.5 * NUM_IMAGES))

if NUM_IMAGES == 1:
    axes = [axes]


# =========================
# INFERENCE + VISUALIZATION
# =========================
with torch.no_grad():
    for i, (img, label) in enumerate(subset):
        img_t = img.unsqueeze(0).to(DEVICE)

        # ---- ViT ----
        vit_out = vit_model(img_t)
        if isinstance(vit_out, tuple):
            vit_out = vit_out[0]
        vit_pred = vit_out.argmax(dim=1).item()

        # ---- DeiT ----
        deit_out = deit_model(img_t)
        if isinstance(deit_out, tuple):
            deit_out = deit_out[0]
        deit_pred = deit_out.argmax(dim=1).item()

        # ---- Labels ----
        gt_wnid = idx_to_class[label]
        vit_wnid = idx_to_class[vit_pred]
        deit_wnid = idx_to_class[deit_pred]

        gt_name = wnid_to_name.get(gt_wnid, gt_wnid)
        vit_name = wnid_to_name.get(vit_wnid, vit_wnid)
        deit_name = wnid_to_name.get(deit_wnid, deit_wnid)

        # ---- Unnormalize image ----
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + \
                 np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        ax = axes[i]
        ax.imshow(img_np)
        ax.axis("off")

        vit_color = "green" if vit_wnid == gt_wnid else "red"
        deit_color = "green" if deit_wnid == gt_wnid else "red"

        # ---- TEXT BURNED INTO IMAGE ----
        ax.imshow(img_np)
        ax.axis("off")

        vit_color = "green" if vit_wnid == gt_wnid else "red"
        deit_color = "green" if deit_wnid == gt_wnid else "red"

        # ---- TEXT BELOW IMAGE (NOT OVER IT) ----
        ax.set_title(
            f"GT: {gt_name}\n"
            f"ViT: {vit_name}\n"
            f"DeiT: {deit_name}",
            fontsize=10,
            loc="left",
            color="black"
        )

        print(
            f"Image {i+1:02d} | "
            f"GT: {gt_name} | "
            f"ViT: {vit_name} | "
            f"DeiT: {deit_name}"
        )

# =========================
# SAVE
# =========================

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close()

print(f"\n[INFO] Saved figure to {OUT_PNG}")
