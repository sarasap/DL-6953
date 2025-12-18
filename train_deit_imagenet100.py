import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# =========================
#  Vision Transformer (ViT / DeiT-Tiny style)
# =========================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Conv2d with kernel=stride=patch_size = patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)        # (B, C, H/ps, W/ps)
        x = x.flatten(2)        # (B, C, N)
        x = x.transpose(1, 2)   # (B, N, C)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=3, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, heads, N, head_dim)
        q, v, k = qkv[0], qkv[2], qkv[1]  # slight reordering but equivalent

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0,
                 attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (per-sample) — used in DeiT/ViT training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class VisionTransformerDeiT(nn.Module):
    """
    Small ViT / DeiT style model.

    If use_distillation=True, model returns (logits_cls, logits_dist).
    Otherwise it returns logits_cls only.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=100,
        embed_dim=192,      # DeiT-Tiny: 192
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_distillation=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = embed_dim
        self.use_distillation = use_distillation

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and optional DIST token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = 2 + num_patches
        else:
            self.dist_token = None
            num_tokens = 1 + num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth rate schedule
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Heads
        self.head = nn.Linear(embed_dim, num_classes)
        if use_distillation:
            self.head_dist = nn.Linear(embed_dim, num_classes)
        else:
            self.head_dist = None

        self._init_weights()

    def _init_weights(self):
        # Simple initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.head_dist is not None:
            nn.init.trunc_normal_(self.head_dist.weight, std=0.02)
            nn.init.zeros_(self.head_dist.bias)

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

        self.apply(_init)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)

        if self.use_distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)  # (B, 1, C)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)  # (B, 2+N, C)
        else:
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, C)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.use_distillation:
            cls_token_final = x[:, 0]          # (B, C)
            dist_token_final = x[:, 1]         # (B, C)
            logits_cls = self.head(cls_token_final)
            logits_dist = self.head_dist(dist_token_final)
            return logits_cls, logits_dist
        else:
            cls_token_final = x[:, 0]
            logits_cls = self.head(cls_token_final)
            return logits_cls


# =========================
#  Distillation Loss
# =========================

class DistillationLoss(nn.Module):
    """
    Combines standard CE loss on CLS output with
    soft distillation loss on DIST output.
    """

    def __init__(self, base_criterion, teacher_model,
                 distill_type="soft", alpha=0.5, tau=1.0):
        super().__init__()
        assert distill_type in ("soft", "none")
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distill_type = distill_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, outputs, inputs, targets):
        if isinstance(outputs, tuple):
            outputs_cls, outputs_dist = outputs
        else:
            outputs_cls, outputs_dist = outputs, None

        # CE loss on CLS head (ground truth)
        loss_cls = self.base_criterion(outputs_cls, targets)

        if self.distill_type == "none" or outputs_dist is None:
            return loss_cls

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        # Soft distillation with temperature
        T = self.tau
        loss_dist = F.kl_div(
            F.log_softmax(outputs_dist / T, dim=1),
            F.softmax(teacher_outputs / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        loss = self.alpha * loss_cls + (1.0 - self.alpha) * loss_dist
        return loss


# =========================
#  Training & Evaluation
# =========================

def accuracy(output, target, topk=(1,)):
    """Compute the top-k accuracy"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@dataclass
class AverageMeter:
    name: str
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, teacher, criterion, optimizer, dataloader, device, epoch):
    model.train()
    if teacher is not None:
        teacher.eval()

    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, targets in loop:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, images, targets)
        if isinstance(outputs, tuple):
            outputs_for_acc = outputs[0]
        else:
            outputs_for_acc = outputs

        acc1 = accuracy(outputs_for_acc, targets, topk=(1,))[0]

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))

        loop.set_postfix(loss=loss_meter.avg, acc1=acc1_meter.avg)

    return loss_meter.avg, acc1_meter.avg


def evaluate(model, dataloader, device):
    model.eval()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Eval", leave=False)
        for images, targets in loop:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs_for_acc = outputs[0]
            else:
                outputs_for_acc = outputs

            loss = criterion(outputs_for_acc, targets)
            acc1 = accuracy(outputs_for_acc, targets, topk=(1,))[0]

            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            loop.set_postfix(loss=loss_meter.avg, acc1=acc1_meter.avg)

    return loss_meter.avg, acc1_meter.avg


# =========================
#  Data
# =========================
import json
from PIL import Image
from torch.utils.data import Dataset

class ImageNetShardDataset(Dataset):
    def __init__(self, shard_dirs, label_json, transform=None):
        """
        shard_dirs: list of directories (train.X1, train.X2, ...)
        label_json: path to label.json
        """
        with open(label_json, "r") as f:
            wnid_to_name = json.load(f)

        # Create stable class index mapping
        self.wnids = sorted(wnid_to_name.keys())
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}


        self.transform = transform
        self.samples = []

        for shard in shard_dirs:
            for wnid in os.listdir(shard):
                wnid_path = os.path.join(shard, wnid)
                if not os.path.isdir(wnid_path):
                    continue

                if wnid not in self.class_to_idx:
                    continue

                label = self.class_to_idx[wnid]  # INTEGER label

                for fname in os.listdir(wnid_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(
                            (os.path.join(wnid_path, fname), label)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def build_dataloaders(data_path, batch_size, num_workers=4):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    label_json = os.path.join(data_path, "Labels.json")

    # ---- TRAIN: multiple shards ----
    train_shards = [
        os.path.join(data_path, d)
        for d in os.listdir(data_path)
        if d.startswith("train.X")
    ]

    train_dataset = ImageNetShardDataset(
        shard_dirs=train_shards,
        label_json=label_json,
        transform=train_transform,
    )

    # ---- VAL: standard folder but mapped via label.json ----
    val_dataset = ImageNetShardDataset(
        shard_dirs=[os.path.join(data_path, "val.X")],
        label_json=label_json,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_classes = len(train_dataset.class_to_idx)

    return train_loader, val_loader, num_classes


# =========================
#  Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DeiT-Tiny / ViT-Tiny on ImageNet-100 with optional distillation"
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to ImageNet-100 root (with train/ and val/).")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (keep <= 5 for course).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="vit_tiny",
        choices=["vit_tiny", "deit_tiny_distill"],
        help="Choose baseline ViT or DeiT with distillation.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="none",
        choices=["none", "resnet18"],
        help="Teacher model (for distillation).",
    )
    parser.add_argument("--distill-alpha", type=float, default=0.5,
                        help="Weight for CE loss vs distillation loss.")
    parser.add_argument("--distill-temp", type=float, default=1.0,
                        help="Temperature for soft distillation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    return parser.parse_args()


def build_teacher(teacher_name, num_classes, device):
    if teacher_name == "none":
        return None

    if teacher_name == "resnet18":
        teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace final layer for ImageNet-100
        in_feats = teacher.fc.in_features
        teacher.fc = nn.Linear(in_feats, num_classes)
        # In a perfect world you'd fine-tune teacher on ImageNet-100 first.
        # For simplicity, we just train it jointly or briefly beforehand.
        return teacher.to(device)
    else:
        raise ValueError(f"Unknown teacher: {teacher_name}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = build_dataloaders(
        args.data_path, args.batch_size, args.num_workers
    )
    print(f"Found {num_classes} classes.")
    print(len(train_loader.dataset))
    print(len(val_loader.dataset)) 
    use_distillation = args.model == "deit_tiny_distill"

    model = VisionTransformerDeiT(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        use_distillation=use_distillation,
    ).to(device)

    teacher = None
    base_criterion = nn.CrossEntropyLoss()

    if use_distillation and args.teacher != "none":
        teacher = build_teacher(args.teacher, num_classes, device)
        # Optionally: freeze teacher if it's pre-trained
        for p in teacher.parameters():
            p.requires_grad = False

        criterion = DistillationLoss(
            base_criterion=base_criterion,
            teacher_model=teacher,
            distill_type="soft",
            alpha=args.distill_alpha,
            tau=args.distill_temp,
        )
    else:
        criterion = DistillationLoss(
            base_criterion=base_criterion,
            teacher_model=None,
            distill_type="none",
            alpha=1.0,
            tau=1.0,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc1 = train_one_epoch(
            model, teacher, criterion, optimizer, train_loader, device, epoch
        )
        val_loss, val_acc1 = evaluate(model, val_loader, device)

        print(
            f"[{args.run_name}] Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc1={train_acc1:.2f}, "
            f"val_loss={val_loss:.4f}, val_acc1={val_acc1:.2f}"
        )

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", f"{args.run_name}_best.pth")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc1": best_acc1,
                    "epoch": epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f" New best model saved to: {ckpt_path}")

    print(f"Best val top-1 accuracy for {args.run_name}: {best_acc1:.2f}")


if __name__ == "__main__":
    main()
