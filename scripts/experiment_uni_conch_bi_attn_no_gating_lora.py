"""
Experiment — UNI+CONCH + Bi-ATTN (No gating) + LoRA
---------------------------------------------------
Dual encoder with bidirectional attention but no gating mechanism.
UNI features are frozen and cached, CONCH uses LoRA adaptation.
Features are combined via bidirectional attention without gating,
then passed through a classifier.

Based on the Bi-ATTN implementation in phase_2_bi_attn_dynamic_feature_wt.py
but without the gating mechanism and with LoRA applied to CONCH.

Results saved to: ./results/experiment_uni_conch_bi_attn_no_gating_lora_results.txt
"""

import warnings
import os
import logging
import math
import gc
import sys
import io
import random

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from conch.open_clip_custom import create_model_from_pretrained
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
CONCH_WEIGHTS  = "/home/d3vd5k/Projects/CANCER-RESEARCH/CONCH.bin"
TRAIN_DIR      = "/home/d3vd5k/Projects/CANCER-RESEARCH/Chaoyang/organized_images_train"
TEST_DIR       = "/home/d3vd5k/Projects/CANCER-RESEARCH/Chaoyang/organized_images_test"
UNI_TRAIN_CACHE = "/home/d3vd5k/Projects/CANCER-RESEARCH/features/uni_train_features.pt"
UNI_TEST_CACHE  = "/home/d3vd5k/Projects/CANCER-RESEARCH/features/uni_test_features.pt"
RESULTS_DIR    = "/home/d3vd5k/Projects/chaoyang-conch-research/results"

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES    = 4
COMMON_DIM     = 256
NUM_HEADS      = 4
HIDDEN_DIM     = 256
DROPOUT        = 0.4
BATCH_SIZE     = 16
EPOCHS         = 100
WARMUP_EPOCHS  = 15
LR_ENCODER     = 1e-4
LR_FUSION      = 1e-4
LR_HEAD        = 1e-3
WEIGHT_DECAY   = 1e-2
NOISE_STDDEV   = 0.3
SEED           = 42

# ── LoRA Hyperparameters ──────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.1
LORA_TARGET_MODULES = ["qkv", "proj"]
LORA_STOCH_DEPTH    = 0.15

CLASS_NAMES = {0: "Normal", 1: "Serrated", 2: "Adenocarcinoma", 3: "Adenoma"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ── Dataset & Transforms ──────────────────────────────────────────────────────

class DualDataset(Dataset):
    def __init__(self, cache_path: str, image_dir: str, transform=None, verify: bool = True):
        cache = torch.load(cache_path, weights_only=False)
        self.uni_features  = cache["features"].float()
        self.labels        = cache["labels"].long()
        self.cached_paths  = cache.get("paths", None)
        self.transform     = transform
        self.image_folder  = datasets.ImageFolder(root=image_dir)

    def __len__(self):
        return len(self.uni_features)

    def __getitem__(self, idx):
        uni_feat = self.uni_features[idx]
        img_path, label = self.image_folder.samples[idx]
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
        image = self.transform(img_rgb) if self.transform else transforms.ToTensor()(img_rgb)
        return uni_feat, image, label

def build_transforms(conch_preprocess):
    conch_normalize = next((t for t in conch_preprocess.transforms if isinstance(t, transforms.Normalize)),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        conch_normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        conch_normalize
    ])
    return train_transform, test_transform

# ── CONCH Encoder Setup ───────────────────────────────────────────────────────

def load_conch_lora(weights_path: str, device: str):
    print(f"Loading CONCH from: {weights_path}")
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=weights_path)
    encoder = model.visual
    encoder.to(device)

    for param in encoder.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
    )
    encoder = get_peft_model(encoder, lora_config)

    if LORA_STOCH_DEPTH > 0:
        def apply_lora_stochastic_depth(model, drop_prob):
            def hook(module, inp, out):
                if module.training:
                    if torch.rand(1).item() < drop_prob:
                        return torch.zeros_like(out)
                    return out / (1.0 - drop_prob)
                return out

            count = 0
            for name, module in model.named_modules():
                if 'lora_B' in name:
                    module.register_forward_hook(hook)
                    count += 1
            print(f"  Applied Stochastic Depth (p={drop_prob}) to {count} LoRA modules.")
        apply_lora_stochastic_depth(encoder, LORA_STOCH_DEPTH)

    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        out   = encoder(dummy)
        feat  = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'last_hidden_state', out)[:, 0] if hasattr(out, 'last_hidden_state') else out
        conch_dim = feat.shape[-1]
    torch.cuda.empty_cache()

    return encoder, preprocess, conch_dim

# ── Bidirectional Attention Fusion Head (NO GATING) ────────────────────────────

class BidirectionalFusionHeadNoGating(nn.Module):
    """
    Bidirectional attention fusion WITHOUT gating mechanism.
    Simply averages the two attended representations.
    """
    def __init__(self, uni_dim, conch_dim, common_dim, num_heads, hidden_dim, num_classes, dropout, noise_stddev):
        super().__init__()
        self.noise_stddev = noise_stddev

        self.uni_proj   = nn.Linear(uni_dim,   common_dim)
        self.conch_proj = nn.Linear(conch_dim, common_dim)

        self.cross_attn_U_C = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_C_U = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(common_dim)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, uni_feat, conch_feat):
        if self.training and self.noise_stddev > 0:
            uni_feat   = uni_feat   + torch.randn_like(uni_feat)   * self.noise_stddev
            conch_feat = conch_feat + torch.randn_like(conch_feat) * self.noise_stddev

        u_proj = self.uni_proj(uni_feat).unsqueeze(1)
        c_proj = self.conch_proj(conch_feat).unsqueeze(1)

        # UNI attends to CONCH
        u_attended, _ = self.cross_attn_U_C(query=u_proj, key=c_proj, value=c_proj)
        # CONCH attends to UNI
        c_attended, _ = self.cross_attn_C_U(query=c_proj, key=u_proj, value=u_proj)

        # Residual connections
        u_out = u_attended.squeeze(1) + u_proj.squeeze(1)
        c_out = c_attended.squeeze(1) + c_proj.squeeze(1)

        # NO GATING: Simply average the two attended features
        fused = (u_out + c_out) / 2.0

        fused = self.norm(fused)
        return self.classifier(fused)

class DualEncoderModel(nn.Module):
    def __init__(self, conch_encoder, fusion_head):
        super().__init__()
        self.conch_encoder = conch_encoder
        self.fusion_head   = fusion_head

    def forward(self, uni_feat, image):
        out = self.conch_encoder(image)
        conch_feat = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'last_hidden_state', out)[:, 0] if hasattr(out, 'last_hidden_state') else out
        return self.fusion_head(uni_feat, conch_feat)

# ── Scheduler & Evaluation ────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        opt = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
        self.scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=self._lr_lambda(warmup_epochs, total_epochs, min_lr_ratio))

    @staticmethod
    def _lr_lambda(warmup, total, min_ratio):
        def fn(epoch):
            if epoch < warmup: return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return fn

    def step(self): self.scheduler.step()

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for uni_feat, images, labels in loader:
            logits = model(uni_feat.to(device), images.to(device))
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

def print_metrics(preds, labels, split_name: str, class_names: dict):
    """Print macro-averaged metrics matching comparison paper format."""
    acc       = accuracy_score(labels, preds) * 100
    precision = precision_score(labels, preds, average='macro',
                                zero_division=0) * 100
    recall    = recall_score(labels, preds, average='macro',
                             zero_division=0) * 100
    f1        = f1_score(labels, preds, average='macro',
                         zero_division=0) * 100

    print(f"\n── {split_name} Results (Macro-Averaged) ──")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {precision:.2f}%")
    print(f"  Recall    : {recall:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")

    print(f"\n── {split_name} Per-Class Breakdown ──")
    report = classification_report(
        labels, preds,
        target_names=list(class_names.values()),
        digits=4
    )
    print(report)

    print(f"── Confusion Matrix ──")
    cm = confusion_matrix(labels, preds)
    header = "         " + "  ".join(f"{n[:6]:>8}" for n in class_names.values())
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:8d}" for v in row)
        print(f"  {list(class_names.values())[i][:6]:>8}  {row_str}")

    return acc, precision, recall, f1

# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Setup ──
    conch_encoder, conch_preprocess, conch_dim = load_conch_lora(CONCH_WEIGHTS, DEVICE)
    train_transform, test_transform = build_transforms(conch_preprocess)

    train_dataset = DualDataset(UNI_TRAIN_CACHE, TRAIN_DIR, transform=train_transform)
    test_dataset  = DualDataset(UNI_TEST_CACHE, TEST_DIR, transform=test_transform)

    # Simple weighting for class imbalance (no focal loss for simplicity in this experiment)
    counts = torch.bincount(train_dataset.labels, minlength=NUM_CLASSES).float()
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    sampler = WeightedRandomSampler(weights=class_weights[train_dataset.labels], num_samples=len(train_dataset.labels), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    fusion_head = BidirectionalFusionHeadNoGating(
        uni_dim=1024, conch_dim=conch_dim, common_dim=COMMON_DIM,
        num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
        dropout=DROPOUT, noise_stddev=NOISE_STDDEV
    )
    model = DualEncoderModel(conch_encoder, fusion_head).to(DEVICE)

    # ── Optimizers & Loss ──
    param_groups = [
        {'params': [p for p in model.conch_encoder.parameters() if p.requires_grad], 'lr': LR_ENCODER},
        {'params': list(model.fusion_head.uni_proj.parameters()) + list(model.fusion_head.conch_proj.parameters()) +
                   list(model.fusion_head.cross_attn_U_C.parameters()) + list(model.fusion_head.cross_attn_C_U.parameters()) +
                   list(model.fusion_head.norm.parameters()), 'lr': LR_FUSION},
        {'params': model.fusion_head.classifier.parameters(), 'lr': LR_HEAD}
    ]

    base_opt = optim.AdamW
    optimizer = base_opt(param_groups, weight_decay=WEIGHT_DECAY)

    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    criterion = nn.CrossEntropyLoss()
    print("  Using CrossEntropy Loss")

    scaler = torch.cuda.amp.GradScaler()

    # Track best metrics
    best_f1 = 0.0
    best_f1_epoch = 0
    best_acc = 0.0
    best_acc_epoch = 0
    best_f1_state = None
    best_acc_state = None

    # Base config for saving
    save_config = {
        'adapter': 'lora',
        'lora_r': LORA_R,
        'lora_alpha': LORA_ALPHA,
        'lora_dropout': LORA_DROPOUT,
        'target_modules': LORA_TARGET_MODULES,
        'common_dim': COMMON_DIM,
        'num_heads': NUM_HEADS,
        'hidden_dim': HIDDEN_DIM,
        'dropout': DROPOUT,
        'lr_encoder': LR_ENCODER,
        'lr_fusion': LR_FUSION,
        'lr_head': LR_HEAD,
        'conch_dim': conch_dim,
        'uni_dim': 1024
    }

    print("=" * 60)
    print("  Experiment — UNI+CONCH + Bi-ATTN (No gating) + LoRA")
    print("=" * 60)
    print(f"Device      : {DEVICE.upper()}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Epochs      : {EPOCHS} (warmup: {WARMUP_EPOCHS})")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Common dim  : {COMMON_DIM}")
    print(f"Attn heads  : {NUM_HEADS}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_steps = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS}", leave=True, dynamic_ncols=True)

        for uni_feat, images, labels in pbar:
            uni_feat, images, labels = uni_feat.to(DEVICE), images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(uni_feat, images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_steps += 1

            pbar.set_postfix(loss=f"{total_loss/total_steps:.4f}")

        scheduler.step()
        avg_loss = total_loss / total_steps

        # Evaluate every epoch
        preds, labels_np = evaluate(model, test_loader, DEVICE)
        f1  = f1_score(labels_np, preds, average='macro', zero_division=0) * 100
        acc = accuracy_score(labels_np, preds) * 100

        warmup_tag = " [warmup]" if epoch <= WARMUP_EPOCHS else ""
        print(f"  Epoch {epoch:3d}/{EPOCHS}{warmup_tag} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.2f}% | F1: {f1:.2f}%")

        # Track and save best F1
        if f1 > best_f1:
            best_f1 = f1
            best_f1_epoch = epoch
            best_f1_state = {k: v.clone() for k, v in model.state_dict().items()}

            torch.save({
                'epoch': best_f1_epoch, 'model_state': best_f1_state,
                'accuracy': acc, 'f1': best_f1, 'config': save_config
            }, os.path.join(RESULTS_DIR, "exp_uni_conch_bi_attn_no_gating_lora_best_f1.pt"))
            print(f"  🟢 [Checkpoint] New Best F1 Saved ({f1:.2f}%)!")

        # Track and save best accuracy
        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch
            best_acc_state = {k: v.clone() for k, v in model.state_dict().items()}

            torch.save({
                'epoch': best_acc_epoch, 'model_state': best_acc_state,
                'accuracy': best_acc, 'f1': f1, 'config': save_config
            }, os.path.join(RESULTS_DIR, "exp_uni_conch_bi_attn_no_gating_lora_best_acc.pt"))
            print(f"  🔵 [Checkpoint] New Best Accuracy Saved ({acc:.2f}%)!")

        if epoch % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ── Final evaluation (Using Best F1 Model) ───────────────────────────────
    print(f"\nTraining Complete!")
    ckpt = torch.load(os.path.join(RESULTS_DIR, "exp_uni_conch_bi_attn_no_gating_lora_best_acc.pt"))
    best_acc_f1 = ckpt['f1']

    print(f"🏆 Best F1 Model       : Epoch {best_f1_epoch} (F1: {best_f1:.2f}%, Acc: {accuracy_score(labels_np, preds)*100:.2f}%)")
    print(f"🎯 Best Accuracy Model : Epoch {best_acc_epoch} (Acc: {best_acc:.2f}%, F1: {best_acc_f1:.2f}%)")

    # Load Best F1 state for final output
    model.load_state_dict(best_f1_state)
    preds, labels_np = evaluate(model, test_loader, DEVICE)
    per_class_f1 = f1_score(labels_np, preds, average=None, zero_division=0) * 100

    print(f"Best Accuracy Model F1: {best_acc_f1:.2f}%")

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("  Experiment — UNI+CONCH + Bi-ATTN (No gating) + LoRA")
    results_lines.append("=" * 60)
    results_lines.append(f"Peak F1 Epoch   : {best_f1_epoch}/{EPOCHS}")
    results_lines.append(f"Peak Acc Epoch  : {best_acc_epoch}/{EPOCHS}")
    results_lines.append(f"Common dim      : {COMMON_DIM}")
    results_lines.append(f"Attn heads      : {NUM_HEADS}")
    results_lines.append("")

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    acc, prec, rec, f1 = print_metrics(preds, labels_np, "TEST (Based on Best F1 Checkpoint)", CLASS_NAMES)
    sys.stdout = old_stdout
    captured = buffer.getvalue()
    print(captured)
    results_lines.append(captured)

    comparison = f"""
── Comparison Table ──
  Method                                    Acc      F1
  ────────────────────────────────────────────────────────
  HSA-NRL (ResNet-34 + noise-robust)        83.40%   76.54%
  Ke et al. (CNN ensemble + MHSA)           86.72%   85.76%
  Belaskri et al. (ViT-B/16 + RCC)         87.14%   82.77%
  Exp 5 — CONCH partial fine-tune           86.02%   80.87%
  UNI+CONCH + Cross-attention               ?        ?
  UNI+CONCH + MHSA                          ?        ?
  UNI+CONCH + Bi-ATTN + GLU + LoRA          87.89%   ?
  UNI+CONCH + Bi-ATTN + Linear gate + LoRA  88.03%   ?
  UNI+CONCH + Bi-ATTN (No gating) + LoRA    {acc:.2f}%   {f1:.2f}%
"""
    print(comparison)
    results_lines.append(comparison)

    # Per-class F1 comparison
    per_class_comparison = f"""
── Per-Class F1 — Comparison with Best Gated Version ──
  Class             Bi-ATTN+LoRA (no gate)   Bi-ATTN+LoRA+gate (from README)   Change
  ───────────────────────────────────────────────────────────────────────────────
  Normal           {per_class_f1[0]:.2f}%   ?                                 ?
  Serrated         {per_class_f1[1]:.2f}%   ?                                 ?
  Adenocarcinoma   {per_class_f1[2]:.2f}%   ?                                 ?
  Adenoma          {per_class_f1[3]:.2f}%   ?                                 ?
"""
    print(per_class_comparison)
    results_lines.append(per_class_comparison)

    # Save results
    results_path = os.path.join(RESULTS_DIR, "experiment_uni_conch_bi_attn_no_gating_lora_results.txt")
    with open(results_path, 'w') as f:
        f.write("\n".join(results_lines))
    print(f"✅ Results saved to: {results_path}")

    # Save final model
    final_model_path = os.path.join(RESULTS_DIR, "experiment_uni_conch_bi_attn_no_gating_lora_final_model.pt")
    torch.save({
        'epoch': best_f1_epoch,
        'model_state': best_f1_state,
        'accuracy': acc,
        'f1': f1,
        'config': save_config
    }, final_model_path)
    print(f"✅ Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()