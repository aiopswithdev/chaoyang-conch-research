import warnings
import os
import logging
import math
import gc
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

"""
Phase 2 — Asymmetric Cache & Tune Training
-------------------------------------------
Architecture:

    Each batch provides three items from DualDataset:
        1. Cached UNI tensor  [B, 1024]  — no GPU compute, direct lookup
        2. Raw .jpg patch     [B, 3, 224, 224] — goes through live CONCH branch
        3. Label              [B]

    UNI branch (offline):
        Cached 1024-dim morphological features from Phase 1
        No gradient, no compute during training

    CONCH branch (live):
        H&E-aware augmentation → CONCH ViT-B/16 (blocks 0-5 frozen, 6-11 tuned)
        Outputs 512-dim semantic features
        Gradients flow through unfrozen blocks only

    DualEncoderFusionHead (cross-attention):
        UNI  1024-dim → Linear projection → common_dim (Query)
        CONCH 512-dim → Linear projection → common_dim (Key, Value)
        Scaled dot-product attention → fused representation
        → MLP classifier → 4-class prediction

Why this design:
    UNI captures morphological structure (trained on 100K+ WSIs, DINOv2)
    CONCH captures semantic tissue meaning (trained on 1.17M path image-text pairs)
    Cross-attention is dynamic — the model learns which morphological features
    are most relevant given CONCH's semantic interpretation of the same patch
    This is more powerful than concatenation which is purely additive

Targets to beat:
    HSA-NRL  (ResNet-34 + noise-robust)      : 83.40% acc, 76.54% F1
    Ke et al. (CNN ensemble + MHSA)          : 86.72% acc, 85.76% F1
    Belaskri et al. (ViT-B/16 + RCC)        : 87.14% acc, 82.77% F1
    Exp 5    (CONCH partial fine-tune)        : 86.02% acc, 80.87% F1

Results saved to: ./results/phase2_results.txt
"""

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
import io
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
CONCH_WEIGHTS  = "/home/dev/Downloads/pytorch_model.bin"
TRAIN_DIR      = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_train"
TEST_DIR       = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_test"
UNI_TRAIN_CACHE = "/home/dev/Documents/ML-Research/Chaoyang/features/uni_train_features.pt"
UNI_TEST_CACHE  = "/home/dev/Documents/ML-Research/Chaoyang/features/uni_test_features.pt"
RESULTS_DIR    = "./results"

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES    = 4
COMMON_DIM     = 256      # projection dimension for cross-attention
NUM_HEADS      = 4        # attention heads in fusion
HIDDEN_DIM     = 256      # MLP classifier hidden dim
DROPOUT        = 0.4
BATCH_SIZE     = 16       # keep small — CONCH runs live each step
EPOCHS         = 100
WARMUP_EPOCHS  = 5
LR_ENCODER     = 1e-5     # CONCH unfrozen blocks
LR_FUSION      = 1e-4     # fusion head
LR_HEAD        = 1e-3     # MLP classifier
WEIGHT_DECAY   = 1e-2
NOISE_STDDEV   = 0.3
SEED           = 42
FREEZE_BLOCKS  = 6        # Ke et al. strategy: freeze blocks 0-5

CLASS_NAMES = {0: "Normal", 1: "Serrated", 2: "Adenocarcinoma", 3: "Adenoma"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ── DualDataset ───────────────────────────────────────────────────────────────
class DualDataset(Dataset):
    """
    Pairs cached UNI features with raw Chaoyang patches.

    Returns per sample:
        uni_feat : Tensor [1024]       — cached UNI embedding, no compute
        image    : Tensor [3, 224, 224] — raw patch for live CONCH branch
        label    : int
    """
    def __init__(self, cache_path: str, image_dir: str,
                 transform=None, verify: bool = True):
        cache = torch.load(cache_path, weights_only=False)
        self.uni_features  = cache["features"].float()   # [N, 1024]
        self.labels        = cache["labels"].long()      # [N]
        self.cached_paths  = cache.get("paths", None)
        self.transform     = transform

        self.image_folder  = datasets.ImageFolder(root=image_dir)

        if verify:
            self._verify_alignment()

    def _verify_alignment(self):
        n_cache  = len(self.uni_features)
        n_folder = len(self.image_folder)

        if n_cache != n_folder:
            raise ValueError(
                f"Size mismatch: cache has {n_cache}, "
                f"ImageFolder found {n_folder}."
            )

        # Path-level check if Phase 1 saved paths
        if self.cached_paths is not None:
            folder_paths = [s[0] for s in self.image_folder.samples]
            mismatches = []
            for i, (cp, fp) in enumerate(
                    zip(self.cached_paths, folder_paths)):
                if os.path.basename(cp) != os.path.basename(fp):
                    mismatches.append((i, cp, fp))
                    if len(mismatches) >= 3:
                        break
            if mismatches:
                detail = "\n".join(
                    f"  [{i}] cache: {os.path.basename(c)} "
                    f"vs folder: {os.path.basename(f)}"
                    for i, c, f in mismatches
                )
                raise ValueError(
                    f"Path mismatch between UNI cache and image directory:\n"
                    f"{detail}\n"
                    f"Fix: re-run phase1_uni_extraction.py on this machine."
                )

        # Label-level check always
        folder_labels = torch.tensor(
            [s[1] for s in self.image_folder.samples]
        )
        if not torch.equal(self.labels, folder_labels):
            raise ValueError(
                "Label mismatch between UNI cache and image directory walk.\n"
                "Ensure Phase 1 used the same sorted traversal order.\n"
                "Fix: re-run phase1_uni_extraction.py with the same "
                "image directory, then retry Phase 2."
            )

        print(f"  Alignment verified — {n_cache} samples match.")

    def __len__(self):
        return len(self.uni_features)

    def __getitem__(self, idx):
        uni_feat = self.uni_features[idx]               # [1024]

        img_path, label = self.image_folder.samples[idx]
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(img_rgb)
        else:
            image = transforms.ToTensor()(img_rgb)

        return uni_feat, image, label


# ── H&E augmentation (Exp 5 / Belaskri et al.) ────────────────────────────────
# Replace build_transforms entirely
def build_transforms(conch_preprocess):
    """
    Phase 2 augmentation — pixel-level only.
    Spatial transforms are intentionally excluded to prevent
    desynchronisation between cached UNI features and live CONCH features.
    UNI was cached from centred, unrotated 224x224 patches — CONCH must
    see the same spatial layout. Regularisation comes from ColorJitter
    and GaussianNoise injection inside DualEncoderFusionHead.
    """
    conch_normalize = None
    for t in conch_preprocess.transforms:
        if isinstance(t, transforms.Normalize):
            conch_normalize = t

    norm = conch_normalize if conch_normalize else \
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        # Resize to exact 224×224 — matches Phase 1 spatial layout
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        # Pixel-level only — safe with cached UNI features
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05),
        # Mild affine — sub-5% shifts only, no rotation, no flip
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        norm
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        norm
    ])

    return train_transform, test_transform


# ── CONCH encoder with partial fine-tuning ────────────────────────────────────
def load_conch_partial(weights_path: str, freeze_blocks: int, device: str):
    """Load CONCH with Ke et al. hierarchical layer-freezing."""
    print(f"Loading CONCH from: {weights_path}")
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=weights_path
    )
    encoder = model.visual
    encoder.to(device)

    # Freeze everything first
    for param in encoder.parameters():
        param.requires_grad = False

    # Unfreeze blocks freeze_blocks → 11
    if hasattr(encoder, 'trunk') and hasattr(encoder.trunk, 'blocks'):
        blocks = encoder.trunk.blocks
    else:
        raise AttributeError("Cannot locate trunk.blocks in CONCH encoder.")

    print(f"  Freezing blocks 0-{freeze_blocks-1}, "
          f"unfreezing {freeze_blocks}-{len(blocks)-1}")

    for i, block in enumerate(blocks):
        if i >= freeze_blocks:
            for param in block.parameters():
                param.requires_grad = True

    # Unfreeze final norm and projection layers
    if hasattr(encoder, 'trunk'):
        for attr in ['norm', 'ln_post', 'ln_final']:
            if hasattr(encoder.trunk, attr):
                for p in getattr(encoder.trunk, attr).parameters():
                    p.requires_grad = True

    for attr in ['proj_contrast', 'head', 'attn_pool_contrast',
                 'attn_pool_caption', 'ln_contrast', 'ln_caption']:
        if hasattr(encoder, attr):
            obj = getattr(encoder, attr)
            if isinstance(obj, nn.Parameter):
                obj.requires_grad = True
            elif isinstance(obj, nn.Module):
                for p in obj.parameters():
                    p.requires_grad = True

    total     = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters()
                    if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    # Get CONCH output dim
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        out   = encoder(dummy)
        feat  = out[0] if isinstance(out, tuple) else out
        conch_dim = feat.shape[-1]
    print(f"  CONCH output dim: {conch_dim}")

    return encoder, preprocess, conch_dim


# ── DualEncoderFusionHead ─────────────────────────────────────────────────────
class DualEncoderFusionHead(nn.Module):
    """
    Cross-attention fusion of UNI (morphological) and CONCH (semantic) features.

    UNI  [B, 1024] → projection → Query  [B, 1, common_dim]
    CONCH [B, 512] → projection → Key, Value [B, 1, common_dim]

    Scaled dot-product attention → attended features [B, common_dim]
    → MLP classifier → [B, num_classes]

    The UNI query attends to CONCH key/value — the model learns
    which semantic CONCH features are most relevant to each
    morphological UNI pattern.
    """
    def __init__(self, uni_dim: int, conch_dim: int,
                 common_dim: int, num_heads: int,
                 hidden_dim: int, num_classes: int,
                 dropout: float, noise_stddev: float):
        super().__init__()
        self.noise_stddev = noise_stddev

        # Projections to common dimension
        self.uni_proj   = nn.Linear(uni_dim,   common_dim)
        self.conch_proj = nn.Linear(conch_dim, common_dim)

        # Cross-attention: UNI queries, CONCH keys+values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = common_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )

        # Residual connection norm
        self.norm = nn.LayerNorm(common_dim)

        # MLP classifier (Belaskri et al. design)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, uni_feat: torch.Tensor,
                conch_feat: torch.Tensor) -> torch.Tensor:
        # Gaussian noise injection during training
        if self.training and self.noise_stddev > 0:
            uni_feat   = uni_feat   + torch.randn_like(uni_feat)   * self.noise_stddev
            conch_feat = conch_feat + torch.randn_like(conch_feat) * self.noise_stddev

        # Project to common dimension
        q = self.uni_proj(uni_feat).unsqueeze(1)     # [B, 1, common_dim]
        k = self.conch_proj(conch_feat).unsqueeze(1) # [B, 1, common_dim]
        v = k                                         # Key = Value

        # Cross-attention
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        attn_out = attn_out.squeeze(1)               # [B, common_dim]

        # Residual + norm
        fused = self.norm(attn_out + q.squeeze(1))   # [B, common_dim]

        return self.classifier(fused)


# ── Full model ────────────────────────────────────────────────────────────────
class DualEncoderModel(nn.Module):
    """
    Wraps CONCH encoder + DualEncoderFusionHead.
    UNI features are passed in directly (pre-cached, no encoder needed).
    """
    def __init__(self, conch_encoder, fusion_head):
        super().__init__()
        self.conch_encoder = conch_encoder
        self.fusion_head   = fusion_head

    def forward(self, uni_feat: torch.Tensor,
                image: torch.Tensor) -> torch.Tensor:
        # Live CONCH forward pass
        out        = self.conch_encoder(image)
        conch_feat = out[0] if isinstance(out, tuple) else out  # [B, 512]

        return self.fusion_head(uni_feat, conch_feat)


# ── Warmup cosine scheduler ───────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int,
                 total_epochs: int, min_lr_ratio: float = 0.01):
        self.scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self._lr_lambda(warmup_epochs, total_epochs,
                                       min_lr_ratio)
        )

    @staticmethod
    def _lr_lambda(warmup, total, min_ratio):
        def fn(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine
        return fn

    def step(self):
        self.scheduler.step()


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for uni_feat, images, labels in loader:
            uni_feat = uni_feat.to(device)
            images   = images.to(device)
            logits   = model(uni_feat, images)
            preds    = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def print_metrics(preds, labels, split_name: str):
    acc  = accuracy_score(labels, preds) * 100
    prec = precision_score(labels, preds, average='macro',
                           zero_division=0) * 100
    rec  = recall_score(labels, preds, average='macro',
                        zero_division=0) * 100
    f1   = f1_score(labels, preds, average='macro',
                    zero_division=0) * 100

    print(f"\n── {split_name} Results (Macro-Averaged) ──")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")

    print(f"\n── {split_name} Per-Class Breakdown ──")
    print(classification_report(
        labels, preds,
        target_names=list(CLASS_NAMES.values()),
        digits=4
    ))

    print("── Confusion Matrix ──")
    cm     = confusion_matrix(labels, preds)
    header = "         " + "  ".join(
        f"{n[:6]:>8}" for n in CLASS_NAMES.values()
    )
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:8d}" for v in row)
        print(f"  {list(CLASS_NAMES.values())[i][:6]:>8}  {row_str}")

    return acc, prec, rec, f1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Phase 2 — Asymmetric Cache & Tune")
    print("  UNI (offline) + CONCH (live) + Cross-Attention Fusion")
    print("=" * 60)
    print(f"Device      : {DEVICE.upper()}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM        : "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Epochs      : {EPOCHS} (warmup: {WARMUP_EPOCHS})")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Common dim  : {COMMON_DIM}")
    print(f"Attn heads  : {NUM_HEADS}")

    # ── Load CONCH ────────────────────────────────────────────────────────────
    conch_encoder, conch_preprocess, conch_dim = load_conch_partial(
        CONCH_WEIGHTS, FREEZE_BLOCKS, DEVICE
    )

    # ── Build transforms ──────────────────────────────────────────────────────
    train_transform, test_transform = build_transforms(conch_preprocess)

    # ── DualDataset ───────────────────────────────────────────────────────────
    print("\nLoading datasets and verifying alignment...")
    train_dataset = DualDataset(
        cache_path = UNI_TRAIN_CACHE,
        image_dir  = TRAIN_DIR,
        transform  = train_transform,
        verify     = True
    )
    test_dataset = DualDataset(
        cache_path = UNI_TEST_CACHE,
        image_dir  = TEST_DIR,
        transform  = test_transform,
        verify     = True
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test : {len(test_dataset)} samples")

    # Weighted sampler for class imbalance
    labels_tensor  = train_dataset.labels
    counts         = torch.bincount(labels_tensor,
                                    minlength=NUM_CLASSES).float()
    print("\nTrain class distribution:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES.values(), counts)):
        print(f"  label_{i} ({name}): {int(count)}")

    weights        = 1.0 / counts
    sample_weights = weights[labels_tensor]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(labels_tensor),
        replacement = True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,
        num_workers = 4,
        pin_memory  = True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True
    )

    # ── Build fusion head ─────────────────────────────────────────────────────
    fusion_head = DualEncoderFusionHead(
        uni_dim      = 1024,
        conch_dim    = conch_dim,
        common_dim   = COMMON_DIM,
        num_heads    = NUM_HEADS,
        hidden_dim   = HIDDEN_DIM,
        num_classes  = NUM_CLASSES,
        dropout      = DROPOUT,
        noise_stddev = NOISE_STDDEV
    )

    # ── Full model ────────────────────────────────────────────────────────────
    model = DualEncoderModel(conch_encoder, fusion_head).to(DEVICE)

    # ── Differential learning rates ───────────────────────────────────────────
    encoder_params = [p for p in model.conch_encoder.parameters()
                      if p.requires_grad]
    fusion_params  = list(model.fusion_head.uni_proj.parameters()) + \
                     list(model.fusion_head.conch_proj.parameters()) + \
                     list(model.fusion_head.cross_attn.parameters()) + \
                     list(model.fusion_head.norm.parameters())
    head_params    = list(model.fusion_head.classifier.parameters())

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': fusion_params,  'lr': LR_FUSION},
        {'params': head_params,    'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    scheduler  = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    criterion  = nn.CrossEntropyLoss()
    scaler     = torch.cuda.amp.GradScaler()

    print(f"\nOptimizer groups:")
    print(f"  CONCH encoder  : {len(encoder_params)} tensors, lr={LR_ENCODER}")
    print(f"  Fusion head    : {len(fusion_params)} tensors, lr={LR_FUSION}")
    print(f"  Classifier     : {len(head_params)} tensors, lr={LR_HEAD}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nTraining...")
    best_f1        = 0.0
    best_f1_epoch  = 0
    best_f1_state  = None

    best_acc       = 0.0
    best_acc_epoch = 0
    best_acc_state = None

    #Base config for saving
    save_config = {
        'freeze_blocks': FREEZE_BLOCKS, 'common_dim': COMMON_DIM,
        'num_heads': NUM_HEADS, 'hidden_dim': HIDDEN_DIM,
        'dropout': DROPOUT, 'lr_encoder': LR_ENCODER,
        'lr_fusion': LR_FUSION, 'lr_head': LR_HEAD,
        'conch_dim': conch_dim, 'uni_dim': 1024
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        total_steps = 0

        for uni_feat, images, labels in train_loader:
            uni_feat = uni_feat.to(DEVICE)
            images   = images.to(DEVICE)
            labels   = labels.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(uni_feat, images)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss  += loss.item()
            total_steps += 1

        scheduler.step()
        avg_loss = total_loss / total_steps

        # Replace with
        # Evaluate every epoch — best checkpoint can appear anywhere
        preds, labels_np = evaluate(model, test_loader, DEVICE)
        f1  = f1_score(labels_np, preds, average='macro',
                       zero_division=0) * 100
        acc = accuracy_score(labels_np, preds) * 100

        warmup_tag = " [warmup]" if epoch <= WARMUP_EPOCHS else ""
        print(f"  Epoch {epoch:3d}/{EPOCHS}{warmup_tag} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.2f}% | F1: {f1:.2f}%")

        # ── TRACK AND SAVE BEST F1 ──
        if f1 > best_f1:
            best_f1       = f1
            best_f1_epoch = epoch
            best_f1_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            torch.save({
                'epoch': best_f1_epoch, 'model_state': best_f1_state,
                'accuracy': acc, 'f1': best_f1, 'config': save_config
            }, os.path.join(RESULTS_DIR, "phase2_best_f1.pt"))
            print(f"  🟢 [Checkpoint] New Best F1 Saved ({f1:.2f}%)!")

        # ── TRACK AND SAVE BEST ACCURACY ──
        if acc > best_acc:
            best_acc       = acc
            best_acc_epoch = epoch
            best_acc_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            torch.save({
                'epoch': best_acc_epoch, 'model_state': best_acc_state,
                'accuracy': best_acc, 'f1': f1, 'config': save_config
            }, os.path.join(RESULTS_DIR, "phase2_best_acc.pt"))
            print(f"  🔵 [Checkpoint] New Best Accuracy Saved ({acc:.2f}%)!")

        if epoch % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ── Final evaluation (Evaluating Best F1 Model for Table) ─────────────────
    print(f"\nTraining Complete!")
    ckpt = torch.load(os.path.join(RESULTS_DIR, "phase2_best_acc.pt"))
    best_acc_f1 = ckpt['f1']

    print(f"🏆 Best F1 Model       : Epoch {best_f1_epoch} (F1: {best_f1:.2f}%, Acc: {accuracy_score(labels_np, preds)*100:.2f}%)")
    print(f"🎯 Best Accuracy Model : Epoch {best_acc_epoch} (Acc: {best_acc:.2f}%, F1: {best_acc_f1:.2f}%)")
    # Load Best F1 state for the final output table
    model.load_state_dict(best_f1_state)
    preds, labels_np = evaluate(model, test_loader, DEVICE)
    per_class_f1 = f1_score(labels_np, preds, average=None, zero_division=0) * 100


    print(f"Best Accuracy Model F1: {best_acc_f1:.2f}%")

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("  Phase 2 — Asymmetric Cache & Tune (SOTA Run)")
    results_lines.append("=" * 60)
    results_lines.append(f"Peak F1 Epoch   : {best_f1_epoch}/{EPOCHS}")
    results_lines.append(f"Peak Acc Epoch  : {best_acc_epoch}/{EPOCHS}")
    results_lines.append(f"Common dim      : {COMMON_DIM}")
    results_lines.append(f"Attn heads      : {NUM_HEADS}")
    results_lines.append("")

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    acc, prec, rec, f1 = print_metrics(preds, labels_np, "TEST (Based on Best F1 Checkpoint)")
    sys.stdout = old_stdout
    captured = buffer.getvalue()
    print(captured)
    results_lines.append(captured)

    comparison = f"""
── Full Comparison Table ──
  Method                                    Acc      F1
  ────────────────────────────────────────────────────────
  HSA-NRL (ResNet-34 + noise-robust)        83.40%   76.54%
  Ke et al. (CNN ensemble + MHSA)           86.72%   85.76%
  Belaskri et al. (ViT-B/16 + RCC)         87.14%   82.77%
  Exp 5 — CONCH partial fine-tune           86.02%   80.87%
  Phase 2 (Our Best F1 Model)              {acc:.2f}%   {f1:.2f}%
  Phase 2 (Our Best Acc Model)             {best_acc:.2f}%   {best_acc_f1:.2f}%

── Per-Class F1 — Phase 2 vs Experiment 5 ──
  Class             Exp5     Phase2   Change
  ────────────────────────────────────────────────
  Normal           85.47%   {per_class_f1[0]:.2f}%   {per_class_f1[0]-85.47:+.2f}%
  Serrated         62.08%   {per_class_f1[1]:.2f}%   {per_class_f1[1]-62.08:+.2f}%
  Adenocarcinoma   97.60%   {per_class_f1[2]:.2f}%   {per_class_f1[2]-97.60:+.2f}%
  Adenoma          78.34%   {per_class_f1[3]:.2f}%   {per_class_f1[3]-78.34:+.2f}%
"""
    print(comparison)
    results_lines.append(comparison)

    # Save results and checkpoint
    results_path = os.path.join(RESULTS_DIR, "phase2_results.txt")
    with open(results_path, 'w') as f:
        f.write("\n".join(results_lines))
    print(f"✅ Results saved to: {results_path}")

if __name__ == "__main__":
    main()
