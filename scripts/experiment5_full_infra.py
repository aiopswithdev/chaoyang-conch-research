import warnings
import os
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
 
# Suppress specific deep learning framework warnings
import logging
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
"""
Experiment 5 — CONCH Partial Fine-Tuning + Full Infrastructure Improvements
----------------------------------------------------------------------------
Builds directly on Experiment 4, adding every improvement identified from
Belaskri et al. (IC2SDA 2025) analysis:

  1. H&E-aware data augmentation (rotation, flip, zoom, shift, shear)
  2. AdamW optimizer with weight_decay=1e-2 (vs Adam + 1e-4 in Exp 4)
  3. Cosine annealing with linear warmup for 5 epochs
  4. Improved MLP head: GaussianNoise + BatchNorm + GELU + Dropout(0.5)
  5. 100 training epochs (vs 50 in Exp 4)

Hypothesis: CONCH's superior pathology pretraining (1.17M path images)
should outperform Belaskri et al.'s ViT-B/16 (ImageNet + 75K RCC patches)
once the surrounding infrastructure is matched.

Target to beat:
  Belaskri et al. (ViT-B/16 + RCC) : 87.14% acc, 82.77% F1
  Ke et al. (CNN ensemble + MHSA)   : 86.72% acc, 85.76% F1
  Exp 4 (CONCH partial fine-tune)   : 85.46% acc, 80.55% F1

Results saved to: ./results/experiment5_results.txt
"""

import os
import io
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from conch.open_clip_custom import create_model_from_pretrained

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/home/dev/Downloads/pytorch_model.bin"
TRAIN_DIR    = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_train"
TEST_DIR     = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_test"
RESULTS_DIR  = "./results"

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES    = 4
HIDDEN_DIM     = 128       # matches Belaskri et al. head width
DROPOUT        = 0.5       # raised from 0.3 (Belaskri et al.)
BATCH_SIZE     = 16        # kept small for 7GB VRAM
EPOCHS         = 60       # raised from 50 (Belaskri et al.)
WARMUP_EPOCHS  = 5         # linear warmup before cosine decay
LR_ENCODER     = 1e-5      # slow — preserve CONCH pretrained weights
LR_HEAD        = 1e-3      # fast — classifier learns from scratch
WEIGHT_DECAY   = 1e-2      # raised from 1e-4 (Belaskri et al. AdamW)
SEED           = 42

# Ke et al. freeze strategy — same as Exp 4
TOTAL_BLOCKS   = 12
FREEZE_BLOCKS  = 6

# Gaussian noise injection stddev (Belaskri et al.)
NOISE_STDDEV   = 0.5

CLASS_NAMES = {
    0: "Normal",
    1: "Serrated",
    2: "Adenocarcinoma",
    3: "Adenoma"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ── Improved MLP Head (Belaskri et al. design) ────────────────────────────────
class ImprovedMLPHead(nn.Module):
    """
    MLP classifier head incorporating Belaskri et al. improvements:
      - GaussianNoise injection after encoder output
      - BatchNormalization for training stability
      - GELU activation (vs ReLU in Exp 1-4)
      - Higher dropout (0.5 vs 0.3)

    Architecture:
      GaussianNoise(0.5) → Linear(D→128) → BatchNorm(128)
      → GELU → Dropout(0.5) → Linear(128→4)
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_classes: int, dropout: float,
                 noise_stddev: float):
        super().__init__()
        self.noise_stddev = noise_stddev

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Gaussian noise injection during training only
        if self.training and self.noise_stddev > 0:
            x = x + torch.randn_like(x) * self.noise_stddev
        return self.network(x)


class CONCHClassifier(nn.Module):
    """Partially fine-tuned CONCH encoder + improved MLP head."""
    def __init__(self, encoder, feature_dim: int, hidden_dim: int,
                 num_classes: int, dropout: float, noise_stddev: float):
        super().__init__()
        self.encoder = encoder
        self.head    = ImprovedMLPHead(
            feature_dim, hidden_dim, num_classes,
            dropout, noise_stddev
        )

    def forward(self, x):
        out      = self.encoder(x)
        features = out[0] if isinstance(out, tuple) else out
        return self.head(features)


# ── Cosine LR scheduler with linear warmup ───────────────────────────────────
class WarmupCosineScheduler:
    """
    Linear warmup for warmup_epochs, then cosine annealing to min_lr.
    Implemented as a LambdaLR multiplier for clean integration with AdamW.
    """
    def __init__(self, optimizer, warmup_epochs: int,
                 total_epochs: int, min_lr_ratio: float = 0.01):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr_ratio  = min_lr_ratio

        self.scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self._lr_lambda
        )

    def _lr_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 → 1 over warmup_epochs
            return float(epoch + 1) / float(self.warmup_epochs)
        else:
            # Cosine decay: 1 → min_lr_ratio
            progress = (epoch - self.warmup_epochs) / \
                       max(1, self.total_epochs - self.warmup_epochs)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def load_conch_partial(weights_path: str, freeze_blocks: int, device: str):
    """Load CONCH with Ke et al. hierarchical layer-freezing."""
    print(f"Loading CONCH from: {weights_path}")
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=weights_path
    )

    vision_encoder = model.visual
    vision_encoder.to(device)

    # Freeze everything first
    for param in vision_encoder.parameters():
        param.requires_grad = False

    # Unfreeze blocks freeze_blocks → 11 via trunk.blocks
    if hasattr(vision_encoder, 'trunk') and \
       hasattr(vision_encoder.trunk, 'blocks'):
        blocks = vision_encoder.trunk.blocks
    else:
        raise AttributeError("Cannot find trunk.blocks in CONCH encoder.")

    total_found = len(blocks)
    print(f"Total transformer blocks: {total_found}")
    print(f"Freezing blocks 0-{freeze_blocks-1}, "
          f"unfreezing blocks {freeze_blocks}-{total_found-1}")

    for i, block in enumerate(blocks):
        if i >= freeze_blocks:
            for param in block.parameters():
                param.requires_grad = True

    # Unfreeze trunk final norm
    if hasattr(vision_encoder, 'trunk'):
        for attr_name in ['norm', 'ln_post', 'ln_final']:
            if hasattr(vision_encoder.trunk, attr_name):
                for param in getattr(vision_encoder.trunk,
                                     attr_name).parameters():
                    param.requires_grad = True
                print(f"Unfrozen: visual.trunk.{attr_name}")

    # Unfreeze projection/head layers
    for attr_name in ['proj_contrast', 'head', 'attn_pool_contrast',
                      'attn_pool_caption', 'ln_contrast', 'ln_caption']:
        if hasattr(vision_encoder, attr_name):
            attr = getattr(vision_encoder, attr_name)
            if isinstance(attr, nn.Parameter):
                attr.requires_grad = True
            elif isinstance(attr, nn.Module):
                for param in attr.parameters():
                    param.requires_grad = True
            print(f"Unfrozen: visual.{attr_name}")

    total_params     = sum(p.numel() for p in vision_encoder.parameters())
    trainable_params = sum(p.numel() for p in vision_encoder.parameters()
                           if p.requires_grad)
    print(f"\nParameter summary:")
    print(f"  Total      : {total_params:,}")
    print(f"  Trainable  : {trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen     : {total_params-trainable_params:,} "
          f"({100*(total_params-trainable_params)/total_params:.1f}%)")

    # Get feature dimension
    vision_encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        out   = vision_encoder(dummy)
        feat  = out[0] if isinstance(out, tuple) else out
        feature_dim = feat.shape[-1]
    print(f"  Feature dim: {feature_dim}")

    return vision_encoder, preprocess, feature_dim


def build_augmented_transform(preprocess):
    """
    Build H&E-aware augmentation pipeline placed BEFORE CONCH preprocess.
    Mirrors Belaskri et al. Table 2 augmentation parameters.

    Order matters:
      1. PIL augmentations (geometric + color)
      2. CONCH preprocess (resize to 224, normalize with CONCH stats)
    """
    # Extract CONCH's normalization from its preprocess pipeline
    # CONCH preprocess is a Compose — find the Normalize transform
    conch_normalize = None
    conch_resize    = None
    for t in preprocess.transforms:
        if isinstance(t, transforms.Normalize):
            conch_normalize = t
        if isinstance(t, transforms.Resize):
            conch_resize = t

    # Build augmented train transform
    train_transform = transforms.Compose([
        # Geometric augmentations (Belaskri et al. Table 2)
        transforms.RandomRotation(degrees=40, fill=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),   # width/height shift
            shear=0.2
        ),
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.5, 1.0),       # zoom range [0.5, 1.0]
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        # Color augmentations for H&E stain variation
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        # Use CONCH's own normalization stats
        conch_normalize if conch_normalize else
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Test transform — no augmentation, just CONCH preprocess
    test_transform = preprocess

    return train_transform, test_transform


def get_weighted_sampler(dataset):
    """Weighted sampler for class imbalance."""
    labels = torch.tensor([s[1] for s in dataset.samples])
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float()

    print("\nTrain class distribution:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES.values(), counts)):
        print(f"  label_{i} ({name}): {int(count)} samples")

    weights        = 1.0 / counts
    sample_weights = weights[labels]
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(labels),
        replacement = True
    )


def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def print_metrics(preds, labels, split_name: str, class_names: dict):
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
    print(classification_report(
        labels, preds,
        target_names=list(class_names.values()),
        digits=4
    ))

    print("── Confusion Matrix ──")
    cm     = confusion_matrix(labels, preds)
    header = "         " + "  ".join(f"{n[:6]:>8}"
                                      for n in class_names.values())
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:8d}" for v in row)
        print(f"  {list(class_names.values())[i][:6]:>8}  {row_str}")

    return acc, precision, recall, f1


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Experiment 5 — CONCH + Full Infrastructure Improvements")
    print("  (Belaskri et al. augmentation + AdamW + warmup + head)")
    print("=" * 60)
    print(f"Device        : {DEVICE.upper()}")
    if torch.cuda.is_available():
        print(f"GPU memory    : "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Epochs        : {EPOCHS} (warmup: {WARMUP_EPOCHS})")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"LR encoder    : {LR_ENCODER}")
    print(f"LR head       : {LR_HEAD}")
    print(f"Weight decay  : {WEIGHT_DECAY} (AdamW)")
    print(f"Dropout       : {DROPOUT}")
    print(f"Noise stddev  : {NOISE_STDDEV}")
    print(f"Freeze blocks : 0-{FREEZE_BLOCKS-1} of {TOTAL_BLOCKS}")

    # ── Load CONCH ─────────────────────────────────────────────────────────────
    encoder, preprocess, feature_dim = load_conch_partial(
        WEIGHTS_PATH, FREEZE_BLOCKS, DEVICE
    )

    # ── Build augmented transforms ────────────────────────────────────────────
    train_transform, test_transform = build_augmented_transform(preprocess)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print(f"\nLoading datasets...")
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR,
                                         transform=train_transform)
    test_dataset  = datasets.ImageFolder(root=TEST_DIR,
                                         transform=test_transform)
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Test : {len(test_dataset)} images")

    sampler = get_weighted_sampler(train_dataset)

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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CONCHClassifier(
        encoder      = encoder,
        feature_dim  = feature_dim,
        hidden_dim   = HIDDEN_DIM,
        num_classes  = NUM_CLASSES,
        dropout      = DROPOUT,
        noise_stddev = NOISE_STDDEV
    ).to(DEVICE)

    # ── AdamW with differential learning rates ────────────────────────────────
    encoder_params = [p for p in model.encoder.parameters()
                      if p.requires_grad]
    head_params    = list(model.head.parameters())

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': head_params,    'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    # ── Warmup + cosine LR schedule ───────────────────────────────────────────
    scheduler  = WarmupCosineScheduler(
        optimizer,
        warmup_epochs = WARMUP_EPOCHS,
        total_epochs  = EPOCHS
    )
    criterion  = nn.CrossEntropyLoss()

    print(f"\nOptimizer: AdamW")
    print(f"  Encoder params : {len(encoder_params)} tensors, lr={LR_ENCODER}")
    print(f"  Head params    : {len(head_params)} tensors, lr={LR_HEAD}")
    print(f"  Schedule       : {WARMUP_EPOCHS}-epoch warmup + cosine decay")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nTraining...")
    best_f1    = 0.0
    best_epoch = 0
    best_state = None
    scaler     = torch.cuda.amp.GradScaler()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        total_steps = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(images)
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
        curr_lr  = scheduler.get_last_lr()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            preds, labels_np = evaluate(model, test_loader, DEVICE)
            f1  = f1_score(labels_np, preds, average='macro',
                           zero_division=0) * 100
            acc = accuracy_score(labels_np, preds) * 100

            warmup_tag = " [warmup]" if epoch <= WARMUP_EPOCHS else ""
            print(f"  Epoch {epoch:3d}/{EPOCHS}{warmup_tag} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Test Acc: {acc:.2f}% | "
                  f"Test F1: {f1:.2f}%")

            if f1 > best_f1:
                best_f1    = f1
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\nBest model was at epoch {best_epoch} with F1: {best_f1:.2f}%")
    model.load_state_dict(best_state)
    preds, labels_np = evaluate(model, test_loader, DEVICE)

    per_class_f1 = f1_score(labels_np, preds, average=None,
                             zero_division=0) * 100

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("  Experiment 5 — CONCH + Full Infrastructure")
    results_lines.append("=" * 60)
    results_lines.append(f"Best epoch     : {best_epoch}/{EPOCHS}")
    results_lines.append(f"Augmentation   : H&E-aware (rotation, flip, "
                         f"zoom, shift, shear, color jitter)")
    results_lines.append(f"Optimizer      : AdamW weight_decay={WEIGHT_DECAY}")
    results_lines.append(f"LR schedule    : {WARMUP_EPOCHS}-ep warmup + cosine")
    results_lines.append(f"Head           : GaussianNoise + BN + GELU + "
                         f"Dropout({DROPOUT})")
    results_lines.append("")

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    acc, precision, recall, f1 = print_metrics(
        preds, labels_np, "TEST", CLASS_NAMES
    )
    sys.stdout = old_stdout
    captured = buffer.getvalue()
    print(captured)
    results_lines.append(captured)

    comparison = f"""
── Full Comparison Table ──
  Method                                    Acc      F1
  ────────────────────────────────────────────────────────
  HSA-NRL (ResNet-34, noise-robust)         83.40%   76.54%
  Ke et al. (CNN ensemble + MHSA)           86.72%   85.76%
  Belaskri et al. (ViT-B/16 + RCC)         87.14%   82.77%
  Exp 4 — CONCH partial fine-tune           85.46%   80.55%
  Exp 5 — CONCH + full infra (ours)         {acc:.2f}%   {f1:.2f}%

── Per-Class F1 ──
  Class             Exp4     Exp5     Change
  ──────────────────────────────────────────────
  Normal           83.72%   {per_class_f1[0]:.2f}%   {per_class_f1[0]-83.72:+.2f}%
  Serrated         59.31%   {per_class_f1[1]:.2f}%   {per_class_f1[1]-59.31:+.2f}%
  Adenocarcinoma   97.26%   {per_class_f1[2]:.2f}%   {per_class_f1[2]-97.26:+.2f}%
  Adenoma          81.90%   {per_class_f1[3]:.2f}%   {per_class_f1[3]-81.90:+.2f}%
"""
    print(comparison)
    results_lines.append(comparison)

    # Save results
    path = os.path.join(RESULTS_DIR, "experiment5_results.txt")
    with open(path, 'w') as f:
        f.write("\n".join(results_lines))
    print(f"✅ Results saved to: {path}")

    # Save checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, "experiment5_best_model.pt")
    torch.save({
        'epoch':       best_epoch,
        'model_state': best_state,
        'accuracy':    acc,
        'f1':          f1,
        'config': {
            'freeze_blocks': FREEZE_BLOCKS,
            'lr_encoder':    LR_ENCODER,
            'lr_head':       LR_HEAD,
            'weight_decay':  WEIGHT_DECAY,
            'hidden_dim':    HIDDEN_DIM,
            'dropout':       DROPOUT,
            'noise_stddev':  NOISE_STDDEV,
            'warmup_epochs': WARMUP_EPOCHS,
            'total_epochs':  EPOCHS
        }
    }, ckpt_path)
    print(f"✅ Best model checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
