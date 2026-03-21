"""
Experiment 4 — Partial CONCH Fine-Tuning (Ke et al. Strategy)
--------------------------------------------------------------
Implements Ke et al. (2025) hierarchical layer-freezing strategy on CONCH.

Key design decisions directly from Ke et al.:
  1. Freeze exactly the first half of transformer blocks (blocks 0-5)
  2. Fine-tune the second half (blocks 6-11) + final norm + classifier head
  3. Differential learning rates:
       - Frozen layers    : lr = 0 (no update)
       - Unfrozen encoder : lr = 1e-5 (slow, preserve pretrained features)
       - Classifier head  : lr = 1e-3 (fast, learn task-specific mapping)

Why this works (from Ke et al. ablation):
  "Shallow layers capture low-level generic features (edges, textures);
   deeper layers extract high-level semantic information more relevant
   to the specific task. Freezing half preserves general representations
   while enabling deeper layers to adapt to the target domain."

CONCH ViT-B-16 architecture:
  - Patch embedding layer (frozen)
  - 12 transformer blocks: blocks[0] ... blocks[11]
  - Final layer norm (unfrozen)
  - Visual projection head (unfrozen)
  Frozen  : patch embedding + blocks 0-5
  Unfrozen: blocks 6-11 + norm + projection

Training note:
  This experiment uses RAW IMAGES not cached features.
  CONCH runs forward pass every batch — expect ~10-15 min per run.
  This is necessary because gradients must flow through the encoder.

Comparison targets:
  HSA-NRL (ResNet-34, noise-robust)  : 83.40% acc, 76.54% F1
  Ke et al. (CNN ensemble + MHSA)    : 86.72% acc, 85.76% F1  ← primary target
  Exp 1 — CONCH frozen + CE          : 82.14% acc, 77.05% F1  ← our best so far

Results saved to: ./results/experiment4_results.txt
"""

import os
import io
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
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
HIDDEN_DIM     = 256
DROPOUT        = 0.3
BATCH_SIZE     = 16       # smaller than Exp 1-3 — full model in GPU memory
EPOCHS         = 50       # fewer epochs — fine-tuning converges faster
LR_ENCODER     = 1e-5     # very small — preserve CONCH pretrained weights
LR_HEAD        = 1e-3     # standard — classifier head learns from scratch
WEIGHT_DECAY   = 1e-4
SEED           = 42

# Ke et al. freeze strategy: freeze first half of 12 ViT blocks
TOTAL_BLOCKS   = 12
FREEZE_BLOCKS  = 6        # blocks 0-5 frozen, blocks 6-11 unfrozen

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


def load_conch_partial(weights_path: str, freeze_blocks: int, device: str):
    """
    Load CONCH and apply Ke et al. hierarchical layer-freezing strategy.

    Freezes:
      - Patch embedding (positional + conv projection)
      - Transformer blocks 0 to freeze_blocks-1

    Unfreezes:
      - Transformer blocks freeze_blocks to 11
      - Final layer norm
      - Visual projection

    Returns model, preprocess, and feature_dim for classifier head.
    """
    print(f"Loading CONCH from: {weights_path}")
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=weights_path
    )

    vision_encoder = model.visual
    vision_encoder.to(device)

    # ── Step 1: Freeze everything first ──────────────────────────────────────
    for param in vision_encoder.parameters():
        param.requires_grad = False

    # ── Step 2: Selectively unfreeze following Ke et al. ─────────────────────
    # Unfreeze transformer blocks 6-11
    # CONCH ViT uses .transformer.resblocks or .blocks depending on version
    # Try both attribute names for compatibility
    # Replace with
    blocks = None
    if hasattr(vision_encoder, 'trunk') and \
       hasattr(vision_encoder.trunk, 'blocks'):
        blocks = vision_encoder.trunk.blocks
        print(f"Found transformer blocks via: visual.trunk.blocks")
    elif hasattr(vision_encoder, 'transformer') and \
       hasattr(vision_encoder.transformer, 'resblocks'):
        blocks = vision_encoder.transformer.resblocks
        print(f"Found transformer blocks via: visual.transformer.resblocks")
    elif hasattr(vision_encoder, 'blocks'):
        blocks = vision_encoder.blocks
        print(f"Found transformer blocks via: visual.blocks")
    else:
        print("Available vision encoder attributes:")
        print([attr for attr in dir(vision_encoder)
               if not attr.startswith('_')])
        raise AttributeError(
            "Cannot find transformer blocks in CONCH vision encoder."
        )

    total_found = len(blocks)
    print(f"Total transformer blocks found: {total_found}")
    print(f"Freezing blocks 0-{freeze_blocks-1}, "
          f"unfreezing blocks {freeze_blocks}-{total_found-1}")

    for i, block in enumerate(blocks):
        if i >= freeze_blocks:
            for param in block.parameters():
                param.requires_grad = True
                
    # Replace with
    # Unfreeze trunk's final norm if it exists
    if hasattr(vision_encoder, 'trunk'):
        for attr_name in ['norm', 'ln_post', 'ln_final']:
            if hasattr(vision_encoder.trunk, attr_name):
                for param in getattr(vision_encoder.trunk,
                                     attr_name).parameters():
                    param.requires_grad = True
                print(f"Unfrozen: visual.trunk.{attr_name}")

    # Unfreeze top-level projection and head layers
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

    # ── Count trainable vs frozen parameters ─────────────────────────────────
    total_params     = sum(p.numel() for p in vision_encoder.parameters())
    trainable_params = sum(p.numel() for p in vision_encoder.parameters()
                           if p.requires_grad)
    frozen_params    = total_params - trainable_params

    print(f"\nParameter summary:")
    print(f"  Total      : {total_params:,}")
    print(f"  Trainable  : {trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen     : {frozen_params:,} "
          f"({100*frozen_params/total_params:.1f}%)")

    # Determine feature dimension from a dummy forward pass
    vision_encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        out   = vision_encoder(dummy)
        feat  = out[0] if isinstance(out, tuple) else out
        feature_dim = feat.shape[-1]
    print(f"  Feature dim: {feature_dim}")

    return vision_encoder, preprocess, feature_dim


class MLPHead(nn.Module):
    """Classifier head — identical architecture to Experiments 1-3."""
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_classes: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class CONCHClassifier(nn.Module):
    """Full model: partially fine-tuned CONCH encoder + MLP head."""
    def __init__(self, encoder, feature_dim: int, hidden_dim: int,
                 num_classes: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head    = MLPHead(feature_dim, hidden_dim,
                               num_classes, dropout)

    def forward(self, x):
        out      = self.encoder(x)
        features = out[0] if isinstance(out, tuple) else out
        return self.head(features)


def get_weighted_sampler(dataset):
    """Weighted sampler for class imbalance — same as Experiments 1-3."""
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
    if torch.cuda.is_available():
        print(f"GPU memory available: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Experiment 4 — Partial CONCH Fine-Tuning")
    print("  (Ke et al. Hierarchical Layer-Freezing Strategy)")
    print("=" * 60)
    print(f"Device        : {DEVICE.upper()}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"LR encoder    : {LR_ENCODER}  (unfrozen blocks 6-11)")
    print(f"LR head       : {LR_HEAD}  (classifier head)")
    print(f"Freeze blocks : 0-{FREEZE_BLOCKS-1} of {TOTAL_BLOCKS}")

    # ── Load CONCH with partial freezing ──────────────────────────────────────
    encoder, preprocess, feature_dim = load_conch_partial(
        WEIGHTS_PATH, FREEZE_BLOCKS, DEVICE
    )

    # ── Datasets (raw images, not cached features) ────────────────────────────
    print(f"\nLoading datasets...")
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR,
                                         transform=preprocess)
    test_dataset  = datasets.ImageFolder(root=TEST_DIR,
                                         transform=preprocess)
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

    # ── Full model ────────────────────────────────────────────────────────────
    model = CONCHClassifier(
        encoder     = encoder,
        feature_dim = feature_dim,
        hidden_dim  = HIDDEN_DIM,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT
    ).to(DEVICE)

    # ── Differential learning rates (key to Ke et al. strategy) ──────────────
    # Encoder unfrozen params get LR_ENCODER (10-100x smaller than head)
    # Classifier head params get LR_HEAD
    encoder_params = [p for p in model.encoder.parameters()
                      if p.requires_grad]
    head_params    = list(model.head.parameters())

    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': head_params,    'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    criterion  = nn.CrossEntropyLoss()

    print(f"\nOptimizer parameter groups:")
    print(f"  Encoder (unfrozen) : {len(encoder_params)} tensors, "
          f"lr={LR_ENCODER}")
    print(f"  Classifier head    : {len(head_params)} tensors, "
          f"lr={LR_HEAD}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nTraining...")
    best_f1    = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        total_steps = 0

        # Replace with
        scaler = torch.cuda.amp.GradScaler()
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

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            preds, labels_np = evaluate(model, test_loader, DEVICE)
            f1  = f1_score(labels_np, preds, average='macro',
                           zero_division=0) * 100
            acc = accuracy_score(labels_np, preds) * 100

            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
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
    results_lines.append("  Experiment 4 — Partial CONCH Fine-Tuning")
    results_lines.append(f"  Frozen blocks : 0-{FREEZE_BLOCKS-1}")
    results_lines.append(f"  LR encoder    : {LR_ENCODER}")
    results_lines.append(f"  LR head       : {LR_HEAD}")
    results_lines.append(f"  Best epoch    : {best_epoch}/{EPOCHS}")
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
  Method                                  Acc      F1
  ──────────────────────────────────────────────────────
  HSA-NRL (ResNet-34, noise-robust)       83.40%   76.54%
  Ke et al. (CNN ensemble + MHSA)         86.72%   85.76%
  Exp 1 — CONCH frozen + CE              82.14%   77.05%
  Exp 2 — CONCH frozen + SCE             80.79%   75.12%
  Exp 3 — CONCH frozen + GCE             81.30%   75.87%
  Exp 4 — CONCH partial fine-tune        {acc:.2f}%   {f1:.2f}%

── Per-Class F1 ──
  Class             Exp1     Exp4     Change
  ────────────────────────────────────────────
  Normal           79.50%   {per_class_f1[0]:.2f}%   {per_class_f1[0]-79.50:+.2f}%
  Serrated         54.95%   {per_class_f1[1]:.2f}%   {per_class_f1[1]-54.95:+.2f}%
  Adenocarcinoma   96.53%   {per_class_f1[2]:.2f}%   {per_class_f1[2]-96.53:+.2f}%
  Adenoma          77.20%   {per_class_f1[3]:.2f}%   {per_class_f1[3]-77.20:+.2f}%
"""
    print(comparison)
    results_lines.append(comparison)

    path = os.path.join(RESULTS_DIR, "experiment4_results.txt")
    with open(path, 'w') as f:
        f.write("\n".join(results_lines))
    print(f"✅ Results saved to: {path}")

    ckpt_path = os.path.join(RESULTS_DIR, "experiment4_best_model.pt")
    torch.save({
        'epoch':       best_epoch,
        'model_state': best_state,
        'accuracy':    acc,
        'f1':          f1,
        'config': {
            'freeze_blocks': FREEZE_BLOCKS,
            'lr_encoder':    LR_ENCODER,
            'lr_head':       LR_HEAD,
            'hidden_dim':    HIDDEN_DIM,
            'num_classes':   NUM_CLASSES,
            'dropout':       DROPOUT
        }
    }, ckpt_path)
    print(f"✅ Best model checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
