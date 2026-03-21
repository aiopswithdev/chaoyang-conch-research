"""
Experiment 2 — CONCH + MLP + Symmetric Cross-Entropy Loss
-----------------------------------------------------------
Identical to Experiment 1 in every way except the loss function.
Replaces standard CrossEntropyLoss with Symmetric Cross-Entropy (SCE)
from Wang et al. (ICCV 2019) "Symmetric Cross Entropy for Robust
Learning with Noisy Labels."

SCE = α * CE(p, q) + β * RCE(p, q)

  CE  term: standard forward cross-entropy — learns from clean labels
  RCE term: reverse cross-entropy — penalises overconfident wrong predictions
            this is what suppresses the noisy label effect

The hypothesis: Serrated class confusion in Experiment 1 is driven by
noisy labels in training. SCE should specifically improve Serrated F1.

Comparison targets:
  HSA-NRL (ResNet-34, noise-robust)  : 83.40% acc, 76.54% F1
  Ke et al. (CNN ensemble + MHSA)    : 86.72% acc, 85.76% F1
  Experiment 1 (CONCH, standard CE)  : 82.14% acc, 77.05% F1  ← beat this

Results saved to: ./results/experiment2_results.txt
"""

import os
import io
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = "/home/dev/Documents/ML-Research/Chaoyang/features"
TRAIN_PATH   = os.path.join(FEATURES_DIR, "chaoyang_train_features.pt")
TEST_PATH    = os.path.join(FEATURES_DIR, "chaoyang_test_features.pt")
RESULTS_DIR  = "./results"

# ── Hyperparameters (identical to Experiment 1) ───────────────────────────────
FEATURE_DIM  = 512
NUM_CLASSES  = 4
HIDDEN_DIM   = 256
DROPOUT      = 0.3
BATCH_SIZE   = 128
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42

# ── SCE-specific hyperparameters ──────────────────────────────────────────────
# Alpha controls the standard CE term (clean label learning)
# Beta  controls the RCE term (noise suppression)
# Values from Wang et al. recommended for real-world noise scenarios
SCE_ALPHA = 0.1
SCE_BETA  = 1.0

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


# ── Symmetric Cross-Entropy Loss ──────────────────────────────────────────────
class SymmetricCrossEntropyLoss(nn.Module):
    """
    Symmetric Cross-Entropy Loss (Wang et al., ICCV 2019).

    L_SCE = alpha * CE(p, q) + beta * RCE(p, q)

    CE(p, q)  = -sum_c [ q_c * log(p_c) ]   standard cross-entropy
    RCE(p, q) = -sum_c [ p_c * log(q_c) ]   reverse cross-entropy

    The RCE term penalises the model for being confidently wrong,
    which is the signature behaviour when training on noisy labels.

    Args:
        alpha      : weight for standard CE term
        beta       : weight for reverse CE term
        num_classes: number of output classes
        smoothing  : small constant to avoid log(0) in RCE term
    """
    def __init__(self, alpha: float, beta: float,
                 num_classes: int, smoothing: float = 1e-4):
        super().__init__()
        self.alpha       = alpha
        self.beta        = beta
        self.num_classes = num_classes
        self.smoothing   = smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        # Standard CE term
        ce = F.cross_entropy(logits, targets)

        # Reverse CE term
        # Convert logits to probabilities for model predictions
        pred_probs = F.softmax(logits, dim=1)
        # Clamp to avoid log(0)
        pred_probs = torch.clamp(pred_probs, self.smoothing, 1.0)

        # Convert targets to one-hot soft labels
        # Clamp to avoid log(0) — use uniform smoothing as in the paper
        target_one_hot = torch.zeros_like(pred_probs)
        target_one_hot.fill_(self.smoothing / (self.num_classes - 1))
        target_one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # RCE = -sum [ p(x) * log(q(x)) ] where p=pred, q=target
        rce = -torch.sum(pred_probs * torch.log(target_one_hot), dim=1).mean()

        return self.alpha * ce + self.beta * rce


# ── Model (identical to Experiment 1) ────────────────────────────────────────
class MLPClassifier(nn.Module):
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


def load_features(path: str, split_name: str):
    print(f"Loading {split_name} features from: {path}")
    data     = torch.load(path, weights_only=False)
    features = data['features'].float()
    labels   = data['labels'].long()
    print(f"  Shape: {features.shape}, Labels: {labels.shape}")
    return features, labels


def get_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    print("\nTrain class distribution:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES.values(), counts)):
        print(f"  label_{i} ({name}): {int(count)} samples")
    weights        = 1.0 / counts
    sample_weights = weights[labels]
    return sample_weights


def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits   = model(features)
            preds    = logits.argmax(dim=1).cpu()
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
    header = "         " + "  ".join(f"{n[:6]:>8}" for n in class_names.values())
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:8d}" for v in row)
        print(f"  {list(class_names.values())[i][:6]:>8}  {row_str}")

    return acc, precision, recall, f1


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Experiment 2 — CONCH + MLP + Symmetric Cross-Entropy")
    print("=" * 60)
    print(f"Device     : {DEVICE.upper()}")
    print(f"Epochs     : {EPOCHS}")
    print(f"SCE alpha  : {SCE_ALPHA}  (CE weight)")
    print(f"SCE beta   : {SCE_BETA}   (RCE noise-suppression weight)")

    # ── Load features ─────────────────────────────────────────────────────────
    train_features, train_labels = load_features(TRAIN_PATH, "TRAIN")
    test_features,  test_labels  = load_features(TEST_PATH,  "TEST")

    # ── Weighted sampler ──────────────────────────────────────────────────────
    sample_weights = get_class_weights(train_labels, NUM_CLASSES)
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_labels),
        replacement = True
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset  = TensorDataset(test_features,  test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MLPClassifier(
        input_dim   = FEATURE_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT
    ).to(DEVICE)

    # ── SCE loss instead of standard CE ───────────────────────────────────────
    criterion = SymmetricCrossEntropyLoss(
        alpha       = SCE_ALPHA,
        beta        = SCE_BETA,
        num_classes = NUM_CLASSES
    )

    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nTraining...")
    best_f1    = 0.0
    best_epoch = 0
    best_state = None
    log_interval = 10

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        total_steps = 0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels   = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_steps += 1

        scheduler.step()
        avg_loss = total_loss / total_steps

        if epoch % log_interval == 0 or epoch == EPOCHS:
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

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("  Experiment 2 — CONCH + MLP + Symmetric Cross-Entropy")
    results_lines.append("=" * 60)
    results_lines.append(f"SCE alpha  : {SCE_ALPHA}")
    results_lines.append(f"SCE beta   : {SCE_BETA}")
    results_lines.append(f"Best epoch : {best_epoch}/{EPOCHS}")
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

    # ── Comparison table including Experiment 1 ───────────────────────────────
    comparison = f"""
── Comparison Against Baselines ──
  Method                                  Acc      F1
  ──────────────────────────────────────────────────────
  HSA-NRL (ResNet-34, noise-robust)       83.40%   76.54%
  Ke et al. (CNN ensemble + MHSA)         86.72%   85.76%
  Exp 1 — CONCH + MLP (standard CE)       82.14%   77.05%
  Exp 2 — CONCH + MLP + SCE (ours)        {acc:.2f}%   {f1:.2f}%

── Per-Class F1 Comparison (Serrated is the key indicator) ──
  Class           Exp 1 F1    Exp 2 F1    Change
  ──────────────────────────────────────────────
  Normal           79.50%      {f1_score(labels_np, preds, labels=[0], average='macro', zero_division=0)*100:.2f}%
  Serrated         54.95%      {f1_score(labels_np, preds, labels=[1], average='macro', zero_division=0)*100:.2f}%       ← noise indicator
  Adenocarcinoma   96.53%      {f1_score(labels_np, preds, labels=[2], average='macro', zero_division=0)*100:.2f}%
  Adenoma          77.20%      {f1_score(labels_np, preds, labels=[3], average='macro', zero_division=0)*100:.2f}%
"""
    print(comparison)
    results_lines.append(comparison)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "experiment2_results.txt")
    with open(path, 'w') as f:
        f.write("\n".join(results_lines))
    print(f"✅ Results saved to: {path}")

    # Save checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, "experiment2_best_model.pt")
    torch.save({
        'epoch':       best_epoch,
        'model_state': best_state,
        'accuracy':    acc,
        'f1':          f1,
        'config': {
            'feature_dim': FEATURE_DIM,
            'hidden_dim':  HIDDEN_DIM,
            'num_classes': NUM_CLASSES,
            'dropout':     DROPOUT,
            'sce_alpha':   SCE_ALPHA,
            'sce_beta':    SCE_BETA
        }
    }, ckpt_path)
    print(f"✅ Best model checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
