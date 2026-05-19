"""
Experiment — UNI Frozen Baseline
---------------------------------------
Single encoder baseline: frozen UNI features + MLP head
Standard cross-entropy loss, no noise handling.

This establishes the UNI-only baseline to compare against:
  - HSA-NRL (ResNet-34, no domain pretraining): 83.40% acc, 76.54% F1
  - Ke et al. (CNN ensemble + MHSA):             86.72% acc, 85.76% F1
  - CONCH + MLP (Experiment 1, ours)             82.14% acc, ? F1

Metrics reported (macro-averaged to match both comparison papers):
  Accuracy, Precision, Recall, F1, per-class breakdown

Results saved to: ./results/experiment_uni_frozen_results.txt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = "/home/d3vd5k/Projects/CANCER-RESEARCH/uni_features"
TRAIN_PATH   = os.path.join(FEATURES_DIR, "uni_train_features.pt")
TEST_PATH    = os.path.join(FEATURES_DIR, "uni_test_features.pt")
RESULTS_DIR  = "./results"

# ── Hyperparameters ───────────────────────────────────────────────────────────
FEATURE_DIM  = 1024      # UNI ViT-L/16 output dimension
NUM_CLASSES  = 4
HIDDEN_DIM   = 256
DROPOUT      = 0.3
BATCH_SIZE   = 128
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42

# Label mapping matching Ke et al. and HSA-NRL
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


class MLPClassifier(nn.Module):
    """
    Two-layer MLP classifier head on top of frozen UNI features.
    """
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
    """Load cached UNI features and labels from disk."""
    print(f"Loading {split_name} features from: {path}")
    data     = torch.load(path, weights_only=False)
    features = data['features'].float()
    labels   = data['labels'].long()
    print(f"  Shape: {features.shape}, Labels: {labels.shape}")
    return features, labels


def get_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted random sampling.
    Addresses Chaoyang class imbalance (adenoma has only 664 train samples).
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    print("\nTrain class distribution:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES.values(), counts)):
        print(f"  label_{i} ({name}): {int(count)} samples")

    weights = 1.0 / counts
    sample_weights = weights[labels]
    return sample_weights


def evaluate(model, loader, device):
    """Run evaluation and return predictions and ground truth."""
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

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return preds, labels


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


def save_results(results_text: str):
    """Save results to a text file for paper reporting."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "experiment_uni_frozen_results.txt")
    with open(path, 'w') as f:
        f.write(results_text)
    print(f"\n✅ Results saved to: {path}")


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Experiment — UNI Frozen Baseline")
    print("=" * 60)
    print(f"Device     : {DEVICE.upper()}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"LR         : {LR}")
    print(f"Hidden dim : {HIDDEN_DIM}")
    print(f"Dropout    : {DROPOUT}")

    # ── Load features ─────────────────────────────────────────────────────────
    train_features, train_labels = load_features(TRAIN_PATH, "TRAIN")
    test_features,  test_labels  = load_features(TEST_PATH,  "TEST")

    # ── Weighted sampler to handle class imbalance ────────────────────────────
    sample_weights = get_class_weights(train_labels, NUM_CLASSES)
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_labels),
        replacement = True
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset  = TensorDataset(test_features,  test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        sampler    = sampler       # weighted sampling, not shuffle
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle    = False
    )

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model = MLPClassifier(
        input_dim   = FEATURE_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nModel parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nTraining...")
    best_f1       = 0.0
    best_epoch    = 0
    best_state    = None
    log_interval  = 10

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

        # Evaluate every log_interval epochs
        if epoch % log_interval == 0 or epoch == EPOCHS:
            preds, labels_np = evaluate(model, test_loader, DEVICE)
            f1 = f1_score(labels_np, preds, average='macro',
                          zero_division=0) * 100
            acc = accuracy_score(labels_np, preds) * 100

            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Test Acc: {acc:.2f}% | "
                  f"Test F1: {f1:.2f}%")

            # Save best model by F1 (not accuracy — fairer for imbalanced data)
            if f1 > best_f1:
                best_f1    = f1
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}

    # ── Final evaluation with best model ──────────────────────────────────────
    print(f"\nBest model was at epoch {best_epoch} with F1: {best_f1:.2f}%")
    model.load_state_dict(best_state)

    preds, labels_np = evaluate(model, test_loader, DEVICE)

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("  Experiment — UNI Frozen Baseline")
    results_lines.append("=" * 60)
    results_lines.append(f"Best epoch : {best_epoch}/{EPOCHS}")
    results_lines.append("")

    import io, sys
    # Capture print output into results string
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    acc, precision, recall, f1 = print_metrics(
        preds, labels_np, "TEST", CLASS_NAMES
    )

    sys.stdout = old_stdout
    captured = buffer.getvalue()
    print(captured)
    results_lines.append(captured)

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison = f"""
── Comparison Against Published Baselines ──
  Method                              Acc      F1
  ─────────────────────────────────────────────────
  HSA-NRL (ResNet-34, noise-robust)   83.40%   76.54%
  Ke et al. (CNN ensemble + MHSA)     86.72%   85.76%
  CONCH + MLP (Experiment 1, ours)    82.14%   ?
  UNI + MLP (Experiment uni_frozen, ours)    {acc:.2f}%   {f1:.2f}%
"""
    print(comparison)
    results_lines.append(comparison)

    # Save everything
    save_results("\n".join(results_lines))

    # Save best model checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, "experiment_uni_frozen_best_model.pt")
    torch.save({
        'epoch':       best_epoch,
        'model_state': best_state,
        'accuracy':    acc,
        'f1':          f1,
        'config': {
            'feature_dim': FEATURE_DIM,
            'hidden_dim':  HIDDEN_DIM,
            'num_classes': NUM_CLASSES,
            'dropout':     DROPOUT
        }
    }, ckpt_path)
    print(f"✅ Best model checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()