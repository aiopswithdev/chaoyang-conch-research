"""
CONCH Feature Extraction for Chaoyang Dataset
----------------------------------------------
Extracts 512-dim patch embeddings from all train and test images
using the frozen CONCH ViT-B-16 encoder.

Outputs (saved to ./features/):
    chaoyang_train_features.pt  -> {'features': Tensor[N,512], 'labels': Tensor[N]}
    chaoyang_test_features.pt   -> {'features': Tensor[M,512], 'labels': Tensor[M]}

Label mapping (from folder names):
    label_0 -> 0  (Normal)
    label_1 -> 1  (Serrated)
    label_2 -> 2  (Adenocarcinoma)
    label_3 -> 3  (Adenoma)

Run once, cache results, never run again unless dataset changes.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from conch.open_clip_custom import create_model_from_pretrained

# ── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/home/dev/Downloads/pytorch_model.bin"
TRAIN_DIR    = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_train"
TEST_DIR     = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_test"
OUTPUT_DIR   = "/home/dev/Documents/ML-Research/Chaoyang/features"

# ── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 64    # reduce to 32 if you get CUDA out-of-memory errors
NUM_WORKERS  = 4     # parallel data loading, reduce to 0 if you get errors
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Class name mapping for sanity check output ───────────────────────────────
CLASS_NAMES = {
    0: "Normal",
    1: "Serrated",
    2: "Adenocarcinoma",
    3: "Adenoma"
}

# Expected counts from HSA-NRL paper (Zhu et al., 2022)
EXPECTED_TRAIN = {0: 1111, 1: 842, 2: 1404, 3: 664}
EXPECTED_TEST  = {0: 705,  1: 321, 2: 840,  3: 273}


def load_conch(weights_path: str, device: str):
    """Load CONCH from local weights file and return encoder + preprocess."""
    print(f"Loading CONCH weights from: {weights_path}")
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=weights_path
    )
    # Extract vision encoder only — we do not need the text encoder
    vision_encoder = model.visual
    vision_encoder.eval()
    vision_encoder.to(device)

    # Freeze all parameters — we never train CONCH
    for param in vision_encoder.parameters():
        param.requires_grad = False

    print(f"CONCH loaded successfully on {device.upper()}")
    return vision_encoder, preprocess


def verify_dataset(dataset, expected_counts: dict, split_name: str):
    """
    Count samples per class and compare against expected values.
    Warns if counts do not match — catches wrong dataset version early.
    """
    print(f"\n── {split_name} dataset class distribution ──")
    actual_counts = {}
    for _, label in dataset.samples:
        actual_counts[label] = actual_counts.get(label, 0) + 1

    all_correct = True
    for label_idx, class_name in CLASS_NAMES.items():
        actual   = actual_counts.get(label_idx, 0)
        expected = expected_counts[label_idx]
        status   = "✅" if actual == expected else "⚠️  MISMATCH"
        print(f"  {status}  label_{label_idx} ({class_name}): {actual} "
              f"(expected {expected})")
        if actual != expected:
            all_correct = False

    print(f"  Total: {len(dataset.samples)} samples")
    if not all_correct:
        print("  ⚠️  Some counts do not match expected values.")
        print("      Check that you have the original HSA-NRL Chaoyang split.")
    else:
        print("  All counts match expected values.")


def extract_features(
    encoder,
    data_dir: str,
    preprocess,
    expected_counts: dict,
    split_name: str,
    batch_size: int,
    num_workers: int,
    device: str
) -> dict:
    """
    Run all images through CONCH encoder and return feature dict.
    Uses CONCH's own preprocess — do not substitute torchvision transforms.
    """
    dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
    verify_dataset(dataset, expected_counts, split_name)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,       # keep order consistent with labels
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )

    all_features = []
    all_labels   = []
    total_batches = len(loader)

    print(f"\nExtracting {split_name} features...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            output = encoder(images)
            # CONCH visual encoder returns a tuple — first element is the embedding
            features = output[0] if isinstance(output, tuple) else output

            all_features.append(features.cpu())
            all_labels.append(labels)

            # Progress indicator every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Batch {batch_idx + 1}/{total_batches} "
                      f"— {(batch_idx + 1) * batch_size} images processed",
                      end="\r")

    features_tensor = torch.cat(all_features, dim=0)  # [N, 512]
    labels_tensor   = torch.cat(all_labels,   dim=0)  # [N]

    print(f"\n  Done. Feature tensor shape: {features_tensor.shape}")
    print(f"  Label tensor shape:   {labels_tensor.shape}")

    return {
        "features":    features_tensor,
        "labels":      labels_tensor,
        "class_names": CLASS_NAMES,
        "source_dir":  data_dir
    }


def main():
    print("=" * 60)
    print("  CONCH Feature Extraction — Chaoyang Dataset")
    print("=" * 60)
    print(f"Device : {DEVICE.upper()}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Output : {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    encoder, preprocess = load_conch(WEIGHTS_PATH, DEVICE)

    # ── Extract train features ────────────────────────────────────────────────
    train_data = extract_features(
        encoder       = encoder,
        data_dir      = TRAIN_DIR,
        preprocess    = preprocess,
        expected_counts = EXPECTED_TRAIN,
        split_name    = "TRAIN",
        batch_size    = BATCH_SIZE,
        num_workers   = NUM_WORKERS,
        device        = DEVICE
    )
    train_output_path = os.path.join(OUTPUT_DIR, "chaoyang_train_features.pt")
    torch.save(train_data, train_output_path)
    print(f"\n✅ Train features saved to: {train_output_path}")

    # ── Extract test features ─────────────────────────────────────────────────
    test_data = extract_features(
        encoder       = encoder,
        data_dir      = TEST_DIR,
        preprocess    = preprocess,
        expected_counts = EXPECTED_TEST,
        split_name    = "TEST",
        batch_size    = BATCH_SIZE,
        num_workers   = NUM_WORKERS,
        device        = DEVICE
    )
    test_output_path = os.path.join(OUTPUT_DIR, "chaoyang_test_features.pt")
    torch.save(test_data, test_output_path)
    print(f"✅ Test features saved to:  {test_output_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Extraction complete. Summary:")
    print("=" * 60)
    print(f"  Train features : {train_data['features'].shape}")
    print(f"  Test features  : {test_data['features'].shape}")
    print(f"  Feature dim    : 512  (CONCH ViT-B-16)")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print("\n  Next step: train your classifier on the cached features.")
    print("  Load with: data = torch.load('chaoyang_train_features.pt')")


if __name__ == "__main__":
    main()
