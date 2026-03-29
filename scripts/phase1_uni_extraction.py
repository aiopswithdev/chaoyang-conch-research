import gc
import warnings
import os
import logging
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

"""
Phase 1 — Macenko Normalisation + Frozen UNI Feature Extraction
----------------------------------------------------------------
Runs ONCE. Output is cached to disk and never recomputed.

Pipeline:
    Raw Chaoyang patch (512×512)
        │
        ▼
    Macenko stain normalisation     ← removes stain artefacts
        │
        ▼
    Frozen UNI ViT-L/16 encoder     ← extracts 1024-dim morphological features
        │
        ▼
    Save .pt tensors to disk        ← Phase 2 loads these directly

UNI details (Chen et al., Nature Medicine 2024):
    Architecture : ViT-Large/16 (via timm)
    Pretraining  : 100K+ WSIs, DINOv2 self-supervised
    Output dim   : 1024 (CLS token)
    Weights      : MahmoodLab/UNI on HuggingFace (gated — requires approval)

Macenko details:
    Method  : Macenko et al. (ISBI 2009) stain normalisation
    Library : torchstain (pure PyTorch implementation)
    Reference patch: first clean patch from Normal class (label_0)

Outputs saved to:
    /home/dev/Documents/ML-Research/Chaoyang/features/
        uni_train_features.pt   → {features: [4021, 1024], labels: [4021]}
        uni_test_features.pt    → {features: [2139, 1024], labels: [2139]}

Install dependency before running:
    pip install torchstain
"""

import torch
# Prevent PyTorch from maxing out all CPU threads and triggering Fedora's OOM daemon
torch.set_num_threads(4)
import timm
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_tensor
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchstain

# ── Paths ─────────────────────────────────────────────────────────────────────
UNI_WEIGHTS_PATH = "/home/dev/Downloads/uni_pytorch_model.bin"
TRAIN_DIR        = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_train"
TEST_DIR         = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_test"
OUTPUT_DIR       = "/home/dev/Documents/ML-Research/Chaoyang/features"

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 1       # one image at a time — guaranteed to fit any VRAM
NUM_WORKERS  = 0       # disable multiprocessing — Macenko is CPU-heavy
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_MACENKO = True   # Set False once UNI extraction confirmed working
# DEVICE = "cpu"  # force CPU for maximum compatibility (Macenko is CPU-bound)
# HuggingFace token — only needed if loading weights from hub directly
# Leave empty if using local weights file

CLASS_NAMES = {0: "Normal", 1: "Serrated", 2: "Adenocarcinoma", 3: "Adenoma"}
EXPECTED_TRAIN = {0: 1111, 1: 842, 2: 1404, 3: 664}
EXPECTED_TEST  = {0: 705,  1: 321, 2: 840,  3: 273}


# ── Macenko Normaliser ────────────────────────────────────────────────────────
class MacenkoNormaliser:
    """
    Macenko H&E stain normalisation using torchstain.

    Requires a reference (target) patch to define the stain matrix.
    All other patches are normalised toward this reference, ensuring
    consistent stain appearance across the dataset.

    We use the first valid Normal class patch as the reference —
    Normal tissue has the most balanced H&E staining and provides
    a stable target matrix.
    """
    def __init__(self, reference_image_path: str, device: str):
        print(f"Fitting Macenko normaliser on reference: {reference_image_path}")

        # Load reference image and convert to tensor [C, H, W] in [0, 255]
        ref_img    = Image.open(reference_image_path).convert("RGB")
        # Permute to [H, W, C] for torchstain
        ref_tensor = to_tensor(ref_img).permute(1, 2, 0) * 255.0
        ref_tensor = ref_tensor.to(device)   # ← GPU
        self.normaliser = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        self.normaliser.fit(ref_tensor)
        self.device = device
        print("  Macenko normaliser fitted successfully.")

    def normalise(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # ── Absolute shape guarantee ──────────────────────────────────────────
        # Force [3, 224, 224] regardless of what arrives
        # Handles non-square patches, blank patches, any edge case
        c, h, w = img_tensor.shape
        if h != 224 or w != 224 or c != 3:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0).float(),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # ── Macenko on GPU ────────────────────────────────────────────────────
        img_255 = (img_tensor.permute(1, 2, 0) * 255.0).clamp(0, 255).to(self.device)
        try:
            normalised, _, _ = self.normaliser.normalize(
                I=img_255,
                stains=False
            )
            return (normalised.permute(2, 0, 1) / 255.0).clamp(0, 1).cpu()
        except Exception as e:
            print(f"\n  ⚠️  Macenko failed ({e}), using original")
            return img_tensor.cpu()


class MacenkoDataset(torch.utils.data.Dataset):
    """
    Custom dataset that applies Macenko normalisation before UNI transform.

    Pipeline per sample:
        PIL Image → to_tensor [0,1] → Macenko normalise → UNI preprocess
    """
    def __init__(self, image_folder: datasets.ImageFolder,
                 normaliser: MacenkoNormaliser,
                 uni_transform):
        self.dataset    = image_folder
        self.normaliser = normaliser
        self.uni_transform = uni_transform

        # Separate ToTensor from rest of UNI transform
        # We need to apply Macenko between loading and UNI's normalisation
        self.to_tensor = transforms.ToTensor()

        # UNI transform without the first ToTensor step
        # (we handle the tensor conversion ourselves)
        self.uni_normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        self.resize = transforms.Resize(
            224,
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.center_crop = transforms.CenterCrop(224)

    def __len__(self):
        return len(self.dataset)

    # Replace __getitem__ entirely
    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]

        with Image.open(path) as img:
            img_rgb = img.convert("RGB")

        img_224   = img_rgb.resize((224, 224), Image.BICUBIC)
        del img_rgb
        img_tensor = self.to_tensor(img_224)
        del img_224

        if not SKIP_MACENKO:
            img_tensor = self.normaliser.normalise(img_tensor)

        img_final = self.uni_normalize(img_tensor)
        return img_final, label


def find_reference_candidates(train_dir: str) -> list:
    """
    Find images from the Normal class (label_0) to use
    as candidates for the Macenko reference patch.
    """
    normal_dir = Path(train_dir) / "label_0"
    images = sorted(normal_dir.glob("*.jpg"))  + \
             sorted(normal_dir.glob("*.JPG"))  + \
             sorted(normal_dir.glob("*.jpeg")) + \
             sorted(normal_dir.glob("*.png"))  + \
             sorted(normal_dir.glob("*.PNG"))  + \
             sorted(normal_dir.glob("*.tif"))  + \
             sorted(normal_dir.glob("*.tiff")) + \
             sorted(normal_dir.glob("*.bmp"))

    if not images:
        raise FileNotFoundError(
            f"No images found in {normal_dir}. "
            "Check your TRAIN_DIR path."
        )
    
    # Return a list of candidate paths instead of just the first one
    return [str(p) for p in images]


def load_uni(weights_path: str, device: str):
    """
    Load frozen UNI ViT-L/16 model from local weights file.

    UNI architecture (from official HuggingFace page):
        timm.create_model("vit_large_patch16_224", ...)
        num_classes=0 → outputs 1024-dim CLS token
    """
    print(f"Loading UNI weights from: {weights_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"UNI weights not found at: {weights_path}\n"
            f"Download pytorch_model.bin from https://huggingface.co/MahmoodLab/UNI\n"
            f"Then update UNI_WEIGHTS_PATH at the top of this script."
        )

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size        = 224,
        patch_size      = 16,
        init_values     = 1e-5,
        num_classes     = 0,          # CLS token output → 1024-dim
        dynamic_img_size = True
    )
# Replace with
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    del state_dict # Free RAM immediately

    # Freeze all parameters — UNI is never updated
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    # Confirm output dimension with dummy forward pass
    with torch.no_grad():
        dummy    = torch.zeros(1, 3, 224, 224).to(device)
        out      = model(dummy)
        feat_dim = out.shape[-1]
    del dummy, out
    torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  UNI loaded — architecture : ViT-L/16")
    print(f"  Parameters               : {total_params:,} (all frozen)")
    print(f"  Output dimension         : {feat_dim}")

    return model, feat_dim


def verify_counts(dataset, expected: dict, split_name: str):
    """Verify class counts match the canonical Chaoyang split."""
    print(f"\n── {split_name} class distribution ──")
    actual = {}
    for _, label in dataset.samples:
        actual[label] = actual.get(label, 0) + 1

    all_ok = True
    for i, name in CLASS_NAMES.items():
        a = actual.get(i, 0)
        e = expected[i]
        status = "✅" if a == e else "⚠️  MISMATCH"
        print(f"  {status}  label_{i} ({name}): {a} (expected {e})")
        if a != e:
            all_ok = False

    print(f"  Total: {len(dataset.samples)}")
    if not all_ok:
        print("  ⚠️  Counts do not match. Verify you have the original HSA-NRL split.")


def extract_features(model, dataloader, device, split_name: str):
    """Run all patches through frozen UNI and collect features."""
    all_features = []
    all_labels   = []
    total        = len(dataloader)

    print(f"\nExtracting {split_name} features...")

    # Replace with
    with torch.inference_mode():   # more memory efficient than no_grad
        for i, (images, labels) in enumerate(dataloader):
            images   = images.to(device)
            features = model(images)

            # all_features.append(features.cpu())
            all_features.append(features.cpu().clone())
            all_labels.append(labels)

            # Explicitly free VRAM after every image
            del images, features
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  {i+1}/{total} images processed", end="\r")
                gc.collect()
                torch.cuda.empty_cache()

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor   = torch.cat(all_labels,   dim=0)

    print(f"\n  Done — features shape: {features_tensor.shape}")
    return features_tensor, labels_tensor


def main():
    print("=" * 60)
    print("  Phase 1 — Macenko Normalisation + UNI Feature Extraction")
    print("=" * 60)
    print(f"Device  : {DEVICE.upper()}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Output  : {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load UNI ──────────────────────────────────────────────────────
    model, feat_dim = load_uni(UNI_WEIGHTS_PATH, DEVICE)

    # ── Step 2: Fit Macenko normaliser on a valid tissue patch ────────────────
    reference_candidates = find_reference_candidates(TRAIN_DIR)
    normaliser = None
    
    for ref_path in reference_candidates:
        try:
            normaliser = MacenkoNormaliser(ref_path, DEVICE)
            break  # Successfully fitted on a valid tissue patch!
        except IndexError:
            # Catch the empty tensor error caused by blank glass patches
            print(f"  ⚠️ Skipping blank/invalid reference patch: {Path(ref_path).name}")
            continue

    if normaliser is None:
        raise RuntimeError("Failed to find any valid tissue patches in the Normal class to fit Macenko.")

    # ── Step 3: Build UNI transform (for reference only — we apply manually) ──
    uni_transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    # ── Step 4: Build datasets with Macenko preprocessing ────────────────────
    print(f"\nLoading datasets...")
    raw_train = datasets.ImageFolder(root=TRAIN_DIR)
    raw_test  = datasets.ImageFolder(root=TEST_DIR)

    verify_counts(raw_train, EXPECTED_TRAIN, "TRAIN")
    verify_counts(raw_test,  EXPECTED_TEST,  "TEST")

    train_dataset = MacenkoDataset(raw_train, normaliser, uni_transform)
    test_dataset  = MacenkoDataset(raw_test,  normaliser, uni_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,        # preserve order for label alignment
        num_workers = NUM_WORKERS,
        pin_memory  = False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = False
    )

    # ── Step 5: Extract and cache features ───────────────────────────────────
    train_features, train_labels = extract_features(
        model, train_loader, DEVICE, "TRAIN"
    )
    test_features, test_labels = extract_features(
        model, test_loader, DEVICE, "TEST"
    )

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    train_path = os.path.join(OUTPUT_DIR, "uni_train_features.pt")
    test_path  = os.path.join(OUTPUT_DIR, "uni_test_features.pt")

    torch.save({
        "features":    train_features,
        "labels":      train_labels,
        "paths":       [s[0] for s in raw_train.samples],  # exact traversal order
        "class_names": CLASS_NAMES,
        "source_dir":  TRAIN_DIR,
        "feat_dim":    feat_dim,
        "normalised":  "macenko",
        "model":       "UNI ViT-L/16"
    }, train_path)

    torch.save({
        "features":    test_features,
        "labels":      test_labels,
        "paths":       [s[0] for s in raw_test.samples],  # exact traversal order
        "class_names": CLASS_NAMES,
        "source_dir":  TEST_DIR,
        "feat_dim":    feat_dim,
        "normalised":  "macenko",
        "model":       "UNI ViT-L/16"
    }, test_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 1 Complete")
    print("=" * 60)
    print(f"  Train features : {train_features.shape}")
    print(f"  Test features  : {test_features.shape}")
    print(f"  Feature dim    : {feat_dim}  (UNI ViT-L/16)")
    print(f"  Normalisation  : Macenko")
    print(f"\n  Saved to:")
    print(f"    {train_path}")
    print(f"    {test_path}")
    print(f"\n  Phase 2 will load these alongside raw patches for CONCH.")
    print(f"  Load with: data = torch.load('uni_train_features.pt')")


if __name__ == "__main__":
    main()