import warnings
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from peft import LoraConfig, get_peft_model
from conch.open_clip_custom import create_model_from_pretrained

# Mute warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("timm").setLevel(logging.ERROR)

# ── Paths & Hyperparameters ───────────────────────────────────────────────────
CONCH_WEIGHTS  = "/home/d3vd5k/projects/DL/CONCH/pytorch_model.bin"
TEST_DIR       = "/home/d3vd5k/projects/DL/chanoyang/organized_images_test"
UNI_TEST_CACHE = "/home/d3vd5k/projects/DL/cmae+conch+uni/extracted_features/uni_test_features.pt"
RESULTS_DIR    = "/home/d3vd5k/projects/DL/conch+uni/using_cached_UNI/dev_phase_1_2/results_lora_advanced"

# 👇 Change this to evaluate your different saved models
CHECKPOINT_NAME = "best_acc.pt"  # Options: best_f1.pt, best_acc.pt, best_f1_swa.pt, best_acc_swa.pt

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, CHECKPOINT_NAME)
OUTPUT_FILE    = os.path.join(RESULTS_DIR, f"eval_report_{CHECKPOINT_NAME.replace('.pt', '.txt')}")

NUM_CLASSES = 4
COMMON_DIM  = 256      
NUM_HEADS   = 4        
HIDDEN_DIM  = 256      
DROPOUT     = 0.4
BATCH_SIZE  = 16       
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {0: "Normal", 1: "Serrated", 2: "Adenocarcinoma", 3: "Adenoma"}

# ── Upgraded Architecture ─────────────────────────────────────────────────────

class DualDataset(Dataset):
    def __init__(self, cache_path, image_dir, transform=None):
        cache = torch.load(cache_path, weights_only=False)
        self.uni_features = cache["features"].float()   
        self.labels       = cache["labels"].long()      
        self.transform    = transform
        self.image_folder = datasets.ImageFolder(root=image_dir)

    def __len__(self): return len(self.uni_features)
    def __getitem__(self, idx):
        uni_feat = self.uni_features[idx]
        img_path, label = self.image_folder.samples[idx]
        with Image.open(img_path) as img: img_rgb = img.convert("RGB")
        image = self.transform(img_rgb) if self.transform else transforms.ToTensor()(img_rgb)
        return uni_feat, image, label

class BidirectionalFusionHead(nn.Module):
    def __init__(self, uni_dim, conch_dim, common_dim, num_heads, hidden_dim, num_classes, dropout):
        super().__init__()
        self.noise_stddev = 0.0 # No noise during eval
        self.uni_proj   = nn.Linear(uni_dim,   common_dim)
        self.conch_proj = nn.Linear(conch_dim, common_dim)
        self.cross_attn_U_C = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_C_U = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Upgraded Scalar Gate
        self.gate = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim // 4),
            nn.GELU(),
            nn.Linear(common_dim // 4, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(common_dim)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, uni_feat, conch_feat):
        u_proj = self.uni_proj(uni_feat).unsqueeze(1)    
        c_proj = self.conch_proj(conch_feat).unsqueeze(1)
        u_attended, _ = self.cross_attn_U_C(query=u_proj, key=c_proj, value=c_proj)
        c_attended, _ = self.cross_attn_C_U(query=c_proj, key=u_proj, value=u_proj)
        u_out = u_attended.squeeze(1) + u_proj.squeeze(1) 
        c_out = c_attended.squeeze(1) + c_proj.squeeze(1) 
        
        gate_weight = self.gate(torch.cat([u_out, c_out], dim=1)) 
        fused = (gate_weight * u_out) + ((1 - gate_weight) * c_out)
        return self.classifier(self.norm(fused))

class DualEncoderModel(nn.Module):
    def __init__(self, conch_encoder, fusion_head):
        super().__init__()
        self.conch_encoder = conch_encoder
        self.fusion_head   = fusion_head

    def forward(self, uni_feat, image):
        out = self.conch_encoder(image)
        conch_feat = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'last_hidden_state', out)[:, 0] if hasattr(out, 'last_hidden_state') else out
        return self.fusion_head(uni_feat, conch_feat)

# ── Evaluation Logic ──────────────────────────────────────────────────────────

def main():
    print("Loading CONCH encoder...")
    model_conch, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=CONCH_WEIGHTS)
    encoder = model_conch.visual
    
    # Upgraded LoRA Config
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["qkv", "proj", "mlp.fc1", "mlp.fc2"], 
        bias="none"
    )
    encoder = get_peft_model(encoder, lora_config)
    
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        out = encoder(dummy)
        conch_dim = out[0].shape[-1] if isinstance(out, (tuple, list)) else getattr(out, 'last_hidden_state', out)[:, 0].shape[-1] if hasattr(out, 'last_hidden_state') else out.shape[-1]

    fusion_head = BidirectionalFusionHead(1024, conch_dim, COMMON_DIM, NUM_HEADS, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
    model = DualEncoderModel(encoder, fusion_head).to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    print(f"Loading weights from {CHECKPOINT_NAME}...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    conch_normalize = next((t for t in preprocess.transforms if isinstance(t, transforms.Normalize)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        conch_normalize
    ])

    print("Loading test dataset...")
    test_dataset = DualDataset(UNI_TEST_CACHE, TEST_DIR, transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Running evaluation (this may take a minute)...")
    all_probs, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for uni_feat, images, labels in test_loader:
            logits = model(uni_feat.to(DEVICE), images.to(DEVICE))
            probs = F.softmax(logits, dim=1) 
            preds = logits.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    y_probs = np.concatenate(all_probs)
    y_preds = np.concatenate(all_preds)
    y_true  = np.concatenate(all_labels)

    # ── CALCULATE METRICS ──
    acc = accuracy_score(y_true, y_preds)
    
    # Macro Metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_preds, average='macro', zero_division=0)
    roc_auc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')

    # Weighted Metrics
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_preds, average='weighted', zero_division=0)
    roc_auc_weight = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')

    # Per-Class Metrics
    prec_per, rec_per, f1_per, support = precision_recall_fscore_support(y_true, y_preds, average=None, zero_division=0)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    roc_auc_per = roc_auc_score(y_true_bin, y_probs, average=None, multi_class='ovr')

    # ── FORMAT OUTPUT STRING ──
    out = []
    out.append("="*60)
    out.append(f" 📊 TEST SET STATISTICS ({CHECKPOINT_NAME})")
    out.append("="*60)
    
    out.append(f"\n[ OVERALL METRICS ]")
    out.append(f"  Accuracy           : {acc * 100:.2f}%")
    out.append("-" * 40)
    out.append(f"  Macro F1           : {f1_macro * 100:.2f}%")
    out.append(f"  Macro Precision    : {prec_macro * 100:.2f}%")
    out.append(f"  Macro Recall       : {rec_macro * 100:.2f}%")
    out.append(f"  Macro ROC-AUC      : {roc_auc_macro:.4f}")
    out.append("-" * 40)
    out.append(f"  Weighted F1        : {f1_weight * 100:.2f}%")
    out.append(f"  Weighted Precision : {prec_weight * 100:.2f}%")
    out.append(f"  Weighted Recall    : {rec_weight * 100:.2f}%")
    out.append(f"  Weighted ROC-AUC   : {roc_auc_weight:.4f}")
    
    out.append("\n[ PER-CLASS METRICS ]")
    out.append("-" * 75)
    header = f"{'Class':<16} | {'Support':<7} | {'F1':<7} | {'Prec':<7} | {'Recall':<7} | {'ROC-AUC':<7}"
    out.append(header)
    out.append("-" * 75)
    
    for i in range(NUM_CLASSES):
        c_name = CLASS_NAMES[i]
        out.append(f"{c_name:<16} | {support[i]:<7} | {f1_per[i]*100:>6.2f}% | {prec_per[i]*100:>6.2f}% | {rec_per[i]*100:>6.2f}% | {roc_auc_per[i]:>7.4f}")

    out.append("\n[ CONFUSION MATRIX ]")
    out.append("-" * 60)
    cm = confusion_matrix(y_true, y_preds)
    cm_header = "         " + "  ".join(f"{n[:6]:>8}" for n in CLASS_NAMES.values())
    out.append(cm_header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:8d}" for v in row)
        out.append(f"  {list(CLASS_NAMES.values())[i][:6]:>8}  {row_str}")
    out.append("="*60 + "\n")

    # Combine all lines into a single string
    final_report = "\n".join(out)
    
    # Print to console
    print(final_report)
    
    # Save to file
    with open(OUTPUT_FILE, "w") as f:
        f.write(final_report)
        
    print(f"✅ Detailed metrics successfully saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()