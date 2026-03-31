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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from conch.open_clip_custom import create_model_from_pretrained
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
CONCH_WEIGHTS  = "/home/d3vd5k/projects/DL/CONCH/pytorch_model.bin"
TRAIN_DIR      = "/home/d3vd5k/projects/DL/chanoyang/organized_images_train"
TEST_DIR       = "/home/d3vd5k/projects/DL/chanoyang/organized_images_test"
UNI_TRAIN_CACHE = "/home/d3vd5k/projects/DL/extracted_features/uni_train_features.pt"
UNI_TEST_CACHE  = "/home/d3vd5k/projects/DL/extracted_features/uni_test_features.pt"
RESULTS_DIR    = "/home/d3vd5k/projects/DL/results"

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

# ── Advanced Toggles & Features ──────────────────────────────────────────
USE_FOCAL_LOSS      = True   
USE_SWA             = True   
SWA_START_EPOCH     = 81    
SWA_LR              = 5e-5   
USE_SAM             = True   
SAM_RHO             = 0.05   
LORA_STOCH_DEPTH    = 0.15   

CLASS_NAMES = {0: "Normal", 1: "Serrated", 2: "Adenocarcinoma", 3: "Adenoma"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ── Advanced Optimizers & Losses ──────────────────────────────────────────────

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        
        # --- Circuit breaker for FP16 NaNs/Infs ---
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            if zero_grad: self.zero_grad()
            return False 
            
        for group in self.param_groups:
            # --- Bumped 1e-12 to 1e-6 for FP16 safety ---
            scale = group["rho"] / (grad_norm + 1e-6)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()
        return True

    @torch.no_grad()
    def restore_weights(self):
        """Restores the original weights before the actual step is taken."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # ONLY restore if we actually saved it in the first step
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
                    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else torch.ones(NUM_CLASSES))

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ── LoRA Stochastic Depth Injection ───────────────────────────────────────────

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
        apply_lora_stochastic_depth(encoder, LORA_STOCH_DEPTH)

    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        out   = encoder(dummy)
        feat  = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'last_hidden_state', out)[:, 0] if hasattr(out, 'last_hidden_state') else out
        conch_dim = feat.shape[-1]
    torch.cuda.empty_cache()

    return encoder, preprocess, conch_dim

# ── Bidirectional Dual Encoder Fusion Head ────────────────────────────────────

class BidirectionalFusionHead(nn.Module):
    def __init__(self, uni_dim, conch_dim, common_dim, num_heads, hidden_dim, num_classes, dropout, noise_stddev):
        super().__init__()
        self.noise_stddev = noise_stddev

        self.uni_proj   = nn.Linear(uni_dim,   common_dim)
        self.conch_proj = nn.Linear(conch_dim, common_dim)

        self.cross_attn_U_C = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_C_U = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.gate = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
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
        if self.training and self.noise_stddev > 0:
            uni_feat   = uni_feat   + torch.randn_like(uni_feat)   * self.noise_stddev
            conch_feat = conch_feat + torch.randn_like(conch_feat) * self.noise_stddev

        u_proj = self.uni_proj(uni_feat).unsqueeze(1)    
        c_proj = self.conch_proj(conch_feat).unsqueeze(1)

        u_attended, _ = self.cross_attn_U_C(query=u_proj, key=c_proj, value=c_proj)
        c_attended, _ = self.cross_attn_C_U(query=c_proj, key=u_proj, value=u_proj)

        u_out = u_attended.squeeze(1) + u_proj.squeeze(1) 
        c_out = c_attended.squeeze(1) + c_proj.squeeze(1) 

        gate_weight = self.gate(torch.cat([u_out, c_out], dim=1)) 
        fused = (gate_weight * u_out) + ((1 - gate_weight) * c_out)

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

# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Setup ──
    conch_encoder, conch_preprocess, conch_dim = load_conch_lora(CONCH_WEIGHTS, DEVICE)
    train_transform, test_transform = build_transforms(conch_preprocess)

    train_dataset = DualDataset(UNI_TRAIN_CACHE, TRAIN_DIR, transform=train_transform)
    test_dataset  = DualDataset(UNI_TEST_CACHE, TEST_DIR, transform=test_transform)

    counts = torch.bincount(train_dataset.labels, minlength=NUM_CLASSES).float()
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES 
    
    sampler = WeightedRandomSampler(weights=class_weights[train_dataset.labels], num_samples=len(train_dataset.labels), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    fusion_head = BidirectionalFusionHead(
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
                   list(model.fusion_head.gate.parameters()) + list(model.fusion_head.norm.parameters()), 'lr': LR_FUSION},
        {'params': model.fusion_head.classifier.parameters(), 'lr': LR_HEAD}
    ]

    base_opt = optim.AdamW
    if USE_SAM:
        optimizer = SAM(param_groups, base_opt, rho=SAM_RHO, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = base_opt(param_groups, weight_decay=WEIGHT_DECAY)

    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights.to(DEVICE), gamma=2.0)
        print(f"  Using Focal Loss (gamma=2.0, alpha={class_weights.cpu().numpy().round(3)})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("  Using CrossEntropy Loss")

    # ── SWA Setup ──
    if USE_SWA:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer.base_optimizer if USE_SAM else optimizer, swa_lr=SWA_LR)
        print(f"  SWA enabled (Starts at epoch {SWA_START_EPOCH})")

    scaler = torch.cuda.amp.GradScaler()

    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_steps = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS}", leave=True, dynamic_ncols=True)

        for uni_feat, images, labels in pbar:
            uni_feat, images, labels = uni_feat.to(DEVICE), images.to(DEVICE), labels.to(DEVICE)

            if USE_SAM:
                # --- PASS 1 ---
                with torch.cuda.amp.autocast():
                    loss = criterion(model(uni_feat, images), labels)
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # --- SAFEGUARD: The Circuit Breaker ---
                if optimizer.first_step(zero_grad=True) is False:
                    # Exploding gradients detected! Reset and skip this batch.
                    optimizer.zero_grad()
                    continue

                # --- PASS 2 ---
                with torch.cuda.amp.autocast():
                    loss_2 = criterion(model(uni_feat, images), labels)
                scaler.scale(loss_2).backward()
                
                optimizer.restore_weights()
                
                scaler.unscale_(optimizer.base_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer.base_optimizer)
                scaler.update()
                
                optimizer.zero_grad()
                step_loss = loss.item()
            else:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = criterion(model(uni_feat, images), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                step_loss = loss.item()
            
            total_loss += step_loss
            total_steps += 1
            
            pbar.set_postfix(loss=f"{total_loss/total_steps:.4f}")

        if USE_SWA and epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        preds, labels_np = evaluate(model, test_loader, DEVICE)
        f1  = f1_score(labels_np, preds, average='macro', zero_division=0) * 100
        acc = accuracy_score(labels_np, preds) * 100

        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/total_steps:.4f} | Acc: {acc:.2f}% | F1: {f1:.2f}%")

        if f1 > best_f1:
            best_f1, best_f1_epoch = f1, epoch
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_f1.pt"))

        if epoch % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if USE_SWA:
        print("\nUpdating SWA BatchNorm statistics...")
        
        # --- NEW: Custom Multi-Input BN Update Loop ---
        swa_model.train() # BN layers only track running stats in train mode
        with torch.no_grad():
            for uni_feat, images, _ in train_loader:
                uni_feat = uni_feat.to(DEVICE)
                images   = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    swa_model(uni_feat, images) # Forward pass updates the stats
        # ----------------------------------------------

        # Now evaluate the updated SWA model
        preds, labels_np = evaluate(swa_model.module, test_loader, DEVICE)
        swa_f1 = f1_score(labels_np, preds, average='macro', zero_division=0) * 100
        print(f"SWA Model F1: {swa_f1:.2f}%")
        
        if swa_f1 > best_f1:
            print("🟢 SWA Outperformed Best Single Epoch! Saving SWA Model.")
            torch.save(swa_model.module.state_dict(), os.path.join(RESULTS_DIR, "best_f1.pt"))
if __name__ == "__main__":
    main()