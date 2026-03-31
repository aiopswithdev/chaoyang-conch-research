import warnings
import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import sys

from peft import LoraConfig, get_peft_model
from conch.open_clip_custom import create_model_from_pretrained

# Mute warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("timm").setLevel(logging.ERROR)

# ── Paths & Hyperparameters ───────────────────────────────────────────────────
CONCH_WEIGHTS   = "/home/d3vd5k/projects/DL/CONCH/pytorch_model.bin"
CHECKPOINT_PATH = "/home/d3vd5k/projects/DL/Results/best_f1.pt"
IMAGE_PATH      = "/home/d3vd5k/projects/DL/chanoyang/organized_images_test/label_0/535943_2-IMG009x015-0.JPG"
UNI_WEIGHTS     = "/home/d3vd5k/projects/DL/UNI/pytorch_model.bin" 

NUM_CLASSES = 4
COMMON_DIM  = 256      
NUM_HEADS   = 4        
HIDDEN_DIM  = 256      
DROPOUT     = 0.4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {0: "Normal", 1: "Serrated", 2: "Adenocarcinoma", 3: "Adenoma"}

class BidirectionalFusionHead(nn.Module):
    def __init__(self, uni_dim, conch_dim, common_dim, num_heads, hidden_dim, num_classes, dropout):
        super().__init__()
        self.noise_stddev = 0.0 # No noise during inference
        self.uni_proj   = nn.Linear(uni_dim,   common_dim)
        self.conch_proj = nn.Linear(conch_dim, common_dim)
        self.cross_attn_U_C = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_C_U = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(common_dim * 2, common_dim), nn.Sigmoid())
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

# ── Helper Functions ──────────────────────────────────────────────────────────

def load_models():
    print(f"[1/3] Loading live UNI model from local weights...")
    try:
        # Initialize the ViT-Large architecture and load local weights
        uni_model = timm.create_model(
            "vit_large_patch16_224", 
            pretrained=False, 
            num_classes=0, 
            init_values=1e-5, 
            dynamic_img_size=True,
            checkpoint_path=UNI_WEIGHTS
        )
    except Exception as e:
        print(f"\n❌ Failed to load local UNI weights. Error: {e}")
        print(f"Ensure the path is correct: {UNI_WEIGHTS}")
        sys.exit(1)
        
    uni_model.eval().to(DEVICE)

    print("[2/3] Loading CONCH and applying LoRA...")
    model_conch, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=CONCH_WEIGHTS)
    encoder = model_conch.visual
    
    lora_config = LoraConfig(r=16, lora_alpha=16, target_modules=["qkv", "proj"], bias="none")
    encoder = get_peft_model(encoder, lora_config)
    
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        out = encoder(dummy)
        conch_dim = out[0].shape[-1] if isinstance(out, (tuple, list)) else getattr(out, 'last_hidden_state', out)[:, 0].shape[-1] if hasattr(out, 'last_hidden_state') else out.shape[-1]

    fusion_head = BidirectionalFusionHead(1024, conch_dim, COMMON_DIM, NUM_HEADS, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
    dual_model = DualEncoderModel(encoder, fusion_head).to(DEVICE)

    print(f"[3/3] Loading trained fusion weights from {os.path.basename(CHECKPOINT_PATH)}...")
    dual_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    dual_model.eval()

    return uni_model, dual_model, preprocess

def build_transforms(conch_preprocess):
    conch_normalize = next((t for t in conch_preprocess.transforms if isinstance(t, transforms.Normalize)), 
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    uni_normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    conch_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        conch_normalize
    ])
    
    uni_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        uni_normalize
    ])
    
    return uni_transform, conch_transform

# ── Main Inference Logic ──────────────────────────────────────────────────────

def predict(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    uni_model, dual_model, conch_preprocess = load_models()
    uni_transform, conch_transform = build_transforms(conch_preprocess)

    print(f"\nProcessing Image: {os.path.basename(image_path)}")
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return

    img_uni   = uni_transform(img).unsqueeze(0).to(DEVICE)   
    img_conch = conch_transform(img).unsqueeze(0).to(DEVICE) 

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            uni_feat = uni_model(img_uni) 
            logits = dual_model(uni_feat, img_conch)
            
            probs = F.softmax(logits, dim=1).squeeze(0) 
            pred_class_idx = logits.argmax(dim=1).item()

    # ── Display Results ──
    print("\n" + "="*40)
    print(" 🎯 PREDICTION RESULTS")
    print("="*40)
    
    pred_name = CLASS_NAMES[pred_class_idx]
    confidence = probs[pred_class_idx].item() * 100
    
    print(f"Predicted Class : ** {pred_name} **")
    print(f"Confidence      : {confidence:.2f}%\n")
    
    print("Confidence Breakdown:")
    print("-" * 25)
    for i in range(NUM_CLASSES):
        c_name = CLASS_NAMES[i]
        c_prob = probs[i].item() * 100
        bar = "█" * int(c_prob / 5)
        print(f"{c_name:<16} | {c_prob:>6.2f}% | {bar}")
    print("="*40 + "\n")

if __name__ == "__main__":
    predict(IMAGE_PATH)