import torch
import os
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from PIL import Image

# ── Import Custom Architecture ────────────────────────────────────────────────
import sys
sys.path.insert(0, "scripts") # Adjust this if your phase2 script is in the same folder
from phase2_self_LoRA import DualEncoderFusionHead, DualEncoderModel
from conch.open_clip_custom import create_model_from_pretrained

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = {0: "Normal", 1: "Serrated", 2: "Adenocarcinoma", 3: "Adenoma"}

TARGET_IMAGE_PATH = "/home/dev/Documents/ML-Research/Chaoyang/archive/organized_images_test/label_1/535958-IMG011x011-1.JPG"
CONCH_BIN_PATH    = "/home/dev/Documents/ML-Research/Chaoyang/archive/pytorch_model.bin"
CHECKPOINT_PATH   = "/home/dev/Downloads/phase2_best_acc_selfLoRA.pt"
UNI_CACHE_PATH    = "/home/dev/Documents/ML-Research/Chaoyang/features/uni_test_features.pt" # Evaluating a test image

def main():
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    ckpt   = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    print(f"Checkpoint Loaded: Epoch {ckpt['epoch']} | Acc {ckpt['accuracy']:.2f}% | F1 {ckpt['f1']:.2f}%\n")

    # 1. Rebuild CONCH encoder with LoRA
    print("Rebuilding CONCH + LoRA architecture...")
    conch_model, _ = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=CONCH_BIN_PATH
    )
    encoder = conch_model.visual
    
    # Freeze base weights
    for p in encoder.parameters():
        p.requires_grad = False

    # Apply LoRA config from the saved checkpoint
    encoder = get_peft_model(encoder, LoraConfig(
        r              = config["lora_r"],
        lora_alpha     = config["lora_alpha"],
        lora_dropout   = config["lora_dropout"],
        target_modules = config["target_modules"],
        bias           = "none"
    ))

    # 2. Rebuild Fusion Head & Full Model
    fusion_head = DualEncoderFusionHead(
        uni_dim=config["uni_dim"], conch_dim=config["conch_dim"],
        common_dim=config["common_dim"], 
        num_heads=config["num_heads"],     # <--- FIXED: Dynamically pulls '8' from config
        hidden_dim=config["hidden_dim"], 
        num_classes=4,
        dropout=config["dropout"], 
        noise_stddev=0.0                   # Turn off noise for pure inference
    )
    
    model = DualEncoderModel(encoder, fusion_head).to(DEVICE)
    
    # Load the trained SOTA weights into the reconstructed architecture
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Architecture perfectly matched and weights loaded.\n")

    # 3. Dynamic UNI Feature Lookup
    target_filename = os.path.basename(TARGET_IMAGE_PATH)
    print(f"Searching cache for UNI features of: {target_filename}")
    cache = torch.load(UNI_CACHE_PATH, map_location="cpu", weights_only=False)
    
    try:
        # Search using ONLY the filename, ignoring the base directories
        patch_index = next(i for i, saved_path in enumerate(cache["paths"]) 
                           if os.path.basename(saved_path) == target_filename)
        
        uni_feat = cache["features"][patch_index].unsqueeze(0).to(DEVICE)  # [1, 1024]
        print(f"Feature found at cache index: {patch_index}")
        
    except StopIteration:
        raise ValueError(f"Image filename '{target_filename}' not found in the UNI cache. "
                         f"Ensure this image was part of the dataset when Phase 1 was run.")

    # 4. Prepare raw image for CONCH
    # <--- FIXED: Using exact CLIP normalizations required by CONCH
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = Image.open(TARGET_IMAGE_PATH).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 5. Execute Inference
    print("-" * 40)
    print("Running Inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(): # Use AMP if available for speed
            logits = model(uni_feat, image_tensor)
            probs  = torch.softmax(logits, dim=-1)
            pred   = logits.argmax(dim=-1).item()

    # 6. Output Results
    print(f"\nFinal Prediction : {CLASS_NAMES[pred]}")
    print("Confidence Scores:")
    for i, (name, p) in enumerate(zip(CLASS_NAMES.values(), probs[0])):
        marker = "⭐" if i == pred else "  "
        print(f"{marker} {name:>16}: {p.item()*100:>5.1f}%")

if __name__ == "__main__":
    main()