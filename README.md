# 🔬 Dual-Encoder Colorectal Cancer Classification on Chaoyang

> A dual-foundation-model architecture combining **CONCH** (vision-language, semantic features) and **UNI** (DINOv2, morphological features) with cross-attention fusion and LoRA fine-tuning for four-class colorectal cancer subtype classification on the noisy, class-imbalanced Chaoyang benchmark.

---

## 📋 Overview

The Chaoyang dataset is one of the most challenging histopathology benchmarks — ~40% of training labels are noisy (three pathologists disagreed; one label was randomly assigned), the classes are imbalanced, and the dataset is small (4,021 training patches). Prior work addressed these challenges through explicit noise-correction pipelines or large CNN ensembles.

This work takes a different approach: use two complementary pathology foundation models whose pretraining scale and diversity already encode noise-resistant representations, fuse them dynamically via cross-attention, and apply LoRA for parameter-efficient adaptation across all transformer blocks. The result is a single-model pipeline that surpasses prior CNN ensemble and ViT-transfer baselines on accuracy while using a fraction of the trainable parameters.

---

## 📊 Results

### Published Baselines

| Method | Acc | F1 |
|---|---|---|
| HSA-NRL — ResNet-34 + noise-robust pipeline *(Zhu et al., TMI 2022)* | 83.40% | 76.54% |
| Ke et al. — 3-CNN ensemble + MHSA fusion *(Scientific Reports 2025)* | 86.72% | 85.76% |
| Belaskri et al. — ViT-B/16 + RCC domain pretraining *(IC2SDA 2025)* | 87.14% | 82.77% |

### Our Experiments — Single CONCH Encoder

| # | Configuration | Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | CONCH frozen + Cross-Entropy | 82.14% | 76.75% | 77.46% | 77.05% |
| 2 | CONCH frozen + Symmetric CE | 80.79% | 75.23% | 75.06% | 75.12% |
| 3 | CONCH frozen + Generalized CE (q=0.4) | 81.30% | 75.60% | 76.20% | 75.87% |
| 4 | CONCH partial fine-tune (blocks 6–11) | 85.46% | 80.95% | 80.27% | 80.55% |
| 5 | CONCH partial fine-tune + H&E augmentation | 86.02% | 81.43% | 80.39% | 80.87% |

### Our Experiments — Dual Encoder (UNI + CONCH)

| Configuration | Acc | Precision | Recall | F1 |
|---|---|---|---|---|
| UNI + CONCH + cross-attention (Phase 2 baseline) | 86.40% | 82.88% | 82.20% | 82.38% |
| UNI + CONCH + MHSA fusion (150 epochs) | 86.77% | 82.65% | 82.76% | 82.66% |
| **UNI + CONCH + LoRA (r=16) — best model** ⭐ | **87.38%** | **84.32%** | **83.01%** | **83.57%** |

### Per-Class F1 — Best Model (UNI + CONCH + LoRA)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 85.57% | 85.82% | 85.69% | 705 |
| Serrated | 64.90% | 68.54% | 66.67% | 321 |
| Adenocarcinoma | 97.18% | 98.57% | 97.87% | 840 |
| Adenoma | 89.63% | 79.12% | 84.05% | 273 |

---

## 🗂️ Repository Structure

```
chaoyang-conch-research/
├── scripts/
│   ├── extract_features.py          # CONCH feature extraction (Experiments 1–3)
│   ├── train_classifier_exp1.py     # Frozen CONCH + Cross-Entropy
│   ├── train_classifier_exp2.py     # Frozen CONCH + Symmetric CE
│   ├── train_classifier_exp3.py     # Frozen CONCH + Generalized CE
│   ├── train_classifier_exp4.py     # CONCH partial fine-tune
│   ├── train_classifier_exp5.py     # CONCH partial fine-tune + H&E augmentation
│   ├── phase1_uni_extraction.py     # UNI feature extraction
│   ├── phase2_dual_encoder.py       # UNI + CONCH + cross-attention baseline
│   ├── phase2_MHSA_150epoch.py      # UNI + CONCH + MHSA fusion
│   └── phase2_self_LoRA.py          # UNI + CONCH + LoRA — best model ⭐
├── results/
│   ├── experiment1_results.txt … experiment5_results.txt
│   ├── phase2_results.txt
│   ├── phase2_results_MHSA_150.txt
│   └── phase2_results_selfLoRA.txt
├── notebooks/
├── requirements.txt
├── requirements-frozen.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10
- CUDA-capable GPU (tested on 7 GB RTX 4060 and 100 GB H100 NVL)
- Access approved for both gated models on HuggingFace (see Step 4)

### 1. Clone and install CONCH

```bash
git clone https://github.com/mahmoodlab/CONCH.git
cd CONCH
git checkout 141cc09c7d4ff33d8eda562bd75169b457f71a62   # pinned version
```

### 2. Create environment

```bash
pyenv local 3.10.13          # or: conda create -n conch_env python=3.10
python -m venv conch_env
source conch_env/bin/activate
pip install --upgrade pip
pip install -e .
pip install peft torchstain   # required for Phase 2 and Phase 1 respectively
```

### 3. Install PyTorch matching your CUDA version

```bash
# CUDA 13.0
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

### 4. Download model weights

**CONCH** — Request access at [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) using your institutional email. Download `pytorch_model.bin` after approval.

**UNI** — Request access separately at [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI). Download `pytorch_model.bin` after approval. Store it separately from the CONCH weights — both files share the same name.

Update `CONCH_WEIGHTS` and `UNI_WEIGHTS_PATH` at the top of each script to point to your local paths.

### 5. Prepare the Chaoyang dataset

Download from the [HSA-NRL repository](https://github.com/bupt-ai-cz/HSA-NRL) and organise as:

```
Chaoyang/
├── organized_images_train/
│   ├── label_0/    # Normal         — 1,111 patches
│   ├── label_1/    # Serrated       —   842 patches
│   ├── label_2/    # Adenocarcinoma — 1,404 patches
│   └── label_3/    # Adenoma        —   664 patches
└── organized_images_test/
    ├── label_0/    # Normal         —   705 patches
    ├── label_1/    # Serrated       —   321 patches
    ├── label_2/    # Adenocarcinoma —   840 patches
    └── label_3/    # Adenoma        —   273 patches
```

Update `TRAIN_DIR` and `TEST_DIR` at the top of each script.

---

## 🚀 Running Experiments

### Single CONCH encoder (Experiments 1–5)

```bash
# Experiments 1–3 use cached CONCH features — run this once first
python scripts/extract_features.py

python scripts/train_classifier_exp1.py    # seconds per run
python scripts/train_classifier_exp2.py
python scripts/train_classifier_exp3.py
python scripts/train_classifier_exp4.py    # ~10–30 min
python scripts/train_classifier_exp5.py
```

### Dual encoder — Phase 1: UNI feature extraction

Run once. Extracts frozen UNI ViT-L/16 embeddings for every patch.

```bash
python scripts/phase1_uni_extraction.py
```

Outputs `uni_train_features.pt` [4021, 1024] and `uni_test_features.pt` [2139, 1024] to `Chaoyang/features/`. Update `UNI_TRAIN_CACHE` and `UNI_TEST_CACHE` in each Phase 2 script to point to these files.

### Dual encoder — Phase 2: training

```bash
python scripts/phase2_dual_encoder.py       # cross-attention baseline
python scripts/phase2_MHSA_150epoch.py      # MHSA fusion variant
python scripts/phase2_self_LoRA.py          # LoRA — best model ⭐
```

---

## 🔍 Inference with the Best Model

All experiment checkpoints are available for download:

> **[📥 Download model checkpoints (Google Drive)](https://drive.google.com/drive/folders/1JMreZz9VLrx8g6OETNXMQww5QANr3t2M?usp=sharing)**

The best model is `phase2_best_acc_selfLoRA.pt` — UNI + CONCH + LoRA, 87.38% accuracy, 83.57% macro F1.

---

## 📦 Dataset

**Chaoyang** — Colorectal tissue histopathology patches. Published by Zhu et al. (IEEE TMI, 2022).

| Property | Value |
|---|---|
| Total patches | 6,160 |
| Resolution | 512 × 512 pixels, 20× magnification |
| Classes | Normal, Serrated, Adenocarcinoma, Adenoma |
| Label noise | ~40% annotation disagreement in train set |
| Test set | Unanimous pathologist agreement only |

---

## 📚 References

```bibtex
@article{zhu2022hsa,
  author  = {Zhu, Chuang and Chen, Wenkai and Peng, Ting and Wang, Ying and Jin, Mulan},
  title   = {Hard Sample Aware Noise Robust Learning for Histopathology Image Classification},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {41}, number = {4}, pages = {881--894}, year = {2022}
}

@article{lu2024conch,
  author  = {Lu, Ming Y. and others},
  title   = {A visual-language foundation model for computational pathology},
  journal = {Nature Medicine}, year = {2024}
}

@article{chen2024uni,
  author  = {Chen, Richard J. and others},
  title   = {Towards a general-purpose foundation model for computational pathology},
  journal = {Nature Medicine}, year = {2024}
}

@article{ke2025domain,
  author  = {Ke, Qi and others},
  title   = {Histopathological classification of colorectal cancer based on
             domain-specific transfer learning and multi-model feature fusion},
  journal = {Scientific Reports}, year = {2025}
}

@inproceedings{belaskri2025vit,
  author    = {Belaskri, Moncef and others},
  title     = {Improving ViT Performance for Colon Cancer Histopathological
               Image Classification Using Transfer Learning},
  booktitle = {IC2SDA}, year = {2025}
}
```

---

## ⚖️ License

Research use only. CONCH and UNI weights are subject to the [MahmoodLab non-commercial academic license](https://huggingface.co/MahmoodLab/CONCH). The Chaoyang dataset is subject to the [HSA-NRL repository terms](https://github.com/bupt-ai-cz/HSA-NRL).