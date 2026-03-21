# 🔬 CONCH for Colorectal Cancer Classification on Chaoyang

> Investigating whether a single pathology foundation model with partial fine-tuning can match or exceed multi-model CNN ensemble approaches on a noisy, class-imbalanced histopathology benchmark — without explicit noise-handling mechanisms.

---

## 📋 Overview

This repository contains the full experimental pipeline for classifying colorectal cancer subtypes on the **Chaoyang dataset** using **CONCH** (Contrastive learning from Captions for Histopathology), a ViT-B/16 foundation model pretrained on 1.17M pathology image-text pairs.

We systematically evaluate five configurations across two axes:

- **Noise handling** — standard CE vs. noise-robust loss functions (SCE, GCE)
- **Feature adaptation** — frozen encoder vs. partial fine-tuning

---

## 📊 Results at a Glance

### Our Experiments

| # | Configuration | Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | CONCH frozen + Cross-Entropy | 82.14% | 76.75% | 77.46% | 77.05% |
| 2 | CONCH frozen + Symmetric CE | 80.79% | 75.23% | 75.06% | 75.12% |
| 3 | CONCH frozen + Generalized CE (q=0.4) | 81.30% | 75.60% | 76.20% | 75.87% |
| 4 | CONCH partial fine-tune (blocks 6–11) | 85.46% | 80.95% | 80.27% | 80.55% |
| **5** | **CONCH partial fine-tune + H&E augmentation** | **86.02%** | **81.43%** | **80.39%** | **80.87%** |

### Comparison Against Published Baselines

| Method | Acc | F1 |
|---|---|---|
| HSA-NRL — ResNet-34 + noise-robust pipeline *(Zhu et al., TMI 2022)* | 83.40% | 76.54% |
| Ke et al. — 3-CNN ensemble + MHSA fusion *(Scientific Reports 2025)* | 86.72% | 85.76% |
| Belaskri et al. — ViT-B/16 + RCC domain pretraining *(IC2SDA 2025)* | 87.14% | 82.77% |
| **Ours — CONCH partial fine-tune + augmentation (Exp 5)** | **86.02%** | **80.87%** |

### Per-Class F1 — Best Model (Experiment 5)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 85.41% | 85.53% | 85.47% | 705 |
| Serrated | 63.82% | 60.44% | 62.08% | 321 |
| Adenocarcinoma | 95.87% | 99.40% | 97.60% | 840 |
| Adenoma | 80.62% | 76.19% | 78.34% | 273 |

---

## 💡 Key Findings

**1. Foundation model features are inherently noise-resistant.**
CONCH frozen (Exp 1) already beats HSA-NRL on F1 (77.05% vs 76.54%) with no explicit noise handling, suggesting pathology-specific pretraining at scale partially substitutes for noise correction.

**2. Noise-robust losses do not help with CONCH features.**
SCE and GCE both degraded performance relative to standard CE (Exp 2 & 3). CONCH's rich feature space appears to reduce noise sensitivity at the representation level, making additional loss-level correction redundant.

**3. Partial fine-tuning is the single most impactful intervention.**
Unfreezing the top 6 of 12 transformer blocks with differential learning rates (encoder: 1e-5, head: 1e-3) produced the largest gain: +3.32% acc and +3.50% F1 over the frozen baseline.

**4. Serrated classification is the persistent bottleneck.**
At 62.08% F1 in Exp 5, Serrated remains the weakest class due to morphological overlap with Normal tissue and the highest concentration of noisy training labels.

---

## 🗂️ Repository Structure

```
chaoyang-conch-research/
├── scripts/
│   ├── extract_features.py          # CONCH feature extraction — run once, cache to disk
│   ├── train_classifier_exp1.py     # Frozen CONCH + Cross-Entropy
│   ├── train_classifier_exp2.py     # Frozen CONCH + Symmetric CE
│   ├── train_classifier_exp3.py     # Frozen CONCH + Generalized CE
│   ├── train_classifier_exp4.py     # Partial fine-tune (Ke et al. strategy)
│   └── train_classifier_exp5.py     # Partial fine-tune + full infrastructure
├── results/
│   ├── experiment1_results.txt
│   ├── experiment2_results.txt
│   ├── experiment3_results.txt
│   ├── experiment4_results.txt
│   ├── experiment5_results.txt
│   └── experiment4_report.md        # Detailed Exp 4 comparative analysis
├── notebooks/                       # Exploratory analysis
├── requirements.txt                 # Direct dependencies
├── requirements-frozen.txt          # Exact pip freeze output
└── README.md
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10 (via pyenv recommended)
- CUDA-capable GPU — tested on 7 GB VRAM with mixed precision (AMP)
- CONCH access approved at [HuggingFace → MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)

### 1. Clone and install CONCH

```bash
git clone https://github.com/mahmoodlab/CONCH.git
cd CONCH
git checkout 141cc09c7d4ff33d8eda562bd75169b457f71a62   # pinned version
```

### 2. Create environment

```bash
pyenv local 3.10.13
python -m venv conch_env
source conch_env/bin/activate
pip install --upgrade pip
pip install -e .
```

### 3. Install dependencies

Install PyTorch first, matching your CUDA version:

```bash
# CUDA 13.0
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

Then install remaining dependencies:

```bash
pip install -r requirements.txt
# or for exact reproducibility:
pip install -r requirements-frozen.txt
```

### 4. Download CONCH weights

Download `pytorch_model.bin` from [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) after access is approved. Then update `WEIGHTS_PATH` at the top of each script.

### 5. Prepare the Chaoyang dataset

Download from the [HSA-NRL repository](https://github.com/bupt-ai-cz/HSA-NRL) and organise as:

```
Chaoyang/
├── organized_images_train/
│   ├── label_0/    # Normal        — 1,111 patches
│   ├── label_1/    # Serrated      —   842 patches
│   ├── label_2/    # Adenocarcinoma— 1,404 patches
│   └── label_3/    # Adenoma       —   664 patches
└── organized_images_test/
    ├── label_0/    # Normal        —   705 patches
    ├── label_1/    # Serrated      —   321 patches
    ├── label_2/    # Adenocarcinoma—   840 patches
    └── label_3/    # Adenoma       —   273 patches
```

Update `TRAIN_DIR` and `TEST_DIR` at the top of each script.

---

## 🚀 Running Experiments

### Step 1 — Extract features (required for Experiments 1–3 only)

```bash
python scripts/extract_features.py
```

Caches 512-dim CONCH embeddings for all patches to disk. Run once — Experiments 4 and 5 use raw images directly.

### Step 2 — Run experiments

```bash
# Frozen encoder experiments (seconds per run — uses cached features)
python scripts/train_classifier_exp1.py    # Cross-Entropy baseline
python scripts/train_classifier_exp2.py    # Symmetric CE
python scripts/train_classifier_exp3.py    # Generalized CE

# Fine-tuning experiments (~10–30 min per run — uses raw images)
python scripts/train_classifier_exp4.py    # Partial fine-tune
python scripts/train_classifier_exp5.py    # Full infrastructure — best result
```

Results and model checkpoints are saved automatically to `./results/`.

---

## 📦 Dataset

**Chaoyang** — Colorectal tissue histopathology patches  
Collected at Chaoyang Hospital, Beijing, China.  
Published by Zhu et al. (IEEE Transactions on Medical Imaging, 2022).

| Property | Value |
|---|---|
| Total patches | 6,160 |
| Patch resolution | 512 × 512 pixels |
| Magnification | 20× |
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
  volume  = {41},
  number  = {4},
  pages   = {881--894},
  year    = {2022}
}

@article{lu2024conch,
  author  = {Lu, Ming Y. and others},
  title   = {A visual-language foundation model for computational pathology},
  journal = {Nature Medicine},
  year    = {2024}
}

@article{ke2025domain,
  author  = {Ke, Qi and others},
  title   = {Histopathological classification of colorectal cancer based on
             domain-specific transfer learning and multi-model feature fusion},
  journal = {Scientific Reports},
  year    = {2025}
}

@inproceedings{belaskri2025vit,
  author    = {Belaskri, Moncef and others},
  title     = {Improving ViT Performance for Colon Cancer Histopathological
               Image Classification Using Transfer Learning},
  booktitle = {IC2SDA},
  year      = {2025}
}
```

---

## ⚖️ License

This project is for **non-commercial academic research only**.  
CONCH model weights are subject to the [MahmoodLab non-commercial academic license](https://huggingface.co/MahmoodLab/CONCH).  
The Chaoyang dataset is subject to the terms of the [HSA-NRL repository](https://github.com/bupt-ai-cz/HSA-NRL).
