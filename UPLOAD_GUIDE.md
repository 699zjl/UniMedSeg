# UniMedSeg - GitHub Upload Guide

## Project Overview

**UniMedSeg** is a unified multi-modal medical image segmentation framework that extends MedCLIPSeg with novel architectural innovations.

**Paper Title**: UniMedSeg: Unified Multi-Modal Medical Image Segmentation via Uncertainty-Guided Encoding and Frequency-Spatial Dual-Prompt Decoding

**Authors**:
- Jingling Zhang - City University of Macau, Putian University
- Shuting Zheng - Putian University  
- Xiangfei Liu - City University of Macau, Shenzhen Institute of Advanced Technology, CAS
- Wen Zhang - Florida Atlantic University
- Jia Gu (Corresponding Author) - City University of Macau

## Project Location

The code ready for upload is located at: `/root/shared-nvme/code/OneModel_Upload/`

## Project Structure

```
UniMedSeg/
├── README.md                  # Main project documentation
├── QUICKSTART.md              # Quick start guide
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore file
├── setup_github.sh            # GitHub repository setup script
├── requirements.txt           # Python dependencies
│
├── config_unified.yaml        # Model configuration
├── model_unified.py           # UniMedSeg main architecture
├── train.py                   # Training script for 16 datasets
├── train_unified.py           # Shared training utilities
├── unified_dataloader.py      # Multi-modal dataloader
│
├── assets/
│   └── overview.png           # Main figure (from Fig1.png)
│
├── trainers/                  # PVL adapters from MedCLIPSeg
├── open_clip_lib/             # CLIP model library
├── datasets/                  # Dataset loading utilities
└── utils/                     # Helper functions
```

## Four Key Innovations

1. **Modality-Aware Encoding**
   - Learnable modality embeddings for ultrasound, MRI, CT, endoscopy, dermoscopy

2. **Cross-Scale Uncertainty Propagation**
   - Multi-layer uncertainty extractor and memory bank
   - Aggregates uncertainty across Transformer layers

3. **Frequency-Spatial Decomposition Decoder**
   - Low-frequency branch for global structure
   - High-frequency branch for fine details
   - Dual-prompt cross-attention mechanism

4. **Boundary-Aware Contrastive Learning**
   - Specialized contrastive loss for boundary regions
   - Improves boundary delineation accuracy

## Datasets

Uses the same 16 datasets as MedCLIPSeg:

**Ultrasound**: BUSI, BUSBRA, BUSUC, BUID, UDIAT, EUS
**MRI**: BRISC, BTMRI
**CT**: Covid19
**Endoscopy**: CVC300, ClinicDB, ColonDB, Kvasir, BKAI
**Dermoscopy**: ISIC, UWaterlooSkinCancer

## Upload to GitHub Steps

### 1. Navigate to Project Directory

```bash
cd /root/shared-nvme/code/OneModel_Upload
```

### 2. Run Setup Script (Optional)

```bash
./setup_github.sh
```

This script will automatically:
- Initialize Git repository
- Add all files
- Create initial commit

### 3. Manual Setup (If Not Using Script)

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: UniMedSeg

- Unified multi-modal medical image segmentation
- Built upon MedCLIPSeg with four key innovations
- 16 datasets, 5 imaging modalities
- Authors: Jingling Zhang et al."

# Add remote repository (after creating on GitHub)
git remote add origin https://github.com/YOUR_USERNAME/UniMedSeg.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 4. Create New Repository on GitHub

1. Visit https://github.com/new
2. Repository name: `UniMedSeg`
3. Description: `Unified Multi-Modal Medical Image Segmentation via Uncertainty-Guided Encoding and Frequency-Spatial Dual-Prompt Decoding`
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have them)
6. Click "Create repository"

### 5. Push Code

Follow the commands shown on GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/UniMedSeg.git
git branch -M main
git push -u origin main
```

### 6. Update Links in README

After pushing, edit README.md to replace `YOUR_USERNAME` with your actual GitHub username.

### 7. Add Repository Topics (Optional but Recommended)

On GitHub repository page, click the gear icon and add these topics:

- `medical-imaging`
- `deep-learning`
- `semantic-segmentation`
- `vision-language-model`
- `clip`
- `multi-modal`
- `pytorch`
- `medical-ai`
- `uncertainty-quantification`

## Project Size

Total size: ~4.6 MB (excluding model weights and datasets)

## Important Notes

1. **Model Weights**: Pretrained weights auto-download from HuggingFace on first run, not included in repository
2. **Datasets**: Datasets need to be downloaded separately following MedCLIPSeg's guide
3. **Dependencies**: All Python dependencies listed in `requirements.txt`
4. **License**: MIT License, compatible with MedCLIPSeg

## Next Steps

After uploading:

1. **Add Model Checkpoints**: Upload trained models to HuggingFace
2. **Update Results**: Fill in experimental results (DSC scores) in README
3. **Prepare Paper**: Add repository link to paper
4. **Create Demo**: Consider creating a Gradio/Streamlit demo
5. **Documentation**: Add more usage examples and visualization results

## File Summary

- **Total files**: 124
- **Python files**: 34
- **Main model**: model_unified.py (27 KB)
- **Training script**: train.py (7.7 KB)
- **Documentation**: README.md, QUICKSTART.md, UPLOAD_GUIDE.md

## Contact

For questions:
- Corresponding Author: Jia Gu (jiagu@cityu.edu.mo)
- First Author: Jingling Zhang (D240921002700@cityu.edu.mo)

---

Prepared: 2026-10-02
Based on: MedCLIPSeg (CVPR 2026)
Source path: Innovation_OneModel/OneModel_Unified_16DS (Ours)
