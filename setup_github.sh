#!/bin/bash

# UniMedSeg GitHub Repository Setup Script

echo "====================================="
echo "UniMedSeg Repository Setup"
echo "====================================="
echo ""

# Check if we're in the right directory
if [ ! -f "model_unified.py" ]; then
    echo "Error: Please run this script from the UniMedSeg root directory"
    exit 1
fi

echo "Step 1: Initializing Git repository..."
git init

echo ""
echo "Step 2: Adding files to Git..."
git add .gitignore
git add LICENSE
git add README.md
git add QUICKSTART.md
git add requirements.txt
git add config_unified.yaml
git add *.py
git add assets/
git add trainers/
git add datasets/
git add open_clip_lib/
git add utils/

echo ""
echo "Step 3: Creating initial commit..."
git commit -m "Initial commit: UniMedSeg - Unified Multi-Modal Medical Image Segmentation

- Added main model architecture (model_unified.py)
- Added training scripts (train.py, train_unified.py)
- Added unified dataloader for 16 datasets
- Added configuration file (config_unified.yaml)
- Added comprehensive README and quick start guide
- Added necessary dependencies and utilities

Built upon MedCLIPSeg with four key innovations:
1. Modality-aware encoding via learnable embeddings
2. Cross-scale uncertainty propagation with memory bank
3. Frequency-spatial decomposition decoder
4. Boundary-aware contrastive learning

Authors: Jingling Zhang, Shuting Zheng, Xiangfei Liu, Wen Zhang, Jia Gu"

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: UniMedSeg"
echo "   - Description: Unified Multi-Modal Medical Image Segmentation"
echo "   - Make it Public or Private (your choice)"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. Add remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/UniMedSeg.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Update the README.md URLs:"
echo "   - Replace YOUR_USERNAME with your GitHub username"
echo ""
echo "4. Add topics/tags to your repository (recommended):"
echo "   medical-imaging, deep-learning, segmentation, vision-language-model"
echo "   clip, multi-modal, pytorch, medical-ai"
echo ""
echo "====================================="
