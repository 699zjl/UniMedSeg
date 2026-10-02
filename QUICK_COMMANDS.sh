#!/bin/bash
# Quick commands for GitHub upload - Copy and paste these

echo "======================================"
echo "UniMedSeg GitHub Upload Quick Commands"
echo "======================================"
echo ""
echo "Step 1: Navigate to project directory"
echo "--------------------------------------"
echo "cd /root/shared-nvme/code/OneModel_Upload"
echo ""
echo ""
echo "Step 2: Initialize Git (choose ONE method)"
echo "-------------------------------------------"
echo ""
echo "Method A - Use automated script:"
echo "./setup_github.sh"
echo ""
echo "Method B - Manual setup:"
echo "git init"
echo "git add ."
echo 'git commit -m "Initial commit: UniMedSeg

- Unified multi-modal medical image segmentation framework
- Built upon MedCLIPSeg with four key innovations
- 16 datasets across 5 imaging modalities
- Authors: Jingling Zhang, Shuting Zheng, Xiangfei Liu, Wen Zhang, Jia Gu

Key innovations:
1. Modality-aware encoding with learnable embeddings
2. Cross-scale uncertainty propagation with memory bank
3. Frequency-spatial decomposition decoder
4. Boundary-aware contrastive learning"'
echo ""
echo ""
echo "Step 3: Create repository on GitHub"
echo "------------------------------------"
echo "1. Visit: https://github.com/new"
echo "2. Repository name: UniMedSeg"
echo "3. Description: Unified Multi-Modal Medical Image Segmentation via Uncertainty-Guided Encoding and Frequency-Spatial Dual-Prompt Decoding"
echo "4. Choose: Public (recommended) or Private"
echo "5. DO NOT check: 'Add README', 'Add .gitignore', 'Choose license'"
echo "6. Click: Create repository"
echo ""
echo ""
echo "Step 4: Connect and push to GitHub"
echo "-----------------------------------"
echo "# Replace YOUR_USERNAME with your actual GitHub username"
echo "git remote add origin https://github.com/YOUR_USERNAME/UniMedSeg.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo ""
echo "Step 5: Post-upload tasks"
echo "-------------------------"
echo "1. Edit README.md on GitHub to replace YOUR_USERNAME with actual username"
echo "2. Add repository topics (Settings → Topics):"
echo "   medical-imaging, deep-learning, semantic-segmentation,"
echo "   vision-language-model, clip, multi-modal, pytorch,"
echo "   medical-ai, uncertainty-quantification"
echo ""
echo "3. Optional: Add description to repository"
echo "4. Optional: Set up GitHub Pages for documentation"
echo ""
echo ""
echo "======================================"
echo "Project Information"
echo "======================================"
echo "Location: /root/shared-nvme/code/OneModel_Upload/"
echo "Size: 4.6 MB"
echo "Files: 126 total (34 Python files)"
echo "Main: model_unified.py, train.py"
echo ""
echo "Contact:"
echo "  Jia Gu (Corresponding): jiagu@cityu.edu.mo"
echo "  Jingling Zhang: D240921002700@cityu.edu.mo"
echo ""
echo "======================================"
