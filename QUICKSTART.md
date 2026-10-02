# UniMedSeg Quick Start Guide

## Prerequisites

1. **Hardware Requirements**
   - NVIDIA GPU with at least 16GB VRAM (recommended: A100 or V100)
   - 32GB+ system RAM
   - 50GB+ free disk space

2. **Software Requirements**
   - Linux OS (tested on Ubuntu 20.04+)
   - CUDA 12.0+
   - Python 3.10+

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/699zjl/UniMedSeg.git
cd UniMedSeg
```

### 2. Setup Environment

```bash
# Using conda (recommended)
conda create -n unimedseg python=3.10
conda activate unimedseg

# Install PyTorch with CUDA support
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt
```

### 3. Download Pretrained Weights

The model automatically downloads the UniMedCLIP pretrained weights from HuggingFace on first run. Make sure you have internet access.

```python
# Weights will be downloaded to: ./checkpoints/unimed_clip_vit_b16.pt
```

### 4. Prepare Data

Follow the [MedCLIPSeg data preparation guide](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md) to download and organize the 16 datasets.

Your data directory should look like:
```
data/
├── BUSI/
│   ├── Train/
│   │   ├── images/
│   │   └── masks/
│   └── Val/
│       ├── images/
│       └── masks/
├── BUSBRA/
├── BUSUC/
...
```

## Training

### Basic Training

Train on all 16 datasets with 100% data:

```bash
python train.py \
    --config-file config_unified.yaml \
    --seed 42 \
    --output-dir ./outputs
```

### Data Efficiency Experiments

Train with partial data (50%, 25%, 10%):

```bash
# 50% data
python train.py \
    --config-file config_unified.yaml \
    --seed 42 \
    --output-dir ./outputs_50pct \
    --data-percentage 50

# 25% data
python train.py \
    --config-file config_unified.yaml \
    --seed 42 \
    --output-dir ./outputs_25pct \
    --data-percentage 25

# 10% data
python train.py \
    --config-file config_unified.yaml \
    --seed 42 \
    --output-dir ./outputs_10pct \
    --data-percentage 10
```

### Resume Training

If training is interrupted, resume from the latest checkpoint:

```bash
python train.py \
    --config-file config_unified.yaml \
    --seed 42 \
    --output-dir ./outputs \
    --resume
```

## Training Configuration

Edit [config_unified.yaml](config_unified.yaml) to customize training:

```yaml
DATASET:
  DATA_ROOT: "./data"           # Path to your data directory
  SIZE: 224                     # Input image size
  TRAIN_DATASETS:               # List of datasets to train on
    - BUSI
    - BUSBRA
    # ... add or remove datasets

TRAIN:
  BATCH_SIZE: 16                # Adjust based on GPU memory
  NUM_EPOCHS: 200               # Total training epochs
  LEARNING_RATE: 0.0002         # Base learning rate
  
  # Loss weights
  DICE_WEIGHT: 0.5
  CE_WEIGHT: 0.5
  CLIP_WEIGHT: 0.1
  BOUNDARY_WEIGHT: 0.3          # Boundary contrastive loss
  CALIBRATION_WEIGHT: 0.05      # Uncertainty calibration loss
  
  # Warmup epochs
  BOUNDARY_WARMUP_EPOCHS: 5     # Warmup for boundary loss
  CALIBRATION_WARMUP_EPOCHS: 10 # Warmup for calibration loss
  
  GRAD_CLIP: 1.0                # Gradient clipping

MODEL:
  BACKBONE: "ViT-B/16"
  DEVICE: "cuda"
  NUM_MODALITIES: 5             # ultrasound, MRI, CT, endoscopy, dermoscopy
  
  # Uncertainty propagation
  UNC_EXTRACTOR_HIDDEN: 128
  UNC_MODULATION_INIT: 0.01
  
  # Frequency-spatial decoder
  DECODER_TAP_LAYERS: [2, 5, 8, 11]
  
  # Boundary-aware learning
  BOUNDARY_PROJ_DIM: 128
```

## Monitoring Training

Training logs are saved to `{output_dir}/seed{seed}/log.txt`:

```bash
# Monitor training progress
tail -f outputs/seed42/log.txt
```

Example log output:
```
2026-10-02 16:30:00 | Epoch 1/200 | Train Loss: 0.4523 | Val Loss: 0.3891 | Val Dice: 0.7845 | LR: 0.000200
2026-10-02 16:35:00 | Epoch 2/200 | Train Loss: 0.3912 | Val Loss: 0.3654 | Val Dice: 0.8023 | LR: 0.000198
...
2026-10-02 17:00:00 |   New best Dice: 0.8234, saving...
```

## Output Files

After training, the output directory contains:

```
outputs/seed42/
├── log.txt              # Training logs
├── best.pth             # Best model checkpoint (highest validation Dice)
└── latest.pth           # Latest checkpoint (for resuming)
```

## Per-Dataset Evaluation

The model evaluates on each dataset individually every 10 epochs:

```
Epoch 10/200 | Train Loss: 0.2145 | Val Loss: 0.1823 | Val Dice: 0.8567
Per-dataset Dice scores:
  BUSI: 0.8723
  BUSBRA: 0.8645
  BUSUC: 0.8512
  BUID: 0.8698
  UDIAT: 0.8456
  ...
```

## Common Issues

### Out of Memory (OOM)

Reduce batch size in config:
```yaml
TRAIN:
  BATCH_SIZE: 8  # or 4
```

### Slow Training

- Reduce `num_workers` in train.py (line 70)
- Use mixed precision training (already enabled by default)

### Data Loading Errors

Check that:
- Data directory path is correct in config
- All datasets are properly organized
- Image and mask files exist

## Next Steps

- Check [README.md](README.md) for detailed architecture information
- See model code in [model_unified.py](model_unified.py)
- Customize training in [train.py](train.py)

## Support

For issues or questions:
- Open an issue on GitHub
- Check the [MedCLIPSeg repository](https://github.com/HealthX-Lab/MedCLIPSeg) for dataset and baseline information
