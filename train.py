"""Training script for OneModel Unified 16DS (pure OneModel without RW/Anatomy/Freq)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.dirname(__file__))

from model_unified import (
    build_unified_model, BoundaryContrastiveLoss,
    get_boundary_patch_masks, UncertaintyCalibrationLoss,
)
from Innovation_OneModel.train_unified import (
    set_random_seed, get_arguments, logger_config, calc_loss,
    evaluate, evaluate_per_dataset,
)
from Innovation_OneModel.unified_dataloader import (
    UnifiedRandomGenerator, UnifiedValGenerator,
    build_unified_datasets, build_balanced_sampler,
)

import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
import random
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss


def main():
    cfg = get_arguments()
    if cfg.seed >= 0:
        set_random_seed(cfg.seed)

    save_root = os.path.join(cfg.output_dir, f"seed{cfg.seed}")
    os.makedirs(save_root, exist_ok=True)
    logger = logger_config(os.path.join(save_root, "log.txt"))

    logger.info("=" * 60)
    logger.info("OneModel Unified 16DS Training")
    logger.info("=" * 60)
    logger.info(cfg)

    data_root = cfg.DATASET.DATA_ROOT
    dataset_names = cfg.DATASET.TRAIN_DATASETS
    image_size = cfg.DATASET.SIZE

    train_tf = UnifiedRandomGenerator(output_size=[image_size, image_size])
    val_tf = UnifiedValGenerator(output_size=[image_size, image_size])
    data_percentage = getattr(cfg, 'data_percentage', 100)
    train_concat, train_datasets = build_unified_datasets(
        data_root, dataset_names, split="Train", image_size=image_size, transform=train_tf,
        data_percentage=data_percentage
    )
    val_concat, val_datasets = build_unified_datasets(
        data_root, dataset_names, split="Val", image_size=image_size, transform=val_tf
    )

    logger.info(f"Data percentage: {data_percentage}%")
    logger.info(f"Total training samples: {len(train_concat)}")
    logger.info(f"Total validation samples: {len(val_concat)}")

    sampler = build_balanced_sampler(train_datasets)

    def worker_init_fn(worker_id):
        random.seed(cfg.seed + worker_id)
        np.random.seed(cfg.seed + worker_id)

    train_loader = DataLoader(train_concat, batch_size=cfg.TRAIN.BATCH_SIZE,
        sampler=sampler, num_workers=8, pin_memory=True,
        worker_init_fn=worker_init_fn, drop_last=True)
    val_loader = DataLoader(val_concat, batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = build_unified_model(cfg)
    model.train().to(cfg.MODEL.DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.TRAIN.NUM_EPOCHS, eta_min=1e-5)
    # Loss functions
    ce_loss = BCEWithLogitsLoss()
    dice_loss = monai.losses.DiceLoss(include_background=False, sigmoid=True, reduction="mean")
    boundary_criterion = BoundaryContrastiveLoss(temperature=0.07)
    calibration_criterion = UncertaintyCalibrationLoss(sparsity_weight=0.01)

    # Resume
    start_epoch = 0
    best_dice = 0.0
    resume_path = os.path.join(save_root, "latest.pth")

    if cfg.resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")

    # Training loop
    boundary_warmup = getattr(cfg.TRAIN, "BOUNDARY_WARMUP_EPOCHS", 5)
    boundary_weight = getattr(cfg.TRAIN, "BOUNDARY_WEIGHT", 0.3)
    cal_warmup = getattr(cfg.TRAIN, "CALIBRATION_WARMUP_EPOCHS", 10)
    cal_weight = getattr(cfg.TRAIN, "CALIBRATION_WEIGHT", 0.05)
    grad_clip = getattr(cfg.TRAIN, "GRAD_CLIP", 1.0)

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        model.train()
        train_losses = []
        bw_factor = min(1.0, (epoch + 1) / boundary_warmup) if boundary_warmup > 0 else 1.0
        cal_factor = min(1.0, (epoch + 1) / cal_warmup) if cal_warmup > 0 else 1.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.TRAIN.NUM_EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(cfg.MODEL.DEVICE)
            masks = batch["ground_truth_mask"].to(cfg.MODEL.DEVICE).long()
            text = batch["text_prompt"]
            modality_ids = batch["modality_id"].to(cfg.MODEL.DEVICE)

            seg_logits, fused_unc, clip_loss, boundary_embeds, boundary_text_embed = \
                model(images, text=text, modality_ids=modality_ids)
            # Segmentation loss
            loss = calc_loss(seg_logits, masks, ce_loss, dice_loss, cfg)

            # Contrastive loss
            if clip_loss is not None:
                loss = loss + cfg.TRAIN.CLIP_WEIGHT * clip_loss

            # Boundary contrastive loss
            if boundary_embeds is not None and bw_factor > 0:
                boundary_fg, boundary_bg, interior_fg = get_boundary_patch_masks(
                    masks.float(), grid_size=image_size // 16)
                b_loss = boundary_criterion(
                    boundary_embeds, boundary_text_embed,
                    boundary_fg, boundary_bg, interior_fg)
                loss = loss + boundary_weight * bw_factor * b_loss

            # Uncertainty calibration loss
            if fused_unc is not None and cal_factor > 0:
                cal_loss_val = calibration_criterion(fused_unc, seg_logits, masks)
                loss = loss + cal_weight * cal_factor * cal_loss_val

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        val_loss, val_dice = evaluate(model, val_loader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cfg)
        logger.info(
            f"Epoch {epoch+1}/{cfg.TRAIN.NUM_EPOCHS} | "
            f"Train Loss: {mean(train_losses):.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Per-dataset evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            per_ds_dice = evaluate_per_dataset(model, val_loader, cfg.MODEL.DEVICE)
            logger.info("Per-dataset Dice scores:")
            for ds_name, dice_val in sorted(per_ds_dice.items()):
                logger.info(f"  {ds_name}: {dice_val:.4f}")

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            logger.info(f"  New best Dice: {best_dice:.4f}, saving...")
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_dice": best_dice},
                       os.path.join(save_root, "best.pth"))

        # Save latest
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice},
                   os.path.join(save_root, "latest.pth"))

    logger.info(f"\nTraining complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
