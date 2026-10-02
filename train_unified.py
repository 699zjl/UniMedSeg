import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
import argparse
import random
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
import logging

from utils.main_utils import load_cfg_from_cfg_file
from Innovation_OneModel.unified_dataloader import (
    UnifiedRandomGenerator, UnifiedValGenerator,
    build_unified_datasets, build_balanced_sampler,
)
from Innovation_OneModel.model_unified import (
    build_unified_model, BoundaryContrastiveLoss,
    get_boundary_patch_masks, UncertaintyCalibrationLoss,
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="Innovation_OneModel/Output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-percentage", type=int, default=100,
                        help="Percentage of training data to use (e.g., 50 for 50%)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def calc_loss(seg_logits, masks, ce_loss, dice_loss, cfg):
    loss_ce = ce_loss(seg_logits, masks.float())
    loss_dice = dice_loss(seg_logits.unsqueeze(1), masks.unsqueeze(1))
    return cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce


def evaluate(model, val_loader, device, ce_loss, dice_loss, cfg):
    model.eval()
    val_losses = []
    dice_scores = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device).long()
            text = batch["text_prompt"]
            modality_ids = batch["modality_id"].to(device)

            seg_logits, _ = model(images, text=text, modality_ids=modality_ids)
            loss = calc_loss(seg_logits, masks, ce_loss, dice_loss, cfg)
            val_losses.append(loss.item())

            preds = (torch.sigmoid(seg_logits) > 0.5).float()
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)
            if masks.ndim == 3:
                masks_4d = masks.unsqueeze(1).float()
            else:
                masks_4d = masks.float()

            intersection = (preds * masks_4d).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks_4d.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())

    model.train()
    return mean(val_losses), mean(dice_scores)


def evaluate_per_dataset(model, val_loader, device):
    """Compute per-dataset Dice scores."""
    model.eval()
    dataset_dice = {}

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device).long()
            text = batch["text_prompt"]
            modality_ids = batch["modality_id"].to(device)
            ds_names = batch["dataset_name"]

            seg_logits, _ = model(images, text=text, modality_ids=modality_ids)
            preds = (torch.sigmoid(seg_logits) > 0.5).float()

            for i, ds_name in enumerate(ds_names):
                pred_i = preds[i]
                mask_i = masks[i].float()
                intersection = (pred_i * mask_i).sum()
                union = pred_i.sum() + mask_i.sum()
                dice_i = (2.0 * intersection + 1e-7) / (union + 1e-7)

                if ds_name not in dataset_dice:
                    dataset_dice[ds_name] = []
                dataset_dice[ds_name].append(dice_i.item())

    model.train()
    return {k: mean(v) for k, v in dataset_dice.items()}


def main():
    cfg = get_arguments()

    if cfg.seed >= 0:
        set_random_seed(cfg.seed)

    save_root = os.path.join(cfg.output_dir, f"seed{cfg.seed}")
    os.makedirs(save_root, exist_ok=True)
    logger = logger_config(os.path.join(save_root, "log.txt"))

    logger.info("=" * 60)
    logger.info("Unified Multi-Modal Medical Segmentation Training")
    logger.info("=" * 60)
    logger.info(cfg)

    # Data
    data_root = cfg.DATASET.DATA_ROOT
    dataset_names = cfg.DATASET.TRAIN_DATASETS
    image_size = cfg.DATASET.SIZE

    logger.info(f"\nBuilding training datasets from {len(dataset_names)} sources...")
    train_tf = UnifiedRandomGenerator(output_size=[image_size, image_size])
    val_tf = UnifiedValGenerator(output_size=[image_size, image_size])

    data_percentage = cfg.data_percentage
    train_concat, train_datasets = build_unified_datasets(
        data_root, dataset_names, split="Train", image_size=image_size, transform=train_tf,
        data_percentage=data_percentage
    )
    val_concat, val_datasets = build_unified_datasets(
        data_root, dataset_names, split="Val", image_size=image_size, transform=val_tf
    )

    logger.info(f"Total training samples: {len(train_concat)}")
    logger.info(f"Total validation samples: {len(val_concat)}")

    sampler = build_balanced_sampler(train_datasets)

    def worker_init_fn(worker_id):
        seed = cfg.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)

    train_loader = DataLoader(
        train_concat, batch_size=cfg.TRAIN.BATCH_SIZE,
        sampler=sampler, num_workers=8, pin_memory=True,
        worker_init_fn=worker_init_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_concat, batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False, num_workers=8, pin_memory=True,
    )

    # Model
    model = build_unified_model(cfg)
    model.train().to(cfg.MODEL.DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.TRAIN.NUM_EPOCHS, eta_min=1e-5
    )

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
    boundary_warmup = getattr(cfg.TRAIN, "BOUNDARY_WARMUP_EPOCHS", 20)
    boundary_weight = getattr(cfg.TRAIN, "BOUNDARY_WEIGHT", 0.15)
    cal_warmup = getattr(cfg.TRAIN, "CALIBRATION_WARMUP_EPOCHS", 30)
    cal_weight = getattr(cfg.TRAIN, "CALIBRATION_WEIGHT", 0.05)
    grad_clip = getattr(cfg.TRAIN, "GRAD_CLIP", 1.0)

    nan_count = 0
    nan_recoveries = 0
    max_nan_batches = 50  # recover from best if exceeded

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
            if clip_loss is not None and not torch.isnan(clip_loss) and not torch.isinf(clip_loss):
                loss = loss + cfg.TRAIN.CLIP_WEIGHT * clip_loss.clamp(max=10.0)

            # Boundary contrastive loss
            if boundary_embeds is not None and bw_factor > 0:
                boundary_fg, boundary_bg, interior_fg = get_boundary_patch_masks(
                    masks.float(), grid_size=image_size // 16
                )
                b_loss = boundary_criterion(
                    boundary_embeds, boundary_text_embed,
                    boundary_fg, boundary_bg, interior_fg
                )
                if not torch.isnan(b_loss) and not torch.isinf(b_loss):
                    loss = loss + boundary_weight * bw_factor * b_loss.clamp(max=5.0)

            # Uncertainty calibration loss
            if fused_unc is not None and cal_factor > 0:
                cal_loss_val = calibration_criterion(fused_unc, seg_logits, masks)
                if not torch.isnan(cal_loss_val) and not torch.isinf(cal_loss_val):
                    loss = loss + cal_weight * cal_factor * cal_loss_val.clamp(max=5.0)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                nan_count += 1
                if nan_count >= max_nan_batches:
                    best_path = os.path.join(save_root, "best.pth")
                    if os.path.exists(best_path):
                        logger.info(f"  NaN detected {nan_count} times, recovering from best checkpoint + resetting optimizer...")
                        ckpt = torch.load(best_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
                        model.load_state_dict(ckpt["model"])
                        optimizer.state.clear()
                        for group in optimizer.param_groups:
                            group['lr'] = group['lr'] * 0.5
                        nan_count = 0
                        nan_recoveries += 1
                        if nan_recoveries >= 3:
                            logger.info("  Too many NaN recoveries (3), stopping training early.")
                            break
                    else:
                        logger.info(f"  NaN detected {nan_count} times but no best checkpoint.")
                        break
                continue

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            nan_count = 0
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if nan_recoveries >= 3:
            logger.info("Stopping training due to repeated NaN.")
            break

        scheduler.step()

        # Validation
        val_loss, val_dice = evaluate(model, val_loader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cfg)
        logger.info(
            f"Epoch {epoch+1}/{cfg.TRAIN.NUM_EPOCHS} | "
            f"Train Loss: {mean(train_losses):.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

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
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
            }, os.path.join(save_root, "best.pth"))

        # Save latest
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
        }, os.path.join(save_root, "latest.pth"))

    logger.info(f"\nTraining complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
