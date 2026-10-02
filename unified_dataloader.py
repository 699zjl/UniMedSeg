import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from datasets.dataloader import CLIP_NORMALIZE, to_long_tensor, correct_dims
from utils.main_utils import read_text


MODALITY_MAP = {
    "BUSI": "ultrasound",
    "BUSBRA": "ultrasound",
    "BUSUC": "ultrasound",
    "BUID": "ultrasound",
    "EUS": "ultrasound",
    "UDIAT": "ultrasound",
    "BRISC": "mri",
    "BTMRI": "mri",
    "Covid19": "ct",
    "CVC300": "endoscopy",
    "ClinicDB": "endoscopy",
    "ColonDB": "endoscopy",
    "Kvasir": "endoscopy",
    "ISIC": "dermoscopy",
    "UWaterlooSkinCancer": "dermoscopy",
    "BKAI": "endoscopy",
}

MODALITY_TO_ID = {
    "ultrasound": 0,
    "mri": 1,
    "ct": 2,
    "endoscopy": 3,
    "dermoscopy": 4,
}


class UnifiedRandomGenerator:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        if random.random() > 0.5:
            angle = random.randint(-20, 20)
            image = image.rotate(angle)
            mask = mask.rotate(angle)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if image.size != tuple(self.output_size):
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != tuple(self.output_size):
            mask = mask.resize(self.output_size, resample=Image.NEAREST)

        image = TF.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        sample['image'] = image
        sample['ground_truth_mask'] = mask
        return sample


class UnifiedValGenerator:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        if image.size != tuple(self.output_size):
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != tuple(self.output_size):
            mask = mask.resize(self.output_size, resample=Image.NEAREST)

        image = TF.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        sample['image'] = image
        sample['ground_truth_mask'] = mask
        return sample


class UnifiedSegDataset(Dataset):
    """Single-dataset loader that also returns modality and dataset metadata."""

    def __init__(self, dataset_path, dataset_name, row_text, transform=None, image_size=224):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'label')
        self.dataset_name = dataset_name
        self.modality = MODALITY_MAP.get(dataset_name, "unknown")
        self.modality_id = MODALITY_TO_ID.get(self.modality, 0)
        self.transform = transform

        self.data_pairs = [
            (row['Image'], row['Ground Truth'], row['Description'])
            for row in row_text
        ]
        self.data_pairs = sorted(self.data_pairs, key=lambda x: x[0])

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_filename, mask_filename, text = self.data_pairs[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        if mask is None:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask[mask < 127] = 0
        mask[mask >= 127] = 1

        image, mask = correct_dims(image, mask)

        modality_prefix = f"[{self.modality}] "
        text_with_modality = modality_prefix + text

        sample = {
            "image": image,
            "ground_truth_mask": mask,
            "image_name": image_filename,
            "mask_name": mask_filename,
            "text_prompt": text_with_modality,
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "modality_id": self.modality_id,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def build_unified_datasets(data_root, dataset_names, split="Train", image_size=224, transform=None, data_percentage=100):
    """Build a ConcatDataset from multiple datasets."""
    datasets = []
    for name in dataset_names:
        if split == "Train":
            path = os.path.join(data_root, name, "Train_Folder")
            if data_percentage != 100:
                text_file = os.path.join(data_root, name, "Prompts_Folder", f"Train_text_{data_percentage}.xlsx")
            else:
                text_file = os.path.join(data_root, name, "Prompts_Folder", "Train_text.xlsx")
        elif split == "Val":
            path = os.path.join(data_root, name, "Val_Folder")
            if data_percentage != 100:
                text_file = os.path.join(data_root, name, "Prompts_Folder", f"Val_text_{data_percentage}.xlsx")
                if not os.path.isfile(text_file):
                    text_file = os.path.join(data_root, name, "Prompts_Folder", "Val_text.xlsx")
            else:
                text_file = os.path.join(data_root, name, "Prompts_Folder", "Val_text.xlsx")
        else:
            path = os.path.join(data_root, name, "Test_Folder")
            text_file = os.path.join(data_root, name, "Prompts_Folder", "Test_text_original.xlsx")

        if not os.path.isdir(path) or not os.path.isfile(text_file):
            print(f"Skipping {name}/{split}: path or text file not found")
            continue

        row_text = read_text(text_file)
        ds = UnifiedSegDataset(path, name, row_text, transform=transform, image_size=image_size)
        if len(ds) > 0:
            datasets.append(ds)
            print(f"  {name}/{split}: {len(ds)} samples ({MODALITY_MAP.get(name, 'unknown')})")

    if not datasets:
        raise ValueError(f"No datasets found for split={split}")

    return ConcatDataset(datasets), datasets


def build_balanced_sampler(datasets):
    """Create a sampler that balances across datasets (not modalities) to prevent
    large datasets from dominating training."""
    total_samples = sum(len(d) for d in datasets)
    weights = []
    for ds in datasets:
        w = total_samples / (len(datasets) * len(ds))
        weights.extend([w] * len(ds))
    return WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)
