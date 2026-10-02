import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "data/EUS"
OUTPUT_DIR = "data/EUS"

XLSX_DIR = "data/EUS/Prompts_Folder"

SPLITS = {
    "train": os.path.join(XLSX_DIR, "Train_text.xlsx"),
    "val": os.path.join(XLSX_DIR, "Val_text.xlsx"),
    "test": os.path.join(XLSX_DIR, "Test_text_original.xlsx")
}

CANCER_DIR = os.path.join(BASE_DIR, "EUS_cancer")
HEALTHY_DIR = os.path.join(BASE_DIR, "EUS_healthy")

# -----------------------------
# HELPERS
# -----------------------------
def extract_actual_filename(name):
    """
    Removes prefix if exists:
    e.g. disease_C0_V1_12345.png → C0_V1_12345.png
    """
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def parse_path_from_name(img_name):
    """
    C0_V1_12345.tif → (C0, V1_12345.tif)
    """
    parts = img_name.split("_", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def find_image_and_source(img_name):
    """
    Returns:
    - path
    - source ("cancer" or "healthy")
    """
    folder, file_name = parse_path_from_name(img_name)

    if folder is None:
        return None, None

    # Try cancer
    cancer_path = os.path.join(CANCER_DIR, folder, file_name)
    if os.path.exists(cancer_path):
        return cancer_path, "cancer"

    # Try healthy
    healthy_path = os.path.join(HEALTHY_DIR, folder, file_name)
    if os.path.exists(healthy_path):
        return healthy_path, "healthy"

    return None, None


def find_mask(img_name):
    """
    Mask only exists in cancer annotations
    """
    folder, file_name = parse_path_from_name(img_name)

    if folder is None:
        return None

    return os.path.join(CANCER_DIR, "Annotations", folder, file_name)


def create_empty_mask(width, height, save_path):
    empty = np.zeros((height, width), dtype=np.uint8)
    Image.fromarray(empty).save(save_path)


# -----------------------------
# MAIN PROCESS
# -----------------------------
for split, xlsx_path in SPLITS.items():
    print(f"\nProcessing {split}: {xlsx_path}")

    if not os.path.exists(xlsx_path):
        print(f"❌ Missing file: {xlsx_path}")
        continue

    df = pd.read_excel(xlsx_path)

    img_out_dir = os.path.join(OUTPUT_DIR, f"{split.capitalize()}_Folder", "img")
    label_out_dir = os.path.join(OUTPUT_DIR, f"{split.capitalize()}_Folder", "label")

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    seen = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_name = str(row["Image"]).strip()

        # Step 1: remove prefix if exists
        img_name = os.path.splitext(raw_name)[0] + ".tif"

        if img_name in seen:
            continue
        seen.add(img_name)

        # -----------------------------
        # FIND IMAGE
        # -----------------------------
        img_path, source = find_image_and_source(img_name)

        if img_path is None:
            print(f"⚠️ Image not found: {img_name}")
            continue

        # Output filename (.png)
        out_name = os.path.splitext(img_name)[0] + ".png"

        img_out_path = os.path.join(img_out_dir, out_name)
        label_out_path = os.path.join(label_out_dir, out_name)

        # -----------------------------
        # SAVE IMAGE
        # -----------------------------
        img = Image.open(img_path)
        img.save(img_out_path)

        # -----------------------------
        # HANDLE LABEL
        # -----------------------------
        if source == "healthy":
            # Always empty mask
            create_empty_mask(img.width, img.height, label_out_path)

        else:  # cancer
            mask_path = find_mask(img_name)

            if mask_path is None or not os.path.exists(mask_path):
                print(f"⚠️ Missing mask: {img_name}")
                create_empty_mask(img.width, img.height, label_out_path)
            else:
                mask = Image.open(mask_path)
                mask.save(label_out_path)

print("\n✅ Dataset ready!")