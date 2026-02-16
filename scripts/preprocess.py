# preprocess images
import os

import torch
import torchvision.transforms.functional as F
from torchvision.io import decode_image
from tqdm import tqdm

SRC_ROOT = "./data/celeba_hq"
DST_ROOT = "./data/celeba_hq_preprocessed"
SIZE = (128, 128)

splits = ["train", "test"]

if os.path.exists(DST_ROOT):
    print(
        f"The destination directory '{DST_ROOT}' already exists. Please remove it before running this script."
    )
    exit(1)


for split in splits:
    src_dir = os.path.join(SRC_ROOT, split)
    dst_dir = os.path.join(DST_ROOT, split)
    os.makedirs(dst_dir, exist_ok=True)

    for class_name in os.listdir(src_dir):
        src_class = os.path.join(src_dir, class_name)
        dst_class = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class, exist_ok=True)

        for fname in tqdm(os.listdir(src_class), desc=f"{split}/{class_name}"):
            src_path = os.path.join(src_class, fname)
            dst_path = os.path.join(dst_class, fname.replace(".png", ".pt"))

            img = decode_image(src_path)  # uint8 C×H×W
            img = F.resize(img, SIZE, antialias=True)  # still uint8
            torch.save(img, dst_path)
