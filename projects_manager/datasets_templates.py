# ML LIBRARIES
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
from torchvision.transforms import InterpolationMode
# STANDARD LIBRARIES
from typing import Optional, Tuple
import json
import pandas as pd
import requests
from pathlib import Path
import zipfile
from PIL import Image

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

class DatasetType:
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_TO_IMAGE = "image_to_image"
    # Add other dataset types as needed

def input_output_type_to_dataset_type(
    input_type: str,
    output_type: str
) -> DatasetType:
    if input_type == "image" and output_type == "image":
        return DatasetType.IMAGE_TO_IMAGE
    elif input_type == "image" and output_type == "vector":
        return DatasetType.IMAGE_CLASSIFICATION
    else:
        raise ValueError(f"Unsupported input/output type combination: {input_type}/{output_type}")

def collect_images(root: str):
    images = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(IMG_EXTENSIONS):
                images.append(os.path.join(dirpath, fname))

    return sorted(images)

class ImageToImageDataset(Dataset):
    def __init__(
        self,
        path_to_images: str,
        input_transform=None,
        output_transform=None,
        augmentation_transform=None,
        train: bool = True,
        same_target: bool = True
    ):
        self.image_paths = collect_images(path_to_images)

        if not self.image_paths:
            raise ValueError(f"No images found in {path_to_images}")

        self.input_transform = input_transform
        self.output_transform = output_transform
        self.augmentation_transform = augmentation_transform
        self.train = train
        self.same_target = same_target

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")

        # -------- INPUT / TARGET --------
        x = img
        y = img if self.same_target else img.copy()

        # -------- AUGMENTATIONS --------
        if self.train and self.augmentation_transform:
            seed = torch.randint(0, 10_000, (1,)).item()

            torch.manual_seed(seed)
            x = self.augmentation_transform(x)

            torch.manual_seed(seed)
            y = self.augmentation_transform(y)

        # -------- FINAL TRANSFORMS --------
        if self.input_transform:
            x = self.input_transform(x)

        if self.output_transform:
            y = self.output_transform(y)

        return x, y

class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        input_preprocess=None,
        augmentations=None,
        train: bool = True,
    ):
        self.root_dir = root_dir
        self.input_preprocess = input_preprocess
        self.augmentations = augmentations
        self.train = train

        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._scan()

    def _scan(self):
        class_names = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(class_names)
        }
        self.idx_to_class = {
            idx: cls_name for cls_name, idx in self.class_to_idx.items()
        }

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(self.root_dir, cls_name)

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, cls_idx))

        if not self.samples:
            raise RuntimeError("No images found in dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.input_preprocess:
            img = self.input_preprocess(img)

        if self.train and self.augmentations:
            img = self.augmentations(img)

        return img, label
