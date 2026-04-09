# ML LIBRARIES
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# STANDARD LIBRARIES
from typing import Optional, Tuple
import json
import pandas as pd
import requests
from pathlib import Path
import zipfile
import datasets_templates as ds_templates
from dataclasses import dataclass

DATASETS_ROOT = "./temp_datasets/"

@dataclass
class ImageAugConfig:
    hflip: float = 0.0
    vflip: float = 0.0
    rotate: float = 0.0
    shift_h: float = 0.0
    shift_v: float = 0.0
    scale_h: float = 0.0
    scale_v: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0

def parse_image_aug(cfg: dict) -> ImageAugConfig:
    aug = cfg.get("augmentations", {})

    cj = aug.get("color_jitter", {})
    aff = aug.get("affine", {})

    return ImageAugConfig(
        hflip=aug.get("horizontal_flip_prob", 0),
        vflip=aug.get("vertical_flip_prob", 0),
        rotate=aff.get("rotate_deg", 0),
        shift_h=aff.get("shift_h", 0),
        shift_v=aff.get("shift_v", 0),
        scale_h=aff.get("scale_h", 0),
        scale_v=aff.get("scale_v", 0),
        brightness=cj.get("brightness", 0),
        contrast=cj.get("contrast", 0),
        saturation=cj.get("saturation", 0),
    )

def build_aug_transforms(aug: ImageAugConfig):
    tfms = []

    if aug.hflip > 0:
        tfms.append(transforms.RandomHorizontalFlip(aug.hflip))

    if aug.vflip > 0:
        tfms.append(transforms.RandomVerticalFlip(aug.vflip))

    if aug.rotate > 0 or aug.shift_h or aug.shift_v or aug.scale_h or aug.scale_v:
        tfms.append(
            transforms.RandomAffine(
                degrees=aug.rotate,
                translate=(aug.shift_h, aug.shift_v),
                scale=(1 - min(aug.scale_h, aug.scale_v),
                       1 + max(aug.scale_h, aug.scale_v))
            )
        )

    if aug.brightness or aug.contrast or aug.saturation:
        tfms.append(
            transforms.ColorJitter(
                brightness=aug.brightness,
                contrast=aug.contrast,
                saturation=aug.saturation
            )
        )

    return transforms.Compose(tfms)

def build_torchvision_transforms(preprocess_json_text: str, task: str):
    if not preprocess_json_text:
        return transforms.ToTensor(), transforms.Compose([]), transforms.ToTensor()

    cfg = json.loads(preprocess_json_text)
    
    # ---------- INPUT ----------
    input_cfg = cfg.get("input", {})
    input_tfms = []
    aug_tfms = []
    output_tfms = []

    # ---------- Input basic ----------
    basic_in = input_cfg.get("basic", {})
    if "width" in basic_in and "height" in basic_in:
        input_tfms.append(
            transforms.Resize((basic_in["height"], basic_in["width"]))
        )
    
    if basic_in.get("color_mode") == "grayscale":
        input_tfms.insert(0, transforms.Grayscale(1))

    # ---------- Input augmentations ----------
    aug_cfg = parse_image_aug(input_cfg)
    aug_tfms = build_aug_transforms(aug_cfg)

    input_tfms.append(transforms.ToTensor())

    # ---------- OUTPUT ----------
    output_cfg = cfg.get("output", {})
    basic_out = output_cfg.get("basic", {})

    if "width" in basic_out and "height" in basic_out:
        output_tfms.append(
            transforms.Resize((basic_out["height"], basic_out["width"]))
        )

    # Цвет для выходов, если надо
    if basic_out.get("color_mode") == "grayscale":
        output_tfms.insert(0, transforms.Grayscale(1))

    output_tfms.append(transforms.ToTensor())

    return (
        transforms.Compose(input_tfms),
        aug_tfms,
        transforms.Compose(output_tfms),
    )

def get_dataset(
    dataset_id: int,
    project_id: int,
    preprocess_json_text: str,
    dataset_type: ds_templates.DatasetType,
    cookies: Optional[dict] = None
):
    print("Dataset preprocess JSON:", preprocess_json_text)
    print(f"get_dataset cookies: {cookies}")
    datasets_resp = requests.get(f"http://datasets_manager:8004/datasets/download/id/{dataset_id}", cookies=cookies) # File request
    if datasets_resp.status_code != 200:
        raise ValueError(f"Failed to fetch dataset: {datasets_resp.text}")

    dataset_info = requests.get(f"http://datasets_manager:8004/datasets/{dataset_id}", cookies=cookies).json()
    if datasets_resp.status_code != 200:
        raise ValueError(f"Failed to fetch dataset info: {datasets_resp.text}")

    print("Dataset info:", dataset_info)
    dataset_path = Path(DATASETS_ROOT) / f"project_{project_id}_dataset_{dataset_id}"
    dataset_path.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_path / dataset_info['storage_id']

    with open(zip_path, "wb") as f:
        f.write(datasets_resp.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)

    zip_path.unlink()

    if dataset_type == ds_templates.DatasetType.IMAGE_TO_IMAGE:
        input_transform, augmentation_transform, output_transform = build_torchvision_transforms(
            preprocess_json_text,
            task="image_to_image"
        )

        print("All transforms:")
        print("input_transform:", input_transform)
        print("output_transform:", output_transform)
        print("augmentation_transform:", augmentation_transform)

        dataset = ds_templates.ImageToImageDataset(
            path_to_images=str(dataset_path),
            input_transform=input_transform,
            output_transform=output_transform,
            augmentation_transform=augmentation_transform,
            train=True,
            same_target=True
        )

        return dataset
    
    elif dataset_type == ds_templates.DatasetType.IMAGE_CLASSIFICATION:
        input_transform, augmentation_transform, output_transform = build_torchvision_transforms(
            preprocess_json_text,
            task="classification"
        )
        dataset = ds_templates.ImageClassificationDataset(
            path_to_images=str(dataset_path),
            input_transform=input_transform,
            output_transform=output_transform,
            augmentation_transform=augmentation_transform,
            train=True
        )
        return dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")