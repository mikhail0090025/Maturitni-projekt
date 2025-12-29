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

DATASETS_ROOT = "./temp_datasets/"


def build_torchvision_transforms(
    preprocess_json_text: str,
    task: str = "classification"
):
    """
    Returns:
        input_transform
        augmentation_transform
        output_transform
    """

    # ---------- DEFAULT ----------
    if not preprocess_json_text:
        return (
            transforms.ToTensor(),
            transforms.Identity(),
            transforms.ToTensor()
        )

    cfg = json.loads(preprocess_json_text)

    input_tfms = []
    aug_tfms = []
    output_tfms = []

    # ---------- INPUT BASE ----------
    input_cfg = cfg.get("input", {})
    width = input_cfg.get("width")
    height = input_cfg.get("height")
    image_type = input_cfg.get("image_type", "rgb")

    if width and height:
        resize = transforms.Resize(
            (height, width),
            interpolation=InterpolationMode.BILINEAR
        )
        input_tfms.append(resize)

        # target resize
        target_interp = (
            InterpolationMode.NEAREST
            if task == "segmentation"
            else InterpolationMode.BILINEAR
        )

        output_tfms.append(
            transforms.Resize(
                (height, width),
                interpolation=target_interp
            )
        )

    if image_type == "grayscale":
        input_tfms.insert(0, transforms.Grayscale(num_output_channels=1))

    # ---------- AUGMENTATIONS ----------
    aug_cfg = cfg.get("augmentations", {})

    if aug_cfg:
        if aug_cfg.get("horizontal_flip", 0) > 0:
            aug_tfms.append(
                transforms.RandomHorizontalFlip(
                    p=aug_cfg["horizontal_flip"]
                )
            )

        if aug_cfg.get("vertical_flip", 0) > 0:
            aug_tfms.append(
                transforms.RandomVerticalFlip(
                    p=aug_cfg["vertical_flip"]
                )
            )

        rotate = aug_cfg.get("rotate", 0)
        if rotate > 0:
            aug_tfms.append(
                transforms.RandomRotation(degrees=rotate)
            )

        shift_h = aug_cfg.get("shift_h", 0)
        shift_v = aug_cfg.get("shift_v", 0)
        scale_h = aug_cfg.get("scale_h", 0)
        scale_v = aug_cfg.get("scale_v", 0)

        if any([shift_h, shift_v, scale_h, scale_v]):
            aug_tfms.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(shift_h, shift_v),
                    scale=(1 - min(scale_h, scale_v),
                           1 + max(scale_h, scale_v))
                )
            )

        brightness = aug_cfg.get("brightness", 0)
        contrast = aug_cfg.get("contrast", 0)
        saturation = aug_cfg.get("saturation", 0)

        if brightness or contrast or saturation:
            aug_tfms.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation
                )
            )

    # ---------- FINAL CONVERSIONS ----------
    input_tfms.append(transforms.ToTensor())

    if task == "segmentation":
        output_tfms.append(transforms.ToTensor())
    else:
        output_tfms.append(transforms.ToTensor())

    return (
        transforms.Compose(input_tfms),
        transforms.Compose(aug_tfms) if aug_tfms else transforms.Identity(),
        transforms.Compose(output_tfms)
    )

def get_dataset(
    dataset_id: int,
    project_id: int,
    preprocess_json_text: str,
    dataset_type: ds_templates.DatasetType
):
    datasets_resp = requests.get(f"http://datasets_manager:8004/datasets/download/id/{dataset_id}") # File request
    if datasets_resp.status_code != 200:
        raise ValueError("Failed to fetch dataset")

    dataset_info = requests.get(f"http://datasets_manager:8004/datasets/{dataset_id}").json()
    if datasets_resp.status_code != 200:
        raise ValueError("Failed to fetch dataset info")

    dataset_path = Path(DATASETS_ROOT) / f"project_{project_id}_dataset_{dataset_id}"
    dataset_path.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_path / dataset_info['filename']

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