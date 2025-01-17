import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def save_crop_visualization(image, x=0, y=0, crop_size=0, output_path=""):
    """Save visualization of crop location on original image"""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    # Create figure and axes
    _, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # If crop coordinates provided, draw rectangle and save
    if crop_size != 0:

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x, y), crop_size, crop_size, linewidth=2, edgecolor="r", facecolor="none"
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add center point
        center_x = x + crop_size / 2
        center_y = y + crop_size / 2
        ax.plot(center_x, center_y, "r.", markersize=10)

    # Save the visualization
    plt.savefig(output_path)
    plt.close()


class UnsupervisedDataset(Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.root = cfg["DATASETS"]["ROOT"]
        self.target_size = (608, 608)  # Fixed size as per RetinaNet requirements

        # Load COCO annotations
        if is_train:
            json_file = os.path.join(self.root, "annotations/instances_train2017.json")
            self.image_dir = os.path.join(self.root, "train2017")
        else:
            json_file = os.path.join(self.root, "annotations/instances_val2017.json")
            self.image_dir = os.path.join(self.root, "val2017")

        with open(json_file, "r") as f:
            self.coco = json.load(f)

        # Get image list
        self.images = self.coco["images"]

        # Use only specified percentage of data
        if cfg["DATASETS"]["PERCENTAGE"] < 100:
            num_images = int(len(self.images) * cfg["DATASETS"]["PERCENTAGE"] / 100)
            self.images = self.images[:num_images]

        # Base transforms (normalization and resize)
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg["MODEL"]["PIXEL_MEAN"], std=cfg["MODEL"]["PIXEL_STD"]
                ),
            ]
        )

        # Augmentations as per paper
        self.augmentations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.images)

    def random_crop(self, image):
        """Generate random crop coordinates ensuring crop is within original image bounds"""
        w, h = image.size

        # Crop size between 10% and 25% of max dimension
        max_dim = max(w, h)
        crop_size = int(random.uniform(0.10 * max_dim, 0.25 * max_dim))

        # Ensure crop fits within actual image bounds (not padded region)
        max_x = w - crop_size
        max_y = h - crop_size

        # If image is too small, adjust crop size
        if max_x < 0 or max_y < 0:
            crop_size = min(w, h)
            max_x = w - crop_size
            max_y = h - crop_size

        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))

        return x, y, crop_size

    # TODO: Crop beforepadding. Pad only to the right and at the bottom. Done!!
    def pad_image(self, image):
        """Pad image only to the right and bottom to make it square"""
        w, h = image.size
        max_dim = max(w, h)
        padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        # Paste original image at top-left corner
        padded.paste(image, (0, 0))
        return padded, (0, 0)  # x_offset and y_offset are always 0

    def __getitem__(self, idx):
        # Load image
        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        original_image = Image.open(image_path).convert("RGB")

        # First get crop from original unpadded image
        x, y, crop_size = self.random_crop(original_image)
        # x_i = original_image.crop((x, y, x + crop_size, y + crop_size))
        cropped_image = original_image.crop((x, y, x + crop_size, y + crop_size))

        # Then pad the original image (not the crop)
        padded_image, _ = self.pad_image(original_image)

        # Generate random flip flags
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5

        # Apply flips to padded image and update coordinates
        if do_hflip:
            padded_image = padded_image.transpose(Image.FLIP_LEFT_RIGHT)
            cropped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)
            W, _ = padded_image.size
            x = W - x - crop_size

        if do_vflip:
            padded_image = padded_image.transpose(Image.FLIP_TOP_BOTTOM)
            cropped_image = cropped_image.transpose(Image.FLIP_TOP_BOTTOM)
            _, H = padded_image.size
            y = H - y - crop_size

        # Apply augmentations
        # x_j = self.augmentations(padded_image)  # augment padded full image
        # # x_i = self.augmentations(x_i)           # augment unpadded crop
        # x_i = self.augmentations(cropped_image)           # augment unpadded crop

        # Save visualization for first 100 images only
        if idx < 100:
            viz_dir = os.path.join(self.cfg["OUTPUT"]["DIR"], "crop_visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # save padded image visualization
            viz_path = os.path.join(viz_dir, f"crop_viz_{idx}.png")
            save_crop_visualization(padded_image, x, y, crop_size, viz_path)

            # save crop image visualization
            viz_path = os.path.join(viz_dir, f"crop_viz_{idx}_small.png")
            save_crop_visualization(cropped_image, output_path=viz_path)

        # Resize and convert both images to tensors
        x_i_tensor = self.base_transform(cropped_image)
        x_j_tensor = self.base_transform(padded_image)

        # # Adjust crop coordinates for resize
        # scale_w = self.target_size[0] / padded_image.size[0]
        # scale_h = self.target_size[1] / padded_image.size[1]

        # # Adjust crop coordinates for resize
        # scale_w = self.target_size[0] / original_image.size[0]
        # scale_h = sel
        # f.target_size[1] / original_image.sze[1]

        # Get padded dimensions for proper scaling
        padded_w, padded_h = padded_image.size

        # Adjust crop coordinates for resize using padded dimensions
        scale_w = self.target_size[0] / padded_w
        scale_h = self.target_size[1] / padded_h

        x = x * scale_w
        y = y * scale_h
        crop_size_w = crop_size * scale_w
        crop_size_h = crop_size * scale_h

        return {
            "image_original": x_j_tensor,
            "image_crop": x_i_tensor,
            "image_id": image_info["id"],
            "crop_coords": torch.tensor(
                [y, x, crop_size_h, crop_size_w], dtype=torch.float32
            ),
        }


def build_unsupervised_train_loader(cfg):
    """Build data loader for training"""
    dataset = UnsupervisedDataset(cfg, is_train=True)

    # Create sampler for distributed training
    sampler = None
    if cfg["DISTRIBUTED"]["ENABLED"]:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["SOLVER"]["IMS_PER_BATCH"],
        shuffle=(sampler is None),
        num_workers=cfg["SYSTEM"]["NUM_WORKERS"],
        # num_workers=0,
        sampler=sampler,
        pin_memory=True,
        # persistent_workers=True,
        drop_last=True,
    )

    return data_loader


class PrepareImage(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, image):
        image[:, :, 0] = (image[:, :, 0] * self.std[0]) + self.mean[0]
        image[:, :, 1] = (image[:, :, 1] * self.std[1]) + self.mean[1]
        image[:, :, 2] = (image[:, :, 2] * self.std[2]) + self.mean[2]

        return image


def save_image(
    image,
    anchors=None,
    name="test",
    color=(0, 0, 0),
    save=True,
    normalize=True,
    thickness=1,
):
    """
    args: image with shape h, w, c
    annot with shape of N, 4
    name for the saved image
    """

    if normalize:
        image = PrepareImage()(image) * 255
        image = np.uint8(image)

    if anchors is not None:
        for ann in anchors:
            y_min = int(ann[0])
            x_min = int(ann[1])
            y_max = y_min + int(ann[2])
            x_max = x_min + int(ann[3])

            image = cv2.rectangle(
                image, (x_min, y_min), (x_max, y_max), color, thickness
            )

    if save:
        im = Image.fromarray(image)
        im.save(f"{name}.png")

    return image
