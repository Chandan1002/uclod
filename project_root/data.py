import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from torch.utils.data.distributed import DistributedSampler
import logging


class RandomCropGenerator:
    """Generates random crops following paper specifications"""
    def __init__(self, min_scale: float = 0.08, max_scale: float = 0.25):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
        W, H = image.size
        max_dim = max(W, H)
        
        # Generate crop size according to paper's section 3.5
        crop_size = int(random.uniform(
            self.min_scale * max_dim,
            self.max_scale * max_dim
        ))
        
        # Ensure crop stays within image bounds
        x = random.randint(0, W - crop_size)
        y = random.randint(0, H - crop_size)
        
        # Extract crop
        crop = image.crop((x, y, x + crop_size, y + crop_size))
        center = (x + crop_size//2, y + crop_size//2)
        
        return crop, center

def get_coco_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    transform_config: Optional[Dict] = None,
    distributed: bool = False,
    training: bool = True
) -> DataLoader:
    """
    Create COCO data loader
    """
    # Set default transform config if none provided
    if transform_config is None:
        transform_config = {}

    # Setup paths
    split = 'train2017' if training else 'val2017'
    img_dir = os.path.join(data_dir, split)
    ann_file = os.path.join(data_dir, 'annotations', f'instances_{split}.json')
    
    # Verify paths exist
    if not os.path.exists(img_dir):
        raise ValueError(f"Images directory not found: {img_dir}")
    if not os.path.exists(ann_file):
        raise ValueError(f"Annotation file not found: {ann_file}")
    
    # Create dataset
    dataset = ContrastiveDataset(
        root_dir=img_dir,
        annotation_file=ann_file,
        transform_config=transform_config,
        training=training
    )
    
    # Setup sampler for distributed training
    sampler = None
    if distributed and training:
        sampler = DistributedSampler(dataset)
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and training),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=training
    )
    
    return loader

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with random crops"""
    def __init__(self, 
                 root_dir: str,
                 annotation_file: str,
                 transform_config: dict,
                 training: bool = True):
        """
        Args:
            root_dir: Path to images directory
            annotation_file: Path to COCO annotation file
            transform_config: Configuration for data augmentation
            training: Whether this is training set
        """
        self.coco = CocoDetection(root_dir, annotation_file)
        self.training = training
        self.crop_size = transform_config.get('crop_size', 224)
        
        # Setup transformations
        if training:
            self.transform = T.Compose([
                T.RandomResizedCrop(self.crop_size),
                T.RandomHorizontalFlip(p=transform_config.get('flip_prob', 0.5)),
                T.ColorJitter(
                    brightness=transform_config.get('color_jitter', {}).get('brightness', 0.4),
                    contrast=transform_config.get('color_jitter', {}).get('contrast', 0.4),
                    saturation=transform_config.get('color_jitter', {}).get('saturation', 0.4),
                    hue=transform_config.get('color_jitter', {}).get('hue', 0.1)
                ),
                T.RandomGrayscale(p=transform_config.get('grayscale_probability', 0.2)),
                T.RandomApply([
                    T.GaussianBlur(
                        kernel_size=(3, 3),
                        sigma=(0.1, 2.0)
                    )
                ], p=transform_config.get('blur_probability', 0.5)),
                T.ToTensor(),
                T.Normalize(
                    mean=transform_config.get('normalize', {}).get('mean', [0.485, 0.456, 0.406]),
                    std=transform_config.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
                )
            ])
        else:
            self.transform = T.Compose([
                T.Resize(self.crop_size),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize(
                    mean=transform_config.get('normalize', {}).get('mean', [0.485, 0.456, 0.406]),
                    std=transform_config.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
                )
            ])

    def __len__(self) -> int:
        return len(self.coco)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, _ = self.coco[idx]
        
        if self.training:
            # Transform the image twice to get two different views
            img_tensor1 = self.transform(img)
            img_tensor2 = self.transform(img)
            
            # Use the center of the image as the reference point
            center = torch.tensor([self.crop_size // 2, self.crop_size // 2])
            
            return img_tensor1, img_tensor2, center
        else:
            # For validation, just return the transformed image
            return self.transform(img), None, None

def custom_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function to handle the batch"""
    images, crops, centers = zip(*batch)
    
    # Images will always be present
    images = torch.stack(images)
    
    # Handle crops and centers which might be None during validation
    if all(crop is not None for crop in crops):
        crops = torch.stack(crops)
    else:
        crops = None
    
    if all(center is not None for center in centers):
        centers = torch.stack(centers)
    else:
        centers = None
    
    return images, crops, centers