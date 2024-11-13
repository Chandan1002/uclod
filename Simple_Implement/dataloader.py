import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UnsupervisedDataset(Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.root = cfg['DATASETS']['ROOT']
        self.target_size = (608, 608)  # Fixed size as per RetinaNet requirements
        
        # Load COCO annotations
        if is_train:
            json_file = os.path.join(self.root, 'annotations/instances_train2017.json')
            self.image_dir = os.path.join(self.root, 'train2017')
        else:
            json_file = os.path.join(self.root, 'annotations/instances_val2017.json')
            self.image_dir = os.path.join(self.root, 'val2017')
            
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
            
        # Get image list
        self.images = self.coco['images']
        
        # Use only specified percentage of data
        if cfg['DATASETS']['PERCENTAGE'] < 100:
            num_images = int(len(self.images) * cfg['DATASETS']['PERCENTAGE'] / 100)
            self.images = self.images[:num_images]
        
        # Base transforms (normalization and resize)
        self.base_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg['MODEL']['PIXEL_MEAN'],
                std=cfg['MODEL']['PIXEL_STD']
            )
        ])
        
        # Augmentations as per paper
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.images)
        
    def random_crop(self, image):
        """Generate random crop as per Section 3.5 of the paper"""
        w, h = image.size
        max_dim = max(w, h)
        
        # Crop size between 10% and 25% of max dimension
        crop_size = random.uniform(0.10 * max_dim, 0.25 * max_dim)
        crop_size = int(crop_size)
        
        # Ensure crop fits within image bounds
        x = random.randint(0, max(0, w - crop_size))
        y = random.randint(0, max(0, h - crop_size))
        
        return x, y, crop_size
    
    def pad_image(self, image):
        """Pad image to square with black borders"""
        w, h = image.size
        max_dim = max(w, h)
        padded = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        # Paste original image in center
        x_offset = (max_dim - w) // 2
        y_offset = (max_dim - h) // 2
        padded.paste(image, (x_offset, y_offset))
        return padded, (x_offset, y_offset)
        
    def __getitem__(self, idx):
        # Load image
        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Pad image to square
        image, (x_offset, y_offset) = self.pad_image(image)
        
        # Apply augmentations to original image (x_j)
        x_j = self.augmentations(image)
        
        # Generate crop (x_i)
        x, y, crop_size = self.random_crop(image)
        x_i = image.crop((x, y, x + crop_size, y + crop_size))
        
        # Apply augmentations to crop
        x_i = self.augmentations(x_i)
        
        # Resize and convert both images to tensors
        x_i_tensor = self.base_transform(x_i)
        x_j_tensor = self.base_transform(x_j)
        
        # Adjust crop coordinates to account for padding and resize
        scale_w = self.target_size[0] / image.size[0]
        scale_h = self.target_size[1] / image.size[1]
        
        x = (x - x_offset) * scale_w if x >= x_offset else 0
        y = (y - y_offset) * scale_h if y >= y_offset else 0
        crop_size_w = crop_size * scale_w
        crop_size_h = crop_size * scale_h
        
        return {
            "image_original": x_j_tensor,
            "image_crop": x_i_tensor,
            "image_id": image_info['id'],
            "crop_coords": torch.tensor([y, x, crop_size_h, crop_size_w], dtype=torch.float32)
        }

def build_unsupervised_train_loader(cfg):
    """Build data loader for training"""
    dataset = UnsupervisedDataset(cfg, is_train=True)
    
    # Create sampler for distributed training
    sampler = None
    if cfg['DISTRIBUTED']['ENABLED']:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=cfg['SOLVER']['IMS_PER_BATCH'],
        shuffle=(sampler is None),
        num_workers=cfg['SYSTEM']['NUM_WORKERS'],
        sampler=sampler,
        pin_memory=True,
        drop_last=True
    )
    
    return data_loader