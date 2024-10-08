import os
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms
import numpy as np
import yaml

# Function to load configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
config = load_config('config.yaml')

# Define COCO DataLoader with DistributedSampler and Subset handling data_percentage
def get_coco_dataloader(data_dir, batch_size, num_workers, rank, world_size, data_percentage=config['data_percentage']):
    # Define augmentations (apply to images before they are converted to tensors)
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.Resize((224, 224)),  # Resize the images to 224x224
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    coco_train = datasets.CocoDetection(
        root=os.path.join(data_dir, 'train2017'),
        annFile=os.path.join(data_dir, 'annotations', 'instances_train2017.json'),
        transform=augmentations  # Apply augmentations directly during data loading
    )

    # Create a subset to use only a specified percentage of the dataset
    dataset_size = len(coco_train)
    subset_size = int(data_percentage * dataset_size)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    subset_indices = indices[:subset_size]
    coco_train_subset = Subset(coco_train, subset_indices)

    # Use DistributedSampler for multi-GPU training
    if world_size > 1:
        train_sampler = DistributedSampler(coco_train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None  # No need for DistributedSampler in single-GPU mode

    train_loader = DataLoader(
        coco_train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))  # Collate function to handle variable-sized annotations
    )

    return train_loader, train_sampler